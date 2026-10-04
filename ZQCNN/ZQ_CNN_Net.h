#ifndef _ZQ_CNN_NET_H_
#define _ZQ_CNN_NET_H_
#pragma once
#include "ZQ_CNN_Layer.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
namespace ZQ
{
	class ZQ_CNN_Net
	{
	protected:
		class Buffer
		{
		public:
			void* data;
			__int64 len;

			Buffer() : data(0), len(0) {}
			~Buffer() { Release(); }
			void Release() { if (data) _aligned_free(data); data = 0; len = 0; }
		};

	public:
		// 初始化列表按**声明顺序**排（审计 2026-10-02，附录 AU.2）：原来把
		// has_innerproduct_layer 写在 ignore_small_value 之前，而它声明在
		// _buffer 之后、ignore_small_value 之后。实际初始化按声明顺序走，
		// 今天的数值不受影响（全是非 0 常量）。
		ZQ_CNN_Net() :has_input_layer(false),show_debug_info(false),use_buffer(true),
			ignore_small_value(0),has_innerproduct_layer(false),
			input_C(0),input_H(0),input_W(0) {}
		~ZQ_CNN_Net() { _clear(); };

	private:
		std::vector<ZQ_CNN_Layer*> layers;
		std::vector<std::string> layer_type_names;
		std::vector<ZQ_CNN_Tensor4D*> blobs; //blobs[0] stores a pointer to input blob
		std::map<std::string, int> map_name_to_layer_idx;
		std::map<std::string, int> map_name_to_blob_idx; 
		std::map<int, int> simplify_inplace_blob_map;
		std::vector<std::vector<int> > bottoms;
		std::vector<std::vector<int> > tops;	//tops[0][0] stores input blob pointer
		std::string input_name;
		bool has_input_layer;
		bool show_debug_info;
		bool use_buffer;
		float ignore_small_value;
		Buffer _buffer;
		bool has_innerproduct_layer;
		// 审计 2026-10-02（附录 AU.3）：input_C/H/W 原来既不在初始化列表里、
		// 也只在 LoadModel 里的 `cur_layer->GetTopDim(input_C, input_H, input_W)`
		// 才被赋值 —— 构造到那之间是未初始化窗口。GetInputDim() 在那之前调用
		// 会返回栈上的垃圾。补 =0 零风险（LoadModel 一定会覆盖）。
		int input_C, input_H, input_W;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		void TurnOnUseBuffer() { use_buffer = true; }
		void TurnOffUseBuffer() { use_buffer = false; }
		void GetInputDim(int& in_C, int& in_H, int& in_W)const { in_C = input_C; in_H = input_H; in_W = input_W; }
		bool LoadFrom(const std::string& param_file, const std::string& model_file, bool merge_bn = false, float ignore_small_value = 1e-12,
			bool merge_prelu = false)
		{
			_clear();
			this->ignore_small_value = ignore_small_value;
			if (!_load_param_file(param_file))
			{
				_clear();
				return false;
			}
			if (!_check_connect())
			{
				_clear();
				return false;
			}
			if (!_load_model_file(model_file))
			{
				_clear();
				return false;
			}

			_simplify_inplace();

			if (merge_bn)
			{
				if (!_merge_bn())
					return false;
			}
			if (merge_prelu)
			{
				if (!_merge_prelu())
					return false;
			}
			return true;
		}

		bool SaveModel(const std::string& file) const
		{
			return _save_model_file(file);
		}

		bool SwapInputRGBandBGR(const std::vector<std::string>& layer_names)
		{
			return _swap_input_RGB_and_BGR(layer_names);
		}

		bool LoadFromBuffer(const char*& param_buffer, __int64 param_buffer_len, const char*& model_buffer, __int64 model_buffer_len, 
			bool merge_bn = false, float ignore_small_value = 1e-12, bool merge_prelu = false)
		{
			_clear();
			this->ignore_small_value = ignore_small_value;
			if (!_load_param_from_buffer(param_buffer, param_buffer_len))
			{
				_clear();
				return false;
			}
			if (!_check_connect())
			{
				_clear();
				return false;
			}
			if (!_load_model_from_buffer(model_buffer, model_buffer_len))
			{
				_clear();
				return false;
			}

			_simplify_inplace();

			if (merge_bn)
			{
				if (!_merge_bn())
					return false;
			}
			if (merge_prelu)
			{
				if (!_merge_prelu())
					return false;
			}
			return true;
		}

		__int64 GetNumOfMulAdd() const
		{
			__int64 sum = 0;
			for (int i = 0; i < layers.size(); i++)
				sum += layers[i]->GetNumOfMulAdd();
			return sum;
		}

		__int64 GetNumOfMulAddConv() const
		{
			__int64 sum = 0;
			for (int i = 0; i < layers.size(); i++)
			{
				if(ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(),"Convolution") == 0
					|| ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "InnerProduct") == 0)
					sum += layers[i]->GetNumOfMulAdd();
			}
			return sum;
		}
		__int64 GetNumOfMulAddDwConv() const
		{
			__int64 sum = 0;
			for (int i = 0; i < layers.size(); i++)
			{
				if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "DepthwiseConvolution") == 0)
					sum += layers[i]->GetNumOfMulAdd();
			}
			return sum;
		}

		float GetLastTimeOfLayerType(const std::string& layer_typename) const
		{
			float sum = 0;
			for (int i = 0; i < layers.size(); i++)
			{
				if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), layer_typename.c_str()) == 0)
					sum += layers[i]->last_cost_time;
			}
			return sum;
		}


		/*it may change input in case of padding, but the data will not be lost*/
		bool Forward(ZQ_CNN_Tensor4D& input)
		{
			if (map_name_to_blob_idx.size() == 0 || map_name_to_layer_idx.size() == 0 || tops.size() == 0)
				return false;
			if (has_innerproduct_layer)
			{
				if (input.GetH() != input_H || input.GetW() != input_W || input.GetC() != input_C)
				{
					std::cout << "The dimenson doesnot match with the needed\n";
					return false;
				}
			}
			blobs[0] = &input;
			
			double sum_cost = 0;
			for (int i = 0; i < layers.size(); i++)
			{
				std::vector<ZQ_CNN_Tensor4D*> bottom_ptrs, top_ptrs;
				for (int j = 0; j < bottoms[i].size(); j++)
					bottom_ptrs.push_back(blobs[bottoms[i][j]]);
				for (int j = 0; j < tops[i].size(); j++)
					top_ptrs.push_back(blobs[tops[i][j]]);

				layers[i]->show_debug_info = show_debug_info;
				layers[i]->use_buffer = use_buffer;
				layers[i]->buffer = &(_buffer.data);
				layers[i]->buffer_len = &(_buffer.len);
				double t1 = omp_get_wtime();
				if (!layers[i]->Forward(&bottom_ptrs, &top_ptrs))
				{
					blobs[0] = 0;
					tops[0][0] = 0;
					printf("failed to run layer: %s\n", layers[i]->name.c_str());
					return false;
				}
				double t2 = omp_get_wtime();
				sum_cost += 1000 * (t2 - t1);
				//printf("%.3f\n", 1000 * (t2 - t1));
				/*printf("%d\n", i);
				if (i == 69)
				{
					printf("here\n");
				}*/
//				char buf[100];
//#if defined(_WIN32)
//				sprintf_s(buf, "NHWC_%d.txt", i);
//#else
//				sprintf(buf, "NHWC_%d.txt", i);
//#endif
//				top_ptrs[0]->SaveToFile(buf);
			}
			//printf("sum=%.3f\n", sum_cost);
			blobs[0] = 0;
			tops[0][0] = 0;
			return true;
		}

		bool Forward(ZQ_CNN_Tensor4D& input, const std::string& start_layer_name, const std::string& end_layer_name)
		{
			if (map_name_to_blob_idx.size() == 0 || map_name_to_layer_idx.size() == 0 || tops.size() == 0)
				return false;
			if (has_innerproduct_layer)
			{
				if (input.GetH() != input_H || input.GetW() != input_W || input.GetC() != input_C)
				{
					std::cout << "The dimenson doesnot match with the needed\n";
					return false;
				}
			}
			blobs[0] = &input;

			bool has_begin = false, has_end = false;
			for (int i = 0; i < layers.size(); i++)
			{
				if (ZQ_CNN_Layer::_my_strcmpi(layers[i]->name.c_str(), start_layer_name.c_str()) == 0)
					has_begin = true;
				if (!has_begin)
					continue;
				std::vector<ZQ_CNN_Tensor4D*> bottom_ptrs, top_ptrs;
				for (int j = 0; j < bottoms[i].size(); j++)
					bottom_ptrs.push_back(blobs[bottoms[i][j]]);
				for (int j = 0; j < tops[i].size(); j++)
					top_ptrs.push_back(blobs[tops[i][j]]);

				layers[i]->show_debug_info = show_debug_info;
				//printf("%d\n", i);
				layers[i]->use_buffer = use_buffer;
				layers[i]->buffer = &(_buffer.data);
				layers[i]->buffer_len = &(_buffer.len);
				if (!layers[i]->Forward(&bottom_ptrs, &top_ptrs))
				{
					blobs[0] = 0;
					tops[0][0] = 0;
					printf("failed to run layer: %s\n", layers[i]->name.c_str());
					return false;
				}
				if (ZQ_CNN_Layer::_my_strcmpi(layers[i]->name.c_str(), end_layer_name.c_str()) == 0)
					has_end = true;
				if (has_end)
					break;
			}

			blobs[0] = 0;
			tops[0][0] = 0;
			return true;
		}

		const ZQ_CNN_Tensor4D* GetBlobByName(std::string name) 
		{
			std::map<std::string, int>::iterator it = map_name_to_blob_idx.find(name);
			if (it == map_name_to_blob_idx.end())
				return 0;
			else
			{
				if (simplify_inplace_blob_map.find(it->second) == simplify_inplace_blob_map.end())
					return blobs[it->second];
				else
					return blobs[simplify_inplace_blob_map[it->second]];
			}
		}

	private:
		void _clear()
		{
			for (int i = 0; i < layers.size(); i++)
			{
				if (layers[i])
					delete layers[i];
			}
			layers.clear();
			layer_type_names.clear();
			map_name_to_layer_idx.clear();
			//blob[0] is a pointer to input blob, DONT FREE
			for (int i = 1; i < blobs.size(); i++)
			{
				if (blobs[i])
					delete blobs[i];
			}
			blobs.clear();
			map_name_to_layer_idx.clear();
			has_input_layer = false;
		}

		bool _getline(std::fstream& fin, const char*& buffer, __int64& buffer_len, std::string& line)
		{
			if (buffer == 0)
			{
				if (!fin.is_open() || fin.eof())
					return false;
				std::getline(fin, line);
				return true;
			}
			else
			{
				if (buffer_len <= 0)
					return false;
				__int64 i = 0;
				for (; i < buffer_len; i++)
				{
					if (buffer[i] != '\n' && buffer[i] != '\r')
						break;
				}
				if (i == buffer_len)
					return false;

				__int64 j = i;
				for (; j < buffer_len; j++)
				{
					if (buffer[j] == '\n' || buffer[j] == '\r')
						break;
				}

				if (j == i)
					return false;
				line.clear();
				line.append(buffer + i, j - i);
				// 指针要推到行尾符所在位置（下一轮开头那个跳过 \n\r 的循环会吃掉它），
				// 而不是只推进 (j - i)。原来推进 cur_len 会少推进开头被跳过的那些
				// 行尾符，于是 buffer 停在**上一行正文最后一个字符**上，下一轮
				// 把它当成一行读出来 —— 每读一行就多产出一行 1 个字符的垃圾。
				// 以前层类型分发链没有 else 兜底，这些垃圾行被静默丢掉，所以看不出来。
				buffer += j;
				buffer_len -= j;
				return true;
			}
		}

		bool _load_param_file(const std::string& file)
		{
			std::fstream fin(file, std::ios::in);
			if (!fin.is_open())
			{
				std::cout << "failed to open file " << file << "\n";
				return false;
			}
			return _load_param_from_file_or_buffer(fin, NULL, 0);
		}

		bool _load_param_from_buffer(const char* buffer, __int64 buffer_len)
		{
			std::fstream fin;
			return _load_param_from_file_or_buffer(fin, buffer, buffer_len);
		}

		bool _load_param_from_file_or_buffer(std::fstream& fin, const char* buffer, __int64 buffer_len)
		{	
			std::string line;
			int buf_len = 2000;
			std::vector<char> buf(buf_len+1);
			while (_getline(fin, buffer, buffer_len, line))
			{
				buf[0] = '\0';
				// sscanf 对空行返回 EOF(-1) 而不是 0，所以这里必须判 != 1；
				// 只判 == 0 的话空行会一路穿到下面的层类型分发里去。
#if defined(_WIN32)
				if (sscanf_s(line.c_str(), "%s", &buf[0], buf_len) != 1)
					continue;
#else
				if (sscanf(line.c_str(), "%2000s", &buf[0]) != 1)
					continue;
#endif
				// zqparams 里用 '#' 注释掉整层（仓库自带的几份权重都这么用）。
				// 这行必须在下面那条"未知层类型就报错"的兜底之前判掉，否则
				// 每一个被注释掉的层都会让整个模型加载失败。
				if (buf[0] == '#')
					continue;
				if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Convolution") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Convolution();
					if (cur_layer == 0) {
						std::cout << "failed to create a Convolution layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Convolution");
				}
				else if(ZQ_CNN_Layer::_my_strcmpi(&buf[0], "DeConvolution") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_DeConvolution();
					if (cur_layer == 0) {
						std::cout << "failed to create a DeConvolution layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("DeConvolution");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "DepthwiseConvolution") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_DepthwiseConvolution();
					if (cur_layer == 0) {
						std::cout << "failed to create a DepthwiseConvolution layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("DepthwiseConvolution");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "BatchNormScale") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_BatchNormScale();
					if (cur_layer == 0) {
						std::cout << "failed to create a BatchNorm layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("BatchNormScale");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "BatchNorm") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_BatchNorm();
					if (cur_layer == 0) {
						std::cout << "failed to create a BatchNorm layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("BatchNorm");
				}	
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Scale") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Scale();
					if (cur_layer == 0) {
						std::cout << "failed to create a Scale layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Scale");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "AddBias") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_AddBias();
					if (cur_layer == 0) {
						std::cout << "failed to create a AddBias layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("AddBias");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "PReLU") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_PReLU();
					if (cur_layer == 0) {
						std::cout << "failed to create a PReLU layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("PReLU");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "ReLU") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_ReLU();
					if (cur_layer == 0) {
						std::cout << "failed to create a ReLU layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("ReLU");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "ReLU6") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_ReLU6();
					if (cur_layer == 0) {
						std::cout << "failed to create a ReLU6 layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("ReLU6");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Softmax") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Softmax();
					if (cur_layer == 0) {
						std::cout << "failed to create a Softmax layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Softmax");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Pooling") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Pooling();
					if (cur_layer == 0) {
						std::cout << "failed to create a Pooling layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Pooling");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Copy") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Copy();
					if (cur_layer == 0) {
						std::cout << "failed to create a Dropout layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Copy");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Dropout") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Dropout();
					if (cur_layer == 0) {
						std::cout << "failed to create a Dropout layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Dropout");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "InnerProduct") == 0)
				{
					has_innerproduct_layer = true;
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_InnerProduct();
					if (cur_layer == 0) {
						std::cout << "failed to create a InnerProduct layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("InnerProduct");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "LSTM_TF") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}

					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_LSTM_TF();
					if (cur_layer == 0) {
						std::cout << "failed to create a LSTM_TF layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("LSTM_TF");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Eltwise") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Eltwise();
					if (cur_layer == 0) {
						std::cout << "failed to create a Eltwise layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Eltwise");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "UpSampling") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_UpSampling();
					if (cur_layer == 0) {
						std::cout << "failed to create a UpSampling layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("UpSampling");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "ScalarOperation") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_ScalarOperation();
					if (cur_layer == 0) {
						std::cout << "failed to create a ScalarOperation layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("ScalarOperation");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "UnaryOperation") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_UnaryOperation();
					if (cur_layer == 0) {
						std::cout << "failed to create a UnaryOperation layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("UnaryOperation");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Sqrt") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Sqrt();
					if (cur_layer == 0) {
						std::cout << "failed to create a Sqrt layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Sqrt");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Tile") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Tile();
					if (cur_layer == 0) {
						std::cout << "failed to create a Tile layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Tile");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Reduction") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Reduction();
					if (cur_layer == 0) {
						std::cout << "failed to create a Reduction layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Reduction");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "LRN") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_LRN();
					if (cur_layer == 0) {
						std::cout << "failed to create a LRN layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("LRN");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Normalize") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Normalize();
					if (cur_layer == 0) {
						std::cout << "failed to create a LRN layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Normalize");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Permute") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Permute();
					if (cur_layer == 0) {
						std::cout << "failed to create a Permute layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Permute");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Flatten") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Flatten();
					if (cur_layer == 0) {
						std::cout << "failed to create a Flatten layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Flatten");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Reshape") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Reshape();
					if (cur_layer == 0) {
						std::cout << "failed to create a Reshape layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Reshape");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Squeeze") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Squeeze();
					if (cur_layer == 0) {
						std::cout << "failed to create a Squeeze layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Squeeze");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "PriorBox") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_PriorBox();
					if (cur_layer == 0) {
						std::cout << "failed to create a PriorBox layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("PriorBox");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "PriorBoxText") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_PriorBoxText();
					if (cur_layer == 0) {
						std::cout << "failed to create a PriorBoxText layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("PriorBoxText");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "PriorBox_MXNET") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_PriorBox_MXNET();
					if (cur_layer == 0) {
						std::cout << "failed to create a PriorBox_MXNET layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("PriorBox_MXNET");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Concat") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Concat();
					if (cur_layer == 0) {
						std::cout << "failed to create a Concat layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("Concat");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "DetectionOutput") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_DetectionOutput();
					if (cur_layer == 0) {
						std::cout << "failed to create a DetectionOutput layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("DetectionOutput");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "DetectionOutput_MXNET") == 0)
				{
					if (layers.size() == 0)
					{
						std::cout << "Input layer must be the first!\n";
						return false;
					}
					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_DetectionOutput_MXNET();
					if (cur_layer == 0) {
						std::cout << "failed to create a DetectionOutput_MXNET layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line, false))
					{
						delete cur_layer;
						return false;
					}
					layer_type_names.push_back("DetectionOutput_MXNET");
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(&buf[0], "Input") == 0)
				{
					if (has_input_layer)
					{
						printf("Already have input layer\n");
						return false;
					}

					ZQ_CNN_Layer* cur_layer = new ZQ_CNN_Layer_Input();
					if (cur_layer == 0) {
						std::cout << "failed to create a Input layer!\n";
						return false;
					}
					if (!_add_layer_and_blobs(cur_layer, line,true))
					{
						delete cur_layer;
						return false;
					}
					cur_layer->GetTopDim(input_C, input_H, input_W);
					layer_type_names.push_back("Input");
				}
				else
				{
					// 链上最后一个分支是 Input, 之前没有 else 兜底: 任何不认识的
					// 首 token 整行被静默丢弃, 后面只会在 _check_connect 报一句
					// 误导性的 "unknown blob", 甚至整网少几层也不报错。
					printf("unknown layer type: %s\n", &buf[0]);
					printf("  in line: %s\n", line.c_str());
					return false;
				}
				line = "";
				
			}
			return true;
		}

		bool _load_model_file(const std::string& file)
		{
			int layer_num = layers.size();
			if (layers.size() == 0)
				return false;
			FILE* in = 0;
#if defined(_WIN32)
			fopen_s(&in, file.c_str(), "rb");
#else
			in = fopen(file.c_str(), "rb");
#endif
			if (in == 0)
			{
				std::cout << "failed to open " << file << "\n";
				return false;
			}
			for (int i = 0; i < layer_num; i++)
			{
				if (!layers[i]->LoadBinary_NCHW(in))
				{
					fclose(in);
					std::cout << "Failed to load Binary for layer " << layers[i]->name << "\n";
					return false;
				}
			}
			// 尾部**多余**的字节（2026-10-04 加，附录 GZ.2）。
			//
			// `LoadFrom(param_file, model_file)` 走的是**这条**路径，而它
			// **没有任何字节记账** —— `LoadBinary_NCHW(in)` 只给 FILE*，
			// 不告诉你读了多少。所以这里用 `ftell` 拿当前位置，跟文件大小比。
			//
			// 另一条路径 `_load_model_from_buffer`（`LoadFromBuffer` 用的）
			// 有 `readed_len_in_bytes`，同一个检查已经加在那里。
			// 两条都要加：2026-10-04 我只改了有记账的那一条，
			// 结果**编译通过、测试全过、实际一点作用都没有** ——
			// 走的是这条。见 GZ.3。
			{
				long long consumed = ftell(in);
				long long total = 0;
				if (fseek(in, 0, SEEK_END) == 0)
					total = ftell(in);
				if (consumed >= 0 && total > consumed)
				{
					std::cout << "warning: " << (total - consumed)
						<< " bytes left in the weight file after loading "
						<< layer_num << " layers" << std::endl;
				}
			}
			fclose(in);
			return true;
		}

		bool _save_model_file(const std::string& file) const
		{
			int layer_num = layers.size();
			if (layers.size() == 0)
				return false;
			FILE* out = 0;
#if defined(_WIN32)
			fopen_s(&out, file.c_str(), "wb");
#else
			out = fopen(file.c_str(), "wb");
#endif
			if (out == 0)
			{
				std::cout << "failed to create " << file << "\n";
				return false;
			}
			for (int i = 0; i < layer_num; i++)
			{
				if (!layers[i]->SaveBinary_NCHW(out))
				{
					fclose(out);
					std::cout << "Failed to save Binary for layer " << layers[i]->name << "\n";
					return false;
				}
			}
			fclose(out);
			return true;
		}

		bool _load_model_from_buffer(const char* model_buffer, __int64 model_buffer_len)
		{
			int layer_num = layers.size();
			if (layers.size() == 0)
				return false;
			
			for (int i = 0; i < layer_num; i++)
			{
				__int64 readed_len_in_bytes = 0;
				if (!layers[i]->LoadBinary_NCHW(model_buffer, model_buffer_len, readed_len_in_bytes))
				{
					std::cout << "Failed to load Binary for layer " << layers[i]->name << "\n";
					return false;
				}
				model_buffer += readed_len_in_bytes;
				model_buffer_len -= readed_len_in_bytes;
			}

			// 尾部**多余**的字节（2026-10-04 加，附录 GZ.2）。
			//
			// 上面只查了"字节**不够**"（`LoadBinary_NCHW` 失败），
			// **没查"字节太多"** —— 而 `model_buffer_len` 是**按值传**进来的
			// 局部副本，循环一结束就只能丢掉，所以"到底消费了多少"从未被看见过。
			// 2026-10-04 实测：给 `det1-dw20-fast.nchwbin` 尾部追加
			// 1 / 4 / 4096 / **65536** 字节（原文件大小的 10 倍）的垃圾，
			// `LoadFrom` **四次全部返回 true**，一声不吭。
			//
			// 为什么这要紧：权重文件被**截断**会**报错**（离原因近），
			// 而被**拼接/被追加**会**照常加载**、只是后面几层的权重读到了
			// 偏移的地方 —— 生产里的表现是"精度慢慢掉了"，不是"跑不起来"。
			//
			// 这里打的是 `warning:` 而不是直接失败：随仓 23 个模型的
			// "连权重一起加载"门禁（附录 GH）把**任何 warning 行都当失败**，
			// 所以一旦真有模型的权重尾部有多余数据，那道门禁会立刻指出来 ——
			// 而不必等到生产环境里精度掉了才发现。
			if (model_buffer_len != 0)
			{
				std::cout << "warning: " << model_buffer_len
					<< " bytes left in the weight file after loading "
					<< layer_num << " layers" << std::endl;
			}
			return true;
		}

		bool _add_layer_and_blobs(ZQ_CNN_Layer* cur_layer, const std::string& line, bool is_input_layer)
		{
			if (!cur_layer->ReadParam(line))
			{
				return false;
			}
			cur_layer->ignore_small_value = this->ignore_small_value;
			std::string layer_name = cur_layer->name;
			if (is_input_layer)
			{
				tops.resize(1);
				bottoms.resize(1);
				tops[0].push_back(0);
				map_name_to_blob_idx[cur_layer->top_names[0]] = 0;
				blobs.push_back(0);
				has_input_layer = true;
				layers.push_back(cur_layer);
				map_name_to_layer_idx[layer_name] = layers.size() - 1;
			}
			else
			{
				if (map_name_to_layer_idx.find(layer_name) == map_name_to_layer_idx.end())
				{
					std::vector<int> bottom_idx, top_idx;
					std::vector<std::string>& cur_bottom_names = cur_layer->bottom_names;
					for (int i = 0; i < cur_bottom_names.size(); i++)
					{
						std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(cur_bottom_names[i]);
						if (name_it == map_name_to_blob_idx.end())
						{
							int idx = blobs.size();
#if __ARM_NEON
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
#else
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align0();
#endif
#endif //__ARM_NEON
							if (blob == 0)
							{
								std::cout << "failed to allocate a ZQ_CNN_Tensor4D\n";
								return false;
							}
							blobs.push_back(blob);
							bottom_idx.push_back(blobs.size()-1);
							map_name_to_blob_idx[cur_bottom_names[i]] = idx;
						}
						else
						{
							bottom_idx.push_back(name_it->second);
						}
					}
					std::vector<std::string>& cur_top_names = cur_layer->top_names;
					for (int i = 0; i < cur_top_names.size(); i++)
					{
						std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(cur_top_names[i]);
						if (name_it == map_name_to_blob_idx.end())
						{
							int idx = blobs.size();
#if __ARM_NEON
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
#else
							ZQ_CNN_Tensor4D* blob = new ZQ_CNN_Tensor4D_NHW_C_Align0();
#endif
#endif //__ARM_NEON
							if (blob == 0)
							{
								std::cout << "failed to allocate a ZQ_CNN_Tensor4D\n";
								return false;
							}
							blobs.push_back(blob);
							top_idx.push_back(blobs.size()-1);
							map_name_to_blob_idx[cur_top_names[i]] = idx;
						}
						else
						{
							top_idx.push_back(name_it->second);
						}
					}
					bottoms.push_back(bottom_idx);
					tops.push_back(top_idx);
					layers.push_back(cur_layer);
					map_name_to_layer_idx[layer_name] = layers.size() - 1;
				}
				else
				{
					std::cout << "There's already a layer named " << layer_name << "!\n";
					return false;
				}
			}
			return true;
		}

		// 逐元素、可以安全地 bottom/top 同名的层。Convolution/Pooling/Reshape
		// 这些会改形状的层不在名单里 —— 它们的 LayerSetup 会把 tops[0] 重排成
		// 输出形状，一旦 tops[0] 就是 bottoms[0]，等于把自己的输入就地毁掉。
		// 名单与 _simplify_inplace() 保持一致，改一处必须改两处。
		bool _is_inplace_safe(int i) const
		{
			const char* t = layer_type_names[i].c_str();
			return ZQ_CNN_Layer::_my_strcmpi(t, "ReLU") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "ReLU6") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "PReLU") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "BatchNormScale") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "BatchNorm") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "Scale") == 0
				|| ZQ_CNN_Layer::_my_strcmpi(t, "AddBias") == 0;
		}

		bool _check_connect()
		{
			int blob_num = blobs.size();
			int layer_num = layers.size();
			if (layers.size() == 0 || blob_num == 0)
				return false;
			ZQ_CNN_Layer_Input* input_layer = (ZQ_CNN_Layer_Input*)(layers[0]);
			if (has_innerproduct_layer)
			{
				if (!input_layer->has_H_val || !input_layer->has_W_val)
				{
					std::cout << "Input dim must be specified for InnerProduct layer\n";
					return false;
				}
			}
			std::vector<bool> visited(blob_num);
			std::vector<int> blob_dim_C(blob_num);
			std::vector<int> blob_dim_H(blob_num);
			std::vector<int> blob_dim_W(blob_num);
			visited[0] = true;
			blob_dim_C[0] = input_layer->C;
			blob_dim_H[0] = input_layer->H;
			blob_dim_W[0] = input_layer->W;

			for (int i = 1; i < blob_num; i++)
				visited[i] = false;

			for (int i = 1; i < layer_num; i++)
			{
				std::vector<std::string>& bottom_names = layers[i]->bottom_names;
				int cur_bottom_c = 0;
				int cur_bottom_h = 0;
				int cur_bottom_w = 0;
				for (int j = 0; j < bottom_names.size(); j++)
				{
					std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(bottom_names[j]);
					if (!visited[name_it->second])
					{
						std::cout << "unknown blob " << bottom_names[j] << " in Layer " << layers[i]->name << "\n";
						return false;
					}
					/*if (j == 0)
					{
						cur_bottom_c = blob_dim_C[name_it->second];
						cur_bottom_h = blob_dim_H[name_it->second];
						cur_bottom_w = blob_dim_W[name_it->second];
						layers[i]->SetBottomDim(cur_bottom_c,cur_bottom_h,cur_bottom_w);
					}
					else
					{
						if (blob_dim_C[name_it->second] != cur_bottom_c || blob_dim_H[name_it->second] != cur_bottom_h 
							|| blob_dim_W[name_it->second] != cur_bottom_w)
						{
							std::cout << "Dimension mismatch detected in layer " << layers[i]->name << "\n";
							return false;
						}
					}*/
				}
				std::vector<std::string>& top_names = layers[i]->top_names;
				// 会改形状的层不能把 top 声明成自己的 bottom: LayerSetup 里
				// (*tops)[0]->SetShape(...) 会在 bottoms[0] 上就地重排, Forward
				// 再拿这个对象当输入读, 读到的是按输出步长解释的旧数据。
				// **必须比全部组合, 不能只比同一下标** —— 原来写的是
				// tops[i][j] == bottoms[i][j], 于是 bottoms=[A,B] top=B 被放行。
				// 而 Concat 的 Forward 把 inputs 收集成**指针**再改 output 的形状,
				// 那个别名会让某一路输入被就地扩容成 out_C, 拷贝循环再用扩容后的
				// in_C 去写, 最后一个像素越出整块分配 (附录 EN, ASan 实证)。
				if (!_is_inplace_safe(i))
				{
					for (int j = 0; j < top_names.size(); j++)
					{
						for (int k = 0; k < bottoms[i].size(); k++)
						{
							if (tops[i][j] == bottoms[i][k])
							{
								std::cout << "Layer " << layers[i]->name << " (" << layer_type_names[i]
									<< ") changes shape but declares top == bottom ("
									<< top_names[j] << "); that destroys its own input\n";
								return false;
							}
						}
					}
				}
				for (int j = 0; j < top_names.size(); j++)
				{
					std::map<std::string, int>::iterator name_it = map_name_to_blob_idx.find(top_names[j]);
					visited[name_it->second] = true;
					//layers[i]->GetTopDim(blob_dim_C[name_it->second], blob_dim_H[name_it->second], blob_dim_W[name_it->second]);
				}
			}

			/*for (int i = 1; i < blob_num; i++)
			{
				blobs[i]->ChangeSize(1, blob_dim_H[i], blob_dim_W[i], blob_dim_C[i], 0, 0);
			}*/

			if (!_setup())
			{
				return false;
			}
			return true;
		}

		bool _setup()
		{
			ZQ_CNN_Tensor4D_NHW_C_Align0 input;
			input.SetShape(1, input_C, input_H, input_W);
			if (map_name_to_blob_idx.size() == 0 || map_name_to_layer_idx.size() == 0 || tops.size() == 0)
				return false;
			
			blobs[0] = &input;

			for (int i = 0; i < layers.size(); i++)
			{
				std::vector<ZQ_CNN_Tensor4D*> bottom_ptrs, top_ptrs;
				for (int j = 0; j < bottoms[i].size(); j++)
					bottom_ptrs.push_back(blobs[bottoms[i][j]]);
				for (int j = 0; j < tops[i].size(); j++)
					top_ptrs.push_back(blobs[tops[i][j]]);

				layers[i]->show_debug_info = show_debug_info;
				if (!layers[i]->LayerSetup(&bottom_ptrs, &top_ptrs))
				{
					blobs[0] = 0;
					tops[0][0] = 0;
					printf("failed to setup layer: %s\n", layers[i]->name.c_str());
					return false;
				}
			}
			blobs[0] = 0;
			tops[0][0] = 0;
			return true;
		}

		void _simplify_inplace()
		{
			for (int i = 0; i < layers.size(); i++)
			{
				if (_is_inplace_safe(i))
				{
					bool later_refer = false;
					for (int j = i + 1; j < layers.size(); j++)
					{
						for (int k = 0; k < bottoms[j].size(); k++)
						{
							if (bottoms[j][k] == bottoms[i][0])
							{
								later_refer = true;
								break;
							}
						}
						if (later_refer)
							break;
					}
					if (later_refer)
						continue;

					for (int j = i + 1; j < layers.size(); j++)
					{
						for (int k = 0; k < bottoms[j].size(); k++)
						{
							if (bottoms[j][k] == tops[i][0])
							{
								bottoms[j][k] = bottoms[i][0];
							}
						}
					}

					if (simplify_inplace_blob_map.find(tops[i][0]) == simplify_inplace_blob_map.end())
					{
						simplify_inplace_blob_map[tops[i][0]] = bottoms[i][0];
					}
					tops[i][0] = bottoms[i][0];
				}
			}
		}

		// 把 i / i+1 两层（卷积 + 紧随其后的 BN）折成一层。
		//
		// 前提（附录 HX.2，加这条之前这里有个**结果会变**的缺陷）：
		//     tops[i][0] == bottoms[i + 1][0]
		// 它说的是"**这个 BN 吃的就是这个卷积的输出**"。
		// 缺了它，融合会把 BN 的系数折进一个**根本没喂给这个 BN**的卷积里：
		// 原图算的是"**另一个 blob** 的值 × 逐通道系数"，
		// 融合后算的是"这个卷积自己的输出 × 逐通道系数"。
		// 随仓的 model/mobilefacenet-v1.zqparams 第 108/109 行就是这个形状 ——
		//     108 DepthwiseConvolution ... bottom=res4_block1_conv top=res4_block5_conv_dw
		//     109 BatchNormScale      ... bottom=res4_block1_conv_dw top=res4_block1_conv_dw
		// 也就是 block5 的 BN 读的是**上一个 block 留下的** res4_block1_conv_dw，
		// 而 block5 的 dwconv 输出 res4_block5_conv_dw **压根没人读**。
		// 原来的 `later_refer` 只问"这个 conv 的输出后面还有没有人要"，
		// 恰好在这一格答"没有"，于是把一个喂错了输入的 BN 照折不误 ——
		// 生产实参下 mobilefacenet-v1 的输出被改了 0.37（后向误差，附录 HE.2）。
		// 折之前必须先确认 BN 的输入就是它；不是就不折，语义与原图一致。
		bool _merge_bn()
		{
			std::vector<ZQ_CNN_Layer*> tmp_layers;
			std::vector<std::string> tmp_layer_type_names;
			std::vector<std::vector<int> > tmp_bottoms;
			std::vector<std::vector<int> > tmp_tops;	
			for (int i = 0; i < layers.size(); i++)
			{
				/*BUG: merge innerproduct will lead MTCNN fail*/
				/*if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "InnerProduct") == 0)
				{
					if (i + 1 < layers.size() && ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i + 1].c_str(), "BatchNormScale") == 0)
					{
						bool later_refer = false;
						for (int j = i + 2; j < layers.size(); j++)
						{
							for (int k = 0; k < bottoms[j].size(); k++)
							{
								if (bottoms[j][k] == tops[i][0])
								{
									later_refer = true;
									break;
								}
							}
							if (later_refer)
								break;
						}
						if (tops[i][0] == bottoms[i + 1][0] && (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))	// 前提：后一层吃的就是这个 conv 的输出，理由见函数头注释
						{
							ZQ_CNN_Layer_InnerProduct* conv_layer = (ZQ_CNN_Layer_InnerProduct*)layers[i];
							ZQ_CNN_Layer_BatchNormScale* bns_layer = (ZQ_CNN_Layer_BatchNormScale*)layers[i + 1];
							if (!_merge_bns_to_innerproduct(conv_layer, bns_layer))
								return false;

							delete bns_layer; bns_layer = 0;
							tmp_layers.push_back(conv_layer);
							tmp_layer_type_names.push_back(layer_type_names[i]);
							tmp_bottoms.push_back(bottoms[i]);
							tmp_tops.push_back(tops[i + 1]);
							i++;
							continue;
						}
					}
				}
				else */if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "Convolution") == 0)
				{
					if (i + 1 < layers.size() && ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i+1].c_str(), "BatchNormScale") == 0)
					{
						bool later_refer = false;
						for (int j = i + 2; j < layers.size(); j++)
						{
							for (int k = 0; k < bottoms[j].size(); k++)
							{
								if (bottoms[j][k] == tops[i][0])
								{
									later_refer = true;
									break;
								}
							}
							if (later_refer)
								break;
						}
						if (tops[i][0] == bottoms[i + 1][0] && (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))	// 前提：后一层吃的就是这个 conv 的输出，理由见函数头注释
						{
							//do merge
							ZQ_CNN_Layer_Convolution* conv_layer = (ZQ_CNN_Layer_Convolution*)layers[i];
							ZQ_CNN_Layer_BatchNormScale* bns_layer = (ZQ_CNN_Layer_BatchNormScale*)layers[i + 1];
							if (!_merge_bns_to_conv(conv_layer, bns_layer))
								return false;

							delete bns_layer; bns_layer = 0;
							layers[i + 1] = NULL;
							tmp_layers.push_back(conv_layer);
							tmp_layer_type_names.push_back(layer_type_names[i]);
							tmp_bottoms.push_back(bottoms[i]);
							tmp_tops.push_back(tops[i + 1]);
							i++;
							continue;
						}
					}
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "DepthwiseConvolution") == 0)
				{
					if (i + 1 < layers.size() && ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i+1].c_str(), "BatchNormScale") == 0)
					{
						bool later_refer = false;
						for (int j = i + 2; j < layers.size(); j++)
						{
							for (int k = 0; k < bottoms[j].size(); k++)
							{
								if (bottoms[j][k] == tops[i][0])
								{
									later_refer = true;
									break;
								}
							}
							if (later_refer)
								break;
						}
						if (tops[i][0] == bottoms[i + 1][0] && (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))	// 前提：后一层吃的就是这个 conv 的输出，理由见函数头注释
						{
							//do merge
							ZQ_CNN_Layer_DepthwiseConvolution* dwconv_layer = (ZQ_CNN_Layer_DepthwiseConvolution*)layers[i];
							ZQ_CNN_Layer_BatchNormScale* bns_layer = (ZQ_CNN_Layer_BatchNormScale*)layers[i + 1];
							if (!_merge_bns_to_dwconv(dwconv_layer, bns_layer))
								return false;

							delete bns_layer; bns_layer = 0;
							layers[i + 1] = NULL;
							tmp_layers.push_back(dwconv_layer);
							tmp_layer_type_names.push_back(layer_type_names[i]);
							tmp_bottoms.push_back(bottoms[i]);
							tmp_tops.push_back(tops[i + 1]);
							i++;
							continue;
						}
					}
				}
				
				tmp_layers.push_back(layers[i]);
				tmp_layer_type_names.push_back(layer_type_names[i]);
				tmp_bottoms.push_back(bottoms[i]);
				tmp_tops.push_back(tops[i]);
			}

			layers = tmp_layers;
			layer_type_names = tmp_layer_type_names;
			bottoms = tmp_bottoms;
			tops = tmp_tops;
			return true;
		}

		// 与 `_merge_bn` 同一件事的 PReLU 版本，所以**同一个前提**：
		//     tops[i][0] == bottoms[i + 1][0]
		// 不是就**不折**。随仓的 17 个模型里这一格一处都没踩到
		// （`tools/check_bn_prelu_pairing.py` 扫出来是 0），
		// 但前提缺失时它和 `_merge_bn` 一样会静默改结果，所以一起补上 ——
		// 缺陷不该等到某个模型踩中了才修。
		bool _merge_prelu()
		{
			std::vector<ZQ_CNN_Layer*> tmp_layers;
			std::vector<std::string> tmp_layer_type_names;
			std::vector<std::vector<int> > tmp_bottoms;
			std::vector<std::vector<int> > tmp_tops;
			for (int i = 0; i < layers.size(); i++)
			{
				if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "Convolution") == 0)
				{
					if (i + 1 < layers.size() && ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i + 1].c_str(), "PReLU") == 0)
					{
						bool later_refer = false;
						for (int j = i + 2; j < layers.size(); j++)
						{
							for (int k = 0; k < bottoms[j].size(); k++)
							{
								if (bottoms[j][k] == tops[i][0])
								{
									later_refer = true;
									break;
								}
							}
							if (later_refer)
								break;
						}
						if (tops[i][0] == bottoms[i + 1][0] && (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))	// 前提：后一层吃的就是这个 conv 的输出，理由见函数头注释
						{
							//do merge
							ZQ_CNN_Layer_Convolution* conv_layer = (ZQ_CNN_Layer_Convolution*)layers[i];
							ZQ_CNN_Layer_PReLU* prelu_layer = (ZQ_CNN_Layer_PReLU*)layers[i + 1];
							if (!_merge_prelu_to_conv(conv_layer, prelu_layer))
								return false;

							delete prelu_layer; prelu_layer = 0;
							layers[i + 1] = NULL;
							tmp_layers.push_back(conv_layer);
							tmp_layer_type_names.push_back(layer_type_names[i]);
							tmp_bottoms.push_back(bottoms[i]);
							tmp_tops.push_back(tops[i + 1]);
							i++;
							continue;
						}
					}
				}
				else if (ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i].c_str(), "DepthwiseConvolution") == 0)
				{
					if (i + 1 < layers.size() && ZQ_CNN_Layer::_my_strcmpi(layer_type_names[i + 1].c_str(), "PReLU") == 0)
					{
						bool later_refer = false;
						for (int j = i + 2; j < layers.size(); j++)
						{
							for (int k = 0; k < bottoms[j].size(); k++)
							{
								if (bottoms[j][k] == tops[i][0])
								{
									later_refer = true;
									break;
								}
							}
							if (later_refer)
								break;
						}
						if (tops[i][0] == bottoms[i + 1][0] && (tops[i + 1][0] == bottoms[i + 1][0] || !later_refer))	// 前提：后一层吃的就是这个 conv 的输出，理由见函数头注释
						{
							//do merge
							ZQ_CNN_Layer_DepthwiseConvolution* dwconv_layer = (ZQ_CNN_Layer_DepthwiseConvolution*)layers[i];
							ZQ_CNN_Layer_PReLU* prelu_layer = (ZQ_CNN_Layer_PReLU*)layers[i + 1];
							if (!_merge_prelu_to_dwconv(dwconv_layer, prelu_layer))
								return false;

							delete prelu_layer; prelu_layer = 0;
							layers[i + 1] = NULL;
							tmp_layers.push_back(dwconv_layer);
							tmp_layer_type_names.push_back(layer_type_names[i]);
							tmp_bottoms.push_back(bottoms[i]);
							tmp_tops.push_back(tops[i + 1]);
							i++;
							continue;
						}
					}
				}

				tmp_layers.push_back(layers[i]);
				tmp_layer_type_names.push_back(layer_type_names[i]);
				tmp_bottoms.push_back(bottoms[i]);
				tmp_tops.push_back(tops[i]);
			}

			layers = tmp_layers;
			layer_type_names = tmp_layer_type_names;
			bottoms = tmp_bottoms;
			tops = tmp_tops;
			return true;
		}

		bool _merge_bns_to_innerproduct(ZQ_CNN_Layer_InnerProduct* conv_layer, ZQ_CNN_Layer_BatchNormScale* bns_layer)
		{
			ZQ_CNN_Tensor4D* filters = conv_layer->filters;
			ZQ_CNN_Tensor4D* b = bns_layer->b;
			ZQ_CNN_Tensor4D* a = bns_layer->a;
			int N = filters->GetN();
			int kH = filters->GetH();
			int kW = filters->GetW();
			int kC = filters->GetC();
			if (b->GetC() != N || a->GetC() != N)
				return false;
			for (int n = 0; n < N; n++)
			{
				float b_v = (b->GetFirstPixelPtr())[n];
				float* slice_ptr = filters->GetFirstPixelPtr() + n*filters->GetSliceStep();
				for (int h = 0; h < kH; h++)
				{
					float* row_ptr = slice_ptr + h*filters->GetWidthStep();
					for (int w = 0; w < kW; w++)
					{
						float* pix_ptr = row_ptr + w*filters->GetPixelStep();
						for (int c = 0; c < kC; c++)
						{
							pix_ptr[c] *= b_v;
							if (fabs(pix_ptr[c]) < this->ignore_small_value)
								pix_ptr[c] = 0;
						}
					}
				}
			}

			if (conv_layer->bias == 0)
			{
				conv_layer->with_bias = true;
				conv_layer->bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (conv_layer->bias == 0 || !conv_layer->bias->ChangeSize(1, 1, 1, N, 0, 0))
					return false;
				for (int n = 0; n < N; n++)
				{
					float a_v = (a->GetFirstPixelPtr())[n];
					(conv_layer->bias->GetFirstPixelPtr())[n] = a_v;
				}
			}
			else
			{
				for (int n = 0; n < N; n++)
				{
					float b_v = (b->GetFirstPixelPtr())[n];
					float bias_v = (conv_layer->bias->GetFirstPixelPtr())[n];
					float a_v = (a->GetFirstPixelPtr())[n];
					(conv_layer->bias->GetFirstPixelPtr())[n] = bias_v*b_v + a_v;
				}
			}
			return true;
		}


		bool _merge_bns_to_conv(ZQ_CNN_Layer_Convolution* conv_layer, ZQ_CNN_Layer_BatchNormScale* bns_layer)
		{
			ZQ_CNN_Tensor4D* filters = conv_layer->filters;
			ZQ_CNN_Tensor4D* b = bns_layer->b;
			ZQ_CNN_Tensor4D* a = bns_layer->a;
			int N = filters->GetN();
			int kH = filters->GetH();
			int kW = filters->GetW();
			int kC = filters->GetC();
			if (b->GetC() != N || a->GetC() != N)
				return false;
			for (int n = 0; n < N; n++)
			{
				float b_v = (b->GetFirstPixelPtr())[n];
				float* slice_ptr = filters->GetFirstPixelPtr() + n*filters->GetSliceStep();
				for (int h = 0; h < kH; h++)
				{
					float* row_ptr = slice_ptr + h*filters->GetWidthStep();
					for (int w = 0; w < kW; w++)
					{
						float* pix_ptr = row_ptr + w*filters->GetPixelStep();
						for (int c = 0; c < kC; c++)
						{
							pix_ptr[c] *= b_v;
							if (fabs(pix_ptr[c]) < this->ignore_small_value)
								pix_ptr[c] = 0;
						}
					}
				}
			}

			if (conv_layer->bias == 0)
			{
				conv_layer->with_bias = true;
				conv_layer->bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (conv_layer->bias == 0 || !conv_layer->bias->ChangeSize(1, 1, 1, N, 0, 0))
					return false;
				for (int n = 0; n < N; n++)
				{
					float a_v = (a->GetFirstPixelPtr())[n];
					(conv_layer->bias->GetFirstPixelPtr())[n] = a_v;
				}
			}
			else
			{
				for (int n = 0; n < N; n++)
				{
					float b_v = (b->GetFirstPixelPtr())[n];
					float bias_v = (conv_layer->bias->GetFirstPixelPtr())[n];
					float a_v = (a->GetFirstPixelPtr())[n];
					(conv_layer->bias->GetFirstPixelPtr())[n] = bias_v*b_v + a_v;
				}
			}
			return true;
		}

		bool _merge_prelu_to_conv(ZQ_CNN_Layer_Convolution* conv_layer, ZQ_CNN_Layer_PReLU* prelu_layer)
		{
			ZQ_CNN_Tensor4D* slope = prelu_layer->slope;
			conv_layer->with_prelu = true;
			conv_layer->prelu_slope = slope;
			prelu_layer->slope = 0;
			return true;
		}

		bool _merge_bns_to_dwconv(ZQ_CNN_Layer_DepthwiseConvolution* dwconv_layer, ZQ_CNN_Layer_BatchNormScale* bns_layer)
		{
			ZQ_CNN_Tensor4D* filters = dwconv_layer->filters;
			ZQ_CNN_Tensor4D* b = bns_layer->b;
			ZQ_CNN_Tensor4D* a = bns_layer->a;
			int kH = filters->GetH();
			int kW = filters->GetW();
			int kC = filters->GetC();
			if (b->GetC() != kC || a->GetC() != kC)
				return false;
			for (int c = 0; c < kC; c++)
			{
				float b_v = (b->GetFirstPixelPtr())[c];
				float* slice_ptr = filters->GetFirstPixelPtr();
				for (int h = 0; h < kH; h++)
				{
					float* row_ptr = slice_ptr + h*filters->GetWidthStep() + c;
					for (int w = 0; w < kW; w++)
					{
						float* pix_ptr = row_ptr + w*filters->GetPixelStep();
						pix_ptr[0] *= b_v;
						if (fabs(pix_ptr[0]) < this->ignore_small_value)
							pix_ptr[0] = 0;
					}
				}
			}

			if (dwconv_layer->bias == 0)
			{
				dwconv_layer->with_bias = true;
				dwconv_layer->bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (dwconv_layer->bias == 0 || !dwconv_layer->bias->ChangeSize(1, 1, 1, kC, 0, 0))
					return false;
				for (int c = 0; c < kC; c++)
				{
					float a_v = (a->GetFirstPixelPtr())[c];
					(dwconv_layer->bias->GetFirstPixelPtr())[c] = a_v;
				}
			}
			else
			{
				for (int c = 0; c < kC; c++)
				{
					float b_v = (b->GetFirstPixelPtr())[c];
					float bias_v = (dwconv_layer->bias->GetFirstPixelPtr())[c];
					float a_v = (a->GetFirstPixelPtr())[c];
					(dwconv_layer->bias->GetFirstPixelPtr())[c] = bias_v*b_v + a_v;
				}
			}
			return true;
		}

		bool _merge_prelu_to_dwconv(ZQ_CNN_Layer_DepthwiseConvolution* dwconv_layer, ZQ_CNN_Layer_PReLU* prelu_layer)
		{
			ZQ_CNN_Tensor4D* slope = prelu_layer->slope;
			dwconv_layer->with_prelu = true;
			dwconv_layer->prelu_slope = slope;
			prelu_layer->slope = 0;
			return true;
		}

		bool _swap_input_RGB_and_BGR(const std::vector<std::string>& layer_names)
		{
			int blob_num = blobs.size();
			int layer_num = layers.size();
			if (layers.size() == 0 || blob_num == 0)
				return false;
			for (int j = 0; j < layer_names.size(); j++)
			{
				bool found = false;
				for (int i = 0; i < layer_num; i++)
				{
					if (ZQ_CNN_Layer::_my_strcmpi(layers[i]->name.c_str(), layer_names[j].c_str()) == 0)
					{
						found = true;
						if (!layers[i]->SwapInputRGBandBGR())
						{
							printf("failed to swap RGB and BGR for layer %s\n", layer_names[j].c_str());
							return false;
						}
					}
				}
				if (!found)
				{
					printf("warning: layer %s does not exists\n", layer_names[j].c_str());
				}
			}
			return true;
		}
	};
}

#endif
