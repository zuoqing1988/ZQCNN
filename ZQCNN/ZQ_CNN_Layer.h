#ifndef _ZQ_CNN_LAYER_H_
#define _ZQ_CNN_LAYER_H_
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "ZQ_CNN_Defines.h"
#include "ZQ_CNN_Tensor4D.h"
#include "ZQ_CNN_BBoxUtils.h"
#include "ZQ_CNN_Forward_SSEUtils.h"
namespace ZQ
{
	class ZQ_CNN_Layer
	{
	public:
		std::string name;
		std::vector<std::string> bottom_names;
		std::vector<std::string> top_names;
		bool show_debug_info;

		ZQ_CNN_Layer() :show_debug_info(false) {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1) = 0;

		virtual bool ReadParam(const std::string& line) = 0;

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops) = 0;

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) = 0;

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& topH, int &top_W) const = 0;

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) = 0;
	public:
		static std::vector<std::vector<std::string>> split_line(const std::string& line)
		{
			std::vector<std::string> first_splits = _split_blank(line.c_str());
			int num = first_splits.size();
			std::vector<std::vector<std::string>> second_splits(num);
			for (int n = 0; n < num; n++)
			{
				second_splits[n] = _split_separater(first_splits[n].c_str());
			}
			return second_splits;
		}
	private:
		static bool _is_blank_c(char c)
		{
			return c == ' ' || c == '\t' || c == '\n';
		}
		static bool _is_separator_c(char c)
		{
			return c == ':' || c == '=';
		}
		static std::vector<std::string>  _split_blank(const char* str)
		{
			std::vector<std::string> out;
			int len = strlen(str);
			std::vector<char> buf(len + 1);
			int i = 0, j = 0;
			while (1)
			{
				//skip blank
				for (; i < len && _is_blank_c(str[i]); i++);
				if (i >= len)
					break;

				for (j = i; j < len && !_is_blank_c(str[j]); j++);
				int tmp_len = j - i;
				if (tmp_len == 0)
					break;
				memcpy(&buf[0], str + i, tmp_len * sizeof(char));
				buf[tmp_len] = '\0';

				out.push_back(std::string(&buf[0]));
				i = j;
			}
			return out;
		}

		static std::vector<std::string>  _split_separater(const char* str)
		{
			std::vector<std::string> out;
			int len = strlen(str);
			std::vector<char> buf(len + 1);
			int i = 0, j = 0;
			while (1)
			{
				//skip blank
				for (; i < len && _is_separator_c(str[i]); i++);
				if (i >= len)
					break;

				for (j = i; j < len && !_is_separator_c(str[j]); j++);
				int tmp_len = j - i;
				if (tmp_len == 0)
					break;
				memcpy(&buf[0], str + i, tmp_len * sizeof(char));
				buf[tmp_len] = '\0';

				out.push_back(std::string(&buf[0]));
				i = j;
			}
			return out;
		}
	};

	class ZQ_CNN_Layer_Input : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Input() :H(0), W(0), C(3), has_H_val(false), has_W_val(false) {}
		virtual ~ZQ_CNN_Layer_Input() {}
		int H, W, C;
		bool has_H_val, has_W_val;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1) 
		{ return true; }

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_C = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Input", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_H_val = true;
						H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_W_val = true;
						W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("C", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_C = true;
						C = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
						top_names.push_back(name);
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}

			if (!has_C)std::cout << "Layer " << name << " missing " << "C\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_C && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) { return true; }

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const { top_C = C; top_H = H; top_W = W; }

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

	};

	class ZQ_CNN_Layer_Convolution : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Convolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1), pad_H(0), pad_W(), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_Convolution() {
			if (filters)delete filters;
			if (bias)delete bias;
		}
		ZQ_CNN_Tensor4D* filters;
		ZQ_CNN_Tensor4D* bias;
		int num_output;
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		int pad_H;
		int pad_W;
		bool with_bias;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (with_bias)
			{
				if (filters == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::ConvolutionWithBias(*((*bottoms)[0]), *filters, *bias, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]),
					num_threads);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Conv layer:%s %.3f ms HW %dx%d filter: NHWC %d x %d x %d x %d\n", 
						name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::Convolution(*((*bottoms)[0]), *filters, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]),
					num_threads);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Conv layer:%s %.3f ms HW %dx%d filter: NHWC %d x %d x %d x %d\n", 
						name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Convolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << name << " missing " << "num_output\n";
			if (!has_kernelH)std::cout << "Layer " << name << " missing " << "kernel_H (kernel_size)\n";
			if (!has_kernelW)std::cout << "Layer " << name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && has_kernelH && has_kernelW && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) {
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (filters)
			{
				if (!filters->ChangeSize(num_output, kernel_H, kernel_W, bottom_C, 0, 0))
					return false;
			}
			else
			{
				/*if (bottom_C <= 4)
					filters = new ZQ_CNN_Tensor4D_NHW_C_Align128bit();
				else*/
				filters = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (filters == 0)return false;
				if (!filters->ChangeSize(num_output, kernel_H, kernel_W, bottom_C, 0, 0))
					return false;
			}
			if (with_bias)
			{
				if (bias)
				{
					if (!bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
				else
				{
					bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
					if (bias == 0)return false;
					if (!bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{ 
			top_C = num_output;  
			top_H = __max(0, floor((float)(bottom_H + pad_H * 2 - kernel_H) / stride_H) + 1);
			top_W = __max(0, floor((float)(bottom_W + pad_W * 2 - kernel_W) / stride_W) + 1);
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
				return false;
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}
	};


	class ZQ_CNN_Layer_DepthwiseConvolution : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_DepthwiseConvolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1), pad_H(0), pad_W(), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_DepthwiseConvolution() {
			if (filters)delete filters;
			if (bias)delete bias;
		}
		ZQ_CNN_Tensor4D* filters;
		ZQ_CNN_Tensor4D* bias;
		int num_output;
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		int pad_H;
		int pad_W;
		bool with_bias;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (with_bias)
			{
				if (filters == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::DepthwiseConvolutionWithBias(*((*bottoms)[0]), *filters, *bias, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]), 
					num_threads);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("DwConv layer:%s %.3f ms HW %dx%d filter: NHWC %d x %d x %d x %d\n",
						name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::DepthwiseConvolution(*((*bottoms)[0]), *filters, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]),
					num_threads);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Conv layer:%s %.3f ms HW %dx%d filter: NHWC %d x %d x %d x %d\n",
						name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("DepthwiseConvolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << name << " missing " << "num_output\n";
			if (!has_kernelH)std::cout << "Layer " << name << " missing " << "kernel_H (kernel_size)\n";
			if (!has_kernelW)std::cout << "Layer " << name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && has_kernelH && has_kernelW && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) {
			if (bottom_C != num_output)
			{
				std::cout << "Layer " << name << "'s num_output should match bottom's C\n";
				return false;
			}

			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (filters)
			{
				if (!filters->ChangeSize(1, kernel_H, kernel_W, num_output, 0, 0))
					return false;
			}
			else
			{
				filters = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (filters == 0)return false;
				if (!filters->ChangeSize(1, kernel_H, kernel_W, num_output, 0, 0))
					return false;
			}
			if (with_bias)
			{
				if (bias)
				{
					if (!bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
				else
				{
					bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
					if (bias == 0)return false;
					if (!bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = num_output;
			top_H = __max(0, floor((float)(bottom_H + pad_H * 2 - kernel_H) / stride_H) + 1);
			top_W = __max(0, floor((float)(bottom_W + pad_W * 2 - kernel_W) / stride_W) + 1);
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
				return false;
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}
	};

	class ZQ_CNN_Layer_BatchNormScale : public ZQ_CNN_Layer
	{
		/*
		a = bias - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		*/
	public:
		ZQ_CNN_Layer_BatchNormScale() : b(0), a(0), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_BatchNormScale() {
			if (b)delete b;
			if (a)delete a;
		}
		/*ZQ_CNN_Tensor4D* mean;
		ZQ_CNN_Tensor4D* var;
		ZQ_CNN_Tensor4D* scale;
		ZQ_CNN_Tensor4D* bias;*/
		ZQ_CNN_Tensor4D* b;
		ZQ_CNN_Tensor4D* a;

		//
		bool with_bias;
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (b == 0 || a == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::BatchNorm_b_a(*((*tops)[0]), *b, *a);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("BatchNorm layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("BatchNormScale", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (b)
			{
				if (!b->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				b = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (b == 0)return false;
				if (!b->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			if (a)
			{
				if (!a->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				a = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (a == 0)return false;
				if (!a->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}

			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			if (with_bias)
			{
				std::vector<float> nchw_raw(dst_len * 4);
				if (dst_len * 4 != fread_s(&nchw_raw[0], dst_len * 4 * sizeof(float), sizeof(float), dst_len * 4, in))
					return false;
				ZQ_CNN_Tensor4D_NHW_C_Align0 mean, var, scale, bias;
				mean.ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				bias.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 3, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScaleBias_Compute_b_a(*b, *a, mean, var, scale, bias);
			}
			else
			{
				std::vector<float> nchw_raw(dst_len * 3);
				if (dst_len * 3 != fread_s(&nchw_raw[0], dst_len * 3 * sizeof(float), sizeof(float), dst_len * 3, in))
					return false;
				ZQ_CNN_Tensor4D_NHW_C_Align0 mean, var, scale;
				mean.ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScale_Compute_b_a(*b, *a, mean, var, scale);
			}
			
		}

	};

	class ZQ_CNN_Layer_BatchNorm : public ZQ_CNN_Layer
	{
		/*
		a = - mean / sqrt(var)
		b = 1 / sqrt(var)
		value = b * value + a
		*/
	public:
		ZQ_CNN_Layer_BatchNorm() : b(0), a(0), bottom_C(0) {}
		~ZQ_CNN_Layer_BatchNorm() {
			if (b)delete b;
			if (a)delete a;
		}
		/*ZQ_CNN_Tensor4D* mean;
		ZQ_CNN_Tensor4D* var;*/
		ZQ_CNN_Tensor4D* b;
		ZQ_CNN_Tensor4D* a;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (b == 0 || a == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::BatchNorm(*((*tops)[0]), *b, *a);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("BatchNorm layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("BatchNorm", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (b)
			{
				if (!b->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				b = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (b == 0)return false;
				if (!b->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			if (a)
			{
				if (!a->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				a = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (a == 0)return false;
				if (!a->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}

			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}
		
		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len * 2);
			if (dst_len * 2 != fread_s(&nchw_raw[0], dst_len * 2 * sizeof(float), sizeof(float), dst_len * 2, in))
				return false;
			ZQ_CNN_Tensor4D_NHW_C_Align0 mean, var;
			mean.ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
			var.ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
			return ZQ_CNN_Forward_SSEUtils::BatchNorm_Compute_b_a(*b, *a, mean, var);
		}

	};

	class ZQ_CNN_Layer_Scale : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Scale() :scale(0), bias(0), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_Scale() {
			if (scale)	delete scale;
			if (bias)	delete bias;
		}
		ZQ_CNN_Tensor4D* scale;
		ZQ_CNN_Tensor4D* bias;
		bool with_bias;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			if (with_bias)
			{
				if (scale == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::ScaleWithBias(*((*tops)[0]), *scale, *bias);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Scale layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}
			else
			{
				if (scale == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::Scale(*((*tops)[0]), *scale);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Scale layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Scale", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}


		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (scale)
			{
				if (!scale->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				scale = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (scale == 0)return false;
				if (!scale->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			if (with_bias)
			{
				if (bias)
				{
					if (!bias->ChangeSize(1, 1, 1, bottom_C, 0, 0))
						return false;
				}
				else
				{
					bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
					if (bias == 0)return false;
					if (!bias->ChangeSize(1, 1, 1, bottom_C, 0, 0))
						return false;
				}
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (scale == 0)
				return false;
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
				return false;
			scale->ConvertFromCompactNCHW(&nchw_raw[0], scale->GetN(), scale->GetC(), scale->GetH(), scale->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}
	};

	class ZQ_CNN_Layer_PReLU : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Tensor4D* slope;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_PReLU() :slope(0), bottom_C(0) {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (slope == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::PReLU(*((*tops)[0]), *slope);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("PReLU layer: %s %.3f ms \n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("PReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			if (slope)
			{
				if (!slope->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				slope = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (slope == 0)return false;
				if (!slope->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (slope == 0)
				return false;
			int dst_len = slope->GetN() * slope->GetH() * slope->GetW() * slope->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
				return false;
			slope->ConvertFromCompactNCHW(&nchw_raw[0], slope->GetN(), slope->GetC(), slope->GetH(), slope->GetW());
			return true;
		}
	};

	class ZQ_CNN_Layer_ReLU : public ZQ_CNN_Layer
	{
	public:
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_ReLU() :bottom_C(0) {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils::ReLU(*((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("ReLU layer: %s %.3f ms \n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("ReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in){	return true;}
	};

	class ZQ_CNN_Layer_Pooling : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Pooling() :kernel_H(3), kernel_W(3), stride_H(2), stride_W(2), type(0) {}
		virtual ~ZQ_CNN_Layer_Pooling() {}
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		static const int TYPE_MAXPOOLING = 0;
		static const int TYPE_AVEPOOLING = 1;
		int type;	//0-MAX,1-AVE
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			if (type == TYPE_MAXPOOLING)
			{
				double t1 = omp_get_wtime();
				ZQ_CNN_Forward_SSEUtils::MaxPooling(*((*bottoms)[0]), *((*tops)[0]), kernel_H, kernel_W, stride_H, stride_W);
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Pooling layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return true;
			}
			else
			{
				printf("unsupported pooling type!\n");
				return false;
			}
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_kernelH = false, has_kernelW = false;
			bool has_strideH = false, has_strideW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Pooling", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_strideH = true;
						stride_H = atoi(paras[n][1].c_str());
						has_strideW = true;
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("pool", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						const char* str = paras[n][1].c_str();
						if (_strcmpi(str,"MAX") == 0)
							type = TYPE_MAXPOOLING;
						else if (_strcmpi(str,"AVE") == 0)
							type = TYPE_AVEPOOLING;
						else
						{
							type = atoi(str);
						}
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_kernelH)std::cout << "Layer " << name << " missing " << "kernel_H (kernel_size)\n";
			if (!has_kernelW)std::cout << "Layer " << name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_strideH)std::cout << "Layer " << name << " missing " << "stride_H (stride)\n";
			if (!has_strideW)std::cout << "Layer " << name << " missing " << "stride_W (stride)\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_kernelH && has_kernelW && has_strideH && has_strideW && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) 
		{ 
			this->bottom_C = bottom_C; 
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true; 
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{
			top_C = bottom_C;
			top_H = __max(0, ceil((float)(bottom_H - kernel_H) / stride_H) + 1);
			top_W = __max(0, ceil((float)(bottom_W - kernel_W) / stride_W) + 1);
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_InnerProduct : public ZQ_CNN_Layer
	{
	public:
		
		ZQ_CNN_Tensor4D* filters;
		ZQ_CNN_Tensor4D* bias;
		bool with_bias;
		int num_output;
		int kernel_H;
		int kernel_W;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_InnerProduct() :filters(0), bias(0), with_bias(false) {}

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if (with_bias)
			{
				if (filters == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::InnerProductWithBias(*((*bottoms)[0]), *filters, *bias, *((*tops)[0]));
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Innerproduct layer: %.3f ms HW %dx%d filter: NHWC %d x %d x %d x %d\n", 1000 * (t2 - t1), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::InnerProduct(*((*bottoms)[0]), *filters, *((*tops)[0]));
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("Innerproduct layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}

		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false/*, has_kernelH = false, has_kernelW = false*/;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("InnerProduct", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				/*else if (_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}*/
				else if (_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << name << " missing " << "num_output\n";
			//if (!has_kernelH)std::cout << "Layer " << name << " missing " << "kernel_H (kernel_size)\n";
			//if (!has_kernelW)std::cout << "Layer " << name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && /*has_kernelH && has_kernelW && */has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			kernel_H = bottom_H;
			kernel_W = bottom_W;
			if (filters)
			{
				if (filters->ChangeSize(num_output, kernel_H, kernel_W, bottom_C, 0, 0))
					return false;
			}
			else
			{
				filters = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (filters == 0) return false;
				if (!filters->ChangeSize(num_output, kernel_H, kernel_W, bottom_C, 0, 0))
					return false;
			}
			if (with_bias)
			{
				if (bias)
				{
					if (bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
				else
				{
					bias = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
					if (bias == 0) return false;
					if (!bias->ChangeSize(1, 1, 1, num_output, 0, 0))
						return false;
				}
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{ 
			top_C = num_output;
			top_H = 1;
			top_W = 1;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) {
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			int readed_len = fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in);
			if (dst_len != readed_len)
				return false;
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len * 2);
				//int pos = ftell(in);
				readed_len = fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in);
				if (dst_len != readed_len)
					return false;
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}
	};

	class ZQ_CNN_Layer_Softmax : public ZQ_CNN_Layer
	{
	public:
		int axis;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_Softmax() { axis = 1; }
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils::Softmax(*((*tops)[0]), axis);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Softmax layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Softmax", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if(axis < 0 || axis > 3)std::cout << "Layer " << name << " invalid axis " << axis << "\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return axis >= 0 && axis <= 3 && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) 
		{ 
			this->bottom_C = bottom_C; 
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true; 
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_Dropout : public ZQ_CNN_Layer
	{
	public:
		float dropout_ratio;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_Dropout() {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			/*dropout ratio is not used in test phase*/
			//ZQ_CNN_Forward_SSEUtils::Dropout(*((*tops)[0]), dropout_ratio);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Dropout layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_dropout_ratio = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Dropout", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("dropout_ratio", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_dropout_ratio = true;
						dropout_ratio = atof(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}
			}
			if (!has_dropout_ratio)std::cout << "Layer " << name << " missing " << "dropout_ratio\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_dropout_ratio && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) 
		{ 
			this->bottom_C = bottom_C; 
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true; 
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{ 
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_Eltwise : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Eltwise() :with_weight(false) {}
		static const int ELTWISE_PROD = 0;
		static const int ELTWISE_SUM = 1;
		static const int ELTWISE_MAX = 2;
		int operation;//
		bool with_weight;
		std::vector<float> weight;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			if (operation == ELTWISE_PROD)
			{
				return ZQ_CNN_Forward_SSEUtils::Eltwise_Prod(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else if (operation == ELTWISE_SUM)
			{
				if (with_weight)
				{
					return ZQ_CNN_Forward_SSEUtils::Eltwise_SumWithWeight(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, weight, *((*tops)[0]));
				}
				else
				{
					return ZQ_CNN_Forward_SSEUtils::Eltwise_Sum(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
				}
			}
			else if (operation == ELTWISE_MAX)
			{
				return ZQ_CNN_Forward_SSEUtils::Eltwise_Max(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown eltwise operation " << operation << " in Layer " << name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Eltwise layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			weight.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			with_weight = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Eltwise", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (_strcmpi(str, "PROD") == 0)
						{
							operation = ELTWISE_PROD;
						}
						else if (_strcmpi(str, "SUM") == 0)
						{
							operation = ELTWISE_SUM;
						}
						else if(_strcmpi(str,"MAX") == 0)
						{
							operation = ELTWISE_MAX;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("weight", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						with_weight = true;
						weight.push_back(atof(paras[n][1].c_str()));
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}

			}
			if (!has_operation)std::cout << "Layer " << name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (bottom_names.size() < 2)std::cout << "Layer " << name << " must have at least 2 bottoms\n";
			if (with_weight && weight.size() != bottom_names.size()) std::cout << "Layer " << name << " weight num should match with bottom num\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_operation && has_bottom && bottom_names.size() >= 2 && has_top && has_name
				&& (!with_weight || (with_weight && weight.size() != bottom_names.size()));
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) 
		{ 
			this->bottom_C = bottom_C; 
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true; 
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{
			top_C = bottom_C;
			top_H = bottom_H;
			top_W = bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_LRN : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_LRN(): k(1.0f) { operation = LRN_ACROSS_CHANNELS;}
		static const int LRN_ACROSS_CHANNELS = 0;
		int operation;//
		int local_size;
		float alpha, beta, k;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			
			if (operation == LRN_ACROSS_CHANNELS)
			{
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::LRN_across_channels(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0],local_size,alpha,beta,k, *((*tops)[0]));
				double t2 = omp_get_wtime();
				if (show_debug_info)
					printf("LRN layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}
			else
			{
				std::cout << "unknown LRN operation " << operation << " in Layer " << name << "\n";
				return false;
			}
			
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_local_size = false, has_alpha = false, has_beta = false;
			
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("LRN", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						operation = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("local_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						local_size = atoi(paras[n][1].c_str());
						has_local_size = true;
					}
				}
				else if (_strcmpi("alpha", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						alpha = atof(paras[n][1].c_str());
						has_alpha = true;
					}
				}
				else if (_strcmpi("beta", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						beta = atof(paras[n][1].c_str());
						has_beta = true;
					}
				}
				else if (_strcmpi("k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						k = atof(paras[n][1].c_str());
					}
				}

			}
			if (!has_operation)std::cout << "Layer " << name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_local_size)std::cout << "Layer " << name << " missing " << "local_size\n";
			if (!has_alpha)std::cout << "Layer " << name << " missing " << "alpha\n";
			if (!has_beta)std::cout << "Layer " << name << " missing " << "beta\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_operation && has_bottom && has_top && has_name
				&& has_local_size && has_alpha && has_beta;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) 
		{ 
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W; 
			return true; 
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const 
		{ 
			top_C = this->bottom_C;
			top_H = this->bottom_H;
			top_W = this->bottom_W;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_Permute : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Permute() { order[0] = 0; order[1] = 1; order[2] = 2; order[3] = 3; }
		int order[4];
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Permute(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], order, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Permute layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			int order_num = 0;
			bool has_order_flag[4] = { false };
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Permute", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("order", paras[n][0].c_str()) == 0)
				{
					if (order_num >= 4)
					{
						order_num++;
						continue;
					}

					if (paras[n].size() >= 2)
					{
						int cur_order = atoi(paras[n][1].c_str());
						order[order_num] = cur_order;
						if(cur_order >= 0 && cur_order < 4)
							has_order_flag[cur_order] = true;
						order_num++;
					}
				}
				
			}
			bool has_all_order = true;
			for (int i = 0; i < 4; i++)
			{
				if (!has_order_flag[i])
				{
					has_all_order = false;
					break;
				}
			}
			if (order_num != 4 || !has_all_order)std::cout << "Layer " << name << " invalid order \n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return order_num == 4 && has_all_order && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			old_dim[0] = 1;
			old_dim[1] = bottom_C;
			old_dim[2] = bottom_H;
			old_dim[3] = bottom_W;
			if (!ZQ_CNN_Tensor4D::Permute_NCHW_get_size(order, old_dim[0], old_dim[1], old_dim[2], old_dim[3],
				new_dim[0], new_dim[1], new_dim[2], new_dim[3]))
				return false;
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = new_dim[1];
			top_H = new_dim[2];
			top_W = new_dim[3];
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_Flatten : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Flatten() { axis = 1; end_axis = -1; }
		int axis;
		int end_axis;
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Flatten(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], axis, end_axis, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Flatten layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Flatten", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("end_axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						end_axis = atoi(paras[n][1].c_str());
					}
				}
			}
			
			bool valid_axis = true, valid_end_axis = true;
			if (axis < -1)
			{
				std::cout << "Layer " << name << " invalid axis " << axis << "\n";
				valid_axis = false;
			}
			if (end_axis < -1)
			{
				std::cout << "Layer " << name << " invalid end_axis " << end_axis << "\n";
				valid_end_axis = false;
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return valid_axis && valid_end_axis && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			old_dim[0] = 1;
			old_dim[1] = bottom_C;
			old_dim[2] = bottom_H;
			old_dim[3] = bottom_W;
			if (axis == -1)
				axis = 3;
			if (end_axis == -1)
				end_axis = 3;
			if (!ZQ_CNN_Tensor4D::Flatten_NCHW_get_size(axis, end_axis, old_dim[0], old_dim[1], old_dim[2], old_dim[3],
				new_dim[0], new_dim[1], new_dim[2], new_dim[3]))
				return false;
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = new_dim[1];
			top_H = new_dim[2];
			top_W = new_dim[3];
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_Reshape : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Reshape() { axis = 0; num_axes = -1; }
		std::vector<int> shape;
		int axis;
		int num_axes;
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Reshape(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], shape, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Reshape layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("Reshape", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("dim", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						shape.push_back(atoi(paras[n][1].c_str()));
					}
				}
				else if (_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("num_axes", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						num_axes = atoi(paras[n][1].c_str());
					}
				}
			}

			bool valid_axis = true, valid_num_axes = true;
			if (axis < -1)
			{
				std::cout << "Layer " << name << " invalid axis " << axis << "\n";
				valid_axis = false;
			}
			if (num_axes < -1 || num_axes >= (int)(shape.size()))
			{
				std::cout << "Layer " << name << " invalid num_axes " << num_axes << "\n";
				valid_num_axes = false;
			}
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return valid_axis && valid_num_axes && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			old_dim[0] = 1;
			old_dim[1] = bottom_C;
			old_dim[2] = bottom_H;
			old_dim[3] = bottom_W;
			std::vector<int> old_shape(shape);
			shape.clear();
			for (int i = 0; i < axis; i++)
				shape.push_back(old_dim[i]);
			if (num_axes >= 0)
			{
				for (int i = 0; i <= num_axes; i++)
					shape.push_back(old_shape[i]);
			}
			else
			{
				for (int i = 0; i < old_shape.size(); i++)
				{
					if(shape.size() < 4)
						shape.push_back(old_shape[i]);				
				}
			}
			
			int i = shape.size();
			while (i < 4)
			{
				shape.push_back(old_dim[i]);
				i++;
			}

			if (!ZQ_CNN_Tensor4D::Reshape_NCHW_get_size(shape, old_dim[0], old_dim[1], old_dim[2], old_dim[3],
				new_dim[0], new_dim[1], new_dim[2], new_dim[3]))
				return false;
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = new_dim[1];
			top_H = new_dim[2];
			top_W = new_dim[3];
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};


	class ZQ_CNN_Layer_PriorBox : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_PriorBox() 
		{
			flip = true;
			num_priors = 0;
			clip = false;
			img_w = img_h = 0;
			step_w = step_h = 0;
			offset = 0;
		}
		
		std::vector<float> min_sizes;
		std::vector<float> max_sizes;
		std::vector<float> aspect_ratios;
		std::vector<float> variance;
		bool flip;
		int num_priors;
		bool clip;
		int img_w;
		int img_h;
		float step_w;
		float step_h;
		float offset;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::PriorBox(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], *(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[1],
				min_sizes, max_sizes, aspect_ratios, variance, flip, num_priors, clip, img_w, img_h, step_w, step_h, offset, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("PriorBox layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_img_h = false, has_img_w = false, has_step_h = false, has_step_w = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("PriorBox", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("min_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						min_sizes.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_strcmpi("max_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						max_sizes.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_strcmpi("aspect_ratio", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						aspect_ratios.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_strcmpi("variance", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						variance.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_strcmpi("flip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						flip = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("clip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						clip = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("img_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_h = atoi(paras[n][1].c_str());
						img_w = img_w;
						has_img_h = true;
						has_img_w = true;
					}
				}
				else if (_strcmpi("img_h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_h = atoi(paras[n][1].c_str());
						has_img_h = true;
					}
				}
				else if (_strcmpi("img_w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_w = atoi(paras[n][1].c_str());
						has_img_w = true;
					}
				}
				else if (_strcmpi("step", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						step_w = step_w;
						has_step_h = true;
						has_step_w = true;
					}
				}
				else if (_strcmpi("step_h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						has_step_h = true;
					}
				}
				else if (_strcmpi("step_w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_w = atoi(paras[n][1].c_str());
						has_step_w = true;
					}
				}
				else if (_strcmpi("offset", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						offset = atof(paras[n][1].c_str());
					}
				}
			}

			/*if (!has_img_h || !has_img_w)
				std::cout << "Layer " << name << " missing img_h/img_w or img_size\n";
			if (!has_step_h || !has_step_w)
				std::cout << "Layer " << name << " missing step_h/step_w or step_size\n";*/

			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if(bottom_names.size() != 2)std::cout << "Layer " << name << " must have 2 bottoms\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) 
			{
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			
			if (/*!has_img_h || !has_img_w || !has_step_h || !has_step_w ||*/ !has_bottom || bottom_names.size() != 2 || !has_top || !has_name)
				return false;

			return _setup();
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || (*bottoms)[0] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[0]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
			if (!SetBottomDim(bottom_C, bottom_H, bottom_W))
				return false;
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			(*tops)[0]->SetShape(bottom_N, top_C, top_H, top_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			this->bottom_C = bottom_C;
			this->bottom_H = bottom_H;
			this->bottom_W = bottom_W;
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = 2;
			top_H = bottom_H * bottom_W * num_priors * 4;
			top_W = 1;
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

	private:
		bool _setup()
		{
			if (min_sizes.size() == 0)
			{
				std::cout << "Layer " << name << " must provide min_size\n";
				return false;
			}
			else
			{
				for (int i = 0; i < min_sizes.size(); i++)
				{
					if (min_sizes[i] <= 0)
					{
						std::cout << "Layer " << name << " min_size " << min_sizes[i] << " must be positive\n";
						return false;
					}
				}
			}
			
			std::vector<float> old_aspect_ratios(aspect_ratios);
			aspect_ratios.clear();
			aspect_ratios.push_back(1.f);
			for (int i = 0; i < old_aspect_ratios.size(); i++) 
			{
				float ar = old_aspect_ratios[i];
				bool already_exist = false;
				for (int j = 0; j < aspect_ratios.size(); ++j) 
				{
					if (fabs(ar - aspect_ratios[j]) < 1e-6) 
					{
						already_exist = true;
						break;
					}
				}
				if (!already_exist) 
				{
					aspect_ratios.push_back(ar);
					if (flip) 
					{
						aspect_ratios.push_back(1.f / ar);
					}
				}
			}
			num_priors = aspect_ratios.size() * min_sizes.size();
			if (max_sizes.size() > 0) 
			{
				if (max_sizes.size() != min_sizes.size())
				{
					std::cout << "Layer " << name << " num of min_size and max_size should be the same\n";
					return false;
				}
				
				for (int i = 0; i < max_sizes.size(); i++) 
				{
					if (max_sizes[i] <= min_sizes[i])
					{
						std::cout << "Layer " << name << " max_size must be greater than min_size\n";
						return false;
					}
					num_priors ++;
				}
			}
			
			if (variance.size() > 1) 
			{
				if (variance.size() != 4)
				{
					std::cout << "Layer " << name << " must provide 4 variance\n";
					return false;
				}
				for (int i = 0; i < variance.size(); i++)
				{
					if (variance[i] <= 0)
					{
						std::cout << "Layer " << name << " must provide positive variance\n";
						return false;
					}
				}
			}
			else if (variance.size() == 1) 
			{
				if (variance[0] <= 0)
				{
					std::cout << "Layer " << name << " must provide positive variance\n";
					return false;
				}
			}
			else 
			{
				// Set default to 0.1.
				variance.push_back(0.1);
			}

			return true;
		}
	};


	class ZQ_CNN_Layer_Concat : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Concat()
		{
			axis = 1;
		}

		int axis;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Concat_NCHW(*bottoms, axis, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Concat layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("PriorBox", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
			}

			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (bottom_names.size() < 2)std::cout << "Layer " << name << " must have >=2 bottoms\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name)
			{
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}

			if (!has_bottom || bottom_names.size() < 2 || !has_top || !has_name)
				return false;
			return true;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int out_N, out_C, out_H, out_W;
			if (!ZQ_CNN_Forward_SSEUtils::Concat_NCHW_get_size(*bottoms, axis, out_N, out_C, out_H, out_W))
				return false;
			(*tops)[0]->SetShape(out_N, out_C, out_H, out_W);
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			
		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};

	class ZQ_CNN_Layer_DetectionOutput : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_DetectionOutput()
		{
			num_classes = 1;
			share_location = true;
			background_label_id = 0;
			nms_threshold = 0.45f;
			nms_top_k = -1;
			nms_eta = 1.0f;
			code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CORNER;
			variance_encoded_in_target = false;
			// -1 means keeping all bboxes after nms step.
			keep_top_k = -1;
			// Only consider detections whose confidences are larger than a threshold.
			confidence_threshold = 0.25f;
		}


		int num_classes;
		bool share_location;
		int background_label_id;
		float nms_threshold;
		int nms_top_k;
		float nms_eta;
		ZQ_CNN_BBoxUtils::PriorBoxCodeType code_type;
		bool variance_encoded_in_target;
		int keep_top_k;		
		float confidence_threshold;
		int num_loc_classes;


		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops, int num_threads = 1)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() < 3 || tops->size() == 0 
				|| (*bottoms)[0] == 0 || (*bottoms)[1] == 0 || (*bottoms)[2] == 0 || (*tops)[0] == 0)
				return false;

			num_loc_classes = share_location ? 1 : num_classes;
			int num_priors = (*bottoms)[2]->GetH() / 4;
			if (num_priors * num_loc_classes* 4 != (*bottoms)[0]->GetC()
				|| num_priors * num_classes != (*bottoms)[1]->GetC())
			{
				printf("Number of priors must match number of location predictions\n");
				return false;
			}
			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::DetectionOuput(*(*bottoms)[0],*(*bottoms)[1],*(*bottoms)[2], 
				num_priors, num_loc_classes, num_classes, share_location, background_label_id, code_type, variance_encoded_in_target,
				nms_threshold, nms_eta, nms_top_k, confidence_threshold, keep_top_k, *((*tops)[0]));
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Concat layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string>> paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_strcmpi("PriorBox", paras[n][0].c_str()) == 0)
				{

				}
				else if (_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_strcmpi("num_classes", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						num_classes = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("share_location", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						share_location = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("background_label_id", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						background_label_id = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("nms_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("nms_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("nms_eta", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_eta = atof(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("code_type", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						if (_strcmpi(paras[n][1].c_str(), "CORNER") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CORNER;
						}
						else if (_strcmpi(paras[n][1].c_str(), "CORNER_SIZE") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CORNER_SIZE;
						}
						else if (_strcmpi(paras[n][1].c_str(), "CENTER_SIZE") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CENTER_SIZE;
						}
						else
						{
							std::cout << "Layer " << name << " unknown para " << paras[n][1] << "name\n";
						}
					}
				}
				else if (_strcmpi("keep_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						keep_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("confidence_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						confidence_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_strcmpi("variance_encoded_in_target", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						variance_encoded_in_target = atoi(paras[n][1].c_str());
					}
				}
			}

			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (bottom_names.size() != 3)std::cout << "Layer " << name << " must have 3 bottoms\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name)
			{
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}

			if (!has_bottom || bottom_names.size() != 3 || !has_top || !has_name)
				return false;
			return true;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() < 3 || tops->size() == 0
				|| (*bottoms)[0] == 0 || (*bottoms)[1] == 0 || (*bottoms)[2] == 0 || (*tops)[0] == 0)
				return false;

			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W)
		{
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{

		}

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }
	};
}

#endif
