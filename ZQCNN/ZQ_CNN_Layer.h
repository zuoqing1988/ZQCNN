#ifndef _ZQ_CNN_LAYER_H_
#define _ZQ_CNN_LAYER_H_
#pragma once
#include <string>
#include <vector>
#include <iostream>
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
		void** buffer;
		__int64* buffer_len;
		bool use_buffer;
		bool show_debug_info;
		float ignore_small_value;
		float last_cost_time;

		ZQ_CNN_Layer() :show_debug_info(false),use_buffer(false),ignore_small_value(0),last_cost_time(0) {}
		virtual ~ZQ_CNN_Layer() {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops) = 0;

		virtual bool ReadParam(const std::string& line) = 0;

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops) = 0;

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) = 0;

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& topH, int &top_W) const = 0;

		virtual bool SwapInputRGBandBGR() = 0;

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) = 0;

		virtual bool SaveBinary_NCHW(FILE* out) const = 0;

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes) = 0;

		virtual __int64 GetNumOfMulAdd() const = 0;
	public:
		static std::vector<std::vector<std::string> > split_line(const std::string& line)
		{
			std::vector<std::string> first_splits = _split_blank(line.c_str());
			int num = first_splits.size();
			std::vector<std::vector<std::string> > second_splits(num);
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

	public:
		static int _my_strcmpi(const char* str1, const char* str2)
		{
			char c1, c2;
			while (true)
			{
				c1 = tolower(*str1);
				c2 = tolower(*str2);
				str1++;
				str2++;
				if (c1 < c2)
					return -1;
				else if (c1 > c2)
					return 1;
				
				if (c1 == '\0')
					break;
			}
			return 0;
		}
	};

	class ZQ_CNN_Layer_Input : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Input() :H(0), W(0), C(3), has_H_val(false), has_W_val(false) {}
		~ZQ_CNN_Layer_Input() {}
		int H, W, C;
		bool has_H_val, has_W_val;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops) 
		{ return true; }

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_C = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Input", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_H_val = true;
						H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_W_val = true;
						W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("C", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_C = true;
						C = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes) 
		{ 
			readed_length_in_bytes = 0;
			return true; 
		}
		
		virtual __int64 GetNumOfMulAdd() const { return 0; }
	};

	class ZQ_CNN_Layer_Convolution : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Convolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1), dilate_H(1), dilate_W(1), pad_H(0), pad_W(), 
			with_bias(false), with_prelu(false), prelu_slope(0), bottom_C(0) {}
		~ZQ_CNN_Layer_Convolution() {
			if (filters)delete filters;
			if (bias)delete bias;
			if (prelu_slope) delete prelu_slope;
		}
		ZQ_CNN_Tensor4D* filters;
		ZQ_CNN_Tensor4D* bias;
		ZQ_CNN_Tensor4D* prelu_slope;
		int num_output;
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		int dilate_H;
		int dilate_W;
		int pad_H;
		int pad_W;
		bool with_bias;
		bool with_prelu;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (with_bias)
			{
				if (with_prelu)
				{
					if (filters == 0 || bias == 0 || prelu_slope == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = use_buffer ? buffer : 0;
					__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
					bool ret = ZQ_CNN_Forward_SSEUtils::ConvolutionWithBiasPReLU(*((*bottoms)[0]),
						*filters, *bias, *prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
				else
				{
					if (filters == 0 || bias == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = use_buffer ? buffer : 0;
					__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
					bool ret = ZQ_CNN_Forward_SSEUtils::ConvolutionWithBias(*((*bottoms)[0]),
						*filters, *bias, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
			}
			else
			{
				if (with_prelu)
				{
					if (filters == 0 || prelu_slope == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = use_buffer ? buffer : 0;
					__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
					bool ret = ZQ_CNN_Forward_SSEUtils::ConvolutionWithPReLU(*((*bottoms)[0]), *filters, *prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
				else
				{
					if (filters == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = use_buffer ? buffer : 0;
					__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
					bool ret = ZQ_CNN_Forward_SSEUtils::Convolution(*((*bottoms)[0]), *filters, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Convolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (_my_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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
			top_H = __max(0, floor((float)(bottom_H + pad_H * 2 - (kernel_H-1)*dilate_H-1) / stride_H) + 1);
			top_W = __max(0, floor((float)(bottom_W + pad_W * 2 - (kernel_W-1)*dilate_W-1) / stride_W) + 1);
		}

		virtual bool SwapInputRGBandBGR()
		{
			if (filters == 0)
				return false;
			if (filters->GetC() != 3)
				return false;
			int N = filters->GetN();
			int H = filters->GetH();
			int W = filters->GetW();
			int sliceStep = filters->GetSliceStep();
			int widthStep = filters->GetWidthStep();
			int pixStep = filters->GetPixelStep();
			float* ptr = filters->GetFirstPixelPtr();
			for (int n = 0; n < N; n++)
			{
				for (int h = 0; h < H; h++)
				{
					for (int w = 0; w < W; w++)
					{
						float* cur_ptr = ptr + n*sliceStep + h*widthStep + w*pixStep;
						float tmp_val = cur_ptr[0];
						cur_ptr[0] = cur_ptr[2];
						cur_ptr[2] = tmp_val;
					}
				}
			}
			return true;
		};


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
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}

		virtual bool SaveBinary_NCHW(FILE* out) const 
		{ 
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			filters->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				bias->ConvertToCompactNCHW(&nchw_raw[0]);
				if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
					return false;	
			}
			return true; 
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{ 

			readed_length_in_bytes = 0;
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = sizeof(float)*dst_len;
			if (buffer_len < dst_len_in_bytes)
				return false;
			std::vector<float> nchw_raw(dst_len);
			memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
			for (int i = 0; i < dst_len; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			buffer += dst_len_in_bytes;
			buffer_len -= dst_len_in_bytes;
			readed_length_in_bytes += dst_len_in_bytes;
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				int dst_len_in_bytes = sizeof(float)*dst_len;
				nchw_raw.resize(dst_len);
				memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
				buffer += dst_len_in_bytes;
				buffer_len -= dst_len_in_bytes;
				readed_length_in_bytes += dst_len_in_bytes;
			}
			return true; 
		}

		virtual __int64 GetNumOfMulAdd() const 
		{ 
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			__int64 total_num = (__int64)top_H*top_W*filters->GetN()*filters->GetH()*filters->GetW()*filters->GetC();
			if (with_bias)
				total_num += (__int64)top_H*top_W*top_C;
			if(with_prelu)
				total_num += (__int64)top_H*top_W*top_C*3;
			return total_num;
		}
	};


	class ZQ_CNN_Layer_DepthwiseConvolution : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_DepthwiseConvolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1),dilate_H(1),dilate_W(1), pad_H(0), pad_W(), with_bias(false), bottom_C(0), 
			with_prelu(false), prelu_slope(0) {}
		~ZQ_CNN_Layer_DepthwiseConvolution() {
			if (filters)delete filters;
			if (bias)delete bias;
			if (prelu_slope)delete prelu_slope;
		}
		ZQ_CNN_Tensor4D* filters;
		ZQ_CNN_Tensor4D* bias;
		ZQ_CNN_Tensor4D* prelu_slope;
		int num_output;
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		int dilate_H;
		int dilate_W;
		int pad_H;
		int pad_W;
		bool with_bias;
		bool with_prelu;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (with_bias)
			{
				if (with_prelu)
				{
					if (filters == 0 || bias == 0 || prelu_slope == 0)
						return false;
					double t1 = omp_get_wtime();
					bool ret = ZQ_CNN_Forward_SSEUtils::DepthwiseConvolutionWithBiasPReLU(*((*bottoms)[0]), *filters, *bias, *prelu_slope, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]));
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);

					}

					return ret;
				}
				else
				{
					if (filters == 0 || bias == 0)
						return false;
					double t1 = omp_get_wtime();
					bool ret = ZQ_CNN_Forward_SSEUtils::DepthwiseConvolutionWithBias(*((*bottoms)[0]), *filters, *bias, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]));
					double t2 = omp_get_wtime();
					last_cost_time = t2 - t1;
					if (show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);

					}

					return ret;
				}
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::DepthwiseConvolution(*((*bottoms)[0]), *filters, stride_H, stride_W, pad_H, pad_W, *((*tops)[0]));
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
				if (show_debug_info)
				{
					double time = __max(1000 * (t2 - t1), 1e-9);
					double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
					mop /= 1024 * 1024;
					printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
						name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						mop, mop / time);
				}
				return ret;
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("DepthwiseConvolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("dilate_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (_my_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };
		
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
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			filters->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				bias->ConvertToCompactNCHW(&nchw_raw[0]);
				if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
					return false;
				
			}
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = dst_len * sizeof(float);
			if (buffer_len < dst_len_in_bytes)
				return false;
			std::vector<float> nchw_raw(dst_len);
			memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
			for (int i = 0; i < dst_len; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			buffer += dst_len_in_bytes;
			buffer_len -= dst_len_in_bytes;
			readed_length_in_bytes += dst_len_in_bytes;
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				int dst_len_in_bytes = dst_len * sizeof(float);
				if (buffer_len < dst_len_in_bytes)
					return false;
				nchw_raw.resize(dst_len);
				memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
				buffer += dst_len_in_bytes;
				buffer_len -= dst_len_in_bytes;
				readed_length_in_bytes += dst_len_in_bytes;
			}
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			__int64 total_num = (__int64)top_H*top_W*filters->GetN()*filters->GetH()*filters->GetW()*filters->GetC();
			if (with_bias)
				total_num += (__int64)top_H*top_W*top_C;
			if (with_prelu)
				total_num += (__int64)top_H*top_W*top_C * 3;
			return total_num;
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
		ZQ_CNN_Layer_BatchNormScale() : mean(0), var(0), scale(0), bias(0), 
			b(0), a(0), eps(0), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_BatchNormScale() {
			if (mean) delete mean;
			if (var) delete var;
			if (scale) delete scale;
			if (bias) delete bias;
			if (b)delete b;
			if (a)delete a;
		}
		ZQ_CNN_Tensor4D* mean;
		ZQ_CNN_Tensor4D* var;
		ZQ_CNN_Tensor4D* scale;
		ZQ_CNN_Tensor4D* bias;
		ZQ_CNN_Tensor4D* b;
		ZQ_CNN_Tensor4D* a;

		float eps;
		//
		bool with_bias;
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
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
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("BatchNorm layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("BatchNormScale", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("eps", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						eps = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("bias", paras[n][0].c_str()) == 0)
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
			if (mean)
			{
				if (!mean->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				mean = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (mean == 0)return false;
				if (!mean->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			if (var)
			{
				if (!var->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				var = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (var == 0)return false;
				if (!var->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (mean == 0 || var == 0 || scale == 0 || (with_bias && bias == 0) || b == 0 || a == 0)
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
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len*4; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				bias->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 3, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScaleBias_Compute_b_a(*b, *a, *mean, *var, *scale, *bias, eps);
			}
			else
			{
				std::vector<float> nchw_raw(dst_len * 3);
				if (dst_len * 3 != fread_s(&nchw_raw[0], dst_len * 3 * sizeof(float), sizeof(float), dst_len * 3, in))
					return false;
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len*3; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScale_Compute_b_a(*b, *a, *mean, *var, *scale, eps);
			}
			
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (mean == 0 || var == 0 || scale == 0 || (with_bias && bias == 0) || b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			if (with_bias)
			{
				std::vector<float> nchw_raw(dst_len * 4);
				mean->ConvertToCompactNCHW(&nchw_raw[0]);
				var->ConvertToCompactNCHW(&nchw_raw[0] + dst_len);
				scale->ConvertToCompactNCHW(&nchw_raw[0] + dst_len * 2);
				bias->ConvertToCompactNCHW(&nchw_raw[0] + dst_len * 3);
				if (dst_len * 4 != fwrite(&nchw_raw[0], sizeof(float), dst_len * 4, out))
					return false;
				return true;
			}
			else
			{
				std::vector<float> nchw_raw(dst_len * 3);
				mean->ConvertToCompactNCHW(&nchw_raw[0]);
				var->ConvertToCompactNCHW(&nchw_raw[0] + dst_len);
				scale->ConvertToCompactNCHW(&nchw_raw[0] + dst_len * 2);
				if (dst_len * 3 != fwrite(&nchw_raw[0], sizeof(float), dst_len * 3, out))
					return false;
				return true;
			}
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{

			readed_length_in_bytes = 0;
			if (mean == 0 || var == 0 || scale == 0 || (with_bias && bias == 0) || b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			if (with_bias)
			{
				std::vector<float> nchw_raw(dst_len * 4);
				if (dst_len * 4 * sizeof(float) > buffer_len)
					return false;
				memcpy(&nchw_raw[0], buffer, dst_len * 4 * sizeof(float));
				readed_length_in_bytes += dst_len * 4 * sizeof(float);
				for (int i = 0; i < dst_len * 4; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				bias->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 3, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScaleBias_Compute_b_a(*b, *a, *mean, *var, *scale, *bias, eps);
			}
			else
			{
				std::vector<float> nchw_raw(dst_len * 3);
				if (dst_len * 3 * sizeof(float) > buffer_len)
					return false;
				memcpy(&nchw_raw[0], buffer, dst_len * 3 * sizeof(float));
				readed_length_in_bytes += dst_len * 3 * sizeof(float);
				for (int i = 0; i < dst_len * 3; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils::BatchNormScale_Compute_b_a(*b, *a, *mean, *var, *scale, eps);
			}

			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_W*bottom_H*bottom_C;
		}
	};

	class ZQ_CNN_Layer_BatchNorm : public ZQ_CNN_Layer
	{
		/*
		a = - mean / sqrt(var+eps)
		b = 1 / sqrt(var+eps)
		value = b * value + a
		*/
	public:
		ZQ_CNN_Layer_BatchNorm() : mean(0), var(0), b(0), a(0),eps(0), bottom_C(0) {}
		~ZQ_CNN_Layer_BatchNorm() {
			if (mean) delete mean;
			if (var) delete var;
			if (b)delete b;
			if (a)delete a;
		}
		ZQ_CNN_Tensor4D* mean;
		ZQ_CNN_Tensor4D* var;
		ZQ_CNN_Tensor4D* b;
		ZQ_CNN_Tensor4D* a;

		float eps;

		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
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
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("BatchNorm layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("BatchNorm", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("eps", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						eps = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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
			if (mean)
			{
				if (!mean->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				mean = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (mean == 0)return false;
				if (!mean->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			if (var)
			{
				if (!var->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
			else
			{
				var = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (var == 0)return false;
				if (!var->ChangeSize(1, 1, 1, bottom_C, 0, 0))
					return false;
			}
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
		
		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in)
		{
			if (mean == 0 || var == 0 || b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len * 2);
			if (dst_len * 2 != fread_s(&nchw_raw[0], dst_len * 2 * sizeof(float), sizeof(float), dst_len * 2, in))
				return false;
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len*2; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
			var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
			return ZQ_CNN_Forward_SSEUtils::BatchNorm_Compute_b_a(*b, *a, *mean, *var, eps);
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (mean == 0 || var == 0 || b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len * 2);
			mean->ConvertToCompactNCHW(&nchw_raw[0]);
			var->ConvertToCompactNCHW(&nchw_raw[0] + dst_len);
			if (dst_len * 2 != fwrite(&nchw_raw[0], sizeof(float), dst_len * 2, out))
				return false;
			
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			if (mean == 0 || var == 0 || b == 0 || a == 0)
				return false;
			int N = b->GetN(), H = b->GetH(), W = b->GetW(), C = b->GetC();
			int dst_len = N*H*W*C;
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len * 2);
			if (dst_len * 2 * sizeof(float) > buffer_len)
				return false;

			memcpy(&nchw_raw[0], buffer, dst_len * 2 * sizeof(float));
			readed_length_in_bytes += dst_len * 2 * sizeof(float);
			for (int i = 0; i < dst_len * 2; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
			var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
			return ZQ_CNN_Forward_SSEUtils::BatchNorm_Compute_b_a(*b, *a, *mean, *var, eps);
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_W*bottom_H*bottom_C;
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
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
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
				last_cost_time = t2 - t1;
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
				last_cost_time = t2 - t1;
				if (show_debug_info)
					printf("Scale layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Scale", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

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
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			scale->ConvertFromCompactNCHW(&nchw_raw[0], scale->GetN(), scale->GetC(), scale->GetH(), scale->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
					return false;
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (scale == 0 || (with_bias && bias == 0))
				return false;
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			scale->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				bias->ConvertToCompactNCHW(&nchw_raw[0]);
				if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
					return false;
			}
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			if (scale == 0)
				return false;
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = dst_len * sizeof(float);
			if (dst_len_in_bytes > buffer_len)
				return false;
			std::vector<float> nchw_raw(dst_len);
			memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
			for (int i = 0; i < dst_len; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			scale->ConvertFromCompactNCHW(&nchw_raw[0], scale->GetN(), scale->GetC(), scale->GetH(), scale->GetW());
			buffer += dst_len_in_bytes;
			buffer_len -= dst_len_in_bytes;
			readed_length_in_bytes += dst_len_in_bytes;
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				int dst_len_in_bytes = dst_len * sizeof(float);
				if (dst_len_in_bytes > buffer_len)
					return false;
				nchw_raw.resize(dst_len);
				memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
				buffer += dst_len_in_bytes;
				buffer_len -= dst_len_in_bytes;
				readed_length_in_bytes += dst_len_in_bytes;
			}
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_W*bottom_H*bottom_C;
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
		~ZQ_CNN_Layer_PReLU() 
		{
			if (slope) delete slope;
		}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
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
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("PReLU layer: %s %.3f ms \n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("PReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

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
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			slope->ConvertFromCompactNCHW(&nchw_raw[0], slope->GetN(), slope->GetC(), slope->GetH(), slope->GetW());
			return true;
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (slope == 0)
				return false;
			int dst_len = slope->GetN() * slope->GetH() * slope->GetW() * slope->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			slope->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			if (slope == 0)
				return false;
			int dst_len = slope->GetN() * slope->GetH() * slope->GetW() * slope->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = dst_len * sizeof(float);
			if (dst_len_in_bytes > buffer_len)
				return false;
			std::vector<float> nchw_raw(dst_len);
			memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
			for (int i = 0; i < dst_len; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			slope->ConvertFromCompactNCHW(&nchw_raw[0], slope->GetN(), slope->GetC(), slope->GetH(), slope->GetW());
			readed_length_in_bytes += dst_len_in_bytes;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return (__int64)bottom_W*bottom_H*bottom_C*3;
		}
	};

	class ZQ_CNN_Layer_ReLU : public ZQ_CNN_Layer
	{
	public:

		float slope;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_ReLU() :slope(0),bottom_C(0) {}
		~ZQ_CNN_Layer_ReLU(){}

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils::ReLU(*((*tops)[0]), slope);
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("ReLU layer: %s %.3f ms \n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}


		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("ReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("slope", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						slope = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in){	return true;}

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_W*bottom_H*bottom_C;
		}
	};

	class ZQ_CNN_Layer_Pooling : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Pooling() :kernel_H(3), kernel_W(3), stride_H(2), stride_W(2), 
			pad_H(0), pad_W(0), global_pool(false), type(0) {}
		~ZQ_CNN_Layer_Pooling() {}
		int kernel_H;
		int kernel_W;
		int stride_H;
		int stride_W;
		int pad_H;
		int pad_W;
		bool global_pool;
		static const int TYPE_MAXPOOLING = 0;
		static const int TYPE_AVGPOOLING = 1;
		int type;	//0-MAX,1-AVG
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			if (type == TYPE_MAXPOOLING)
			{
				double t1 = omp_get_wtime();
				ZQ_CNN_Forward_SSEUtils::MaxPooling(*((*bottoms)[0]), *((*tops)[0]), kernel_H, kernel_W, stride_H, stride_W, global_pool);
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
				if (show_debug_info)
					printf("Pooling layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return true;
			}
			else if (type == TYPE_AVGPOOLING)
			{
				double t1 = omp_get_wtime();
				ZQ_CNN_Forward_SSEUtils::AVGPooling(*((*bottoms)[0]), *((*tops)[0]), kernel_H, kernel_W, stride_H, stride_W, global_pool);
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
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
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_kernelH = false, has_kernelW = false;
			bool has_strideH = false, has_strideW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Pooling", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_strideH = true;
						stride_H = atoi(paras[n][1].c_str());
						has_strideW = true;
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("global_pool", paras[n][0].c_str()) == 0)
				{
					global_pool = true;
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("pool", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						const char* str = paras[n][1].c_str();
						if (_my_strcmpi(str,"MAX") == 0)
							type = TYPE_MAXPOOLING;
						else if (_my_strcmpi(str,"AVG") == 0 || _my_strcmpi(str, "AVE") == 0)
							type = TYPE_AVGPOOLING;
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
			if (!global_pool)
			{
				if (!has_kernelH)std::cout << "Layer " << name << " missing " << "kernel_H (kernel_size)\n";
				if (!has_kernelW)std::cout << "Layer " << name << " missing " << "kernel_W (kernel_size)\n";
				if (!has_strideH)std::cout << "Layer " << name << " missing " << "stride_H (stride)\n";
				if (!has_strideW)std::cout << "Layer " << name << " missing " << "stride_W (stride)\n";
			}
			
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			if(!global_pool)
				return has_kernelH && has_kernelW && has_strideH && has_strideW && has_bottom && has_top && has_name;
			else
			{
				return has_bottom && has_top && has_name;
			}
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
			if (global_pool)
			{
				top_C = bottom_C;
				top_H = 1;
				top_W = 1;
			}
			else
			{
				top_C = bottom_C;
				top_H = __max(0, ceil((float)(bottom_H - kernel_H) / stride_H) + 1);
				top_W = __max(0, ceil((float)(bottom_W - kernel_W) / stride_W) + 1);
			}
		}

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			return (__int64)top_H*top_W*top_C*kernel_H*kernel_W;
		}
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
		~ZQ_CNN_Layer_InnerProduct()
		{
			if (filters) delete filters;
			if (bias) delete bias;
		}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if (with_bias)
			{
				if (filters == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				void** tmp_buffer = use_buffer ? buffer : 0;
				__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
				bool ret = ZQ_CNN_Forward_SSEUtils::InnerProductWithBias(*((*bottoms)[0]), 
					*filters, *bias, *((*tops)[0]), tmp_buffer, tmp_buffer_len);
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
				if (show_debug_info)
					printf("Innerproduct layer: %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d\n", 
						1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), 
						filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC());
				return ret;
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				void** tmp_buffer = use_buffer ? buffer : 0;
				__int64* tmp_buffer_len = use_buffer ? buffer_len : 0;
				bool ret = ZQ_CNN_Forward_SSEUtils::InnerProduct(*((*bottoms)[0]), *filters, *((*tops)[0]),
					tmp_buffer,tmp_buffer_len);
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
				if (show_debug_info)
					printf("Innerproduct layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
				return ret;
			}

		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_num_output = false/*, has_kernelH = false, has_kernelW = false*/;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("InnerProduct", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				/*else if (_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}*/
				else if (_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR()
		{
			if (filters == 0)
				return false;
			if (filters->GetC() != 3)
				return false;
			int N = filters->GetN();
			int H = filters->GetH();
			int W = filters->GetW();
			int sliceStep = filters->GetSliceStep();
			int widthStep = filters->GetWidthStep();
			int pixStep = filters->GetPixelStep();
			float* ptr = filters->GetFirstPixelPtr();
			for (int n = 0; n < N; n++)
			{
				for (int h = 0; h < H; h++)
				{
					for (int w = 0; w < W; w++)
					{
						float* cur_ptr = ptr + n*sliceStep + h*widthStep + w*pixStep;
						float tmp_val = cur_ptr[0];
						cur_ptr[0] = cur_ptr[2];
						cur_ptr[2] = tmp_val;
					}
				}
			}
			return true;
		};

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) 
		{
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			int readed_len = fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in);
			if (dst_len != readed_len)
				return false;
			if (ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				//int pos = ftell(in);
				readed_len = fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in);
				if (dst_len != readed_len)
					return false;
				if (ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				bias->ConvertFromCompactNCHW(&nchw_raw[0], bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
			}
			return true;
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (filters == 0 || (with_bias && bias == 0))
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			filters->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				nchw_raw.resize(dst_len);
				bias->ConvertToCompactNCHW(&nchw_raw[0]);
				if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
					return false;
			}
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			if (filters == 0)
				return false;
			int dst_len = filters->GetN() * filters->GetH() * filters->GetW() * filters->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = dst_len * sizeof(float);
			if (dst_len_in_bytes > buffer_len)
				return false;
			std::vector<float> nchw_raw(dst_len);
			memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
			for (int i = 0; i < dst_len; i++)
			{
				if (fabs(nchw_raw[i]) < ignore_small_value)
					nchw_raw[i] = 0;
			}
			filters->ConvertFromCompactNCHW(&nchw_raw[0], filters->GetN(), filters->GetC(), filters->GetH(), filters->GetW());
			buffer += dst_len_in_bytes;
			buffer_len -= dst_len_in_bytes;
			readed_length_in_bytes += dst_len_in_bytes;
			if (with_bias)
			{
				int dst_len = bias->GetN() * bias->GetH() * bias->GetW() * bias->GetC();
				if (dst_len <= 0)
					return false;
				int dst_len_in_bytes = dst_len * sizeof(float);
				if (dst_len_in_bytes > buffer_len)
					return false;
				nchw_raw.resize(dst_len);
				memcpy(&nchw_raw[0], buffer, dst_len_in_bytes);
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ignore_small_value)
						nchw_raw[i] = 0;
				}
				bias->ConvertFromCompactNCHW((const float*)buffer, bias->GetN(), bias->GetC(), bias->GetH(), bias->GetW());
				buffer += dst_len_in_bytes;
				buffer_len -= dst_len_in_bytes;
				readed_length_in_bytes += dst_len_in_bytes;
			}
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			int top_C, top_H, top_W;
			GetTopDim(top_C, top_H, top_W);
			return (__int64)top_H*top_W*filters->GetN()*filters->GetH()*filters->GetW()*filters->GetC();
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
		~ZQ_CNN_Layer_Softmax() {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils::Softmax(*((*tops)[0]), axis);
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Softmax layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Softmax", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("axis", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Dropout : public ZQ_CNN_Layer
	{
	public:
		float dropout_ratio;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_Dropout():dropout_ratio(1.0f) {}
		~ZQ_CNN_Layer_Dropout(){}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			/*dropout ratio is not used in test phase*/
			//ZQ_CNN_Forward_SSEUtils::Dropout(*((*tops)[0]), dropout_ratio);
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Dropout layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Dropout", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("dropout_ratio", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dropout_ratio = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Copy : public ZQ_CNN_Layer
	{
	public:
		float dropout_ratio;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_Copy() {}
		~ZQ_CNN_Layer_Copy() {}
		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Copy layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Copy", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Eltwise : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Eltwise() :with_weight(false) {}
		~ZQ_CNN_Layer_Eltwise(){}

		static const int ELTWISE_MUL = 0;
		static const int ELTWISE_SUM = 1;
		static const int ELTWISE_MAX = 2;
		int operation;//
		bool with_weight;
		std::vector<float> weight;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			bool ret = false;
			double t1 = omp_get_wtime();
			if (operation == ELTWISE_MUL)
			{
				ret = ZQ_CNN_Forward_SSEUtils::Eltwise_Mul(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else if (operation == ELTWISE_SUM)
			{
				if (with_weight)
				{
					ret = ZQ_CNN_Forward_SSEUtils::Eltwise_SumWithWeight(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, weight, *((*tops)[0]));
				}
				else
				{
					ret = ZQ_CNN_Forward_SSEUtils::Eltwise_Sum(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
				}
			}
			else if (operation == ELTWISE_MAX)
			{
				ret = ZQ_CNN_Forward_SSEUtils::Eltwise_Max(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown eltwise operation " << operation << " in Layer " << name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Eltwise layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			weight.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			with_weight = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Eltwise", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (_my_strcmpi(str, "PROD") == 0 || _my_strcmpi(str, "MUL") == 0)
						{
							operation = ELTWISE_MUL;
						}
						else if (_my_strcmpi(str, "SUM") == 0 || _my_strcmpi(str, "ADD") == 0 || _my_strcmpi(str, "PLUS") == 0)
						{
							operation = ELTWISE_SUM;
						}
						else if(_my_strcmpi(str,"MAX") == 0)
						{
							operation = ELTWISE_MAX;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("weight", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};

	class ZQ_CNN_Layer_ScalarOperation : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_ScalarOperation():scalar(1),operation(0){}
		~ZQ_CNN_Layer_ScalarOperation(){}

		static const int SCALAR_MUL = 0;
		static const int SCALAR_DIV = 1;
		static const int SCALAR_ADD = 2;
		static const int SCALAR_MINUS = 3;
		static const int SCALAR_MAX = 4;
		static const int SCALAR_MIN = 5;
		static const int SCALAR_POW = 6;
		static const int SCALAR_RDIV = 7;
		static const int SCALAR_RMINUS = 8;
		
		int operation;//
		float scalar;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			if (operation == SCALAR_MUL)
			{
				if((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[0],scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_DIV)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[0], 1.0/scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[0], 1.0/scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_ADD)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_MINUS)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[0], -scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[0], -scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_MAX)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Max(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Max(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_MIN)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Min(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Min(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_POW)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Pow(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Pow(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_RDIV)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rdiv(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rdiv(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else if (operation == SCALAR_RMINUS)
			{
				if ((*bottoms)[0] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rminus(*(*bottoms)[0], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rminus(*(*bottoms)[0], scalar, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown ScalarOperation " << operation << " in Layer " << name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("ScalarOperation layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_scalar = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("ScalarOperation", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (_my_strcmpi(str, "MUL") == 0 || _my_strcmpi(str, "PROD") == 0)
						{
							operation = SCALAR_MUL;
						}
						else if (_my_strcmpi(str, "DIV") == 0)
						{
							operation = SCALAR_DIV;
						}
						else if (_my_strcmpi(str, "ADD") == 0 || _my_strcmpi(str, "SUM") == 0 || _my_strcmpi(str, "PLUS") == 0)
						{
							operation = SCALAR_ADD;
						}
						else if (_my_strcmpi(str, "MINUS") == 0 || _my_strcmpi(str, "SUB") == 0)
						{
							operation = SCALAR_MINUS;
						}
						else if (_my_strcmpi(str, "MAX") == 0)
						{
							operation = SCALAR_MAX;
						}
						else if (_my_strcmpi(str, "MIN") == 0)
						{
							operation = SCALAR_MIN;
						}
						else if (_my_strcmpi(str, "POW") == 0)
						{
							operation = SCALAR_POW;
						}
						else if (_my_strcmpi(str, "RDIV") == 0)
						{
							operation = SCALAR_RDIV;
						}
						else if (_my_strcmpi(str, "RMINUS") == 0)
						{
							operation = SCALAR_RMINUS;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("scalar", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_scalar = true;
						scalar = atof(paras[n][1].c_str());
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << name << "\n";
				}

			}
			if (!has_operation)std::cout << "Layer " << name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_scalar) std::cout << "Layer " << name << " missing " << "scalar\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_operation && has_bottom && has_scalar && has_top && has_name;
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};

	class ZQ_CNN_Layer_UnaryOperation : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_UnaryOperation() :operation(0) {}
		~ZQ_CNN_Layer_UnaryOperation() {}

		static const int UNARY_MUL = 0;
		static const int UNARY_DIV = 1;
		static const int UNARY_ADD = 2;
		static const int UNARY_MINUS = 3;
		static const int UNARY_MAX = 4;
		static const int UNARY_MIN = 5;
		static const int UNARY_POW = 6;
		static const int UNARY_RDIV = 7;
		static const int UNARY_RMINUS = 8;

		int operation;//
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() != 2 || tops->size() == 0 
				|| (*bottoms)[0] == 0 || (*bottoms)[1] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			float scalar = (*bottoms)[0]->GetFirstPixelPtr()[0];
			if (operation == UNARY_MUL)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_DIV)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[1], 1.0 / scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Mul(*(*bottoms)[1], 1.0 / scalar , *((*tops)[0]));
			}
			else if (operation == UNARY_ADD)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_MINUS)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[1], -scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Add(*(*bottoms)[1], -scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_MAX)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Max(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Max(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_MIN)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Min(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Min(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_POW)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Pow(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Pow(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_RDIV)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rdiv(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rdiv(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else if (operation == UNARY_RMINUS)
			{
				if ((*bottoms)[1] == (*tops)[0])
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rminus(*(*bottoms)[1], scalar);
				else
					return ZQ_CNN_Forward_SSEUtils::ScalarOperation_Rminus(*(*bottoms)[1], scalar, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown UnaryOperation " << operation << " in Layer " << name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("UnaryOperation layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("UnaryOperation", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (_my_strcmpi(str, "MUL") == 0 || _my_strcmpi(str, "PROD") == 0)
						{
							operation = UNARY_MUL;
						}
						else if (_my_strcmpi(str, "DIV") == 0)
						{
							operation = UNARY_DIV;
						}
						else if (_my_strcmpi(str, "ADD") == 0 || _my_strcmpi(str, "SUM") == 0 || _my_strcmpi(str, "PLUS") == 0)
						{
							operation = UNARY_ADD;
						}
						else if (_my_strcmpi(str, "MINUS") == 0 || _my_strcmpi(str, "SUB") == 0)
						{
							operation = UNARY_MINUS;
						}
						else if (_my_strcmpi(str, "MAX") == 0)
						{
							operation = UNARY_MAX;
						}
						else if (_my_strcmpi(str, "MIN") == 0)
						{
							operation = UNARY_MIN;
						}
						else if (_my_strcmpi(str, "POW") == 0)
						{
							operation = UNARY_POW;
						}
						else if (_my_strcmpi(str, "RDIV") == 0)
						{
							operation = UNARY_RDIV;
						}
						else if (_my_strcmpi(str, "RMINUS") == 0)
						{
							operation = UNARY_RMINUS;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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
			if (!has_operation)std::cout << "Layer " << name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			int bottom_num = bottom_names.size();
			if (bottom_num != 2) std::cout << "Layer " << name << " need 2 bottoms\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_operation && has_bottom && bottom_num == 2 && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() != 2  || 
				(*bottoms)[0] == 0 || (*bottoms)[1] == 0 || tops->size() == 0 || (*tops)[0] == 0)
				return false;
			int bottom_N, bottom_C, bottom_H, bottom_W;
			(*bottoms)[1]->GetShape(bottom_N, bottom_C, bottom_H, bottom_W);
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};

	class ZQ_CNN_Layer_LRN : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_LRN(): k(1.0f) { operation = LRN_ACROSS_CHANNELS;}
		~ZQ_CNN_Layer_LRN(){ }

		static const int LRN_ACROSS_CHANNELS = 0;
		int operation;//
		int local_size;
		float alpha, beta, k;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			
			if (operation == LRN_ACROSS_CHANNELS)
			{
				double t1 = omp_get_wtime();
				bool ret = ZQ_CNN_Forward_SSEUtils::LRN_across_channels(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0],local_size,alpha,beta,k, *((*tops)[0]));
				double t2 = omp_get_wtime();
				last_cost_time = t2 - t1;
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
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_local_size = false, has_alpha = false, has_beta = false;
			
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("LRN", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						operation = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("local_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						local_size = atoi(paras[n][1].c_str());
						has_local_size = true;
					}
				}
				else if (_my_strcmpi("alpha", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						alpha = atof(paras[n][1].c_str());
						has_alpha = true;
					}
				}
				else if (_my_strcmpi("beta", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						beta = atof(paras[n][1].c_str());
						has_beta = true;
					}
				}
				else if (_my_strcmpi("k", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Normalize : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Normalize():across_spatial(false),channel_shared(false), eps(1e-10),scale(0){}
		~ZQ_CNN_Layer_Normalize(){
			if (scale) delete scale;
		}

		bool across_spatial;
		bool channel_shared;
		ZQ_CNN_Tensor4D* scale;
		float eps;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Normalize(*((*tops)[0]), *scale, across_spatial, channel_shared);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("Normalize layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Normalize", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("across_spatial", paras[n][0].c_str()) == 0)
				{
					across_spatial = true;
				}
				else if (_my_strcmpi("channel_shared", paras[n][0].c_str()) == 0)
				{
					across_spatial = true;
				}
				else if (_my_strcmpi("eps", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						eps = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
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

			int num_channel = channel_shared ? 1 : bottom_C;
			if (scale)
			{
				if (!scale->ChangeSize(1, 1, 1, num_channel, 0, 0))
					return false;
			}
			else
			{
				scale = new ZQ_CNN_Tensor4D_NHW_C_Align256bit();
				if (scale == 0)return false;
				if (!scale->ChangeSize(1, 1, 1, num_channel, 0, 0))
					return false;
			}
			return true;
		}

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const
		{
			top_C = this->bottom_C;
			top_H = this->bottom_H;
			top_W = this->bottom_W;
		}

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) 
		{
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			if (dst_len != fread_s(&nchw_raw[0], dst_len * sizeof(float), sizeof(float), dst_len, in))
				return false;
			scale->ConvertFromCompactNCHW(&nchw_raw[0], scale->GetN(), scale->GetC(), scale->GetH(), scale->GetW());
			
			return true; 
		}

		virtual bool SaveBinary_NCHW(FILE* out) const
		{
			if (scale == 0)
				return false;
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			std::vector<float> nchw_raw(dst_len);
			scale->ConvertToCompactNCHW(&nchw_raw[0]);
			if (dst_len != fwrite(&nchw_raw[0], sizeof(float), dst_len, out))
				return false;
			
			return true;
		}

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			int dst_len = scale->GetN() * scale->GetH() * scale->GetW() * scale->GetC();
			if (dst_len <= 0)
				return false;
			int dst_len_in_bytes = dst_len * sizeof(float);
			scale->ConvertFromCompactNCHW((const float*)buffer, scale->GetN(), scale->GetC(), scale->GetH(), scale->GetW());
			readed_length_in_bytes += dst_len_in_bytes;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Permute : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Permute() { order[0] = 0; order[1] = 1; order[2] = 2; order[3] = 3; }
		~ZQ_CNN_Layer_Permute() {}

		int order[4];
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Permute(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], order, *((*tops)[0]));
			int C = (*bottoms)[0]->GetC();
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Permute layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			int order_num = 0;
			bool has_order_flag[4] = { false };
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Permute", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("order", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Flatten : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Flatten() { axis = 1; end_axis = -1; }
		~ZQ_CNN_Layer_Flatten() {}

		int axis;
		int end_axis;
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Flatten(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], axis, end_axis, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Flatten layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Flatten", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("end_axis", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Reshape : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Reshape() { axis = 0; num_axes = -1; }
		~ZQ_CNN_Layer_Reshape() {}

		std::vector<int> shape;
		int axis;
		int num_axes;
		int old_dim[4];
		int new_dim[4];
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Reshape(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], shape, *((*tops)[0]));
			const float* ptr = (*bottoms)[0]->GetFirstPixelPtr();
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Reshape layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;

			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Reshape", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("dim", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						shape.push_back(atoi(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("num_axes", paras[n][0].c_str()) == 0)
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


		virtual bool SwapInputRGBandBGR() { return true; };


		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
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
		~ZQ_CNN_Layer_PriorBox() {}
		
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

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			img_w = (*bottoms)[1]->GetW();
			img_h = (*bottoms)[1]->GetH();
			for (int i = 0; i < min_sizes.size(); i++)
			{
				if (min_sizes[i] < 0)
					min_sizes[i] = (-min_sizes[i])*img_w;
			}
			for (int i = 0; i < max_sizes.size(); i++)
			{
				if (max_sizes[i] < 0)
					max_sizes[i] = (-max_sizes[i])*img_w;
			}
			bool ret = ZQ_CNN_Forward_SSEUtils::PriorBox(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], *(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[1],
				min_sizes, max_sizes, aspect_ratios, variance, flip, num_priors, clip, img_w, img_h, step_w, step_h, offset, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("PriorBox layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_img_h = false, has_img_w = false, has_step_h = false, has_step_w = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("PriorBox", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("min_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						min_sizes.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("max_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						max_sizes.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("aspect_ratio", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						aspect_ratios.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("variance", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						variance.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("flip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						flip = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("clip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						clip = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("img_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_h = atoi(paras[n][1].c_str());
						img_w = img_w;
						has_img_h = true;
						has_img_w = true;
					}
				}
				else if (_my_strcmpi("img_h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_h = atoi(paras[n][1].c_str());
						has_img_h = true;
					}
				}
				else if (_my_strcmpi("img_w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						img_w = atoi(paras[n][1].c_str());
						has_img_w = true;
					}
				}
				else if (_my_strcmpi("step", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						step_w = step_w;
						has_step_h = true;
						has_step_w = true;
					}
				}
				else if (_my_strcmpi("step_h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						has_step_h = true;
					}
				}
				else if (_my_strcmpi("step_w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_w = atoi(paras[n][1].c_str());
						has_step_w = true;
					}
				}
				else if (_my_strcmpi("offset", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}

	private:
		virtual bool _setup()
		{
			if (min_sizes.size() == 0)
			{
				std::cout << "Layer " << name << " must provide min_size\n";
				return false;
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


	class ZQ_CNN_Layer_PriorBoxText : public ZQ_CNN_Layer_PriorBox
	{
	public:
		ZQ_CNN_Layer_PriorBoxText()
		{
			ZQ_CNN_Layer_PriorBox();
		}
		~ZQ_CNN_Layer_PriorBoxText() {}


		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::PriorBoxText(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], *(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[1],
				min_sizes, max_sizes, aspect_ratios, variance, flip, num_priors, clip, img_w, img_h, step_w, step_h, offset, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("PriorBox layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		
	private:
		virtual bool _setup()
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
					num_priors++;
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

			num_priors *= 2;
			return true;
		}
	};

	class ZQ_CNN_Layer_PriorBox_MXNET : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_PriorBox_MXNET()
		{
			num_priors = 0;
			clip = false;
			step_w = step_h = 0;
			offset = 0;
		}
		~ZQ_CNN_Layer_PriorBox_MXNET() {}

		std::vector<float> sizes;
		std::vector<float> aspect_ratios;
		std::vector<float> variances;
		int num_priors;
		bool clip;
		float step_w;
		float step_h;
		float offset;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			for (int i = 0; i < sizes.size(); i++)
			{
				if (sizes[i] < 0)
					sizes[i] = (-sizes[i]);
			}
			bool ret = ZQ_CNN_Forward_SSEUtils::PriorBox_MXNET(*(*(std::vector<const ZQ_CNN_Tensor4D*>*)bottoms)[0], 
				sizes, aspect_ratios, variances, num_priors, clip, step_w, step_h, offset, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("PriorBox_MXNET layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			bool has_img_h = false, has_img_w = false, has_step_h = false, has_step_w = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("PriorBox_MXNET", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						sizes.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("aspect_ratio", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						aspect_ratios.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("variance", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						variances.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("clip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						clip = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("step", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						step_w = step_w;
						has_step_h = true;
						has_step_w = true;
					}
				}
				else if (_my_strcmpi("step_h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_h = atoi(paras[n][1].c_str());
						has_step_h = true;
					}
				}
				else if (_my_strcmpi("step_w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						step_w = atoi(paras[n][1].c_str());
						has_step_w = true;
					}
				}
				else if (_my_strcmpi("offset", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						offset = atof(paras[n][1].c_str());
					}
				}
			}

			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name)
			{
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}

			if (!has_bottom || !has_top || !has_name)
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
			top_C = 1;
			top_H = bottom_H * bottom_W * num_priors;
			top_W = 4;
		}

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}

	private:
		virtual bool _setup()
		{
			if (sizes.size() == 0)
			{
				std::cout << "Layer " << name << " must provide size\n";
				return false;
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
				}
			}
			num_priors = aspect_ratios.size() + sizes.size() - 1;
			
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
		~ZQ_CNN_Layer_Concat() {}

		int axis;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::Concat_NCHW(*bottoms, axis, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Concat layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Concat", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("axis", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
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
		~ZQ_CNN_Layer_DetectionOutput() {}


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


		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
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
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Concat layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("DetectionOutput", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("num_classes", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						num_classes = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("share_location", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						share_location = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("background_label_id", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						background_label_id = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("nms_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("nms_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("nms_eta", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_eta = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("code_type", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						if (_my_strcmpi(paras[n][1].c_str(), "CORNER") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CORNER;
						}
						else if (_my_strcmpi(paras[n][1].c_str(), "CORNER_SIZE") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CORNER_SIZE;
						}
						else if (_my_strcmpi(paras[n][1].c_str(), "CENTER_SIZE") == 0)
						{
							code_type = ZQ_CNN_BBoxUtils::PriorBoxCodeType_CENTER_SIZE;
						}
						else
						{
							std::cout << "Layer " << name << " unknown para " << paras[n][1] << "name\n";
						}
					}
				}
				else if (_my_strcmpi("keep_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						keep_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("confidence_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						confidence_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("variance_encoded_in_target", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_DetectionOutput_MXNET : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_DetectionOutput_MXNET()
		{
			nms_threshold = 0.45f;
			nms_top_k = -1;
			clip = false;
			// -1 means keeping all bboxes after nms step.
			keep_top_k = -1;
			// Only consider detections whose confidences are larger than a threshold.
			confidence_threshold = 0.25f;
		}
		~ZQ_CNN_Layer_DetectionOutput_MXNET() {}
		
		float nms_threshold;
		int nms_top_k;
		std::vector<float> variances;
		bool clip;
		int keep_top_k;
		float confidence_threshold;
		int num_loc_classes;


		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() < 3 || tops->size() == 0
				|| (*bottoms)[0] == 0 || (*bottoms)[1] == 0 || (*bottoms)[2] == 0 || (*tops)[0] == 0)
				return false;

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils::DetectionOuput_MXNET(*(*bottoms)[1], *(*bottoms)[0], *(*bottoms)[2],
				variances,	clip, nms_threshold, nms_top_k, confidence_threshold, keep_top_k, *((*tops)[0]));
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Concat layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("DetectionOutput_MXNET", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						name = paras[n][1];
					}
				}
				else if (_my_strcmpi("nms_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("nms_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						nms_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("keep_top_k", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						keep_top_k = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("confidence_threshold", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						confidence_threshold = atof(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("variance", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						variances.push_back(atof(paras[n][1].c_str()));
					}
				}
				else if (_my_strcmpi("clip", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						clip = atoi(paras[n][1].c_str());
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return 0;
		}
	};

	class ZQ_CNN_Layer_Reduction : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Reduction(): axis(1), keepdims(false){}
		~ZQ_CNN_Layer_Reduction() {}

		static const int REDUCTION_SUM = 0;
		static const int REDUCTION_MEAN = 1;

		int axis;
		bool keepdims;
		int operation;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			bool ret = false;
			double t1 = omp_get_wtime();
			if(operation == REDUCTION_SUM)
				ret = ZQ_CNN_Forward_SSEUtils::ReductionSum(*((*bottoms)[0]), axis, keepdims, *((*tops)[0]));
			else if(operation == REDUCTION_MEAN)
			{
				ret = ZQ_CNN_Forward_SSEUtils::ReductionMean(*((*bottoms)[0]), axis, keepdims, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown reduction operation " << operation << " in Layer " << name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Reduction layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Reduction", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (_my_strcmpi(str, "SUM") == 0 || _my_strcmpi(str, "ADD") == 0 || _my_strcmpi(str, "PLUS") == 0)
						{
							operation = REDUCTION_SUM;
						}
						else if (_my_strcmpi(str, "MEAN") == 0 || _my_strcmpi(str, "AVG") == 0)
						{
							operation = REDUCTION_MEAN;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (_my_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("keepdims", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						keepdims = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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
			if (!has_operation)std::cout << "Layer " << name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			bool invalid_axis = false;
			if (axis < 0 || axis > 3)
			{
				std::cout << "Layer " << name << " invalid axis\n";
				std::cout << line << "\n";
				invalid_axis = true;
			}
			return has_operation && has_bottom && has_top && has_name && !invalid_axis;
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
			if (keepdims)
			{
				if (axis == 0)
				{
					top_C = bottom_C;
					top_H = bottom_H;
					top_W = bottom_W;
				}
				else if (axis == 1)
				{
					top_C = 1;
					top_H = bottom_H;
					top_W = bottom_W;
				}
				else if (axis == 2)
				{
					top_C = bottom_C;
					top_H = 1;
					top_W = bottom_W;
				}
				else if (axis == 3)
				{
					top_C = bottom_C;
					top_H = bottom_H;
					top_W = 1;
				}
			}
			else
			{
				top_C = 1;
				top_H = 1;
				top_W = 1;
			}
		}

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};

	class ZQ_CNN_Layer_Sqrt : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Sqrt() {}
		~ZQ_CNN_Layer_Sqrt() {}

		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			bool ret = false;
			double t1 = omp_get_wtime();
			if ((*bottoms)[0] != (*tops)[0])
			{
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			}
			ret = ZQ_CNN_Forward_SSEUtils::Sqrt(*((*tops)[0]));

			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Sqrt layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Sqrt", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};

	class ZQ_CNN_Layer_Tile : public ZQ_CNN_Layer
	{
	public:
		ZQ_CNN_Layer_Tile() :tile_n(1), tile_h(1), tile_w(1), tile_c(1) {}
		~ZQ_CNN_Layer_Tile() {}

		int tile_n;
		int tile_h;
		int tile_w;
		int tile_c;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		virtual bool Forward(std::vector<ZQ_CNN_Tensor4D*>* bottoms, std::vector<ZQ_CNN_Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			bool ret = false;
			double t1 = omp_get_wtime();
			ret = ZQ_CNN_Forward_SSEUtils::Tile(*((*bottoms)[0]),tile_n, tile_h, tile_w, tile_c, *((*tops)[0]));

			double t2 = omp_get_wtime();
			last_cost_time = t2 - t1;
			if (show_debug_info)
				printf("Tile layer: %s cost : %.3f ms\n", name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			bottom_names.clear();
			top_names.clear();
			std::vector<std::vector<std::string> > paras = split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (_my_strcmpi("Tile", paras[n][0].c_str()) == 0)
				{

				}
				else if (_my_strcmpi("n", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						tile_n = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("h", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						tile_h = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("w", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						tile_w = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("c", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						tile_c = atoi(paras[n][1].c_str());
					}
				}
				else if (_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						top_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						bottom_names.push_back(paras[n][1]);
					}
				}
				else if (_my_strcmpi("name", paras[n][0].c_str()) == 0)
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
			top_C = bottom_C*tile_c;
			top_H = bottom_H*tile_h;
			top_W = bottom_W*tile_w;
		}

		virtual bool SwapInputRGBandBGR() { return true; };

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) { return true; }

		virtual bool SaveBinary_NCHW(FILE* out) const { return true; }

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes)
		{
			readed_length_in_bytes = 0;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_C*bottom_H*bottom_W;
		}
	};
}

#endif
