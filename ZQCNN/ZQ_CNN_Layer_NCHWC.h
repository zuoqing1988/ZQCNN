#ifndef _ZQ_CNN_LAYER_NCHWC_H_
#define _ZQ_CNN_LAYER_NCHWC_H_
#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "ZQ_CNN_Tensor4D_NCHWC.h"
#include "ZQ_CNN_BBoxUtils.h"
#include "ZQ_CNN_Forward_SSEUtils_NCHWC.h"
namespace ZQ
{
	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC
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

		ZQ_CNN_Layer_NCHWC() :show_debug_info(false), use_buffer(false), ignore_small_value(0), last_cost_time(0) {}
		virtual ~ZQ_CNN_Layer_NCHWC() {}
		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops) = 0;

		virtual bool ReadParam(const std::string& line) = 0;

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops) = 0;

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) = 0;

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& topH, int &top_W) const = 0;

		//should be called after ZQ_CNN_Net have allocated necessery data
		virtual bool LoadBinary_NCHW(FILE* in) = 0;

		virtual bool SaveBinary_NCHW(FILE* out) const = 0;

		virtual bool LoadBinary_NCHW(const char* buffer, __int64 buffer_len, __int64& readed_length_in_bytes) = 0;

		virtual __int64 GetNumOfMulAdd() const = 0;

		virtual void Prepack() {}
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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_Input : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		ZQ_CNN_Layer_NCHWC_Input() :H(0), W(0), C(3), has_H_val(false), has_W_val(false) {}
		~ZQ_CNN_Layer_NCHWC_Input() {}
		int H, W, C;
		bool has_H_val, has_W_val;

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_C = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("Input", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_H_val = true;
						H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_W_val = true;
						W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("C", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_C = true;
						C = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(ZQ_CNN_Layer_NCHWC<Tensor4D>::name);
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}

			if (!has_C)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "C\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_C && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			return true;
		}

		//should called after ReadParam, allocate memory in this func
		virtual bool SetBottomDim(int bottom_C, int bottom_H, int bottom_W) { return true; }

		//should called after SetBottomDim
		virtual void GetTopDim(int& top_C, int& top_H, int& top_W) const { top_C = C; top_H = H; top_W = W; }

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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_Convolution : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		ZQ_CNN_Layer_NCHWC_Convolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1), dilate_H(1), dilate_W(1), pad_H(0), pad_W(),
			with_bias(false), with_prelu(false), prelu_slope(0), bottom_C(0) {}
		~ZQ_CNN_Layer_NCHWC_Convolution() {
			if (filters)delete filters;
			if (bias)delete bias;
			if (prelu_slope) delete prelu_slope;
		}
		Tensor4D* filters;
		Tensor4D* bias;
		Tensor4D* prelu_slope;
		ZQ_CNN_Tensor4D_NCHWC::Buffer packedfilters;
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

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
					void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
					__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
					bool ret = false;
#if __ARM_NEON
					if ((kernel_H == 1 && kernel_W == 1) || filters->GetN() <= 32)
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(*((*bottoms)[0]),
						packedfilters, filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						*bias, *prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);

#endif
					if(!ret)
					{
						ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBiasPReLU(*((*bottoms)[0]),
							*filters, *bias, *prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
							tmp_buffer, tmp_buffer_len);
					}
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
				else
				{
					if (filters == 0 || bias == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
					__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
					bool ret = false;
#if __ARM_NEON
					if ((kernel_H == 1 && kernel_W == 1) || filters->GetN() <= 32)
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(*((*bottoms)[0]),
						packedfilters, filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						*bias, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
#endif
					if(!ret)
					{
						ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithBias(*((*bottoms)[0]),
							*filters, *bias, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
							tmp_buffer, tmp_buffer_len);
					}
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
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
					void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
					__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
					bool ret = false;
#if __ARM_NEON
					if ((kernel_H == 1 && kernel_W == 1) || filters->GetN() <= 32)
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(*((*bottoms)[0]),
						packedfilters, filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						*prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
#endif
					if(!ret)
					{
						ret = ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionWithPReLU(*((*bottoms)[0]), *filters, *prelu_slope, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
							tmp_buffer, tmp_buffer_len);
					}
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
				else
				{
					if (filters == 0)
						return false;
					double t1 = omp_get_wtime();
					void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
					__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
					bool ret = false;
#if __ARM_NEON
					if ((kernel_H == 1 && kernel_W == 1) || filters->GetN() <= 32)
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(*((*bottoms)[0]),
						packedfilters, filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
						tmp_buffer, tmp_buffer_len);
#endif
					if(!ret)
					{
						ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Convolution(*((*bottoms)[0]), *filters, stride_H, stride_W, dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]),
							tmp_buffer, tmp_buffer_len);
					}
					//((*tops)[0])->SaveToFile("truth.txt");
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("Conv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);
					}
					return ret;
				}
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("Convolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "num_output\n";
			if (!has_kernelH)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_H (kernel_size)\n";
			if (!has_kernelW)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && has_kernelH && has_kernelW && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
				filters = new Tensor4D();
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
					bias = new Tensor4D();
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
			top_H = __max(0, floor((float)(bottom_H + pad_H * 2 - (kernel_H - 1)*dilate_H - 1) / stride_H) + 1);
			top_W = __max(0, floor((float)(bottom_W + pad_W * 2 - (kernel_W - 1)*dilate_W - 1) / stride_W) + 1);
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
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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

		virtual void Prepack()
		{
			ZQ_CNN_Forward_SSEUtils_NCHWC::ConvolutionPrePack(*filters, packedfilters);
		}
	};

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_DepthwiseConvolution : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		ZQ_CNN_Layer_NCHWC_DepthwiseConvolution() :filters(0), bias(0), num_output(0), kernel_H(0), kernel_W(0),
			stride_H(1), stride_W(1), dilate_H(1), dilate_W(1), pad_H(0), pad_W(), with_bias(false), bottom_C(0),
			with_prelu(false), prelu_slope(0) {}
		~ZQ_CNN_Layer_NCHWC_DepthwiseConvolution() {
			if (filters)delete filters;
			if (bias)delete bias;
			if (prelu_slope)delete prelu_slope;
		}
		Tensor4D* filters;
		Tensor4D* bias;
		Tensor4D* prelu_slope;
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

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
					bool ret = ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBiasPReLU(*((*bottoms)[0]), *filters, *bias, *prelu_slope, stride_H, stride_W, 
						dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]));
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
							mop, mop / time);

					}

					return ret;
				}
				else
				{
					if (filters == 0 || bias == 0)
						return false;
					double t1 = omp_get_wtime();
					bool ret = ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolutionWithBias(*((*bottoms)[0]), *filters, *bias, stride_H, stride_W, 
						dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]));
					double t2 = omp_get_wtime();
					ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
					if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					{
						double time = __max(1000 * (t2 - t1), 1e-9);
						double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
						mop /= 1024 * 1024;
						printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
							ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
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
				bool ret = ZQ_CNN_Forward_SSEUtils_NCHWC::DepthwiseConvolution(*((*bottoms)[0]), *filters, stride_H, stride_W, 
					dilate_H, dilate_W, pad_H, pad_W, *((*tops)[0]));
				double t2 = omp_get_wtime();
				ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				{
					double time = __max(1000 * (t2 - t1), 1e-9);
					double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
					mop /= 1024 * 1024;
					printf("DwConv layer:%s %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(), filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						mop, mop / time);
				}
				return ret;
			}
		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_num_output = false, has_kernelH = false, has_kernelW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("DepthwiseConvolution", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("dilate_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						dilate_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						int pad_num = atoi(paras[n][1].c_str());
						pad_H = pad_W = pad_num;
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride_H", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_H = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride_W", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "num_output\n";
			if (!has_kernelH)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_H (kernel_size)\n";
			if (!has_kernelW)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_W (kernel_size)\n";
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && has_kernelH && has_kernelW && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "'s num_output should match bottom's C\n";
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
				filters = new Tensor4D();
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
					bias = new Tensor4D();
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
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_BatchNormScale : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
		/*
		a = bias - slope * mean / sqrt(var)
		b = slope / sqrt(var)
		value = b * value + a
		*/
	public:
		ZQ_CNN_Layer_NCHWC_BatchNormScale() : mean(0), var(0), scale(0), bias(0),
			b(0), a(0), eps(0), with_bias(false), bottom_C(0) {}
		~ZQ_CNN_Layer_NCHWC_BatchNormScale() {
			if (mean) delete mean;
			if (var) delete var;
			if (scale) delete scale;
			if (bias) delete bias;
			if (b)delete b;
			if (a)delete a;
		}
		Tensor4D* mean;
		Tensor4D* var;
		Tensor4D* scale;
		Tensor4D* bias;
		Tensor4D* b;
		Tensor4D* a;

		float eps;
		//
		bool with_bias;
		int bottom_C;
		int bottom_H;
		int bottom_W;

	public:
		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (b == 0 || a == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNorm_b_a(*((*tops)[0]), *b, *a);
			double t2 = omp_get_wtime();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				printf("BatchNorm layer: %s cost : %.3f ms\n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("BatchNormScale", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("eps", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						eps = atof(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
				mean = new Tensor4D();
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
				var = new Tensor4D();
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
				scale = new Tensor4D();
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
				bias = new Tensor4D();
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
				b = new Tensor4D();
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
				a = new Tensor4D();
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
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len * 4; i++)
					{
						if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				bias->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 3, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNormScaleBias_Compute_b_a(*b, *a, *mean, *var, *scale, *bias, eps);
			}
			else
			{
				std::vector<float> nchw_raw(dst_len * 3);
				if (dst_len * 3 != fread_s(&nchw_raw[0], dst_len * 3 * sizeof(float), sizeof(float), dst_len * 3, in))
					return false;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len * 3; i++)
					{
						if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
							nchw_raw[i] = 0;
					}
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNormScale_Compute_b_a(*b, *a, *mean, *var, *scale, eps);
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
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
						nchw_raw[i] = 0;
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				bias->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 3, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNormScaleBias_Compute_b_a(*b, *a, *mean, *var, *scale, *bias, eps);
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
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
						nchw_raw[i] = 0;
				}
				mean->ConvertFromCompactNCHW(&nchw_raw[0], N, C, H, W);
				var->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len, N, C, H, W);
				scale->ConvertFromCompactNCHW(&nchw_raw[0] + dst_len * 2, N, C, H, W);
				return ZQ_CNN_Forward_SSEUtils_NCHWC::BatchNormScale_Compute_b_a(*b, *a, *mean, *var, *scale, eps);
			}

			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return bottom_W*bottom_H*bottom_C;
		}
	};

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_PReLU : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		Tensor4D* slope;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_NCHWC_PReLU() :slope(0), bottom_C(0) {}
		~ZQ_CNN_Layer_NCHWC_PReLU()
		{
			if (slope) delete slope;
		}
		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if (slope == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			bool ret = ZQ_CNN_Forward_SSEUtils_NCHWC::PReLU(*((*tops)[0]), *slope);
			double t2 = omp_get_wtime();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				printf("PReLU layer: %s %.3f ms \n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
			return ret;
		}


		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("PReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
				slope = new Tensor4D();
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
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
					nchw_raw[i] = 0;
			}
			slope->ConvertFromCompactNCHW(&nchw_raw[0], slope->GetN(), slope->GetC(), slope->GetH(), slope->GetW());
			readed_length_in_bytes += dst_len_in_bytes;
			return true;
		}

		virtual __int64 GetNumOfMulAdd() const
		{
			return (__int64)bottom_W*bottom_H*bottom_C * 3;
		}
	};

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_ReLU : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:

		float slope;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_NCHWC_ReLU() :slope(0), bottom_C(0) {}
		~ZQ_CNN_Layer_NCHWC_ReLU() {}

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;
			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);

			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils_NCHWC::ReLU(*((*tops)[0]), slope);
			double t2 = omp_get_wtime();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				printf("ReLU layer: %s %.3f ms \n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
			return true;
		}


		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("ReLU", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("slope", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						slope = atof(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_Pooling : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		ZQ_CNN_Layer_NCHWC_Pooling() :kernel_H(3), kernel_W(3), stride_H(2), stride_W(2),
			pad_H(0), pad_W(0), global_pool(false), type(0) {}
		~ZQ_CNN_Layer_NCHWC_Pooling() {}
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

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			if (type == TYPE_MAXPOOLING)
			{
				double t1 = omp_get_wtime();
				ZQ_CNN_Forward_SSEUtils_NCHWC::MaxPooling(*((*bottoms)[0]), *((*tops)[0]), kernel_H, kernel_W, stride_H, stride_W, global_pool);
				double t2 = omp_get_wtime();
				ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					printf("Pooling layer: %s cost : %.3f ms\n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
				return true;
			}
			else if (type == TYPE_AVGPOOLING)
			{
				double t1 = omp_get_wtime();
				ZQ_CNN_Forward_SSEUtils_NCHWC::AVGPooling(*((*bottoms)[0]), *((*tops)[0]), kernel_H, kernel_W, stride_H, stride_W, global_pool);
				double t2 = omp_get_wtime();
				ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					printf("Pooling layer: %s cost : %.3f ms\n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
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
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_kernelH = false, has_kernelW = false;
			bool has_strideH = false, has_strideW = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("Pooling", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("kernel_size", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_kernelH = true;
						kernel_H = atoi(paras[n][1].c_str());
						has_kernelW = true;
						kernel_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("stride", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_strideH = true;
						stride_H = atoi(paras[n][1].c_str());
						has_strideW = true;
						stride_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pad", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						pad_H = atoi(paras[n][1].c_str());
						pad_W = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("global_pool", paras[n][0].c_str()) == 0)
				{
					global_pool = true;
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("pool", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						const char* str = paras[n][1].c_str();
						if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "MAX") == 0)
							type = TYPE_MAXPOOLING;
						else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "AVG") == 0 || ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "AVE") == 0)
							type = TYPE_AVGPOOLING;
						else
						{
							type = atoi(str);
						}
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!global_pool)
			{
				if (!has_kernelH)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_H (kernel_size)\n";
				if (!has_kernelW)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "kernel_W (kernel_size)\n";
				if (!has_strideH)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "stride_H (stride)\n";
				if (!has_strideW)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "stride_W (stride)\n";
			}

			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			if (!global_pool)
				return has_kernelH && has_kernelW && has_strideH && has_strideW && has_bottom && has_top && has_name;
			else
			{
				return has_bottom && has_top && has_name;
			}
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_InnerProduct : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:

		Tensor4D* filters;
		Tensor4D* bias;
		ZQ_CNN_Tensor4D_NCHWC::Buffer packedfilters;
		bool with_bias;
		int num_output;
		int kernel_H;
		int kernel_W;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_NCHWC_InnerProduct() :filters(0), bias(0), with_bias(false) {}
		~ZQ_CNN_Layer_NCHWC_InnerProduct()
		{
			if (filters) delete filters;
			if (bias) delete bias;
		}
		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if (with_bias)
			{
				if (filters == 0 || bias == 0)
					return false;
				double t1 = omp_get_wtime();
				void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
				__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
				bool ret = false;
				ret = ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(*((*bottoms)[0]),
					packedfilters, filters->GetN(), *bias, *((*tops)[0]), tmp_buffer, tmp_buffer_len);
				if (!ret)
				{
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductWithBias(*((*bottoms)[0]),
						*filters, *bias, *((*tops)[0]), tmp_buffer, tmp_buffer_len);
				}
				double t2 = omp_get_wtime();
				ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
				double time = __max(1000 * (t2 - t1), 1e-9);
				double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
				mop /= 1024 * 1024;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					printf("Innerproduct layer: %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
						1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(),
						filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						mop, mop / time);
				return ret;
			}
			else
			{
				if (filters == 0)
					return false;
				double t1 = omp_get_wtime();
				void** tmp_buffer = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer : 0;
				__int64* tmp_buffer_len = ZQ_CNN_Layer_NCHWC<Tensor4D>::use_buffer ? ZQ_CNN_Layer_NCHWC<Tensor4D>::buffer_len : 0;
				bool ret = false;
				ret = ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(*((*bottoms)[0]),
					packedfilters, filters->GetN(), *((*tops)[0]), tmp_buffer, tmp_buffer_len);
				if (!ret)
				{
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProduct(*((*bottoms)[0]),
						*filters, *((*tops)[0]), tmp_buffer, tmp_buffer_len);
				}
				double t2 = omp_get_wtime();
				ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
				double time = __max(1000 * (t2 - t1), 1e-9);
				double mop = (double)(*tops)[0]->GetN()*(*tops)[0]->GetH()* (*tops)[0]->GetW()* filters->GetN()* filters->GetH()* filters->GetW()* filters->GetC();
				mop /= 1024 * 1024;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
					printf("Innerproduct layer: %.3f ms NHW %dx%dx%d filter: NHWC %d x %d x %d x %d, MUL = %.3f M, GFLOPS=%.3f\n",
						1000 * (t2 - t1), (*tops)[0]->GetN(), (*tops)[0]->GetH(), (*tops)[0]->GetW(),
						filters->GetN(), filters->GetH(), filters->GetW(), filters->GetC(),
						mop, mop / time);
				return ret;
			}

		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_num_output = false;
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("InnerProduct", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("num_output", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_num_output = true;
						num_output = atoi(paras[n][1].c_str());
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bias", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() > 1)
						with_bias = atoi(paras[n][1].c_str());
					else
						with_bias = true;
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (!has_num_output)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "num_output\n";
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_num_output && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
				filters = new Tensor4D();
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
					bias = new Tensor4D();
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
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
			{
				for (int i = 0; i < dst_len; i++)
				{
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value != 0)
				{
					for (int i = 0; i < dst_len; i++)
					{
						if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
				if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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
					if (fabs(nchw_raw[i]) < ZQ_CNN_Layer_NCHWC<Tensor4D>::ignore_small_value)
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

		virtual void Prepack()
		{
			ZQ_CNN_Forward_SSEUtils_NCHWC::InnerProductPrePack(*filters, packedfilters);
		}

	};

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_Softmax : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		int axis;
		//
		int bottom_C;
		int bottom_H;
		int bottom_W;

		ZQ_CNN_Layer_NCHWC_Softmax() { axis = 1; }
		~ZQ_CNN_Layer_NCHWC_Softmax() {}
		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;


			if ((*tops)[0] != (*bottoms)[0])
				(*tops)[0]->CopyData(*(*bottoms)[0]);
			double t1 = omp_get_wtime();
			ZQ_CNN_Forward_SSEUtils_NCHWC::Softmax(*((*tops)[0]), axis);
			double t2 = omp_get_wtime();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				printf("Softmax layer: %s cost : %.3f ms\n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
			return true;
		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_top = false, has_bottom = false, has_name = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("Softmax", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("axis", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						axis = atoi(paras[n][1].c_str());
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}
			}
			if (axis < 0 || axis > 3)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " invalid axis " << axis << "\n";
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return axis >= 0 && axis <= 3 && has_bottom && has_top && has_name;
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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

	template<class Tensor4D>
	class ZQ_CNN_Layer_NCHWC_Eltwise : public ZQ_CNN_Layer_NCHWC<Tensor4D>
	{
	public:
		ZQ_CNN_Layer_NCHWC_Eltwise() :with_weight(false) {}
		~ZQ_CNN_Layer_NCHWC_Eltwise() {}

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

		virtual bool Forward(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
		{
			if (bottoms == 0 || tops == 0 || bottoms->size() == 0 || tops->size() == 0 || (*bottoms)[0] == 0 || (*tops)[0] == 0)
				return false;

			bool ret = false;
			double t1 = omp_get_wtime();
			if (operation == ELTWISE_MUL)
			{
				ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Mul(*(std::vector<const Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else if (operation == ELTWISE_SUM)
			{
				if (with_weight)
				{
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_SumWithWeight(*(std::vector<const Tensor4D*>*)bottoms, weight, *((*tops)[0]));
				}
				else
				{
					ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Sum(*(std::vector<const Tensor4D*>*)bottoms, *((*tops)[0]));
				}
			}
			else if (operation == ELTWISE_MAX)
			{
				ret = ZQ_CNN_Forward_SSEUtils_NCHWC::Eltwise_Max(*(std::vector<const Tensor4D*>*)bottoms, *((*tops)[0]));
			}
			else
			{
				std::cout << "unknown eltwise operation " << operation << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				return false;
			}
			double t2 = omp_get_wtime();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::last_cost_time = t2 - t1;
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::show_debug_info)
				printf("Eltwise layer: %s cost : %.3f ms\n", ZQ_CNN_Layer_NCHWC<Tensor4D>::name.c_str(), 1000 * (t2 - t1));
			return ret;
		}

		virtual bool ReadParam(const std::string& line)
		{
			ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.clear();
			ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.clear();
			weight.clear();
			std::vector<std::vector<std::string> > paras = ZQ_CNN_Layer_NCHWC<Tensor4D>::split_line(line);
			int num = paras.size();
			bool has_operation = false;
			bool has_top = false, has_bottom = false, has_name = false;
			with_weight = false;
			for (int n = 0; n < num; n++)
			{
				if (paras[n].size() == 0)
					continue;
				if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("Eltwise", paras[n][0].c_str()) == 0)
				{

				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("operation", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_operation = true;
						const char* str = paras[n][1].c_str();

						if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "PROD") == 0 || ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "MUL") == 0)
						{
							operation = ELTWISE_MUL;
						}
						else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "SUM") == 0 || ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "ADD") == 0 || ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "PLUS") == 0)
						{
							operation = ELTWISE_SUM;
						}
						else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi(str, "MAX") == 0)
						{
							operation = ELTWISE_MAX;
						}
						else
						{
							operation = atoi(str);
						}
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("top", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_top = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::top_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("bottom", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_bottom = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.push_back(paras[n][1]);
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("name", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						has_name = true;
						ZQ_CNN_Layer_NCHWC<Tensor4D>::name = paras[n][1];
					}
				}
				else if (ZQ_CNN_Layer_NCHWC<Tensor4D>::_my_strcmpi("weight", paras[n][0].c_str()) == 0)
				{
					if (paras[n].size() >= 2)
					{
						with_weight = true;
						weight.push_back(atof(paras[n][1].c_str()));
					}
				}
				else
				{
					std::cout << "warning: unknown para " << paras[n][0] << " in Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << "\n";
				}

			}
			if (!has_operation)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "operation\n";
			if (!has_bottom)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "bottom\n";
			if (ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.size() < 2)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " must have at least 2 bottoms\n";
			if (with_weight && weight.size() != ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.size()) std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " weight num should match with bottom num\n";
			if (!has_top)std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "top\n";
			if (!has_name) {
				std::cout << "Layer " << ZQ_CNN_Layer_NCHWC<Tensor4D>::name << " missing " << "name\n";
				std::cout << line << "\n";
			}
			return has_operation && has_bottom && ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.size() >= 2 && has_top && has_name
				&& (!with_weight || (with_weight && weight.size() != ZQ_CNN_Layer_NCHWC<Tensor4D>::bottom_names.size()));
		}

		virtual bool LayerSetup(std::vector<Tensor4D*>* bottoms, std::vector<Tensor4D*>* tops)
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
