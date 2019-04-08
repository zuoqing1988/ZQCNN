#ifndef _ZQ_CNN_TENSOR_4D_NCHWC_H_
#define _ZQ_CNN_TENSOR_4D_NCHWC_H_
#pragma once
#include "ZQ_CNN_CompileConfig.h"
#include <string.h>
#include <stdlib.h>
#include <vector>
namespace ZQ
{
	class ZQ_CNN_Tensor4D_NCHWC
	{
	public:
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
		enum ALIGN_TYPE {
			ALIGN_C1 = 0,
			ALIGN_C4,
			ALIGN_C8,
			ALIGN_C16
		};

	public:
		virtual int GetAlignSize() const = 0;
		virtual ~ZQ_CNN_Tensor4D_NCHWC() {}
		float* const GetFirstPixelPtr() { return firstPixelData; }
		const float* GetFirstPixelPtr() const { return firstPixelData; }
		void SetShape(int in_N, int in_C, int in_H, int in_W) { shape_nchw[0] = in_N; shape_nchw[1] = in_C; shape_nchw[2] = in_H; shape_nchw[3] = in_W; }
		void GetShape(int& out_N, int& out_C, int& out_H, int& out_W) const { out_N = shape_nchw[0]; out_C = shape_nchw[1]; out_H = shape_nchw[2]; out_W = shape_nchw[3]; }
		const int GetN() const { return N; }
		const int GetH() const { return H; }
		const int GetW() const { return W; }
		const int GetC() const { return C; }
		const int GetBorderH() const { return borderH; }
		const int GetBorderW() const { return borderW; }
		const int GetWidthStep() const { return widthStep; }
		const int GetSliceStep() const { return sliceStep; }
		const int GetImageStep() const { return imageStep; }
		ALIGN_TYPE GetAlignType() const { return align_type; }
		void Reset()
		{
			if (rawData)
				memset(rawData, 0, rawDataLen);
		}

	public:
		virtual bool Padding(int padW, int padH, int mode) = 0;
		virtual bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH) = 0;
		virtual void ShrinkToFit() = 0;
		virtual bool IsBorderEnabled() const = 0;

		bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0)
		{
			if (data == 0 || !ChangeSize(N, H, W, C, borderW, borderH))
				return false;
			memset(rawData, 0, sizeof(float)*N*imageStep);
			int align_size = GetAlignSize();
			int rest_c = C %align_size;
			int floor_c = C - rest_c;
			int CHW = C*H*W;
			int HW = H*W;
			float* im_ptr = firstPixelData;
			float* slice_ptr, *row_ptr, *pix_ptr;
			for (int n = 0; n < N; n++, im_ptr += imageStep)
			{
				slice_ptr = im_ptr;
				
				for (int c = 0; c < C - rest_c; c+=align_size, slice_ptr+=sliceStep)
				{
					row_ptr = slice_ptr;
					for (int h = 0; h < H; h++, row_ptr += widthStep)
					{
						pix_ptr = row_ptr;
						for (int w = 0; w < W; w++, pix_ptr += align_size)
						{
							for (int k = 0; k < align_size; k++)
								pix_ptr[k] = data[n*CHW + (c + k)*HW + h*W + w];
						}
					}
				}
				if (rest_c > 0)
				{
					row_ptr = slice_ptr;
					for (int h = 0; h < H; h++, row_ptr += widthStep)
					{
						pix_ptr = row_ptr;
						for (int w = 0; w < W; w++, pix_ptr += align_size)
						{
							for (int k = 0; k < rest_c; k++)
								pix_ptr[k] = data[n*CHW + (floor_c + k)*HW + h*W + w];
						}
					}
				}
			}
			return true;
		}
		
		void ConvertToCompactNCHW(float* data) const
		{
			int align_size = GetAlignSize();
			int rest_c = C %align_size;
			int floor_c = C - rest_c;
			int CHW = C*H*W;
			int HW = H*W;
			float* im_ptr = firstPixelData;
			float* slice_ptr, *row_ptr, *pix_ptr;
			for (int n = 0; n < N; n++, im_ptr += imageStep)
			{
				slice_ptr = im_ptr;

				for (int c = 0; c < C - rest_c; c += align_size, slice_ptr += sliceStep)
				{
					row_ptr = slice_ptr;
					for (int h = 0; h < H; h++, row_ptr += widthStep)
					{
						pix_ptr = row_ptr;
						for (int w = 0; w < W; w++, pix_ptr += align_size)
						{
							for (int k = 0; k < align_size; k++)
								data[n*CHW + (c + k)*HW + h*W + w] = pix_ptr[k];
						}
					}
				}
				if (rest_c > 0)
				{
					row_ptr = slice_ptr;
					for (int h = 0; h < H; h++, row_ptr += widthStep)
					{
						pix_ptr = row_ptr;
						for (int w = 0; w < W; w++, pix_ptr += align_size)
						{
							for (int k = 0; k < rest_c; k++)
								data[n*CHW + (floor_c + k)*HW + h*W + w] = pix_ptr[k];
						}
					}
				}
			}
		}

		bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f)
		{
			int align_size = GetAlignSize();	
			//static const float mean_val = 127.5f;
			//static const float scale = 0.0078125f;
			if (align_size == 1)
			{
				if (!ChangeSize(1, _height, _width, 3, 1, 1))
					return false;
				memset(rawData, 0, rawDataLen);
				const unsigned char* bgr_row = BGR_img;
				float* row_ptr = firstPixelData;
				int sliceStep2 = sliceStep * 2;
				for (int h = 0; h < H; h++, row_ptr += widthStep, bgr_row += _widthStep)
				{
					float* pix_ptr = row_ptr;
					const unsigned char* bgr_pix = bgr_row;
					for (int w = 0; w < W; w++, pix_ptr ++, bgr_pix += 3)
					{
						pix_ptr[0] = (bgr_pix[0] - mean_val)*scale;
						pix_ptr[sliceStep] = (bgr_pix[1] - mean_val)*scale;
						pix_ptr[sliceStep2] = (bgr_pix[2] - mean_val)*scale;
					}
				}
				return true;
			}
			else if (align_size >= 4)
			{
				if (!ChangeSize(1, _height, _width, 3, 1, 1))
					return false;
				const unsigned char* bgr_row = BGR_img;
				float* row_ptr = firstPixelData;
				for (int h = 0; h < H; h++, row_ptr += widthStep, bgr_row += _widthStep)
				{
					float* pix_ptr = row_ptr;
					const unsigned char* bgr_pix = bgr_row;
					for (int w = 0; w < W; w++, pix_ptr += align_size, bgr_pix += 3)
					{
						pix_ptr[0] = (bgr_pix[0] - mean_val)*scale;
						pix_ptr[1] = (bgr_pix[1] - mean_val)*scale;
						pix_ptr[2] = (bgr_pix[2] - mean_val)*scale;
					}
				}
				if (borderH > 0)
				{
					memset(firstPixelData - align_size*borderW - widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
					memset(firstPixelData - align_size*borderW + widthStep*H, 0, sizeof(float)*widthStep*borderH);
				}
				if (borderW > 0)
				{
					for (int h = 0; h < H; h++)
					{
						memset(firstPixelData - align_size*borderW + widthStep*h, 0, sizeof(float)*align_size*borderW);
						memset(firstPixelData - align_size*(borderW << 1) + widthStep*(h + 1), 0, sizeof(float)*align_size*borderW);
					}
				}
				return true;
			}
			else
				return false;
		}

		virtual bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f)
		{
			int align_size = GetAlignSize();
			//static const float mean_val = 127.5f;
			//static const float scale = 0.0078125f;
			if (align_size == 1)
			{
				if (!ChangeSize(1, _height, _width, 1, 1, 1))
					return false;
				memset(rawData, 0, rawDataLen);
				const unsigned char* gray_row = gray_img;
				float* row_ptr = firstPixelData;
				for (int h = 0; h < H; h++, row_ptr += widthStep, gray_row += _widthStep)
				{
					float* pix_ptr = row_ptr;
					const unsigned char* gray_pix = gray_row;
					for (int w = 0; w < W; w++, pix_ptr++, gray_pix ++)
					{
						pix_ptr[0] = (gray_pix[0] - mean_val)*scale;
					}
				}
				return true;
			}
			else if (align_size >= 4)
			{
				if (!ChangeSize(1, _height, _width, 1, 1, 1))
					return false;
				const unsigned char* gray_row = gray_img;
				float* row_ptr = firstPixelData;
				for (int h = 0; h < H; h++, row_ptr += widthStep, gray_row += _widthStep)
				{
					float* pix_ptr = row_ptr;
					const unsigned char* gray_pix = gray_row;
					for (int w = 0; w < W; w++, pix_ptr += align_size, gray_pix ++)
					{
						pix_ptr[0] = (gray_pix[0] - mean_val)*scale;
					}
				}
				if (borderH > 0)
				{
					memset(firstPixelData - align_size*borderW - widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
					memset(firstPixelData - align_size*borderW + widthStep*H, 0, sizeof(float)*widthStep*borderH);
				}
				if (borderW > 0)
				{
					for (int h = 0; h < H; h++)
					{
						memset(firstPixelData - align_size*borderW + widthStep*h, 0, sizeof(float)*align_size*borderW);
						memset(firstPixelData - align_size*(borderW << 1) + widthStep*(h + 1), 0, sizeof(float)*align_size*borderW);
					}
				}
				return true;
			}
			else
				return false;
		}

		/*image size should match*/
		virtual bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const
		{
			int align_size = GetAlignSize();
			if (W != _width || H != _height || n_id < 0 || n_id >= N)
				return false;

			static const float scale = 127.5f;

			if (align_size == 1)
			{
				unsigned char* bgr_row = BGR_img;
				const float* row_ptr = firstPixelData + n_id*imageStep;
				int sliceStep2 = sliceStep * 2;
				float tmp;
				for (int h = 0; h < H; h++, row_ptr += widthStep, bgr_row += _widthStep)
				{
					const float* pix_ptr = row_ptr;
					unsigned char* bgr_pix = bgr_row;
					for (int w = 0; w < W; w++, pix_ptr++, bgr_pix += 3)
					{
						tmp = (pix_ptr[0] + 1.0f)*scale + 0.5f;
						bgr_pix[0] = __min(255, __max(0, (int)tmp));
						tmp = (pix_ptr[sliceStep] + 1.0f)*scale + 0.5f;
						bgr_pix[1] = __min(255, __max(0, (int)tmp));
						tmp = (pix_ptr[sliceStep2] + 1.0f)*scale + 0.5f;
						bgr_pix[2] = __min(255, __max(0, (int)tmp));
					}
				}
				return true;
			}
			else if (align_size >= 4)
			{
				float tmp;
				float* cur_row = firstPixelData + n_id*imageStep;
				int widthStep = GetWidthStep();
				unsigned char* bgr_row = BGR_img;
				for (int h = 0; h < H; h++, cur_row += widthStep, bgr_row += _widthStep)
				{
					float* cur_pix = cur_row;
					unsigned char* bgr_pix = bgr_row;
					for (int w = 0; w < W; w++, cur_pix += align_size, bgr_pix += 3)
					{
						tmp = (cur_pix[0] + 1.0f)*scale + 0.5f;
						bgr_pix[0] = __min(255, __max(0, (int)tmp));
						tmp = (cur_pix[1] + 1.0f)*scale + 0.5f;
						bgr_pix[1] = __min(255, __max(0, (int)tmp));
						tmp = (cur_pix[2] + 1.0f)*scale + 0.5f;
						bgr_pix[2] = __min(255, __max(0, (int)tmp));
					}
				}
				return true;
			}
			else
				return false;
		}

		static bool Permute_NCHW_get_size(const int order[4], int in_N, int in_C, int in_H, int in_W,
			int& out_N, int& out_C, int& out_H, int& out_W)
		{
			bool check_valid = true;
			bool has_order_flag[4] = { false };
			for (int i = 0; i < 4; i++)
			{
				if (order[i] < 0 || order[i] >= 4)
				{
					check_valid = false;
					break;
				}
				has_order_flag[order[i]] = true;
			}
			if (!check_valid)
				return false;
			for (int i = 0; i < 4; i++)
			{
				if (!has_order_flag[i])
				{
					check_valid = false;
					break;
				}
			}
			if (!check_valid)
				return false;

			int old_dim[4] = { in_N,in_C,in_H,in_W };
			int new_dim[4];
			for (int i = 0; i < 4; i++)
				new_dim[i] = old_dim[order[i]];
			out_N = new_dim[0];
			out_C = new_dim[1];
			out_H = new_dim[2];
			out_W = new_dim[3];
			return true;
		}

		bool Permute_NCHW(ZQ_CNN_Tensor4D_NCHWC& output, const int order[4], int num_threads = 1) const
		{
			int out_N, out_C, out_H, out_W;
			if (!Permute_NCHW_get_size(order, N, C, H, W, out_N, out_C, out_H, out_W))
				return false;
			if (!output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0))
				return false;

			int old_steps[4] = { C*H*W,H*W,W,1 };
			int new_steps[4] = { out_C*out_H*out_W, out_H*out_W, out_W,1 };
			int count = old_steps[0] * N;
			if (count)
			{
				std::vector<float> in_buf(count);
				std::vector<float> out_buf(count);
				ConvertToCompactNCHW(&in_buf[0]);
				for (int i = 0; i < count; i++)
				{
					int old_idx = 0;
					int idx = i;
					for (int j = 0; j < 4; j++)
					{
						int cur_order = order[j];
						old_idx += (idx / new_steps[j]) * old_steps[cur_order];
						idx %= new_steps[j];
					}
					out_buf[i] = in_buf[old_idx];
				}
				return output.ConvertFromCompactNCHW(&out_buf[0], out_N, out_C, out_H, out_W);
			}

			return true;
		}

		static bool Flatten_NCHW_get_size(int start_axis, int end_axis, int in_N, int in_C, int in_H, int in_W,
			int& out_N, int& out_C, int& out_H, int& out_W)
		{
			int old_shape[4] = { in_N,in_C,in_H,in_W };
			std::vector<int> shape;
			for (int i = 0; i < start_axis; ++i) {
				shape.push_back(old_shape[i]);
			}
			int flattened_dim = 1;
			for (int i = start_axis; i <= end_axis; i++)
				flattened_dim *= old_shape[i];
			shape.push_back(flattened_dim);

			for (int i = end_axis + 1; i < 4; ++i)
			{
				shape.push_back(old_shape[i]);
			}
			while (shape.size() < 4)
			{
				shape.push_back(1);
			}
			out_N = shape[0];
			out_C = shape[1];
			out_H = shape[2];
			out_W = shape[3];
			return true;
		}

		bool Flatten_NCHW(ZQ_CNN_Tensor4D_NCHWC& output, int start_axis, int end_axis, int num_threads = 1) const
		{
			int old_shape[4] = { N,C,H,W };
			std::vector<int> shape;
			for (int i = 0; i < start_axis; ++i) {
				shape.push_back(old_shape[i]);
			}
			int flattened_dim = 1;
			for (int i = start_axis; i <= end_axis; i++)
				flattened_dim *= old_shape[i];
			shape.push_back(flattened_dim);
			for (int i = end_axis + 1; i < 4; ++i) {
				shape.push_back(old_shape[i]);
			}
			return Reshape_NCHW(output, shape);
		}

		static bool Reshape_NCHW_get_size(const std::vector<int>& shape, int in_N, int in_C, int in_H, int in_W,
			int& out_N, int& out_C, int& out_H, int& out_W)
		{
			if (in_N <= 0 || in_C <= 0 || in_H <= 0 || in_W <= 0)
				return false;
			int shape_dim = shape.size();
			if (shape_dim > 4)
				return false;
			int old_dim[4] = { in_N, in_C, in_H, in_W };
			int new_dim[4];
			int count = in_N*in_C*in_H*in_W;
			for (int i = shape_dim; i < 4; i++)
				new_dim[i] = 1;
			int unknown_num = 0;
			int id = -1;
			for (int i = 0; i < shape_dim; i++)
			{
				if (shape[i] == 0)
				{
					new_dim[i] = old_dim[i];
				}
				else if (shape[i] > 0)
				{
					new_dim[i] = shape[i];
				}
				else
				{
					id = i;
					unknown_num++;
				}
			}

			if (unknown_num == 0)
			{
				out_N = new_dim[0];
				out_C = new_dim[1];
				out_H = new_dim[2];
				out_W = new_dim[3];
				return out_N*out_C*out_H*out_W == count;
			}
			else if (unknown_num == 1)
			{
				int total = count;
				for (int i = 0; i < 4; i++)
				{
					if (shape[i] >= 0)
					{
						if (total % new_dim[i] != 0)
							return false;
						total /= new_dim[i];
					}
				}
				new_dim[id] = total;
				out_N = new_dim[0];
				out_C = new_dim[1];
				out_H = new_dim[2];
				out_W = new_dim[3];
				return out_N*out_C*out_H*out_W == count;
			}
			else
			{
				return false;
			}
		}

		bool Reshape_NCHW(ZQ_CNN_Tensor4D_NCHWC& output, const std::vector<int>& shape, int num_threads = 1) const
		{
			int out_N, out_C, out_H, out_W;
			if (!Reshape_NCHW_get_size(shape, N, C, H, W, out_N, out_C, out_H, out_W))
				return false;
			output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);
			int in_HW = H*W;
			int in_CHW = C*in_HW;
			int count = in_CHW*N;
			if (count > 0)
			{
				std::vector<float> buf(count);
				ConvertToCompactNCHW(&buf[0]);
				output.ConvertFromCompactNCHW(&buf[0], out_N, out_C, out_H, out_W, 0, 0);
			}
			return true;
		}

		bool SaveToFile(const char* file)
		{
			int HW = H*W;
			int CHW = C*HW;
			int buf_len = N*CHW;
			std::vector<float> buffer(buf_len);
			FILE* out;
#if defined(_WIN32)
			if (0 != fopen_s(&out, file, "w"))
				return false;
#else
			out = fopen(file, "w");
			if (out == 0)
				return false;
#endif
			if (buf_len > 0)
			{
				ConvertToCompactNCHW(&buffer[0]);
				for (int n = 0; n < N; n++)
				{
					for (int h = 0; h < H; h++)
					{
						for (int w = 0; w < W; w++)
						{
							fprintf(out, "[n,h,w]=[%04d,%04d,%04d]: ",n,h,w);
								for (int c = 0; c < C; c++)
									fprintf(out, " %4d:%12.7f", c,buffer[n*CHW + c*HW + h*W + w]);
							fprintf(out, "\n");
						}
					}
				}
			}
			fclose(out);
			return true;
		}

	protected:
		int shape_nchw[4];
		int N;
		int W;
		int H;
		int C;
		int borderH;
		int borderW;
		int realHeight;
		int realWidth;
		int widthStep;
		int sliceStep;
		int imageStep;
		float* firstPixelData;
		unsigned char* rawData;
		long long rawDataLen;

		ALIGN_TYPE align_type;
	};

	class ZQ_CNN_Tensor4D_NCHWC1: public ZQ_CNN_Tensor4D_NCHWC
	{
	public:
		/*interface*/
		int GetAlignSize() const { return 1; }
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*other*/
		ZQ_CNN_Tensor4D_NCHWC1();
		~ZQ_CNN_Tensor4D_NCHWC1();
		
		void Swap(ZQ_CNN_Tensor4D_NCHWC1& other);

		bool ResizeBilinear(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH) const
		{
			return ResizeBilinearRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H);
		}

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC1& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC1& other);
	};

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

	class ZQ_CNN_Tensor4D_NCHWC4 : public ZQ_CNN_Tensor4D_NCHWC
	{
	public:
		/*interface*/
		int GetAlignSize() const { return 4; }
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*other*/
		ZQ_CNN_Tensor4D_NCHWC4();
		~ZQ_CNN_Tensor4D_NCHWC4();

		void Swap(ZQ_CNN_Tensor4D_NCHWC4& other);

		bool ResizeBilinear(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH) const
		{
			return ResizeBilinearRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H);
		}


		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC4& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC4& other);

	};

#endif //__ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

	class ZQ_CNN_Tensor4D_NCHWC8 : public ZQ_CNN_Tensor4D_NCHWC
	{
	public:
		/*interface*/
		int GetAlignSize() const { return 8; }
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*other*/
		ZQ_CNN_Tensor4D_NCHWC8();
		~ZQ_CNN_Tensor4D_NCHWC8();

		void Swap(ZQ_CNN_Tensor4D_NCHWC8& other);

		bool ResizeBilinear(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH) const
		{
			return ResizeBilinearRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H);
		}


		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC8& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC8& other);

	};

#endif //ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
}


#endif
