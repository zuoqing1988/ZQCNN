#ifndef _ZQ_CNN_TENSOR_4D_H_
#define _ZQ_CNN_TENSOR_4D_H_
#pragma once
#include "ZQ_CNN_CompileConfig.h"
#include <string.h>
#include <stdlib.h>
#include <vector>
namespace ZQ
{

	class ZQ_CNN_Tensor4D
	{
	public:
		enum ALIGN_TYPE {
			ALIGN_0 = 0,
			ALIGN_128bit = ALIGN_0 + 1,
			ALIGN_256bit = ALIGN_128bit + 1
		};

	public:
		virtual ~ZQ_CNN_Tensor4D() {}
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
		const int GetPixelStep() const { return pixelStep; }
		const int GetWidthStep() const { return widthStep; }
		const int GetSliceStep() const { return sliceStep; }
		ALIGN_TYPE GetAlignType() const { return align_type; }
		inline bool ResizeBilinear(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH) const
		{
			return ResizeBilinearRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H);
		}
		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const = 0;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const = 0;

		virtual bool Padding(int padW, int padH, int mode) = 0;
		virtual bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH) = 0;
		virtual void ShrinkToFit() = 0;
		virtual bool IsBorderEnabled() const = 0;

		virtual bool ROI(ZQ_CNN_Tensor4D& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const 
		{
			if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
				return false;

			if (!dst.ChangeSize(N, height, width, C, dst_borderH, dst_borderW))
				return false;
			int dstWidthStep = dst.GetWidthStep();
			int dstPixelStep = dst.GetPixelStep();
			int dstSliceStep = dst.GetSliceStep();
			int align_mode = __min(GetAlignType(), dst.GetAlignType());
			const float* src_slice_ptr = GetFirstPixelPtr() + off_y * widthStep + off_x*pixelStep;
			float* dst_slice_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++)
			{
				const float* src_row_ptr = src_slice_ptr;
				float* dst_row_ptr = dst_slice_ptr;
				for (int h = 0; h < height; h++)
				{
					const float* src_pix_ptr = src_row_ptr;
					float* dst_pix_ptr = dst_row_ptr;
					for (int w = 0; w < width; w++)
					{
						memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*C);
						if (C < dstPixelStep)
							memset(dst_pix_ptr + C, 0, sizeof(float)*(dstPixelStep - C));
						src_pix_ptr += pixelStep;
						dst_pix_ptr += dstPixelStep;
					}
					src_row_ptr += widthStep;
					dst_row_ptr += dstWidthStep;
				}
				src_slice_ptr += sliceStep;
				dst_slice_ptr += dstSliceStep;
			}
			dst_slice_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_slice_ptr += dstSliceStep)
			{
				if (dst_borderH > 0)
				{
					memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
					memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*height, 0, sizeof(float)*dstWidthStep*dst_borderH);
				}
				if (dst_borderW > 0)
				{
					for (int h = 0; h < dst_borderH; h++)
					{
						memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*h, 0, sizeof(float)*dstPixelStep*dst_borderW);
						memset(dst_slice_ptr - dstPixelStep*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*dstPixelStep*dst_borderW);
					}
				}
			}
			return true;
		}

		virtual bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0)
		{
			if (data == 0 || !ChangeSize(N, H, W, C, borderW, borderH))
				return false;
			memset(rawData, 0, sizeof(float)*N*sliceStep);
			int CHW = C*H*W;
			int HW = H*W;
			for (int n = 0; n < N; n++)
			{
				for (int c = 0; c < C; c++)
				{
					for (int h = 0; h < H; h++)
					{
						for (int w = 0; w < W; w++)
						{
							firstPixelData[n*sliceStep + h*widthStep + w*pixelStep + c] = data[n*CHW + c*HW + h*W + w];
						}
					}
				}
			}
			return true;
		}

		virtual void ConvertToCompactNCHW(float* data) const
		{
			int CHW = C*H*W;
			int HW = H*W;
			for (int n = 0; n < N; n++)
			{
				for (int c = 0; c < C; c++)
				{
					for (int h = 0; h < H; h++)
					{
						for (int w = 0; w < W; w++)
						{
							data[n*CHW + c*HW + h*W + w] = firstPixelData[n*sliceStep + h*widthStep + w*pixelStep + c];
						}
					}
				}
			}
		}

		virtual bool CopyData(const ZQ_CNN_Tensor4D& other)
		{
			if (!ChangeSize(other.GetN(), other.GetH(), other.GetW(), other.GetC(), other.GetBorderW(), other.GetBorderH()))
				return false;
			Reset();
			for (int n = 0; n < N; n++)
			{
				for (int h = -borderH; h < H + borderH; h++)
				{
					for (int w = -borderW; w < W + borderW; w++)
					{
						memcpy(firstPixelData + n*sliceStep+ h*widthStep + w*pixelStep,
							other.GetFirstPixelPtr() + n*other.GetSliceStep()+ h*other.GetWidthStep() + w*other.GetPixelStep(), sizeof(float)*C);
					}
				}

			}
			return true;
		}

		virtual bool Tile(ZQ_CNN_Tensor4D& out, int tile_n, int tile_h, int tile_w, int tile_c) const
		{
			int out_N = N*tile_n;
			int out_H = H*tile_h;
			int out_W = W*tile_w;
			int out_C = C*tile_c;
			if (out_N <= 0 || out_H <= 0 || out_W <= 0 || out_C <= 0)
				return false;
			if (out.N != out_N || out.H != out_H || out.W != out_W || out.C != out_C)
				out.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);
			const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
			float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
			int n, h, w;

			// Tile C
			for (n = 0, in_slice_ptr = firstPixelData, out_slice_ptr = out.firstPixelData;
				n < N;
				n++, in_slice_ptr += sliceStep, out_slice_ptr += sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += widthStep, out_row_ptr += out.widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += pixelStep, out_pix_ptr += out.pixelStep)
					{
						in_c_ptr = in_pix_ptr;
						out_c_ptr = out_pix_ptr;
						for (int tc = 0; tc < tile_c; tc++)
						{
							memcpy(out_c_ptr, in_c_ptr, sizeof(float)*C);
							out_c_ptr += C;
						}
					}
				}
			}

			//Tile W
			for (n = 0, out_slice_ptr = out.firstPixelData; n < tile_n; n++,out_slice_ptr += out.sliceStep)
			{
				for (h = 0, out_row_ptr = out_slice_ptr; h < tile_h; h++, out_row_ptr += out.widthStep)
				{
					int elt_num = out.pixelStep*W;
					in_pix_ptr = out_row_ptr;
					out_pix_ptr = out_row_ptr;
					for (w = 1; w < tile_w; w++)
					{
						memcpy(out_pix_ptr+w*elt_num, in_pix_ptr, sizeof(float)*elt_num);
					}
				}
			}

			//Tile H
			for (n = 0, out_slice_ptr = out.firstPixelData; n < tile_n; n++, out_slice_ptr += out.sliceStep)
			{
				int elt_num = out.widthStep*H;
				in_row_ptr = out_slice_ptr;
				out_row_ptr = out_slice_ptr;
				for (h = 1; h < tile_h; h++)
				{
					memcpy(out_row_ptr+h*elt_num, in_row_ptr, sizeof(float)*elt_num);
				}
			}

			//Tile N
			int elt_num = out.sliceStep*N;
			out_slice_ptr = out.firstPixelData;
			in_slice_ptr = out_slice_ptr;
			for (n = 1; n < tile_n; n++)
			{
				memcpy(out_slice_ptr+n*elt_num, in_slice_ptr, sizeof(float)*elt_num);
			}
			return true;
		}

		virtual void Reset()
		{
			if(rawData)
				memset(rawData, 0, rawDataLen);
		}

		virtual bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f)
		{
			if (!ChangeSize(1, _height, _width, 3, 1, 1))
				return false;

			//static const float mean_val = 127.5f;
			//static const float scale = 0.0078125f;
			float* cur_row = firstPixelData;
			const unsigned char* bgr_row = BGR_img;
			for (int h = 0; h < H; h++, cur_row += widthStep, bgr_row += _widthStep)
			{
				float* cur_pix = cur_row;
				const unsigned char* bgr_pix = bgr_row;
				for (int w = 0; w < W; w++, cur_pix += pixelStep, bgr_pix += 3)
				{
					cur_pix[0] = (bgr_pix[0] - mean_val)*scale;
					cur_pix[1] = (bgr_pix[1] - mean_val)*scale;
					cur_pix[2] = (bgr_pix[2] - mean_val)*scale;
				}
			}


			if (borderH > 0)
			{
				memset(firstPixelData - pixelStep*borderW - widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
				memset(firstPixelData - pixelStep*borderW + widthStep*H, 0, sizeof(float)*widthStep*borderH);
			}
			if (borderW > 0)
			{
				for (int h = 0; h < H; h++)
				{
					memset(firstPixelData - pixelStep*borderW + widthStep*h, 0, sizeof(float)*pixelStep*borderW);
					memset(firstPixelData - pixelStep*(borderW << 1) + widthStep*(h + 1), 0, sizeof(float)*pixelStep*borderW);
				}
			}
			return true;
		}

		virtual bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f)
		{
			if (!ChangeSize(1, _height, _width, 1, 1, 1))
				return false;

			//static const float mean_val = 127.5f;
			//static const float scale = 0.0078125f;
			float* cur_row = firstPixelData;
			const unsigned char* gray_row = gray_img;
			for (int h = 0; h < H; h++, cur_row += widthStep, gray_row += _widthStep)
			{
				float* cur_pix = cur_row;
				const unsigned char* gray_pix = gray_row;
				for (int w = 0; w < W; w++, cur_pix += pixelStep, gray_pix ++)
				{
					cur_pix[0] = (gray_pix[0] - mean_val)*scale;
					
				}
			}

			if (borderH > 0)
			{
				memset(firstPixelData - pixelStep*borderW - widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
				memset(firstPixelData - pixelStep*borderW + widthStep*H, 0, sizeof(float)*widthStep*borderH);
			}
			if (borderW > 0)
			{
				for (int h = 0; h < H; h++)
				{
					memset(firstPixelData - pixelStep*borderW + widthStep*h, 0, sizeof(float)*pixelStep*borderW);
					memset(firstPixelData - pixelStep*(borderW << 1) + widthStep*(h + 1), 0, sizeof(float)*pixelStep*borderW);
				}
			}
			return true;
		}

		/*image size should match*/
		bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const
		{
			if (W != _width || H != _height || n_id < 0 || n_id >= N)
				return false;

			static const float scale = 127.5f;

			float tmp;
			float* cur_row = firstPixelData + n_id*sliceStep;
			int widthStep = GetWidthStep();
			int pixelStep = GetPixelStep();
			unsigned char* bgr_row = BGR_img;
			for (int h = 0; h < H; h++, cur_row += widthStep, bgr_row += _widthStep)
			{
				float* cur_pix = cur_row;
				unsigned char* bgr_pix = bgr_row;
				for (int w = 0; w < W; w++, cur_pix += pixelStep, bgr_pix += 3)
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

		bool Permute_NCHW(ZQ_CNN_Tensor4D& output, const int order[4], int num_threads = 1) const
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

		bool Flatten_NCHW(ZQ_CNN_Tensor4D& output, int start_axis, int end_axis, int num_threads = 1) const
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
			else if(unknown_num == 1)
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

		bool Reshape_NCHW(ZQ_CNN_Tensor4D& output, const std::vector<int>& shape, int num_threads = 1) const
		{
			int out_N, out_C, out_H, out_W;
			if (!Reshape_NCHW_get_size(shape, N, C, H, W, out_N, out_C, out_H, out_W))
				return false;
			output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0);
			int in_HW = H*W;
			int in_CHW = C*in_HW;
			int out_HW = out_H*out_W;
			int out_CHW = out_C*out_HW;
			int idx = 0, rest, i_n, i_c, i_h, i_w;
			int out_SliceStep = output.GetSliceStep();
			int out_WidthStep = output.GetWidthStep();
			int out_PixelStep = output.GetPixelStep();
			int in_SliceStep = GetSliceStep();
			int in_WidthStep = GetWidthStep();
			int in_PixelStep = GetPixelStep();
			float* out_ptr = output.GetFirstPixelPtr();
			const float* in_ptr = GetFirstPixelPtr();
			float* out_slice_ptr = out_ptr;
			for (int nn = 0; nn < out_N; nn++)
			{
				float* out_c_ptr = out_slice_ptr;
				for (int cc = 0; cc < out_C; cc++)
				{
					float* out_row_ptr = out_c_ptr;
					for (int hh = 0; hh < out_H; hh++)
					{
						float* out_pix_ptr = out_row_ptr;
						for (int ww = 0; ww < out_W; ww++)
						{
							rest = idx;
							i_n = rest / in_CHW;
							rest %= in_CHW;
							i_c = rest / in_HW;
							rest %= in_HW;
							i_h = rest / W;
							i_w = rest % W;
							*out_pix_ptr = in_ptr[i_n*in_SliceStep + i_c + i_h*in_WidthStep + i_w*in_PixelStep];

							idx++;
							out_pix_ptr += out_PixelStep;
						}
						out_row_ptr += out_WidthStep;
					}
					out_c_ptr++;
				}
				out_slice_ptr += out_SliceStep;
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
							fprintf(out, "[n,h,w]=[%04d,%04d,%04d]: ", n, h, w);
							for (int c = 0; c < C; c++)
								fprintf(out, " %4d:%12.7f", c, buffer[n*CHW + c*HW + h*W + w]);
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
		int pixelStep;		
		int widthStep;		
		int sliceStep;		
		float* firstPixelData;
		unsigned char* rawData;
		long long rawDataLen;

		ALIGN_TYPE align_type;
	};


	class ZQ_CNN_Tensor4D_NHW_C_Align0 : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/	
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align0();
		~ZQ_CNN_Tensor4D_NHW_C_Align0();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align0& other);

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};


	class ZQ_CNN_Tensor4D_NHW_C_Align128bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		~ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align128bit& other);

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};

	class ZQ_CNN_Tensor4D_NHW_C_Align256bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		~ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align256bit& other);

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};
}


#endif
