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
		enum ALIGN_TYPE {
			ALIGN_C1 = 0,
			ALIGN_C4,
			ALIGN_C8,
			ALIGN_C16
		};

	public:
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

		virtual bool Padding(int padW, int padH, int mode) = 0;
		virtual bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH) = 0;
		virtual void ShrinkToFit() = 0;
		virtual bool IsBorderEnabled() const = 0;
		virtual bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0) = 0;
		virtual void ConvertToCompactNCHW(float* data) const = 0;
		virtual bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f) = 0;
		virtual bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f) = 0;
		/*image size should match*/
		virtual bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const = 0;

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
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit();
		bool IsBorderEnabled() const;
		bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0);
		void ConvertToCompactNCHW(float* data) const;
		bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		/*image size should match*/
		bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const;

		/*other*/
		ZQ_CNN_Tensor4D_NCHWC1();
		~ZQ_CNN_Tensor4D_NCHWC1();

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC1& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC1& other);

		bool Tile(ZQ_CNN_Tensor4D_NCHWC1& out, int tile_n, int tile_h, int tile_w, int tile_c) const;
	};

	class ZQ_CNN_Tensor4D_NCHWC4 : public ZQ_CNN_Tensor4D_NCHWC
	{
	public:
		/*interface*/
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit();
		bool IsBorderEnabled() const;
		bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0);
		void ConvertToCompactNCHW(float* data) const;
		bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		/*image size should match*/
		bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const;

		/*other*/
		ZQ_CNN_Tensor4D_NCHWC4();
		~ZQ_CNN_Tensor4D_NCHWC4();

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC4& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC4& other);

		bool Tile(ZQ_CNN_Tensor4D_NCHWC4& out, int tile_n, int tile_h, int tile_w, int tile_c) const;
	};

	class ZQ_CNN_Tensor4D_NCHWC8 : public ZQ_CNN_Tensor4D_NCHWC
	{
	public:
		/*interface*/
		bool Padding(int padW, int padH, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit();
		bool IsBorderEnabled() const;
		bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W, int borderW = 0, int borderH = 0);
		void ConvertToCompactNCHW(float* data) const;
		bool ConvertFromBGR(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		bool ConvertFromGray(const unsigned char* gray_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f);
		/*image size should match*/
		bool ConvertToBGR(unsigned char* BGR_img, int _width, int _height, int _widthStep, int n_id = 0) const;

		/*other*/
		ZQ_CNN_Tensor4D_NCHWC8();
		~ZQ_CNN_Tensor4D_NCHWC8();

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;

		bool ROI(ZQ_CNN_Tensor4D_NCHWC8& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const;

		bool CopyData(const ZQ_CNN_Tensor4D_NCHWC8& other);

		bool Tile(ZQ_CNN_Tensor4D_NCHWC8& out, int tile_n, int tile_h, int tile_w, int tile_c) const;
	};
}


#endif
