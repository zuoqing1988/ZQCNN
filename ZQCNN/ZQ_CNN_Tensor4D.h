#ifndef _ZQ_CNN_TENSOR_4D_H_
#define _ZQ_CNN_TENSOR_4D_H_
#pragma once
#include "ZQ_CNN_CompileConfig.h"
#include <string.h>
#include <stdlib.h>
#include <new>
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
		enum SAMPLE_ALIGN_TYPE {
			SAMPLE_ALIGN_CENTER = 0,
			SAMPLE_ALIGN_CORNER = 1
		};

	public:
		virtual ~ZQ_CNN_Tensor4D() {}
		float* GetFirstPixelPtr() { return firstPixelData; }
		const float* GetFirstPixelPtr() const { return firstPixelData; }
		void SetShape(int in_N, int in_C, int in_H, int in_W) { shape_nchw[0] = in_N; shape_nchw[1] = in_C; shape_nchw[2] = in_H; shape_nchw[3] = in_W; }
		void GetShape(int& out_N, int& out_C, int& out_H, int& out_W) const { out_N = shape_nchw[0]; out_C = shape_nchw[1]; out_H = shape_nchw[2]; out_W = shape_nchw[3]; }
		int GetN() const { return N; }
		int GetH() const { return H; }
		int GetW() const { return W; }
		int GetC() const { return C; }
		int GetBorderH() const { return borderH; }
		int GetBorderW() const { return borderW; }
		int GetPixelStep() const { return pixelStep; }
		int GetWidthStep() const { return widthStep; }
		int GetSliceStep() const { return sliceStep; }
		ALIGN_TYPE GetAlignType() const { return align_type; }

		inline bool ResizeNearest(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const
		{
			return ResizeNearestRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H, sample_align_type);
		}
		virtual bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const = 0;

		virtual bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const = 0;


		inline bool ResizeBilinear(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const
		{
			return ResizeBilinearRect(dst, dst_W, dst_H, dst_borderW, dst_borderH, 0, 0, W, H, sample_align_type);
		}
		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const = 0;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const = 0;

		virtual bool Remap(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<float>& map_x, const std::vector<float>& map_y, bool use_fill_val = false, float fill_val = 0) const = 0;

		virtual bool Padding(int padW, int padH, int mode) = 0;
		virtual bool Padding(int padW_left, int padW_right, int padH_top, int padH_bottom, int mode) = 0;
		virtual bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH) = 0;
		virtual void ShrinkToFit() = 0;
		virtual bool IsBorderEnabled() const = 0;

		virtual bool ROI(ZQ_CNN_Tensor4D& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const 
		{
			// **审计修复 2026-10-03（附录 DX.2）**：原来写的是
			//     if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
			// `off_x + width` 是 **int 加法**。off_x 来自 MTCNN 的 P-net 检测框输出
			// （ZQ_CNN_MTCNN.h:615 / 621 等处），**是数据/模型可控的**；
			// off_x 足够大时这个加法**回绕成负数**，于是 `> W` 不成立、
			// **边界检查被整条绕过**，紧接着
			//     src_slice_ptr = GetFirstPixelPtr() + off_y*widthStep + off_x*pixelStep;
			// 就是一次**越界读**。
			// UBSan 坐实：ZQ_CNN_Tensor4D.h:74:40: runtime error: signed integer overflow:
			//     2147483645 + 8 cannot be represented in type 'int'
			// 改法：**先各自比、再用减法比**，全程不产生可能溢出的加法；
			// 顺带把 `width < 0` / `height < 0` 也拒掉
			//（原来负的 width/height 会让 `off_x + width` 变小而漏过检查）。
			if (off_x < 0 || off_y < 0 || width < 0 || height < 0)
				return false;
			if (off_x > W || off_y > H)
				return false;
			if (width > W - off_x || height > H - off_y)
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
					for (int h = 0; h < height; h++)
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

		virtual bool FlipX()
		{
			if (C > 0)
			{
				float* buffer = new(std::nothrow) float[pixelStep];
				if (buffer == 0)
					return false;
				for (int n = 0; n < N; n++)
				{
					for (int h = 0; h < H; h++)
					{
						float* row_ptr = firstPixelData + n*sliceStep + h*widthStep;
						for (int w = 0; w < W/2; w++)
						{
							memcpy(buffer, row_ptr + w*pixelStep, sizeof(float)*pixelStep);
							memcpy(row_ptr + w*pixelStep, row_ptr + (W - 1 - w)*pixelStep, sizeof(float)*pixelStep);
							memcpy(row_ptr + (W - 1 - w)*pixelStep, buffer, sizeof(float)*pixelStep);
						}
					}
				}
				delete []buffer;
			}
			return true;
		}

		virtual bool FlipY()
		{
			if (C > 0)
			{
				float* buffer = new(std::nothrow) float[pixelStep];
				if (buffer == 0)
					return false;
				for (int n = 0; n < N; n++)
				{
					for (int w = 0; w < W; w++)
					{
						float* pix_ptr = firstPixelData + n*sliceStep + w*pixelStep;

						for (int h = 0; h < H/2; h++)
						{
							memcpy(buffer, pix_ptr + h*widthStep, sizeof(float)*pixelStep);
							memcpy(pix_ptr + h*widthStep, pix_ptr + (H - 1 - h)*widthStep, sizeof(float)*pixelStep);
							memcpy(pix_ptr + (H - 1 - h)*widthStep, buffer, sizeof(float)*pixelStep);
						}
					}
				}
				delete[]buffer;
			}
			return true;
		}

		virtual bool Tile(ZQ_CNN_Tensor4D& out, int tile_n, int tile_h, int tile_w, int tile_c) const
		{
			// 审计修复 2026-10-02（附录 BF）：tile_* 来自**模型文件**（不可信输入）。
			// 原来这里是**未检查的整数乘法**：
			//     int out_N = N*tile_n;  int out_H = H*tile_h;
			//     int out_W = W*tile_w;  int out_C = C*tile_c;
			// 而下面三个循环都是"按 tile_* 的次数、每次前进对应步长"地 memcpy/写：
			//     for (tc = 0; tc < tile_c; tc++) { memcpy(out_c_ptr, in_c_ptr, 4*C); out_c_ptr += C; }
			//     for (w = 1; w < tile_w; w++) memcpy(out_pix_ptr + w*elt_num, in_pix_ptr, 4*elt_num);
			//     for (h = 0; h < tile_h; h++) ...
			// 只要乘积**回绕**后落在一个小的正数上，ChangeSize 就按这个小值分配，
			// 循环却按 tile_* 的原始值写 —— **堆缓冲区溢出写**。
			// 例：N=H=W=1, C=3, tile_c=0x55555556
			//     3 * 0x55555556 = 0x100000002，截成 int 是 **2**
			//     -> out_C=2，ChangeSize 成功；循环却 memcpy 0x55555556 次。
			// 修法：乘积用 __int64 算，并要求落在 [1, 0x7FFFFFFF]。
			// 顺带把 tile_* <= 0 也拒掉（原来 0 会让 out_* <= 0 提前 return，
			// 但负数会让 out_sliceStep 之类走到负步长，语义上也没意义）。
			if (tile_n <= 0 || tile_h <= 0 || tile_w <= 0 || tile_c <= 0)
				return false;
			if (N <= 0 || H <= 0 || W <= 0 || C <= 0)
				return false;
			__int64 out_N64 = (__int64)N * tile_n;
			__int64 out_H64 = (__int64)H * tile_h;
			__int64 out_W64 = (__int64)W * tile_w;
			__int64 out_C64 = (__int64)C * tile_c;
			const __int64 kTileMax = 0x7FFFFFFF;
			if (out_N64 <= 0 || out_N64 > kTileMax
				|| out_H64 <= 0 || out_H64 > kTileMax
				|| out_W64 <= 0 || out_W64 > kTileMax
				|| out_C64 <= 0 || out_C64 > kTileMax)
				return false;
			int out_N = (int)out_N64;
			int out_H = (int)out_H64;
			int out_W = (int)out_W64;
			int out_C = (int)out_C64;
			if (out.N != out_N || out.H != out_H || out.W != out_W || out.C != out_C)
			{
				if (!out.ChangeSize(out_N, out_H, out_W, out_C, 0, 0))
					return false;
			}
			const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
			float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
			int n, h, w;

			// Tile C
			// **审计修复 2026-10-03（附录 DD.9）**：n 循环的增量原来写成了
			// `out_slice_ptr += sliceStep`——用的是**输入**张量的 sliceStep，而不是 out.sliceStep。
			// 只要 tile_c/tile_h/tile_w 中有任何一个 >1，out.sliceStep 就大于 sliceStep，
			// 第 2 个及以后的 slice 整体前移，最后一部分永远不被写。
			// 全部 tile=1 时 out 的尺寸与 in 一致、sliceStep 恰好相等，**所以看不出错**。
			// 触发条件：N>1 且任一 tile 方向 >1。实测 N=2,C=5,H=2,W=3,tile=1x1x1x2 → 90 格错。
			for (n = 0, in_slice_ptr = firstPixelData, out_slice_ptr = out.firstPixelData;
				n < N;
				n++, in_slice_ptr += sliceStep, out_slice_ptr += out.sliceStep)
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

			// 审计修复 2026-10-03（附录 DD.3）：三个扩展循环的**外层上界与步长都写错了**。
			// 正确的分工是：每一步只复制**输入**范围，剩下的扩展由后面的循环接手。
			// 原来三个循环都用 tile_n / tile_h 作为外层上界（应当是 N / H），
			// 且 Tile N 的目标步长是 out.sliceStep*N（应当是 tile_n*out.sliceStep）。
			//
			// 触发条件：只要 **tile_w > 1 且 H > tile_h**（或 N > tile_n）就会错。
			// 实测：N=1,C=3,H=4,W=5, tile=1x1x2x1 -> 45 格错，第一个错在 h=1。
			// 而 tile_h==1 时恰好只复制第 0 行（那时就是错的），H==1 时又恰好覆盖全部：
			// 两种情况都不错。所以以前的测试覆盖一直是漏的。
			// 附录 BF 修的回绕那一段没动过，两个缺陷叠在同一个函数里。

			//Tile W：每个**输入行**里的 W 个像素块，复制 tile_w 次
			for (n = 0, out_slice_ptr = out.firstPixelData; n < N; n++, out_slice_ptr += out.sliceStep)
			{
				for (h = 0, out_row_ptr = out_slice_ptr; h < H; h++, out_row_ptr += out.widthStep)
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

			//Tile H：每个**输入行**，复制 tile_h 次。
			// 步长必须是 H*out.widthStep（即一个输入行的高度）：
			// 输入第 h 行映射到输出的 h, h+H, h+2H, ... 那几行；
			// 写成 hh*out.widthStep（步长 1）会与上一个 h 的输出区重叠并覆盖它们。
			for (n = 0, out_slice_ptr = out.firstPixelData; n < N; n++, out_slice_ptr += out.sliceStep)
			{
				for (h = 0, out_row_ptr = out_slice_ptr; h < H; h++, out_row_ptr += out.widthStep)
				{
					for (int hh = 1; hh < tile_h; hh++)
					{
						memcpy(out_row_ptr+hh*H*out.widthStep, out_row_ptr, sizeof(float)*out.widthStep);
					}
				}
			}

			//Tile N：每个**输入 slice**，复制 tile_n 次。
			// **与 H / W 保持一致的取模语义**：out[r] <- in[r % N]，
			// 所以源 slice 逐个向后步进 n 个，目标下标是 n + nn*N。
			// 原来写成 `out.sliceStep*N`（一个步长）时 N=2/tile_n=2 会把第一个 slice
			// 写到**第三个** slice 位置，第二个反而永远不被写。
			out_slice_ptr = out.firstPixelData;
			for (n = 0; n < N; n++, out_slice_ptr += out.sliceStep)
			{
				for (int nn = 1; nn < tile_n; nn++)
				{
					memcpy(out_slice_ptr+nn*N*out.sliceStep, out_slice_ptr, sizeof(float)*out.sliceStep);
				}
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

		virtual bool ConvertFromBGR2GRAY(const unsigned char* BGR_img, int _width, int _height, int _widthStep, const float mean_val = 127.5f, const float scale = 0.0078125f)
		{
			if (!ChangeSize(1, _height, _width, 1, 1, 1))
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
					float gray_pix = bgr_pix[0] * 0.114f + bgr_pix[1] * 0.587f + bgr_pix[2] * 0.299f;
					cur_pix[0] = (gray_pix - mean_val)*scale;
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

		virtual bool ConvertColor_BGR2GRAY(ZQ_CNN_Tensor4D& dst, int dst_borderW, int dst_borderH) const
		{
			if (C != 3)
				return false;
			if (dst.GetN() != N || dst.GetH() != H || dst.GetW() != W || dst.GetC() != 1)
			{
				if (!dst.ChangeSize(N, H, W, 1, __max(0, dst_borderH), __max(0, dst_borderW)))
					return false;
			}
			else
			{
				if (dst_borderH >= 0 || dst_borderW >= 0)
				{
					if (!dst.ChangeSize(N, H, W, 1, dst_borderH, dst_borderW))
						return false;
				}
			}
			//printf("C = %d\n", dst.GetC());

			int widthStep = GetWidthStep();
			int pixelStep = GetPixelStep();
			int dstWidthStep = dst.GetWidthStep();
			int dstPixelStep = dst.GetPixelStep();
			int dstSliceStep = dst.GetSliceStep();


			float* dst_slice_ptr = dst.GetFirstPixelPtr();
			const float* src_slice_ptr = GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_slice_ptr += dstSliceStep, src_slice_ptr += sliceStep)
			{
				float* dst_row_ptr = dst_slice_ptr;
				const float* src_row_ptr = src_slice_ptr;
				for (int h = 0; h < H; h++, dst_row_ptr += dstWidthStep, src_row_ptr += widthStep)
				{
					float* dst_pix_ptr = dst_row_ptr;
					const float* src_pix_ptr = src_row_ptr;
					for (int w = 0; w < W; w++, dst_pix_ptr += dstPixelStep, src_pix_ptr += pixelStep)
					{
						dst_pix_ptr[0] = src_pix_ptr[0] * 0.114f + src_pix_ptr[1] * 0.587f + src_pix_ptr[2] * 0.299f;
					}
				}

				if (dst_borderH > 0)
				{
					memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
					memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*H, 0, sizeof(float)*dstWidthStep*dst_borderH);
				}
				if (dst_borderW > 0)
				{
					for (int h = 0; h < H; h++)
					{
						memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*h, 0, sizeof(float)*dstPixelStep*dst_borderW);
						memset(dst_slice_ptr - dstPixelStep*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*dstPixelStep*dst_borderW);
					}
				}
			}
			return true;
		}

		virtual bool MulScalar(float scalar) 
		{
			float* src_slice_ptr = GetFirstPixelPtr();
			for (int n = 0; n < N; n++, src_slice_ptr += sliceStep)
			{
				float* src_row_ptr = src_slice_ptr;
				for (int h = 0; h < H; h++, src_row_ptr += widthStep)
				{
					float* src_pix_ptr = src_row_ptr;
					for (int w = 0; w < W; w++, src_pix_ptr += pixelStep)
					{
						for (int c = 0; c < C; c++)
						{
							src_pix_ptr[c] *= scalar;
						}
					}
				}
			}
			return true;
		}

		virtual bool AddScalar(float scalar)
		{
			float* src_slice_ptr = GetFirstPixelPtr();
			for (int n = 0; n < N; n++, src_slice_ptr += sliceStep)
			{
				float* src_row_ptr = src_slice_ptr;
				for (int h = 0; h < H; h++, src_row_ptr += widthStep)
				{
					float* src_pix_ptr = src_row_ptr;
					for (int w = 0; w < W; w++, src_pix_ptr += pixelStep)
					{
						for (int c = 0; c < C; c++)
						{
							src_pix_ptr[c] += scalar;
						}
					}
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
			int shape_dim = (int)shape.size();
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
			if (!output.ChangeSize(out_N, out_H, out_W, out_C, 0, 0))
				return false;
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
		bool Padding(int padW_left, int padW_right, int padH_top, int padH_bottom, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align0();
		~ZQ_CNN_Tensor4D_NHW_C_Align0();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align0& other);

		
		bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool Remap(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<float>& map_x, const std::vector<float>& map_y, bool use_fill_val = false, float fill_val = 0) const;
	};


	class ZQ_CNN_Tensor4D_NHW_C_Align128bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		bool Padding(int padW, int padH, int mode);
		bool Padding(int padW_left, int padW_right, int padH_top, int padH_bottom, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		~ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align128bit& other);

		bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool Remap(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<float>& map_x, const std::vector<float>& map_y, bool use_fill_val = false, float fill_val = 0) const;
	};

	class ZQ_CNN_Tensor4D_NHW_C_Align256bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		bool Padding(int padW, int padH, int mode);
		bool Padding(int padW_left, int padW_right, int padH_top, int padH_bottom, int mode);
		bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		~ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align256bit& other);

		bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeNearestRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h, SAMPLE_ALIGN_TYPE sample_align_type = SAMPLE_ALIGN_CENTER) const;

		virtual bool Remap(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<float>& map_x, const std::vector<float>& map_y, bool use_fill_val = false, float fill_val = 0) const;
	};
}


#endif
