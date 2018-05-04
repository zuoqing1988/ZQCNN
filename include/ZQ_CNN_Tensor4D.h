#ifndef _ZQ_CNN_TENSOR_4D_H_
#define _ZQ_CNN_TENSOR_4D_H_
#pragma once
#include "ZQ_CNN_Defines.h"
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

		float* const GetFirstPixelPtr() { return firstPixelData; }
		const float* GetFirstPixelPtr() const { return firstPixelData; }
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
		virtual bool ConvertFromCompactNCHW(const float* data, int N, int C, int H, int W)
		{
			if (data == 0 || !ChangeSize(N, H, W, C, 0, 0))
				return false;
			memset(firstPixelData, 0, sizeof(float)*N*sliceStep);
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

		virtual bool CopyData(const ZQ_CNN_Tensor4D& other)
		{
			if (!ChangeSize(other.GetN(), other.GetH(), other.GetW(), other.GetC(), other.GetBorderW(), other.GetBorderH()))
				return false;
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
					cur_pix[0] = (bgr_pix[2] - mean_val)*scale;
					cur_pix[1] = (bgr_pix[1] - mean_val)*scale;
					cur_pix[2] = (bgr_pix[0] - mean_val)*scale;
				}
			}


			if (borderH > 0)
			{
				memset(firstPixelData - pixelStep*borderW - widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
				memset(firstPixelData - pixelStep*borderW + widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
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
				memset(firstPixelData - pixelStep*borderW + widthStep*borderH, 0, sizeof(float)*widthStep*borderH);
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
					bgr_pix[2] = __min(255, __max(0, (int)tmp));
					tmp = (cur_pix[1] + 1.0f)*scale + 0.5f;
					bgr_pix[1] = __min(255, __max(0, (int)tmp));
					tmp = (cur_pix[2] + 1.0f)*scale + 0.5f;
					bgr_pix[0] = __min(255, __max(0, (int)tmp));
				}
			}
			return true;
		}

	protected:
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
		ZQ_CNN_EXPORT bool Padding(int padW, int padH, int mode);
		ZQ_CNN_EXPORT bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_EXPORT ZQ_CNN_Tensor4D_NHW_C_Align0();
		ZQ_CNN_EXPORT ~ZQ_CNN_Tensor4D_NHW_C_Align0();
		ZQ_CNN_EXPORT void Swap(ZQ_CNN_Tensor4D_NHW_C_Align0& other);

		ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};


	class ZQ_CNN_Tensor4D_NHW_C_Align128bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		ZQ_CNN_EXPORT bool Padding(int padW, int padH, int mode);
		ZQ_CNN_EXPORT bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_EXPORT ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		ZQ_CNN_EXPORT ~ZQ_CNN_Tensor4D_NHW_C_Align128bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align128bit& other);

		ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};

	class ZQ_CNN_Tensor4D_NHW_C_Align256bit : public ZQ_CNN_Tensor4D
	{
	public:
		/*********************   Interface functions ********************/
		ZQ_CNN_EXPORT bool Padding(int padW, int padH, int mode);
		ZQ_CNN_EXPORT bool ChangeSize(int N, int H, int W, int C, int borderW, int borderH);
		void ShrinkToFit() { ChangeSize(0, 0, 0, 0, 0, 0); }
		bool IsBorderEnabled() const { return true; }
		
		/*********************   other functions ********************/
		ZQ_CNN_EXPORT ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		ZQ_CNN_EXPORT ~ZQ_CNN_Tensor4D_NHW_C_Align256bit();
		void Swap(ZQ_CNN_Tensor4D_NHW_C_Align256bit& other);

		ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const;

		virtual ZQ_CNN_EXPORT bool ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
			const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const;
	};
}


#endif
