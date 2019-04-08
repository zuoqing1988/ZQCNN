#include "ZQ_CNN_Tensor4D.h"
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "layers_c/zq_cnn_resize_32f_align_c.h"

using namespace ZQ;


ZQ_CNN_Tensor4D_NHW_C_Align0::ZQ_CNN_Tensor4D_NHW_C_Align0()
{
	shape_nchw[0] = 0;
	shape_nchw[1] = 0;
	shape_nchw[2] = 0;
	shape_nchw[3] = 0;
	N = 0;
	W = 0;
	H = 0;
	C = 0;
	borderW = 0;
	borderH = 0;
	realWidth = 0;
	realHeight = 0;
	pixelStep = 0;
	widthStep = 0;
	sliceStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_0;
}


ZQ_CNN_Tensor4D_NHW_C_Align0::~ZQ_CNN_Tensor4D_NHW_C_Align0()
{
	if (rawData)
	{
		free(rawData);
		rawData = 0;
	}
}



void ZQ_CNN_Tensor4D_NHW_C_Align0::Swap(ZQ_CNN_Tensor4D_NHW_C_Align0& other)
{
	int tmp_shape[4];
	memcpy(tmp_shape, shape_nchw, sizeof(int) * 4);
	memcpy(shape_nchw, other.shape_nchw, sizeof(int) * 4);
	memcpy(other.shape_nchw,tmp_shape,sizeof(int) * 4);
	int tmp_N = N; N = other.N; other.N = tmp_N;
	int tmp_H = H; H = other.H; other.H = tmp_H;
	int tmp_W = W; W = other.W; other.W = tmp_W;
	int tmp_C = C; C = other.C; other.C = tmp_C;
	int tmp_borderH = borderH; borderH = other.borderH; other.borderH = tmp_borderH;
	int tmp_borderW = borderW; borderW = other.borderW; other.borderW = tmp_borderW;
	int tmp_realHeight = realHeight; realHeight = other.realHeight; other.realHeight = tmp_realHeight;
	int tmp_realWidth = realWidth; realWidth = other.realWidth; other.realWidth = tmp_realWidth;
	int tmp_pixStep = pixelStep; pixelStep = other.pixelStep; other.pixelStep = tmp_pixStep;
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}



bool ZQ_CNN_Tensor4D_NHW_C_Align0::Padding(int padW, int padH, int mode)
{
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NHW_C_Align0 tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_slice_ptr = tmp.firstPixelData;
		float* cur_slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
		{
			for (int h = 0; h <tmp.borderH; h++)
			{
				memset(tmp_slice_ptr - (h + 1)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
				memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
			}

			float* tmp_row_ptr = tmp_slice_ptr;
			float* cur_row_ptr = cur_slice_ptr;
			for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
			{
				memset(tmp_row_ptr - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memset(tmp_row_ptr + tmp.W*pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*pixelStep);
			}
		}
		Swap(tmp);
	}
	else
	{
		float* slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, slice_ptr += sliceStep)
		{
			for (int h = 0; h < borderH; h++)
			{
				memset(slice_ptr - (h + 1)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
				memset(slice_ptr + (H + h)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
			}

			float* row_ptr = slice_ptr;
			for (int h = 0; h < H; h++, row_ptr += widthStep)
			{
				memset(row_ptr - borderW*pixelStep, 0, sizeof(float)*borderW*pixelStep);
				memset(row_ptr + W*pixelStep, 0, sizeof(float)*borderW*pixelStep);
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NHW_C_Align0::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_pixelStep = dst_C;
	int dst_widthStep = dst_pixelStep*dst_realW;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_tensor_raw_size = dst_sliceStep*dst_N * sizeof(float);
	int needed_dst_raw_len = dst_tensor_raw_size;
	if (dst_tensor_raw_size == 0)
	{
		free(rawData);
		rawData = 0;
		firstPixelData = 0;
		rawDataLen = 0;

		N = 0;
		W = 0;
		H = 0;
		C = 0;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realWidth = 0;
		realHeight = 0;
		pixelStep = 0;
		widthStep = 0;
		sliceStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)malloc(needed_dst_raw_len);
			if (tmp_data == 0)
				return false;
			//memset(tmp_data, 0, needed_dst_raw_len);
			if(rawData)	
				free(rawData);
			rawData = tmp_data;
		}

		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*dst_pixelStep;
		rawDataLen = needed_dst_raw_len;


		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		pixelStep = dst_pixelStep;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
	}

	return true;
}


bool ZQ_CNN_Tensor4D_NHW_C_Align0::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	if (src_off_x < 0 || src_off_y < 0 || src_off_x + src_rect_w > W || src_off_y + src_rect_h > H)
		return false;
	if (dst_W == src_rect_w && dst_H == src_rect_h)
	{
		return ROI(dst, src_off_x, src_off_y, src_rect_w, src_rect_h, dst_borderH, dst_borderW);
	}
	else
	{
		if (!dst.ChangeSize(N, dst_H, dst_W, C, dst_borderH, dst_borderW))
			return false;

		int widthStep = GetWidthStep();
		int pixelStep = GetPixelStep();
		int dstWidthStep = dst.GetWidthStep();
		int dstPixelStep = dst.GetPixelStep();
		int dstSliceStep = dst.GetSliceStep();

		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}

		float* dst_slice_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_slice_ptr += dstSliceStep)
		{

			if (dst_borderH > 0)
			{
				memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NHW_C_Align0::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int rect_num = src_off_x.size();
	if (rect_num == 0 || rect_num != src_off_y.size() || rect_num != src_rect_w.size() || rect_num != src_rect_h.size())
		return false;

	for (int i = 0; i < rect_num; i++)
	{
		if (src_off_x[i] < 0 || src_off_y[i] < 0 || src_off_x[i] + src_rect_w[i] > W || src_off_y[i] + src_rect_h[i] > H)
			return false;
	}
	
	if (!dst.ChangeSize(rect_num, dst_H, dst_W, C, dst_borderH, dst_borderW))
		return false;

	int widthStep = GetWidthStep();
	int pixelStep = GetPixelStep();
	int dstWidthStep = dst.GetWidthStep();
	int dstPixelStep = dst.GetPixelStep();
	int dstSliceStep = dst.GetSliceStep();

	int align_mode = __min(GetAlignType(), dst.GetAlignType());

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_slice_ptr = dst.GetFirstPixelPtr() + dstSliceStep*i;
		bool can_call_safeborder = true;
		if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
			|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
			can_call_safeborder = false;

		if (can_call_safeborder)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (align_mode == ALIGN_128bit)
				zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else 
#endif
				zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (align_mode == ALIGN_128bit)
				zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else 
#endif
				zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
	}
	float* dst_slice_ptr = dst.GetFirstPixelPtr();
	for (int i = 0; i < rect_num; i++, dst_slice_ptr += dstSliceStep)
	{

		if (dst_borderH > 0)
		{
			memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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

ZQ_CNN_Tensor4D_NHW_C_Align128bit::ZQ_CNN_Tensor4D_NHW_C_Align128bit()
{
	shape_nchw[0] = 0;
	shape_nchw[1] = 0;
	shape_nchw[2] = 0;
	shape_nchw[3] = 0;
	N = 0;
	W = 0;
	H = 0;
	C = 0;
	borderW = 0;
	borderH = 0;
	realWidth = 0;
	realHeight = 0;
	pixelStep = 0;
	widthStep = 0;
	sliceStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_128bit;
}


ZQ_CNN_Tensor4D_NHW_C_Align128bit::~ZQ_CNN_Tensor4D_NHW_C_Align128bit()
{
	if (rawData)
	{
		_aligned_free(rawData);
		rawData = 0;
	}
}


void ZQ_CNN_Tensor4D_NHW_C_Align128bit::Swap(ZQ_CNN_Tensor4D_NHW_C_Align128bit& other)
{
	int tmp_shape[4];
	memcpy(tmp_shape, shape_nchw, sizeof(int) * 4);
	memcpy(shape_nchw, other.shape_nchw, sizeof(int) * 4);
	memcpy(other.shape_nchw, tmp_shape, sizeof(int) * 4);
	int tmp_N = N; N = other.N; other.N = tmp_N;
	int tmp_H = H; H = other.H; other.H = tmp_H;
	int tmp_W = W; W = other.W; other.W = tmp_W;
	int tmp_C = C; C = other.C; other.C = tmp_C;
	int tmp_borderH = borderH; borderH = other.borderH; other.borderH = tmp_borderH;
	int tmp_borderW = borderW; borderW = other.borderW; other.borderW = tmp_borderW;
	int tmp_realHeight = realHeight; realHeight = other.realHeight; other.realHeight = tmp_realHeight;
	int tmp_realWidth = realWidth; realWidth = other.realWidth; other.realWidth = tmp_realWidth;
	int tmp_pixStep = pixelStep; pixelStep = other.pixelStep; other.pixelStep = tmp_pixStep;
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}



bool ZQ_CNN_Tensor4D_NHW_C_Align128bit::Padding(int padW, int padH, int mode)
{
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NHW_C_Align128bit tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_slice_ptr = tmp.firstPixelData;
		float* cur_slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
		{
			for (int h = 0; h <tmp.borderH; h++)
			{
				memset(tmp_slice_ptr - (h + 1)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
				memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
			}

			float* tmp_row_ptr = tmp_slice_ptr;
			float* cur_row_ptr = cur_slice_ptr;
			for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
			{
				memset(tmp_row_ptr - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memset(tmp_row_ptr + tmp.W*pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*pixelStep);
			}
		}
		Swap(tmp);
	}
	else
	{
		float* slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, slice_ptr += sliceStep)
		{
			for (int h = 0; h < borderH; h++)
			{
				memset(slice_ptr - (h + 1)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
				memset(slice_ptr + (H + h)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
			}

			float* row_ptr = slice_ptr;
			for (int h = 0; h < H; h++, row_ptr += widthStep)
			{
				memset(row_ptr - borderW*pixelStep, 0, sizeof(float)*borderW*pixelStep);
				memset(row_ptr + W*pixelStep, 0, sizeof(float)*borderW*pixelStep);
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NHW_C_Align128bit::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_pixelStep = (dst_C + 3) >> 2 << 2;
	int dst_widthStep = dst_pixelStep*dst_realW;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_tensor_raw_size = dst_sliceStep*dst_N * sizeof(float);
	int needed_dst_raw_len = dst_tensor_raw_size;
	if (dst_tensor_raw_size == 0)
	{
		_aligned_free(rawData);
		rawData = 0;
		firstPixelData = 0;
		rawDataLen = 0;

		N = 0;
		W = 0;
		H = 0;
		C = 0;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realWidth = 0;
		realHeight = 0;
		pixelStep = 0;
		widthStep = 0;
		sliceStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)_aligned_malloc(needed_dst_raw_len, 16);
			if (tmp_data == 0)
				return false;
#if __ARM_NEON
			memset(tmp_data, 0, needed_dst_raw_len);
#endif
			_aligned_free(rawData);
			rawData = tmp_data;
		}

		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*dst_pixelStep;
		rawDataLen = needed_dst_raw_len;

		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		pixelStep = dst_pixelStep;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
	}

	return true;
}


bool ZQ_CNN_Tensor4D_NHW_C_Align128bit::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	int dstWidthStep, dstPixelStep, dstSliceStep;
	int align_mode;
	if (src_off_x < 0 || src_off_y < 0 || src_off_x + src_rect_w > W || src_off_y + src_rect_h > H)
	{
		if (!dst.ChangeSize(N, dst_H, dst_W, C, dst_borderH, dst_borderW))
			return false;
		dstWidthStep = dst.GetWidthStep();
		dstPixelStep = dst.GetPixelStep();
		dstSliceStep = dst.GetSliceStep();
		align_mode = __min(GetAlignType(), dst.GetAlignType());
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
		if (align_mode == ALIGN_256bit)
			zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (align_mode == ALIGN_128bit)
				zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
				zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
	}
	else
	{
		if (dst_W == src_rect_w && dst_H == src_rect_h)
			return ROI(dst, src_off_x, src_off_y, src_rect_w, src_rect_h, dst_borderH, dst_borderW);

		if (!dst.ChangeSize(N, dst_H, dst_W, C, dst_borderH, dst_borderW))
			return false;

		dstWidthStep = dst.GetWidthStep();
		dstPixelStep = dst.GetPixelStep();
		dstSliceStep = dst.GetSliceStep();
		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
	}

	float* dst_slice_ptr = dst.GetFirstPixelPtr();
	for (int n = 0; n < N; n++, dst_slice_ptr += dstSliceStep)
	{

		if (dst_borderH > 0)
		{
			memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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


bool ZQ_CNN_Tensor4D_NHW_C_Align128bit::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int rect_num = src_off_x.size();
	if (rect_num == 0 || rect_num != src_off_y.size() || rect_num != src_rect_w.size() || rect_num != src_rect_h.size())
		return false;

	if (!dst.ChangeSize(rect_num, dst_H, dst_W, C, dst_borderH, dst_borderW))
		return false;

	int widthStep = GetWidthStep();
	int pixelStep = GetPixelStep();
	int dstWidthStep = dst.GetWidthStep();
	int dstPixelStep = dst.GetPixelStep();
	int dstSliceStep = dst.GetSliceStep();

	int align_mode = __min(GetAlignType(), dst.GetAlignType());

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_slice_ptr = dst.GetFirstPixelPtr() + dstSliceStep*i;
		if (src_off_x[i] < 0 || src_off_x[i] + src_rect_w[i] > W || src_off_y[i] < 0 || src_off_y[i] + src_rect_h[i] > H)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
						src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
						dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
						src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
						dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
			bool can_call_safeborder = true;
			if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
				|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
				can_call_safeborder = false;

			if (can_call_safeborder)
			{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
				if (align_mode == ALIGN_256bit)
					zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
						src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
						dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					if (align_mode == ALIGN_128bit)
						zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
							src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
							dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
					else
#endif
						zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
							src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
							dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			}
			else
			{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
				if (align_mode == ALIGN_256bit)
					zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
						src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
						dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
					if (align_mode == ALIGN_128bit)
						zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
							src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
							dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
					else
#endif
						zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
							src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
							dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			}
		}
	}
	float* dst_slice_ptr = dst.GetFirstPixelPtr();
	for (int i = 0; i < rect_num; i++, dst_slice_ptr += dstSliceStep)
	{

		if (dst_borderH > 0)
		{
			memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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

ZQ_CNN_Tensor4D_NHW_C_Align256bit::ZQ_CNN_Tensor4D_NHW_C_Align256bit()
{
	shape_nchw[0] = 0;
	shape_nchw[1] = 0;
	shape_nchw[2] = 0;
	shape_nchw[3] = 0;
	N = 0;
	W = 0;
	H = 0;
	C = 0;
	borderW = 0;
	borderH = 0;
	realWidth = 0;
	realHeight = 0;
	pixelStep = 0;
	widthStep = 0;
	sliceStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_256bit;
}


ZQ_CNN_Tensor4D_NHW_C_Align256bit::~ZQ_CNN_Tensor4D_NHW_C_Align256bit()
{
	if (rawData)
	{
		_aligned_free(rawData);
		rawData = 0;
	}
}

void ZQ_CNN_Tensor4D_NHW_C_Align256bit::Swap(ZQ_CNN_Tensor4D_NHW_C_Align256bit& other)
{
	int tmp_shape[4];
	memcpy(tmp_shape, shape_nchw, sizeof(int) * 4);
	memcpy(shape_nchw, other.shape_nchw, sizeof(int) * 4);
	memcpy(other.shape_nchw, tmp_shape, sizeof(int) * 4);
	int tmp_N = N; N = other.N; other.N = tmp_N;
	int tmp_H = H; H = other.H; other.H = tmp_H;
	int tmp_W = W; W = other.W; other.W = tmp_W;
	int tmp_C = C; C = other.C; other.C = tmp_C;
	int tmp_borderH = borderH; borderH = other.borderH; other.borderH = tmp_borderH;
	int tmp_borderW = borderW; borderW = other.borderW; other.borderW = tmp_borderW;
	int tmp_realHeight = realHeight; realHeight = other.realHeight; other.realHeight = tmp_realHeight;
	int tmp_realWidth = realWidth; realWidth = other.realWidth; other.realWidth = tmp_realWidth;
	int tmp_pixStep = pixelStep; pixelStep = other.pixelStep; other.pixelStep = tmp_pixStep;
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}


bool ZQ_CNN_Tensor4D_NHW_C_Align256bit::Padding(int padW, int padH, int mode)
{
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NHW_C_Align256bit tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_slice_ptr = tmp.firstPixelData;
		float* cur_slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
		{
			for (int h = 0; h <tmp.borderH; h++)
			{
				memset(tmp_slice_ptr - (h+1)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
				memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.widthStep);
			}

			float* tmp_row_ptr = tmp_slice_ptr;
			float* cur_row_ptr = cur_slice_ptr;
			for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
			{
				memset(tmp_row_ptr - tmp.borderW*tmp.pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memset(tmp_row_ptr + tmp.W*pixelStep, 0, sizeof(float)*tmp.borderW*tmp.pixelStep);
				memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*pixelStep);
			}
		}
		Swap(tmp);
	}
	else
	{
		float* slice_ptr = firstPixelData;
		for (int n = 0; n < N; n++, slice_ptr += sliceStep)
		{
			for (int h = 0; h < borderH; h++)
			{
				memset(slice_ptr - (h+1)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
				memset(slice_ptr + (H + h)*widthStep - borderW*pixelStep, 0, sizeof(float)*widthStep);
			}

			float* row_ptr = slice_ptr;
			for (int h = 0; h < H; h++, row_ptr += widthStep)
			{
				memset(row_ptr - borderW*pixelStep, 0, sizeof(float)*borderW*pixelStep);
				memset(row_ptr + W*pixelStep, 0, sizeof(float)*borderW*pixelStep);
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NHW_C_Align256bit::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_pixelStep = (dst_C + 7) >> 3 << 3;
	int dst_widthStep = dst_pixelStep*dst_realW;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_tensor_raw_size = dst_sliceStep*dst_N * sizeof(float);
	int needed_dst_raw_len = dst_tensor_raw_size;
	if (dst_tensor_raw_size == 0)
	{
		_aligned_free(rawData);
		rawData = 0;
		firstPixelData = 0;
		rawDataLen = 0;

		N = 0;
		W = 0;
		H = 0;
		C = 0;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realWidth = 0;
		realHeight = 0;
		pixelStep = 0;
		widthStep = 0;
		sliceStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)_aligned_malloc(needed_dst_raw_len, 32);
			if (tmp_data == 0)
				return false;
			//memset(tmp_data, 0, needed_dst_raw_len);
			_aligned_free(rawData);
			rawData = tmp_data;
		}
		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*dst_pixelStep;
		rawDataLen = needed_dst_raw_len;

		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		pixelStep = dst_pixelStep;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
	}

	return true;
}

bool ZQ_CNN_Tensor4D_NHW_C_Align256bit::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	if (src_off_x < 0 || src_off_y < 0 || src_off_x + src_rect_w > W || src_off_y + src_rect_h > H)
		return false;
	if (dst_W == src_rect_w && dst_H == src_rect_h)
	{
		return ROI(dst, src_off_x, src_off_y, src_rect_w, src_rect_h, dst_borderH, dst_borderW);
	}
	else
	{
		if (!dst.ChangeSize(N, dst_H, dst_W, C, dst_borderH, dst_borderW))
			return false;

		int widthStep = GetWidthStep();
		int pixelStep = GetPixelStep();
		int dstWidthStep = dst.GetWidthStep();
		int dstPixelStep = dst.GetPixelStep();
		int dstSliceStep = dst.GetSliceStep();

		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
					dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
				if (align_mode == ALIGN_128bit)
					zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
				else
#endif
					zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, N, H, W, C, pixelStep, widthStep, sliceStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}

		float* dst_slice_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_slice_ptr += dstSliceStep)
		{

			if (dst_borderH > 0)
			{
				memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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
	}
	return true;
}



bool ZQ_CNN_Tensor4D_NHW_C_Align256bit::ResizeBilinearRect(ZQ_CNN_Tensor4D& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int rect_num = src_off_x.size();
	if (rect_num == 0 || rect_num != src_off_y.size() || rect_num != src_rect_w.size() || rect_num != src_rect_h.size())
		return false;

	for (int i = 0; i < rect_num; i++)
	{
		if (src_off_x[i] < 0 || src_off_y[i] < 0 || src_off_x[i] + src_rect_w[i] > W || src_off_y[i] + src_rect_h[i] > H)
			return false;
	}

	if (!dst.ChangeSize(rect_num, dst_H, dst_W, C, dst_borderH, dst_borderW))
		return false;

	int widthStep = GetWidthStep();
	int pixelStep = GetPixelStep();
	int dstWidthStep = dst.GetWidthStep();
	int dstPixelStep = dst.GetPixelStep();
	int dstSliceStep = dst.GetSliceStep();

	int align_mode = __min(GetAlignType(), dst.GetAlignType());

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_slice_ptr = dst.GetFirstPixelPtr() + dstSliceStep*i;
		bool can_call_safeborder = true;
		if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
			|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
			can_call_safeborder = false;

		if (can_call_safeborder)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_with_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (align_mode == ALIGN_128bit)
				zq_cnn_resize_with_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else 
#endif
				zq_cnn_resize_with_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
		else
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			if (align_mode == ALIGN_256bit)
				zq_cnn_resize_without_safeborder_32f_align256bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			if (align_mode == ALIGN_128bit)
				zq_cnn_resize_without_safeborder_32f_align128bit(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
			else 
#endif
				zq_cnn_resize_without_safeborder_32f_align0(firstPixelData, 1, H, W, C, pixelStep, widthStep, sliceStep,
					src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
					dst_slice_ptr, dst_H, dst_W, dstPixelStep, dstWidthStep, dstSliceStep);
		}
	}
	float* dst_slice_ptr = dst.GetFirstPixelPtr();
	for (int i = 0; i < rect_num; i++, dst_slice_ptr += dstSliceStep)
	{

		if (dst_borderH > 0)
		{
			memset(dst_slice_ptr - dstPixelStep*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			memset(dst_slice_ptr - dstPixelStep*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
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
