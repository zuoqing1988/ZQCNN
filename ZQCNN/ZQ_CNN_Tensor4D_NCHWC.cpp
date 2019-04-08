#include "ZQ_CNN_Tensor4D_NCHWC.h"
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "layers_nchwc/zq_cnn_resize_nchwc.h"

using namespace ZQ;

ZQ_CNN_Tensor4D_NCHWC1::ZQ_CNN_Tensor4D_NCHWC1()
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
	widthStep = 0;
	sliceStep = 0;
	imageStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_C1;
}


ZQ_CNN_Tensor4D_NCHWC1::~ZQ_CNN_Tensor4D_NCHWC1()
{
	if (rawData)
	{
		free(rawData);
		rawData = 0;
	}
}



void ZQ_CNN_Tensor4D_NCHWC1::Swap(ZQ_CNN_Tensor4D_NCHWC1& other)
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
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	int tmp_imageStep = imageStep; imageStep = other.imageStep; other.imageStep = tmp_imageStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}

bool ZQ_CNN_Tensor4D_NCHWC1::Padding(int padW, int padH, int mode)
{
	int align_size = GetAlignSize();
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NCHWC1 tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_im_ptr = tmp.firstPixelData;
		float* cur_im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_im_ptr += tmp.imageStep, cur_im_ptr += imageStep)
		{
			float* tmp_slice_ptr = tmp_im_ptr;
			float* cur_slice_ptr = cur_im_ptr;
			for (int c = 0; c < C; c+=align_size, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
			{
				for (int h = 0; h <tmp.borderH; h++)
				{
					memset(tmp_slice_ptr - (h + 1)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
					memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
				}

				float* tmp_row_ptr = tmp_slice_ptr;
				float* cur_row_ptr = cur_slice_ptr;
				for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
				{
					memset(tmp_row_ptr - tmp.borderW*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memset(tmp_row_ptr + tmp.W*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*align_size);
				}
			}
		}
		Swap(tmp);
	}
	else
	{
		float* im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, im_ptr += imageStep)
		{
			float* slice_ptr = im_ptr;
			for (int c = 0; c < C; c+=align_size, slice_ptr += sliceStep)
			{
				for (int h = 0; h < borderH; h++)
				{
					memset(slice_ptr - (h + 1)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
					memset(slice_ptr + (H + h)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
				}

				float* row_ptr = slice_ptr;
				for (int h = 0; h < H; h++, row_ptr += widthStep)
				{
					memset(row_ptr - borderW*align_size, 0, sizeof(float)*borderW*align_size);
					memset(row_ptr + W*align_size, 0, sizeof(float)*borderW*align_size);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC1::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	int align_size = GetAlignSize();
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_slice = (dst_C + align_size -1)/ align_size;
	int dst_widthStep = dst_realW * align_size;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_imStep = dst_slice * dst_sliceStep;
	int dst_tensor_raw_size = dst_imStep*dst_N * sizeof(float);
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
		widthStep = 0;
		sliceStep = 0;
		imageStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)malloc(needed_dst_raw_len);
			if (tmp_data == 0)
				return false;
			//memset(tmp_data, 0, needed_dst_raw_len);
			if (rawData)
				free(rawData);
			rawData = tmp_data;
		}

		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*align_size;
		rawDataLen = needed_dst_raw_len;


		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
		imageStep = dst_imStep;
	}

	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC1::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	int align_size = GetAlignSize();
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

		int dstWidthStep = dst.GetWidthStep();
		int dstSliceStep = dst.GetSliceStep();
		int dstImageStep = dst.GetImageStep();

		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc1(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
						dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc1(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				}
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					for (int h = 0; h < dst_borderH; h++)
					{
						memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
						memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
					}
				}
			}
		}
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC1::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC1& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int align_size = GetAlignSize();
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

	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_im_ptr = dst.GetFirstPixelPtr() + dstImageStep*i;
		bool can_call_safeborder = true;

		if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
			|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc1(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc1(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}	
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC1::ROI(ZQ_CNN_Tensor4D_NCHWC1& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const
{
	int align_size = GetAlignSize();
	if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
		return false;

	if (!dst.ChangeSize(N, height, width, C, dst_borderH, dst_borderW))
		return false;
	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();
	const float* src_im_ptr = GetFirstPixelPtr() + off_y * widthStep + off_x*align_size;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	float* dst_im_ptr = dst.GetFirstPixelPtr();
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	for (int n = 0; n < N; n++, src_im_ptr += imageStep, dst_im_ptr += dstImageStep)
	{
		src_slice_ptr = src_im_ptr;
		dst_slice_ptr = dst_im_ptr;
		for (int c = 0; c < C; c += align_size, src_slice_ptr += sliceStep, dst_slice_ptr += dstSliceStep)
		{
			src_row_ptr = src_slice_ptr;
			dst_row_ptr = dst_slice_ptr;
			for (int h = 0; h < height; h++, src_row_ptr += widthStep, dst_row_ptr += dstWidthStep)
			{
				src_pix_ptr = src_row_ptr;
				dst_pix_ptr = dst_row_ptr;
				for (int w = 0; w < width; w++, src_pix_ptr += align_size, dst_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}

	if (dst_borderH > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*height, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}
			
	}
	
	if (dst_borderW > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC1::CopyData(const ZQ_CNN_Tensor4D_NCHWC1& other)
{
	int align_size = GetAlignSize();
	if (!ChangeSize(other.GetN(), other.GetH(), other.GetW(), other.GetC(), other.GetBorderW(), other.GetBorderH()))
		return false;
	float* dst_im_ptr = firstPixelData;
	const float* src_im_ptr = other.firstPixelData;
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	for (int n = 0; n < N; n++, dst_im_ptr += imageStep, src_im_ptr += other.imageStep)
	{
		dst_slice_ptr = dst_im_ptr;
		src_slice_ptr = src_im_ptr;
		for (int c = 0; c < C; c += align_size, dst_slice_ptr += sliceStep, src_slice_ptr += other.sliceStep)
		{
			dst_row_ptr = dst_slice_ptr;
			src_row_ptr = src_slice_ptr;
			for (int h = -borderH; h < H + borderH; h++, dst_row_ptr += widthStep, src_row_ptr += other.widthStep)
			{
				dst_pix_ptr = dst_row_ptr;
				src_pix_ptr = src_row_ptr;
				for (int w = -borderW; w < W + borderW; w++, dst_pix_ptr += align_size, src_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}
	return true;
}

/**************************************/

#if __ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

ZQ_CNN_Tensor4D_NCHWC4::ZQ_CNN_Tensor4D_NCHWC4()
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
	widthStep = 0;
	sliceStep = 0;
	imageStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_C4;
}


ZQ_CNN_Tensor4D_NCHWC4::~ZQ_CNN_Tensor4D_NCHWC4()
{
	if (rawData)
	{
		_aligned_free(rawData);
		rawData = 0;
	}
}



void ZQ_CNN_Tensor4D_NCHWC4::Swap(ZQ_CNN_Tensor4D_NCHWC4& other)
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
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	int tmp_imageStep = imageStep; imageStep = other.imageStep; other.imageStep = tmp_imageStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}

bool ZQ_CNN_Tensor4D_NCHWC4::Padding(int padW, int padH, int mode)
{
	int align_size = GetAlignSize();
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NCHWC4 tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_im_ptr = tmp.firstPixelData;
		float* cur_im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_im_ptr += tmp.imageStep, cur_im_ptr += imageStep)
		{
			float* tmp_slice_ptr = tmp_im_ptr;
			float* cur_slice_ptr = cur_im_ptr;
			for (int c = 0; c < C; c+=align_size, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
			{
				for (int h = 0; h <tmp.borderH; h++)
				{
					memset(tmp_slice_ptr - (h + 1)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
					memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
				}

				float* tmp_row_ptr = tmp_slice_ptr;
				float* cur_row_ptr = cur_slice_ptr;
				for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
				{
					memset(tmp_row_ptr - tmp.borderW*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memset(tmp_row_ptr + tmp.W*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*align_size);
				}
			}
		}
		Swap(tmp);
	}
	else
	{
		float* im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, im_ptr += imageStep)
		{
			float* slice_ptr = im_ptr;
			for (int c = 0; c < C; c+=align_size, slice_ptr += sliceStep)
			{
				for (int h = 0; h < borderH; h++)
				{
					memset(slice_ptr - (h + 1)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
					memset(slice_ptr + (H + h)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
				}

				float* row_ptr = slice_ptr;
				for (int h = 0; h < H; h++, row_ptr += widthStep)
				{
					memset(row_ptr - borderW*align_size, 0, sizeof(float)*borderW*align_size);
					memset(row_ptr + W*align_size, 0, sizeof(float)*borderW*align_size);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC4::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	int align_size = GetAlignSize();
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_slice = (dst_C + align_size - 1) / align_size;
	int dst_widthStep = dst_realW * align_size;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_imStep = dst_slice * dst_sliceStep;
	int dst_tensor_raw_size = dst_imStep*dst_N * sizeof(float);
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
		widthStep = 0;
		sliceStep = 0;
		imageStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)_aligned_malloc(needed_dst_raw_len,16);
			if (tmp_data == 0)
				return false;
			//memset(tmp_data, 0, needed_dst_raw_len);
			if (rawData)
				_aligned_free(rawData);
			rawData = tmp_data;
		}

		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*align_size;
		rawDataLen = needed_dst_raw_len;


		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
		imageStep = dst_imStep;
	}

	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC4::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	int align_size = GetAlignSize();
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

		int dstWidthStep = dst.GetWidthStep();
		int dstSliceStep = dst.GetSliceStep();
		int dstImageStep = dst.GetImageStep();

		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc4(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc4(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				}
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					for (int h = 0; h < dst_borderH; h++)
					{
						memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
						memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
					}
				}
			}
		}
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC4::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC4& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int align_size = GetAlignSize();
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

	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_im_ptr = dst.GetFirstPixelPtr() + dstImageStep*i;
		bool can_call_safeborder = true;

		if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
			|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc4(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc4(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC4::ROI(ZQ_CNN_Tensor4D_NCHWC4& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const
{
	int align_size = GetAlignSize();
	if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
		return false;

	if (!dst.ChangeSize(N, height, width, C, dst_borderH, dst_borderW))
		return false;
	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();
	const float* src_im_ptr = GetFirstPixelPtr() + off_y * widthStep + off_x*align_size;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	float* dst_im_ptr = dst.GetFirstPixelPtr();
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	for (int n = 0; n < N; n++, src_im_ptr += imageStep, dst_im_ptr += dstImageStep)
	{
		src_slice_ptr = src_im_ptr;
		dst_slice_ptr = dst_im_ptr;
		for (int c = 0; c < C; c += align_size, src_slice_ptr += sliceStep, dst_slice_ptr += dstSliceStep)
		{
			src_row_ptr = src_slice_ptr;
			dst_row_ptr = dst_slice_ptr;
			for (int h = 0; h < height; h++,src_row_ptr += widthStep, dst_row_ptr += dstWidthStep)
			{
				src_pix_ptr = src_row_ptr;
				dst_pix_ptr = dst_row_ptr;
				for (int w = 0; w < width; w++, src_pix_ptr += align_size, dst_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}

	if (dst_borderH > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*height, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}

	}

	if (dst_borderW > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC4::CopyData(const ZQ_CNN_Tensor4D_NCHWC4& other)
{
	int align_size = GetAlignSize();
	if (!ChangeSize(other.GetN(), other.GetH(), other.GetW(), other.GetC(), other.GetBorderW(), other.GetBorderH()))
		return false;
	float* dst_im_ptr = firstPixelData;
	const float* src_im_ptr = other.firstPixelData;
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	for (int n = 0; n < N; n++, dst_im_ptr += imageStep, src_im_ptr += other.imageStep)
	{
		dst_slice_ptr = dst_im_ptr;
		src_slice_ptr = src_im_ptr;
		for (int c = 0; c < C; c += align_size, dst_slice_ptr += sliceStep, src_slice_ptr += other.sliceStep)
		{
			dst_row_ptr = dst_slice_ptr;
			src_row_ptr = src_slice_ptr;
			for (int h = -borderH; h < H + borderH; h++, dst_row_ptr += widthStep, src_row_ptr += other.widthStep)
			{
				dst_pix_ptr = dst_row_ptr;
				src_pix_ptr = src_row_ptr;
				for (int w = -borderW; w < W + borderW; w++, dst_pix_ptr += align_size, src_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}
	return true;
}

#endif //__ARM_NEON || (ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE)

/****************************/

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX

ZQ_CNN_Tensor4D_NCHWC8::ZQ_CNN_Tensor4D_NCHWC8()
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
	widthStep = 0;
	sliceStep = 0;
	imageStep = 0;

	firstPixelData = 0;
	rawData = 0;
	rawDataLen = 0;

	align_type = ALIGN_C8;
}


ZQ_CNN_Tensor4D_NCHWC8::~ZQ_CNN_Tensor4D_NCHWC8()
{
	if (rawData)
	{
		_aligned_free(rawData);
		rawData = 0;
	}
}



void ZQ_CNN_Tensor4D_NCHWC8::Swap(ZQ_CNN_Tensor4D_NCHWC8& other)
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
	int tmp_widthStep = widthStep; widthStep = other.widthStep; other.widthStep = tmp_widthStep;
	int tmp_sliceStep = sliceStep; sliceStep = other.sliceStep; other.sliceStep = tmp_sliceStep;
	int tmp_imageStep = imageStep; imageStep = other.imageStep; other.imageStep = tmp_imageStep;
	float* tmp_firstPixelData = firstPixelData; firstPixelData = other.firstPixelData; other.firstPixelData = tmp_firstPixelData;
	unsigned char* tmp_rawData = rawData; rawData = other.rawData; other.rawData = tmp_rawData;
	long long tmp_rawDataLen = rawDataLen; rawDataLen = other.rawDataLen; other.rawDataLen = tmp_rawDataLen;
}

bool ZQ_CNN_Tensor4D_NCHWC8::Padding(int padW, int padH, int mode)
{
	int align_size = GetAlignSize();
	if (padW > borderW || padH > borderH)
	{
		ZQ_CNN_Tensor4D_NCHWC8 tmp;
		if (!tmp.ChangeSize(N, H, W, C, padW, padH))
			return false;
		//
		float* tmp_im_ptr = tmp.firstPixelData;
		float* cur_im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, tmp_im_ptr += tmp.imageStep, cur_im_ptr += imageStep)
		{
			float* tmp_slice_ptr = tmp_im_ptr;
			float* cur_slice_ptr = cur_im_ptr;
			for (int c = 0; c < C; c+=align_size, tmp_slice_ptr += tmp.sliceStep, cur_slice_ptr += sliceStep)
			{
				for (int h = 0; h <tmp.borderH; h++)
				{
					memset(tmp_slice_ptr - (h + 1)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
					memset(tmp_slice_ptr + (H + h)*tmp.widthStep - tmp.borderW*align_size, 0, sizeof(float)*tmp.widthStep);
				}

				float* tmp_row_ptr = tmp_slice_ptr;
				float* cur_row_ptr = cur_slice_ptr;
				for (int h = 0; h < H; h++, tmp_row_ptr += tmp.widthStep, cur_row_ptr += widthStep)
				{
					memset(tmp_row_ptr - tmp.borderW*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memset(tmp_row_ptr + tmp.W*align_size, 0, sizeof(float)*tmp.borderW*align_size);
					memcpy(tmp_row_ptr, cur_row_ptr, sizeof(float)* W*align_size);
				}
			}
		}
		Swap(tmp);
	}
	else
	{
		float* im_ptr = firstPixelData;
		for (int n = 0; n < N; n++, im_ptr += imageStep)
		{
			float* slice_ptr = im_ptr;
			for (int c = 0; c < C; c+=align_size, slice_ptr += sliceStep)
			{
				for (int h = 0; h < borderH; h++)
				{
					memset(slice_ptr - (h + 1)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
					memset(slice_ptr + (H + h)*widthStep - borderW*align_size, 0, sizeof(float)*widthStep);
				}

				float* row_ptr = slice_ptr;
				for (int h = 0; h < H; h++, row_ptr += widthStep)
				{
					memset(row_ptr - borderW*align_size, 0, sizeof(float)*borderW*align_size);
					memset(row_ptr + W*align_size, 0, sizeof(float)*borderW*align_size);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC8::ChangeSize(int dst_N, int dst_H, int dst_W, int dst_C, int dst_borderW, int dst_borderH)
{
	int align_size = GetAlignSize();
	if (N == dst_N && H == dst_H && W == dst_W && C == dst_C && borderW == dst_borderW && borderH == dst_borderH)
		return true;
	shape_nchw[0] = dst_N;
	shape_nchw[1] = dst_C;
	shape_nchw[2] = dst_H;
	shape_nchw[3] = dst_W;
	int dst_realW = dst_W + (dst_borderW << 1);
	int dst_realH = dst_H + (dst_borderH << 1);
	int dst_slice = (dst_C + align_size - 1) / align_size;
	int dst_widthStep = dst_realW * align_size;
	int dst_sliceStep = dst_widthStep*dst_realH;
	int dst_imStep = dst_slice * dst_sliceStep;
	int dst_tensor_raw_size = dst_imStep*dst_N * sizeof(float);
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
		widthStep = 0;
		sliceStep = 0;
		imageStep = 0;
	}
	else
	{
		if (rawDataLen != needed_dst_raw_len)
		{
			unsigned char* tmp_data = (unsigned char*)_aligned_malloc(needed_dst_raw_len, 32);
			if (tmp_data == 0)
				return false;
			//memset(tmp_data, 0, needed_dst_raw_len);
			if (rawData)
				_aligned_free(rawData);
			rawData = tmp_data;
		}

		firstPixelData = (float*)rawData + dst_borderH*dst_widthStep + dst_borderW*align_size;
		rawDataLen = needed_dst_raw_len;


		N = dst_N;
		W = dst_W;
		H = dst_H;
		C = dst_C;
		borderW = dst_borderW;
		borderH = dst_borderH;
		realHeight = dst_realH;
		realWidth = dst_realW;
		widthStep = dst_widthStep;
		sliceStep = dst_sliceStep;
		imageStep = dst_imStep;
	}

	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC8::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	int src_off_x, int src_off_y, int src_rect_w, int src_rect_h) const
{
	int align_size = GetAlignSize();
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

		int dstWidthStep = dst.GetWidthStep();
		int dstSliceStep = dst.GetSliceStep();
		int dstImageStep = dst.GetImageStep();

		bool can_call_safeborder = true;

		if (dst_W > src_rect_w && (src_off_x == 0 || src_off_x + src_rect_w == W)
			|| dst_H > src_rect_h && (src_off_y == 0 || src_off_y + src_rect_h == H))
			can_call_safeborder = false;

		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc8(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc8(firstPixelData, N, H, W, C, widthStep, sliceStep, imageStep, src_off_x, src_off_y, src_rect_w, src_rect_h,
				dst.GetFirstPixelPtr(), dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				}
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_im_ptr = dst.GetFirstPixelPtr();
			for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
			{
				float* dst_slice_ptr = dst_im_ptr;
				for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
				{
					for (int h = 0; h < dst_borderH; h++)
					{
						memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
						memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
					}
				}
			}
		}
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC8::ResizeBilinearRect(ZQ_CNN_Tensor4D_NCHWC8& dst, int dst_W, int dst_H, int dst_borderW, int dst_borderH,
	const std::vector<int>& src_off_x, const std::vector<int>& src_off_y, const std::vector<int>& src_rect_w, const std::vector<int>& src_rect_h) const
{
	int align_size = GetAlignSize();
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

	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();

	for (int i = 0; i < rect_num; i++)
	{
		float* dst_im_ptr = dst.GetFirstPixelPtr() + dstImageStep*i;
		bool can_call_safeborder = true;

		if (dst_W > src_rect_w[i] && (src_off_x[i] == 0 || src_off_x[i] + src_rect_w[i] == W)
			|| dst_H > src_rect_h[i] && (src_off_y[i] == 0 || src_off_y[i] + src_rect_h[i] == H))
			can_call_safeborder = false;

		int align_mode = __min(GetAlignType(), dst.GetAlignType());
		if (can_call_safeborder)
		{
			zq_cnn_resize_with_safeborder_nchwc8(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}
		else
		{
			zq_cnn_resize_without_safeborder_nchwc8(firstPixelData, 1, H, W, C, widthStep, sliceStep, imageStep, src_off_x[i], src_off_y[i], src_rect_w[i], src_rect_h[i],
				dst_im_ptr, dst_H, dst_W, dstWidthStep, dstSliceStep, dstImageStep);
		}

		if (dst_borderH > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}
		if (dst_borderW > 0)
		{
			float* dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}
	return true;
}


bool ZQ_CNN_Tensor4D_NCHWC8::ROI(ZQ_CNN_Tensor4D_NCHWC8& dst, int off_x, int off_y, int width, int height, int dst_borderH, int dst_borderW) const
{
	int align_size = GetAlignSize();
	if (off_x < 0 || off_y < 0 || off_x + width > W || off_y + height > H)
		return false;

	if (!dst.ChangeSize(N, height, width, C, dst_borderH, dst_borderW))
		return false;
	int dstWidthStep = dst.GetWidthStep();
	int dstSliceStep = dst.GetSliceStep();
	int dstImageStep = dst.GetImageStep();
	const float* src_im_ptr = GetFirstPixelPtr() + off_y * widthStep + off_x*align_size;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	float* dst_im_ptr = dst.GetFirstPixelPtr();
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	for (int n = 0; n < N; n++, src_im_ptr += imageStep, dst_im_ptr += dstImageStep)
	{
		src_slice_ptr = src_im_ptr;
		dst_slice_ptr = dst_im_ptr;
		for (int c = 0; c < C; c += align_size, src_slice_ptr += sliceStep, dst_slice_ptr += dstSliceStep)
		{
			src_row_ptr = src_slice_ptr;
			dst_row_ptr = dst_slice_ptr;
			for (int h = 0; h < height; h++, src_row_ptr += widthStep, dst_row_ptr += dstWidthStep)
			{
				src_pix_ptr = src_row_ptr;
				dst_pix_ptr = dst_row_ptr;
				for (int w = 0; w < width; w++, src_pix_ptr += align_size, dst_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}

	if (dst_borderH > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				memset(dst_slice_ptr - align_size*dst_borderW - dstWidthStep*dst_borderH, 0, sizeof(float)*dstWidthStep*dst_borderH);
				memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*height, 0, sizeof(float)*dstWidthStep*dst_borderH);
			}
		}

	}

	if (dst_borderW > 0)
	{
		dst_im_ptr = dst.GetFirstPixelPtr();
		for (int n = 0; n < N; n++, dst_im_ptr += dstImageStep)
		{
			dst_slice_ptr = dst_im_ptr;
			for (int c = 0; c < C; c += align_size, dst_slice_ptr += dstSliceStep)
			{
				for (int h = 0; h < dst_borderH; h++)
				{
					memset(dst_slice_ptr - align_size*dst_borderW + dstWidthStep*h, 0, sizeof(float)*align_size*dst_borderW);
					memset(dst_slice_ptr - align_size*(dst_borderW << 1) + dstWidthStep*(h + 1), 0, sizeof(float)*align_size*dst_borderW);
				}
			}
		}
	}
	return true;
}

bool ZQ_CNN_Tensor4D_NCHWC8::CopyData(const ZQ_CNN_Tensor4D_NCHWC8& other)
{
	int align_size = GetAlignSize();
	if (!ChangeSize(other.GetN(), other.GetH(), other.GetW(), other.GetC(), other.GetBorderW(), other.GetBorderH()))
		return false;
	float* dst_im_ptr = firstPixelData;
	const float* src_im_ptr = other.firstPixelData;
	float* dst_slice_ptr, *dst_row_ptr, *dst_pix_ptr;
	const float* src_slice_ptr, *src_row_ptr, *src_pix_ptr;
	for (int n = 0; n < N; n++, dst_im_ptr += imageStep, src_im_ptr += other.imageStep)
	{
		dst_slice_ptr = dst_im_ptr;
		src_slice_ptr = src_im_ptr;
		for (int c = 0; c < C; c += align_size, dst_slice_ptr += sliceStep, src_slice_ptr += other.sliceStep)
		{
			dst_row_ptr = dst_slice_ptr;
			src_row_ptr = src_slice_ptr;
			for (int h = -borderH; h < H + borderH; h++, dst_row_ptr += widthStep, src_row_ptr += other.widthStep)
			{
				dst_pix_ptr = dst_row_ptr;
				src_pix_ptr = src_row_ptr;
				for (int w = -borderW; w < W + borderW; w++, dst_pix_ptr += align_size, src_pix_ptr += align_size)
				{
					memcpy(dst_pix_ptr, src_pix_ptr, sizeof(float)*align_size);
				}
			}
		}
	}
	return true;
}

#endif //ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX