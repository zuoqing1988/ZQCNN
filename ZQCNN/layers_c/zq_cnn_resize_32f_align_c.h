#ifndef _ZQ_CNN_RESIZE_32F_ALIGN_C_H_
#define _ZQ_CNN_RESIZE_32F_ALIGN_C_H_
#include "../ZQ_CNN_CompileConfig.h"
#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif



#if __ARM_NEON

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align128bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align128bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if __ARM_NEON_FP16
	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_16f_align0(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float16_t* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_16f_align0(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float16_t* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_16f_align128bit(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float16_t* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_16f_align128bit(
		const float16_t* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float16_t* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif//__ARM_NEON_FP16

#else

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align0(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align128bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align128bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1].
	so, you should allocate the input image with border.
	Make sure in_off_x >= 0 && in_off_y >=0 && in_off_x+in_rect_width<=in_W && in_off_y+in_rect_height<= in_H,
	this function will not check it.

	if (out_W > in_rect_w && (in_off_x == 0 || in_off_x + in_rect_w == W)
	|| out_H > in_rect_h && (in_off_y == 0 || in_off_y + in_rect_h == H))
	can_call_safeborder = false;
	*/
	void zq_cnn_resize_with_safeborder_32f_align256bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);

	/*WARNING: when scaling to larger images, it may visit the coordinate input[-1][?] or input[?][-1], or input[H][?], input[?][W].
	this function will clamp to input[0][?] or input[?][0], or input[H-1][?], input[?][W-1]
	*/
	void zq_cnn_resize_without_safeborder_32f_align256bit(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_pixelStep,
		int in_widthStep,
		int in_sliceStep,
		int in_off_x,
		int in_off_y,
		int in_rect_width,
		int in_rect_height,
		float* out_tensor4D_data,
		int out_H,
		int out_W,
		int out_pixelStep,
		int out_widthStep,
		int out_sliceStep
	);
#endif

#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif
#endif
