
#define op_0_4 \
	a0 = zq_mm_load_ps(cur_in_c_ptr);\
	b0 = zq_mm_load_ps(cur_filter_c_ptr);\
	c0 = zq_mm_load_ps(out_c_ptr);\
	a1 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size);\
	c1 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size2);\
	c2 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size3);\
	c3 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size3);\
	d0 = zq_mm_fmadd_ps(a0,b0,c0);\
	d1 = zq_mm_fmadd_ps(a1,b1,c1);\
	d2 = zq_mm_fmadd_ps(a2,b2,c2);\
	d3 = zq_mm_fmadd_ps(a3,b3,c3);\
	zq_mm_store_ps(out_c_ptr, d0);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, d1);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, d2);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, d3);\
	cur_in_c_ptr += zq_mm_align_size4;\
	cur_filter_c_ptr += zq_mm_align_size4;\
	out_c_ptr += zq_mm_align_size4

#if WITH_BIAS
#define op_0_4_first \
	a0 = zq_mm_load_ps(cur_in_c_ptr);\
	b0 = zq_mm_load_ps(cur_filter_c_ptr);\
	c0 = zq_mm_load_ps(cur_bias_ptr);\
	a1 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size);\
	c1 = zq_mm_load_ps(cur_bias_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size2);\
	c2 = zq_mm_load_ps(cur_bias_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size3);\
	c3 = zq_mm_load_ps(cur_bias_ptr+zq_mm_align_size3);\
	d0 = zq_mm_fmadd_ps(a0,b0,c0);\
	d1 = zq_mm_fmadd_ps(a1,b1,c1);\
	d2 = zq_mm_fmadd_ps(a2,b2,c2);\
	d3 = zq_mm_fmadd_ps(a3,b3,c3);\
	zq_mm_store_ps(out_c_ptr, d0);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, d1);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, d2);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, d3);\
	cur_in_c_ptr += zq_mm_align_size4;\
	cur_filter_c_ptr += zq_mm_align_size4;\
	cur_bias_ptr += zq_mm_align_size4;\
	out_c_ptr += zq_mm_align_size4
#else
#define op_0_4_first \
	a0 = zq_mm_load_ps(cur_in_c_ptr);\
	b0 = zq_mm_load_ps(cur_filter_c_ptr);\
	a1 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(cur_in_c_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(cur_filter_c_ptr+zq_mm_align_size3);\
	d0 = zq_mm_mul_ps(a0,b0);\
	d1 = zq_mm_mul_ps(a1,b1);\
	d2 = zq_mm_mul_ps(a2,b2);\
	d3 = zq_mm_mul_ps(a3,b3);\
	zq_mm_store_ps(out_c_ptr, d0);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, d1);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, d2);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, d3);\
	cur_in_c_ptr += zq_mm_align_size4;\
	cur_filter_c_ptr += zq_mm_align_size4;\
	out_c_ptr += zq_mm_align_size4
#endif//WITH_BIAS

#define op_0_8 \
	op_0_4;\
	op_0_4

#define op_0_16 \
	op_0_8;\
	op_0_8;

#define op_0_32 \
	op_0_16;\
	op_0_16

#define op_0_64 \
	op_0_32;\
	op_0_32

#define op_0_8_first \
	op_0_4_first;\
	op_0_4_first

#define op_0_16_first \
	op_0_8_first;\
	op_0_8_first

#define op_0_32_first \
	op_0_16_first;\
	op_0_16_first

#define op_0_64_first \
	op_0_32_first;\
	op_0_32_first

#if WITH_PRELU

#define op_0_4_last \
	a0 = zq_mm_load_ps(cur_in_c_ptr); \
	b0 = zq_mm_load_ps(cur_filter_c_ptr); \
	c0 = zq_mm_load_ps(out_c_ptr); \
	e0 = zq_mm_load_ps(cur_slope_ptr); \
	a1 = zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size); \
	b1 = zq_mm_load_ps(cur_filter_c_ptr + zq_mm_align_size); \
	c1 = zq_mm_load_ps(out_c_ptr + zq_mm_align_size); \
	e1 = zq_mm_load_ps(cur_slope_ptr + zq_mm_align_size); \
	a2 = zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2); \
	b2 = zq_mm_load_ps(cur_filter_c_ptr + zq_mm_align_size2); \
	c2 = zq_mm_load_ps(out_c_ptr + zq_mm_align_size2); \
	e2 = zq_mm_load_ps(cur_slope_ptr + zq_mm_align_size2); \
	a3 = zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3); \
	b3 = zq_mm_load_ps(cur_filter_c_ptr + zq_mm_align_size3); \
	c3 = zq_mm_load_ps(out_c_ptr + zq_mm_align_size3); \
	e3 = zq_mm_load_ps(cur_slope_ptr + zq_mm_align_size3); \
	d0 = zq_mm_fmadd_ps(a0, b0, c0); \
	d1 = zq_mm_fmadd_ps(a1, b1, c1); \
	d2 = zq_mm_fmadd_ps(a2, b2, c2); \
	d3 = zq_mm_fmadd_ps(a3, b3, c3); \
	c0 = zq_mm_min_ps(zero_v,d0);\
	c1 = zq_mm_min_ps(zero_v,d1);\
	c2 = zq_mm_min_ps(zero_v,d2);\
	c3 = zq_mm_min_ps(zero_v,d3);\
	b0 = zq_mm_max_ps(zero_v,d0);\
	b1 = zq_mm_max_ps(zero_v,d1);\
	b2 = zq_mm_max_ps(zero_v,d2);\
	b3 = zq_mm_max_ps(zero_v,d3);\
	zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(e0,c0,b0));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, zq_mm_fmadd_ps(e1,c1,b1));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, zq_mm_fmadd_ps(e2,c2,b2));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, zq_mm_fmadd_ps(e3,c3,b3));\
	cur_in_c_ptr += zq_mm_align_size4; \
	cur_filter_c_ptr += zq_mm_align_size4; \
	out_c_ptr += zq_mm_align_size4; \
	cur_slope_ptr += zq_mm_align_size4

#else // not WITH_PRELU

#define op_0_4_last op_0_4

#endif // WITH_PRELU

#define op_0_8_last \
	op_0_4_last;\
	op_0_4_last

#define op_0_16_last \
	op_0_8_last;\
	op_0_8_last

#define op_0_32_last \
	op_0_16_last;\
	op_0_16_last

#define op_0_64_last \
	op_0_32_last;\
	op_0_32_last


void zq_cnn_depthwise_conv_no_padding_32f_general(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, //must be 1
	int filter_H,
	int filter_W,
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type value_v;
	register zq_mm_type slope_v;
#endif
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, kc;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
#if WITH_BIAS
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_bias_ptr = bias; out_c < in_C;
					out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size, cur_bias_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_load_ps(cur_bias_ptr));
#else
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
#endif

				for (kh = 0, cur_in_row_ptr = in_pix_ptr, cur_filter_row_ptr = filters_data;
					kh < filter_H;
					kh++, cur_in_row_ptr += in_widthStep, cur_filter_row_ptr += filter_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
						kw < filter_W;
						kw++, cur_in_pix_ptr += in_pixelStep, cur_filter_pix_ptr += filter_pixelStep)
					{
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
							kc < in_C;
							kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
						{
							zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
						}
					}
				}
#if WITH_PRELU
				for (kc = 0, out_c_ptr = out_pix_ptr, cur_slope_ptr = slope; kc < in_C; kc += zq_mm_align_size)
				{
					value_v = zq_mm_load_ps(out_c_ptr);
					slope_v = zq_mm_load_ps(cur_slope_ptr);
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
					out_c_ptr += zq_mm_align_size; cur_slope_ptr += zq_mm_align_size;
				}
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type value_v;
	register zq_mm_type slope_v;
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kc;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
#if WITH_BIAS
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_bias_ptr = bias; out_c < in_C; 
					out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size, cur_bias_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_load_ps(cur_bias_ptr));
#else
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
#endif
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

#if WITH_PRELU
				for (kc = 0, out_c_ptr = out_pix_ptr, cur_slope_ptr = slope; kc < in_C; kc += zq_mm_align_size)
				{
					value_v = zq_mm_load_ps(out_c_ptr);
					slope_v = zq_mm_load_ps(cur_slope_ptr);
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
					out_c_ptr += zq_mm_align_size; cur_slope_ptr += zq_mm_align_size;
				}
#endif
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type value_v;
	register zq_mm_type slope_v;
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kc;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
#if WITH_BIAS
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_bias_ptr = bias; out_c < in_C;
					out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size, cur_bias_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_load_ps(cur_bias_ptr));
#else
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
#endif

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr, out_c_ptr = out_pix_ptr;
					kc < in_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				}

#if WITH_PRELU
				for (kc = 0, out_c_ptr = out_pix_ptr, cur_slope_ptr = slope; kc < in_C; kc += zq_mm_align_size)
				{
					value_v = zq_mm_load_ps(out_c_ptr);
					slope_v = zq_mm_load_ps(cur_slope_ptr);
					zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
					out_c_ptr += zq_mm_align_size; cur_slope_ptr += zq_mm_align_size;
				}
#endif
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_1(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2, *cur_in_pix_ptr3;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a13, a21, a22, a23, a31, a32, a33;
	register zq_mm_type b11, b12, b13, b21, b22, b23, b31, b32, b33;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v = zq_mm_load_ps(slope);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11, c1);
#else
				sum = zq_mm_mul_ps(a11, b11);
#endif
				sum = zq_mm_fmadd_ps(a21, b21, sum);
				sum = zq_mm_fmadd_ps(a31, b31, sum);
				sum = zq_mm_fmadd_ps(a12, b12, sum);
				sum = zq_mm_fmadd_ps(a22, b22, sum);
				sum = zq_mm_fmadd_ps(a32, b32, sum);
				sum = zq_mm_fmadd_ps(a13, b13, sum);
				sum = zq_mm_fmadd_ps(a23, b23, sum);
				sum = zq_mm_fmadd_ps(a33, b33, sum);

#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);
			}
		}
	}

}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_1(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a21, a22;
	register zq_mm_type b11, b12, b21, b22;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v = zq_mm_load_ps(slope);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11, c1);
#else
				sum = zq_mm_mul_ps(a11, b11);
#endif
				sum = zq_mm_fmadd_ps(a21, b21, sum);
				sum = zq_mm_fmadd_ps(a12, b12, sum);
				sum = zq_mm_fmadd_ps(a22, b22, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);
			}
		}
	}

}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_2(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2, *cur_in_pix_ptr3;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a13, a21, a22, a23, a31, a32, a33;
	register zq_mm_type b11_1, b12_1, b13_1, b21_1, b22_1, b23_1, b31_1, b32_1, b33_1;
	register zq_mm_type b11_2, b12_2, b13_2, b21_2, b22_2, b23_2, b31_2, b32_2, b33_2;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1,c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a31, b31_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
				sum = zq_mm_fmadd_ps(a32, b32_1, sum);
				sum = zq_mm_fmadd_ps(a13, b13_1, sum);
				sum = zq_mm_fmadd_ps(a23, b23_1, sum);
				sum = zq_mm_fmadd_ps(a33, b33_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a31, b31_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
				sum = zq_mm_fmadd_ps(a32, b32_2, sum);
				sum = zq_mm_fmadd_ps(a13, b13_2, sum);
				sum = zq_mm_fmadd_ps(a23, b23_2, sum);
				sum = zq_mm_fmadd_ps(a33, b33_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_2(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a21, a22;
	register zq_mm_type b11_1, b12_1, b21_1, b22_1;
	register zq_mm_type b11_2, b12_2, b21_2, b22_2;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1, c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_3(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2, *cur_in_pix_ptr3;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a13, a21, a22, a23, a31, a32, a33;
	register zq_mm_type b11_1, b12_1, b13_1, b21_1, b22_1, b23_1, b31_1, b32_1, b33_1;
	register zq_mm_type b11_2, b12_2, b13_2, b21_2, b22_2, b23_2, b31_2, b32_2, b33_2;
	register zq_mm_type b11_3, b12_3, b13_3, b21_3, b22_3, b23_3, b31_3, b32_3, b33_3;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
	register zq_mm_type c3 = zq_mm_load_ps(bias + zq_mm_align_size2);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type slope_v3 = zq_mm_load_ps(slope + zq_mm_align_size2);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);

		cur_filter_pix_ptr = filters_data + zq_mm_align_size2;
		b11_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1, c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a31, b31_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
				sum = zq_mm_fmadd_ps(a32, b32_1, sum);
				sum = zq_mm_fmadd_ps(a13, b13_1, sum);
				sum = zq_mm_fmadd_ps(a23, b23_1, sum);
				sum = zq_mm_fmadd_ps(a33, b33_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a31, b31_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
				sum = zq_mm_fmadd_ps(a32, b32_2, sum);
				sum = zq_mm_fmadd_ps(a13, b13_2, sum);
				sum = zq_mm_fmadd_ps(a23, b23_2, sum);
				sum = zq_mm_fmadd_ps(a33, b33_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size2;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_3, c3);
#else
				sum = zq_mm_mul_ps(a11, b11_3);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_3, sum);
				sum = zq_mm_fmadd_ps(a31, b31_3, sum);
				sum = zq_mm_fmadd_ps(a12, b12_3, sum);
				sum = zq_mm_fmadd_ps(a22, b22_3, sum);
				sum = zq_mm_fmadd_ps(a32, b32_3, sum);
				sum = zq_mm_fmadd_ps(a13, b13_3, sum);
				sum = zq_mm_fmadd_ps(a23, b23_3, sum);
				sum = zq_mm_fmadd_ps(a33, b33_3, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v3, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, sum);
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_3(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a21, a22;
	register zq_mm_type b11_1, b12_1, b21_1, b22_1;
	register zq_mm_type b11_2, b12_2, b21_2, b22_2;
	register zq_mm_type b11_3, b12_3, b21_3, b22_3;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
	register zq_mm_type c3 = zq_mm_load_ps(bias + zq_mm_align_size2);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type slope_v3 = zq_mm_load_ps(slope + zq_mm_align_size2);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size2;
		b11_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1, c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size2;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_3, c3);
#else
				sum = zq_mm_mul_ps(a11, b11_3);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_3, sum);
				sum = zq_mm_fmadd_ps(a12, b12_3, sum);
				sum = zq_mm_fmadd_ps(a22, b22_3, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v3, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, sum);
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_6(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2, *cur_in_pix_ptr3;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a13, a21, a22, a23, a31, a32, a33;
	register zq_mm_type b11_1, b12_1, b13_1, b21_1, b22_1, b23_1, b31_1, b32_1, b33_1;
	register zq_mm_type b11_2, b12_2, b13_2, b21_2, b22_2, b23_2, b31_2, b32_2, b33_2;
	register zq_mm_type b11_3, b12_3, b13_3, b21_3, b22_3, b23_3, b31_3, b32_3, b33_3;
	register zq_mm_type b11_4, b12_4, b13_4, b21_4, b22_4, b23_4, b31_4, b32_4, b33_4;
	register zq_mm_type b11_5, b12_5, b13_5, b21_5, b22_5, b23_5, b31_5, b32_5, b33_5;
	register zq_mm_type b11_6, b12_6, b13_6, b21_6, b22_6, b23_6, b31_6, b32_6, b33_6;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
	register zq_mm_type c3 = zq_mm_load_ps(bias + zq_mm_align_size2);
	register zq_mm_type c4 = zq_mm_load_ps(bias + zq_mm_align_size3);
	register zq_mm_type c5 = zq_mm_load_ps(bias + zq_mm_align_size4);
	register zq_mm_type c6 = zq_mm_load_ps(bias + zq_mm_align_size5);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type slope_v3 = zq_mm_load_ps(slope + zq_mm_align_size2);
	register zq_mm_type slope_v4 = zq_mm_load_ps(slope + zq_mm_align_size3);
	register zq_mm_type slope_v5 = zq_mm_load_ps(slope + zq_mm_align_size4);
	register zq_mm_type slope_v6 = zq_mm_load_ps(slope + zq_mm_align_size5);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size2;
		b11_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size3;
		b11_4 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_4 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_4 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size4;
		b11_5 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_5 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_5 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size5;
		b11_6 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b13_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b21_6 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b23_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);
		cur_filter_pix_ptr += filter_widthStep;
		b31_6 = zq_mm_load_ps(cur_filter_pix_ptr);
		b32_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		b33_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep * 2);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1, c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a31, b31_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
				sum = zq_mm_fmadd_ps(a32, b32_1, sum);
				sum = zq_mm_fmadd_ps(a13, b13_1, sum);
				sum = zq_mm_fmadd_ps(a23, b23_1, sum);
				sum = zq_mm_fmadd_ps(a33, b33_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a31, b31_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
				sum = zq_mm_fmadd_ps(a32, b32_2, sum);
				sum = zq_mm_fmadd_ps(a13, b13_2, sum);
				sum = zq_mm_fmadd_ps(a23, b23_2, sum);
				sum = zq_mm_fmadd_ps(a33, b33_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size2;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_3, c3);
#else
				sum = zq_mm_mul_ps(a11, b11_3);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_3, sum);
				sum = zq_mm_fmadd_ps(a31, b31_3, sum);
				sum = zq_mm_fmadd_ps(a12, b12_3, sum);
				sum = zq_mm_fmadd_ps(a22, b22_3, sum);
				sum = zq_mm_fmadd_ps(a32, b32_3, sum);
				sum = zq_mm_fmadd_ps(a13, b13_3, sum);
				sum = zq_mm_fmadd_ps(a23, b23_3, sum);
				sum = zq_mm_fmadd_ps(a33, b33_3, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v3, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size3;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_4, c4);
#else
				sum = zq_mm_mul_ps(a11, b11_4);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_4, sum);
				sum = zq_mm_fmadd_ps(a31, b31_4, sum);
				sum = zq_mm_fmadd_ps(a12, b12_4, sum);
				sum = zq_mm_fmadd_ps(a22, b22_4, sum);
				sum = zq_mm_fmadd_ps(a32, b32_4, sum);
				sum = zq_mm_fmadd_ps(a13, b13_4, sum);
				sum = zq_mm_fmadd_ps(a23, b23_4, sum);
				sum = zq_mm_fmadd_ps(a33, b33_4, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v4, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size4;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_5, c5);
#else
				sum = zq_mm_mul_ps(a11, b11_5);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_5, sum);
				sum = zq_mm_fmadd_ps(a31, b31_5, sum);
				sum = zq_mm_fmadd_ps(a12, b12_5, sum);
				sum = zq_mm_fmadd_ps(a22, b22_5, sum);
				sum = zq_mm_fmadd_ps(a32, b32_5, sum);
				sum = zq_mm_fmadd_ps(a13, b13_5, sum);
				sum = zq_mm_fmadd_ps(a23, b23_5, sum);
				sum = zq_mm_fmadd_ps(a33, b33_5, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v5, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size4, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size5;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;
				cur_in_pix_ptr3 = cur_in_pix_ptr2 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a13 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixStep2);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
				a23 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixStep2);
				a31 = zq_mm_load_ps(cur_in_pix_ptr3);
				a32 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixelStep);
				a33 = zq_mm_load_ps(cur_in_pix_ptr3 + in_pixStep2);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_6, c6);
#else
				sum = zq_mm_mul_ps(a11, b11_6);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_6, sum);
				sum = zq_mm_fmadd_ps(a31, b31_6, sum);
				sum = zq_mm_fmadd_ps(a12, b12_6, sum);
				sum = zq_mm_fmadd_ps(a22, b22_6, sum);
				sum = zq_mm_fmadd_ps(a32, b32_6, sum);
				sum = zq_mm_fmadd_ps(a13, b13_6, sum);
				sum = zq_mm_fmadd_ps(a23, b23_6, sum);
				sum = zq_mm_fmadd_ps(a33, b33_6, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v6, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size5, sum);
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_6(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;

	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_in_pix_ptr1, *cur_in_pix_ptr2;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a11, a12, a21, a22;
	register zq_mm_type b11_1, b12_1, b21_1, b22_1;
	register zq_mm_type b11_2, b12_2, b21_2, b22_2;
	register zq_mm_type b11_3, b12_3, b21_3, b22_3;
	register zq_mm_type b11_4, b12_4, b21_4, b22_4;
	register zq_mm_type b11_5, b12_5, b21_5, b22_5;
	register zq_mm_type b11_6, b12_6, b21_6, b22_6;
	register zq_mm_type sum;
#if WITH_BIAS
	register zq_mm_type c1 = zq_mm_load_ps(bias);
	register zq_mm_type c2 = zq_mm_load_ps(bias + zq_mm_align_size);
	register zq_mm_type c3 = zq_mm_load_ps(bias + zq_mm_align_size2);
	register zq_mm_type c4 = zq_mm_load_ps(bias + zq_mm_align_size3);
	register zq_mm_type c5 = zq_mm_load_ps(bias + zq_mm_align_size4);
	register zq_mm_type c6 = zq_mm_load_ps(bias + zq_mm_align_size5);
#endif
#if WITH_PRELU
	register zq_mm_type slope_v1 = zq_mm_load_ps(slope);
	register zq_mm_type slope_v2 = zq_mm_load_ps(slope + zq_mm_align_size);
	register zq_mm_type slope_v3 = zq_mm_load_ps(slope + zq_mm_align_size2);
	register zq_mm_type slope_v4 = zq_mm_load_ps(slope + zq_mm_align_size3);
	register zq_mm_type slope_v5 = zq_mm_load_ps(slope + zq_mm_align_size4);
	register zq_mm_type slope_v6 = zq_mm_load_ps(slope + zq_mm_align_size5);
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type e, f;
#endif
	register int in_pixStep2 = in_pixelStep << 1;
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		cur_filter_pix_ptr = filters_data;
		b11_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_1 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_1 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size;
		b11_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_2 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_2 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size2;
		b11_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_3 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_3 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size3;
		b11_4 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_4 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_4 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size4;
		b11_5 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_5 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_5 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr = filters_data + zq_mm_align_size5;
		b11_6 = zq_mm_load_ps(cur_filter_pix_ptr);
		b12_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);
		cur_filter_pix_ptr += filter_widthStep;
		b21_6 = zq_mm_load_ps(cur_filter_pix_ptr);
		b22_6 = zq_mm_load_ps(cur_filter_pix_ptr + filter_pixelStep);

		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_pix_ptr1 = in_pix_ptr;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_1, c1);
#else
				sum = zq_mm_mul_ps(a11, b11_1);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_1, sum);
				sum = zq_mm_fmadd_ps(a12, b12_1, sum);
				sum = zq_mm_fmadd_ps(a22, b22_1, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v1, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_2, c2);
#else
				sum = zq_mm_mul_ps(a11, b11_2);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_2, sum);
				sum = zq_mm_fmadd_ps(a12, b12_2, sum);
				sum = zq_mm_fmadd_ps(a22, b22_2, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v2, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size2;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_3, c3);
#else
				sum = zq_mm_mul_ps(a11, b11_3);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_3, sum);
				sum = zq_mm_fmadd_ps(a12, b12_3, sum);
				sum = zq_mm_fmadd_ps(a22, b22_3, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v3, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size3;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_4, c4);
#else
				sum = zq_mm_mul_ps(a11, b11_4);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_4, sum);
				sum = zq_mm_fmadd_ps(a12, b12_4, sum);
				sum = zq_mm_fmadd_ps(a22, b22_4, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v4, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size4;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_5, c5);
#else
				sum = zq_mm_mul_ps(a11, b11_5);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_5, sum);
				sum = zq_mm_fmadd_ps(a12, b12_5, sum);
				sum = zq_mm_fmadd_ps(a22, b22_5, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v5, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size4, sum);

				cur_in_pix_ptr1 = in_pix_ptr + zq_mm_align_size5;
				cur_in_pix_ptr2 = cur_in_pix_ptr1 + in_widthStep;

				a11 = zq_mm_load_ps(cur_in_pix_ptr1);
				a12 = zq_mm_load_ps(cur_in_pix_ptr1 + in_pixelStep);
				a21 = zq_mm_load_ps(cur_in_pix_ptr2);
				a22 = zq_mm_load_ps(cur_in_pix_ptr2 + in_pixelStep);
#if WITH_BIAS
				sum = zq_mm_fmadd_ps(a11, b11_6, c6);
#else
				sum = zq_mm_mul_ps(a11, b11_6);
#endif
				sum = zq_mm_fmadd_ps(a21, b21_6, sum);
				sum = zq_mm_fmadd_ps(a12, b12_6, sum);
				sum = zq_mm_fmadd_ps(a22, b22_6, sum);
#if WITH_PRELU
				e = zq_mm_min_ps(zero_v, sum);
				f = zq_mm_max_ps(zero_v, sum);
				sum = zq_mm_fmadd_ps(slope_v6, e, f);
#endif
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size5, sum);
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_4(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_4_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_4_last;
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_4(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_4_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_4;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_4_last;
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_8(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_8_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_8_last;
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_8(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_8_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_8;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_8_last;
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_div_8(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
#if WITH_BIAS
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_bias_ptr = bias; out_c < in_C; 
					out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size, cur_bias_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_load_ps(cur_bias_ptr));
#else
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
#endif
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}


				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8_last;
				}
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_div_8(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
#if WITH_BIAS
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_bias_ptr = bias; out_c < in_C;
					out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size, cur_bias_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_load_ps(cur_bias_ptr));
#else
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
#endif

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}


				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8;
				}

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				for (out_c = 0; out_c < in_C; out_c += zq_mm_align_size8)
				{
					op_0_8_last;
				}
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_16(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_16_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_16_last;
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_16(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_16_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_16;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_16_last;
			}
		}
	}
}



void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_32(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_32_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_32_last;
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_32(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_32_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_32;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_32_last;
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_64(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_64_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_64_last;
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel2x2_mul_64(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 2
	int filter_W, // must be 2
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
#if WITH_BIAS
	, const zq_base_type* bias
#endif
#if WITH_PRELU
	, const zq_base_type* slope
#endif
)
{
	const zq_base_type* in_slice_ptr;
	const zq_base_type* in_row_ptr;
	const zq_base_type* in_pix_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_row_ptr;
	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_row_ptr;
	const zq_base_type* cur_filter_pix_ptr;
	const zq_base_type* cur_filter_c_ptr;
#if WITH_BIAS
	const zq_base_type* cur_bias_ptr;
#endif
#if WITH_PRELU
	const zq_base_type* cur_slope_ptr;
	register zq_mm_type e0, e1, e2, e3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_pixelStep)
			{
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_BIAS
				cur_bias_ptr = bias;
#endif
				op_0_64_first;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
				op_0_64;

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr;
				out_c_ptr = out_pix_ptr;
#if WITH_PRELU
				cur_slope_ptr = slope;
#endif
				op_0_64_last;
			}
		}
	}
}

#undef op_0_4
#undef op_0_4_first
#undef op_0_4_last
#undef op_0_8
#undef op_0_8_first
#undef op_0_8_last
#undef op_0_16
#undef op_0_16_first
#undef op_0_16_last
#undef op_0_32
#undef op_0_32_first
#undef op_0_32_last
#undef op_0_64
#undef op_0_64_first
#undef op_0_64_last