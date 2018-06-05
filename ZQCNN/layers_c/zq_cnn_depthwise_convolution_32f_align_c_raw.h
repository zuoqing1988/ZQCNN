void zq_cnn_depthwise_conv_no_padding_32f_general(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, //must be 1
	int filter_H,
	int filter_W,
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

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
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());


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
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

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
				for (out_c = 0, out_c_ptr = out_pix_ptr; out_c < in_C; out_c += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

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
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_1(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_2(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_4(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}



void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_8(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_16(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{	
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}



void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_32(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
			}
		}
	}
}



void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_64(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N, // must be 1
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be in_C
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_in_c_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;
	
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
				out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());
				out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_setzero_ps());

				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = filters_data;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));

			}
		}
	}
}