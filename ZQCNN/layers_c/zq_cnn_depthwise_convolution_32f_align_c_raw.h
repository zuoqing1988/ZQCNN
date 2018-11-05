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

void zq_cnn_depthwise_conv_no_padding_32f_general_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, kc;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
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

	free(in_offsets);
	free(out_offsets);
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

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kc;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
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
	free(in_offsets);
	free(out_offsets);
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

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_1_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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

	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr+zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr+zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr+zq_mm_align_size), zq_mm_load_ps(out_pix_ptr+zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_2_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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
	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size*2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size*2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size*2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size*2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size*3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size*3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size*3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size*3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
#else
				cur_in_c_ptr = cur_in_pix_ptr; cur_filter_c_ptr = cur_filter_pix_ptr; out_c_ptr = out_pix_ptr;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
				cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
				zq_mm_store_ps(out_c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), zq_mm_load_ps(out_c_ptr)));
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_4_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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
	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif
				

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
#else
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
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_8_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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
	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif
				

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
#else
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
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_16_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count) 
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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
	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif
				

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
#else
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
#endif
			}
		}
	}
}


void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_32_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];

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
	free(in_offsets);
	free(out_offsets);
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
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif
				

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif


				cur_in_row_ptr += in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif

				cur_in_pix_ptr += in_pixelStep; cur_filter_pix_ptr += filter_pixelStep;
#if ZQ_CNN_USE_PTR_PLUS_CONST
				zq_mm_store_ps(out_pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 2, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 2), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 2)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 3, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 3), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 3)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 4, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 4), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 4)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 5, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 5), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 5)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 6, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 6), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 6)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 7, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 7), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 7)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 8, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 8), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 8)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 9, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 9), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 9)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 10, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 10), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 10)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 11, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 11), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 11)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 12, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 12), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 12)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 13, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 13), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 13)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 14, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 14), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 14)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 15, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 15), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 15)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 16, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 16), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 16)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 17, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 17), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 17)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 18, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 18), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 18)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 19, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 19), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 19)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 20, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 20), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 20)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 21, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 21), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 21)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 22, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 22), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 22)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 23, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 23), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 23)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 24, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 24), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 24)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 25, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 25), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 25)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 26, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 26), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 26)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 27, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 27), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 27)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 28, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 28), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 28)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 29, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 29), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 29)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 30, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 30), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 30)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 31, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 31), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 31)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 32, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 32), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 32)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 33, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 33), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 33)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 34, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 34), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 34)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 35, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 35), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 35)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 36, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 36), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 36)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 37, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 37), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 37)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 38, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 38), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 38)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 39, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 39), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 39)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 40, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 40), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 40)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 41, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 41), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 41)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 42, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 42), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 42)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 43, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 43), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 43)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 44, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 44), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 44)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 45, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 45), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 45)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 46, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 46), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 46)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 47, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 47), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 47)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 48, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 48), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 48)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 49, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 49), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 49)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 50, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 50), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 50)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 51, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 51), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 51)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 52, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 52), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 52)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 53, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 53), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 53)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 54, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 54), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 54)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 55, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 55), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 55)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 56, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 56), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 56)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 57, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 57), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 57)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 58, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 58), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 58)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 59, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 59), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 59)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 60, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 60), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 60)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 61, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 61), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 61)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 62, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 62), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 62)));
				zq_mm_store_ps(out_pix_ptr + zq_mm_align_size * 63, zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(cur_filter_pix_ptr + zq_mm_align_size * 63), zq_mm_load_ps(out_pix_ptr + zq_mm_align_size * 63)));
#else
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
#endif
			}
		}
	}
}

void zq_cnn_depthwise_conv_no_padding_32f_kernel3x3_mul_64_omp(
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
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int out_n, out_h, out_w;

	int out_NHW = out_N*out_H*out_W;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;
	for (out_n = 0; out_n < out_N; out_n++)
	{
		for (out_h = 0; out_h < out_H; out_h++)
		{
			for (out_w = 0; out_w < out_W; out_w++)
			{
				in_offsets[idx] = out_n*in_sliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_sliceStep + out_h*out_widthStep + out_w*out_pixelStep;
				idx++;
			}
		}
	}

#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		
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
	free(in_offsets);
	free(out_offsets);
}