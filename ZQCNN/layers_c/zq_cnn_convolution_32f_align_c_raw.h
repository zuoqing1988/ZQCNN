void zq_cnn_conv_no_padding_32f_kernel1x1(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 1
	int filter_W, // must be 1
	int filter_C, // must be in_C
	int filter_alignPixelStep,
	int filter_widthStep,
	int filter_SliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	//float result[zq_mm_align_size << 2];
	//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

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
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_alignPixelStep;
	//int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	//int dilate_W_mul_in_pixStep = dilation_W*in_alignPixelStep;
	int out_n, out_h, out_w, out_c, /*kh, kw, */kc;
	int rest_c;


	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
	{
		for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_alignWidthStep)
		{
			for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_alignPixelStep)
			{
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_SliceStep, out_c_ptr++)
				{
					//the full channels
					sum = zq_mm_setzero_ps();
					for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
						kc < filter_C - zq_mm_align_size;
						kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
					{
						cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					}

					zq_mm_store_ps(q, sum);
					*out_c_ptr = zq_final_sum_q;

					//the rest channels
					sum = zq_mm_setzero_ps();
					cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);


					zq_mm_store_ps(q, sum);
					for (rest_c = 0; kc < filter_C; kc++, rest_c++)
						*out_c_ptr += q[rest_c];
				}
			}
		}
	}
}

	void zq_cnn_conv_no_padding_32f_kernel1x1_C4(
		const float* in_tensor4D_data,
		int in_N,
		int in_H,
		int in_W,
		int in_C,
		int in_alignPixelStep,
		int in_widthStep,
		int in_SliceStep,
		const float* filters_data,
		int filter_N,
		int filter_H, // must be 1
		int filter_W, // must be 1
		int filter_C, // must be in_C
		int filter_alignPixelStep,
		int filter_widthStep,
		int filter_SliceStep,
		int stride_H,
		int stride_W,
		int dilation_H,
		int dilation_W,
		float* out_tensor4D_data,
		int out_N,	// must be in_N
		int out_H,	// must be (in_H - filter_H)/stride_H + 1
		int out_W,	// must be (in_W - filter_W)/stride_W + 1
		int out_C,	// must be filter_N
		int out_alignPixelStep,
		int out_alignWidthStep,
		int out_alignSliceStep
	)
	{
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		//float result[zq_mm_align_size << 2];
		//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

		const float* in_slice_ptr;
		const float* in_row_ptr;
		const float* in_pix_ptr;
		float* out_slice_ptr;
		float* out_row_ptr;
		float* out_pix_ptr;
		float* out_c_ptr;

		
		const float* cur_filter_slice_ptr;

		int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
		int stride_W_mul_in_pixStep = stride_W*in_alignPixelStep;
		//int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
		//int dilate_W_mul_in_pixStep = dilation_W*in_alignPixelStep;
		int out_n, out_h, out_w, out_c;
		


		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_SliceStep, out_slice_ptr += out_alignSliceStep)
		{
			for (out_h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, in_row_ptr += stride_H_mul_in_WidthStep, out_row_ptr += out_alignWidthStep)
			{
				for (out_w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, in_pix_ptr += stride_W_mul_in_pixStep, out_pix_ptr += out_alignPixelStep)
				{
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_SliceStep, out_c_ptr++)
					{
						
						sum = zq_mm_mul_ps(zq_mm_load_ps(in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr));

						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}
		}
	}

void zq_cnn_conv_no_padding_32f_kernel1x1_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_alignPixelStep,
	int in_widthStep,
	int in_SliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 1
	int filter_W, // must be 1
	int filter_C, // must be in_C
	int filter_alignPixelStep,
	int filter_widthStep,
	int filter_SliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_alignPixelStep,
	int out_alignWidthStep,
	int out_alignSliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_alignPixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_alignPixelStep;
	int out_n, out_h, out_w;
	int rest_c;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
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
				in_offsets[idx] = out_n*in_SliceStep + out_h*stride_H_mul_in_WidthStep + out_w*stride_W_mul_in_pixStep;
				out_offsets[idx] = out_n*out_alignSliceStep + out_h*out_alignWidthStep + out_w*out_alignPixelStep;
				idx++;
			}
		}
	}


#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
	for (idx = 0; idx < out_NHW; idx++)
	{
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		int out_c, kc;
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
			out_c < out_C;
			out_c++, cur_filter_slice_ptr += filter_SliceStep, out_c_ptr++)
		{
			//the full channels
			sum = zq_mm_setzero_ps();
			for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
				kc < filter_C - zq_mm_align_size;
				kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
			{
				cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			}

			zq_mm_store_ps(q, sum);
			*out_c_ptr = zq_final_sum_q;

			//the rest channels
			sum = zq_mm_setzero_ps();
			cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);


			zq_mm_store_ps(q, sum);
			for (rest_c = 0; kc < filter_C; kc++, rest_c++)
				*out_c_ptr += q[rest_c];
		}
	}
	free(in_offsets);
	free(out_offsets);
}

void zq_cnn_conv_no_padding_32f_kernel3x3(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	//float result[zq_mm_align_size << 2];
	//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

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
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, /*kh, kw, */kc;
	int rest_c;
	double t1, t2;

	t1 = omp_get_wtime();
	if (filter_C % zq_mm_align_size_mul_32 == 0)
	{
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
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_32)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}
		}
	}
	else if (filter_C % zq_mm_align_size_mul_16 == 0)
	{
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
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}


						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_16)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}
		}
	}
	else if (filter_C % zq_mm_align_size_mul_8 == 0)
	{
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
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}


						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
						cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
							kc < filter_C; kc += zq_mm_align_size_mul_8)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
							cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
						}
						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;
					}
				}
			}
		}
	}
	else if (filter_C % zq_mm_align_size == 0)
	{
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
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						sum = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
							kc < filter_C;
							kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
						{
							cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

							cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

							cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						}

						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;

					}
				}
			}
		}
	}
	else
	{
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
					for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
					{
						//the full channels
						sum = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
							kc < filter_C - zq_mm_align_size;
							kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
						{
							cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

							cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

							cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

							cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						}

						zq_mm_store_ps(q, sum);
						*out_c_ptr = zq_final_sum_q;

						//the rest channels
						sum = zq_mm_setzero_ps();
						cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						zq_mm_store_ps(q, sum);
						for (rest_c = 0; kc < filter_C; kc++, rest_c++)
							*out_c_ptr += q[rest_c];
					}
				}
			}
		}
	}
	t2 = omp_get_wtime();
	//printf("base:conv3x3: %.3f ms\n", (t2 - t1) * 1000);

}

void zq_cnn_conv_no_padding_32f_kernel3x3_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w;
	int rest_c;
	double t1, t2;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;

	t1 = omp_get_wtime();

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

	if (filter_C % zq_mm_align_size_mul_32 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			zq_mm_type sum;
			__declspec(align(32)) float q[8];
			int out_c, kc;
			float* out_c_ptr;
			const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
			const float* cur_filter_slice_ptr, *cur_filter_row_ptr,*cur_filter_pix_ptr, *cur_filter_c_ptr;
			const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
			for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
				out_c < out_C;
				out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
			{
				sum = zq_mm_setzero_ps();
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}


				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_32)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum);
				*out_c_ptr = zq_final_sum_q;
			}
		}
	}
	else if (filter_C % zq_mm_align_size_mul_16 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			zq_mm_type sum;
			__declspec(align(32)) float q[8];
			int out_c, kc;
			float* out_c_ptr;
			const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
			const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
			const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
			for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
				out_c < out_C;
				out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
			{
				sum = zq_mm_setzero_ps();
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}


				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_16)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum);
				*out_c_ptr = zq_final_sum_q;
			}
		}
	}
	else if (filter_C % zq_mm_align_size_mul_8 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			zq_mm_type sum;
			__declspec(align(32)) float q[8];
			int out_c, kc;
			float* out_c_ptr;
			const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
			const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
			const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
			for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
				out_c < out_C;
				out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
			{
				sum = zq_mm_setzero_ps();
				cur_in_row_ptr = in_pix_ptr; cur_filter_row_ptr = cur_filter_slice_ptr;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}


				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;
				cur_in_pix_ptr = cur_in_row_ptr; cur_filter_pix_ptr = cur_filter_row_ptr;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				cur_in_pix_ptr += dilate_W_mul_in_pixStep; cur_filter_pix_ptr += filter_pixelStep;
				for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, cur_filter_c_ptr = cur_filter_pix_ptr;
					kc < filter_C; kc += zq_mm_align_size_mul_8)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr), sum);
					cur_in_c_ptr += zq_mm_align_size; cur_filter_c_ptr += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum);
				*out_c_ptr = zq_final_sum_q;
			}
		}
	}
	else if (filter_C % zq_mm_align_size == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			zq_mm_type sum;
			__declspec(align(32)) float q[8];
			int out_c, kc;
			float* out_c_ptr;
			const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
			const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
			const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
			for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
				out_c < out_C;
				out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
			{
				sum = zq_mm_setzero_ps();
				for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
					kc < filter_C;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
				{
					cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				}

				zq_mm_store_ps(q, sum);
				*out_c_ptr = zq_final_sum_q;

			}
		}
	}
	else
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < out_NHW; idx++)
		{
			zq_mm_type sum;
			__declspec(align(32)) float q[8];
			int out_c, kc;
			float* out_c_ptr;
			const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
			const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
			const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
			for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
				out_c < out_C;
				out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
			{
				//the full channels
				sum = zq_mm_setzero_ps();
				for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
					kc < filter_C - zq_mm_align_size;
					kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
				{
					cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				}

				zq_mm_store_ps(q, sum);
				*out_c_ptr = zq_final_sum_q;

				//the rest channels
				sum = zq_mm_setzero_ps();
				cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				zq_mm_store_ps(q, sum);
				for (rest_c = 0; kc < filter_C; kc++, rest_c++)
					*out_c_ptr += q[rest_c];
			}
		}
	}
	free(in_offsets);
	free(out_offsets);
	t2 = omp_get_wtime();
	//printf("base:conv3x3: %.3f ms\n", (t2 - t1) * 1000);

}

void zq_cnn_conv_no_padding_32f_kernel3x3_C3(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,	//must be 3
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be 3
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	//float result[zq_mm_align_size << 2];
	//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

	const float* in_slice_ptr;
	const float* in_row_ptr;
	const float* in_pix_ptr;
	float* out_slice_ptr;
	float* out_row_ptr;
	float* out_pix_ptr;
	float* out_c_ptr;

	const float* cur_in_row_ptr;
	const float* cur_in_pix_ptr;
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w, out_c;
	double t1, t2;
	t1 = omp_get_wtime();

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
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
				{
					sum = zq_mm_setzero_ps();
					cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					zq_mm_store_ps(q, sum);
					*out_c_ptr = q[0] + q[1] + q[2];
				}
			}
		}

	}
	t2 = omp_get_wtime();
	//printf("base:conv3x3x3: %.3f ms\n", (t2 - t1) * 1000);

}

void zq_cnn_conv_no_padding_32f_kernel3x3_C3_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,	//must be 3
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be 3
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w;
	double t1, t2;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
	int chunk_size = (out_NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(out_NHW * sizeof(int));
	int* out_offsets = (int*)malloc(out_NHW * sizeof(int));
	int idx = 0;

	t1 = omp_get_wtime();

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
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		int out_c;
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr;
		const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
			out_c < out_C;
			out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
		{
			sum = zq_mm_setzero_ps();
			cur_in_row_ptr = in_pix_ptr;	cur_filter_row_ptr = cur_filter_slice_ptr;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			zq_mm_store_ps(q, sum);
			*out_c_ptr = q[0] + q[1] + q[2];
		}
	}
	free(in_offsets);
	free(out_offsets);
	t2 = omp_get_wtime();
	//printf("base:conv3x3x3: %.3f ms\n", (t2 - t1) * 1000);

}

void zq_cnn_conv_no_padding_32f_kernel2x2(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,	
	int stride_W,	
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	//float result[zq_mm_align_size << 2];
	//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

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
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, /*kh, kw, */kc;
	int rest_c;
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
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
				{
					//the full channels
					sum = zq_mm_setzero_ps();
					for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
						kc < filter_C - zq_mm_align_size;
						kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
					{
						cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					}

					zq_mm_store_ps(q, sum);
					*out_c_ptr = zq_final_sum_q;

					//the rest channels
					sum = zq_mm_setzero_ps();
					cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					zq_mm_store_ps(q, sum);
					for (rest_c = 0; kc < filter_C; kc++, rest_c++)
						*out_c_ptr += q[rest_c];
				}
			}
		}
	}
}

void zq_cnn_conv_no_padding_32f_kernel2x2_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 3
	int filter_W, // must be 3
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,	
	int stride_W,	
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w;
	int rest_c;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
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
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		int out_c, kc;
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
			out_c < out_C;
			out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
		{
			//the full channels
			sum = zq_mm_setzero_ps();
			for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
				kc < filter_C - zq_mm_align_size;
				kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
			{
				cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			}

			zq_mm_store_ps(q, sum);
			*out_c_ptr = zq_final_sum_q;

			//the rest channels
			sum = zq_mm_setzero_ps();
			cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			zq_mm_store_ps(q, sum);
			for (rest_c = 0; kc < filter_C; kc++, rest_c++)
				*out_c_ptr += q[rest_c];
		}
	}
	free(in_offsets);
	free(out_offsets);
}

void zq_cnn_conv_no_padding_32f_kernel5x5(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 5
	int filter_W, // must be 5
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,	
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	//float result[zq_mm_align_size << 2];
	//float* q = (float*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);

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
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, /*kh, kw, */kc;
	int rest_c;

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
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
				{
					//the full channels
					sum = zq_mm_setzero_ps();
					for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
						kc < filter_C - zq_mm_align_size;
						kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
					{
						cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

						cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

						cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					}

					zq_mm_store_ps(q, sum);
					*out_c_ptr = zq_final_sum_q;

					//the rest channels
					sum = zq_mm_setzero_ps();
					cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

					cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

					zq_mm_store_ps(q, sum);
					for (rest_c = 0; kc < filter_C; kc++, rest_c++)
						*out_c_ptr += q[rest_c];
				}
			}
		}
	}

}

void zq_cnn_conv_no_padding_32f_kernel5x5_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H, // must be 5
	int filter_W, // must be 5
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,	
	int stride_W,	
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int thread_count
)
{
	
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w;
	int rest_c;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
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
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		int out_c, kc;
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
			out_c < out_C;
			out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
		{
			//the full channels
			sum = zq_mm_setzero_ps();
			for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
				kc < filter_C - zq_mm_align_size;
				kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
			{
				cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

				cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

				cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
				sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			}

			zq_mm_store_ps(q, sum);
			*out_c_ptr = zq_final_sum_q;

			//the rest channels
			sum = zq_mm_setzero_ps();
			cur_in_row_ptr = cur_in_c_ptr;	cur_filter_row_ptr = cur_filter_c_ptr;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			cur_in_row_ptr += dilate_H_mul_in_widthStep; cur_filter_row_ptr += filter_widthStep;

			cur_in_pix_ptr = cur_in_row_ptr;	cur_filter_pix_ptr = cur_filter_row_ptr;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
			cur_in_pix_ptr += dilate_W_mul_in_pixStep;	cur_filter_pix_ptr += filter_pixelStep;
			sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);

			zq_mm_store_ps(q, sum);
			for (rest_c = 0; kc < filter_C; kc++, rest_c++)
				*out_c_ptr += q[rest_c];
		}
	}
	free(in_offsets);
	free(out_offsets);

}

void zq_cnn_conv_no_padding_32f_general(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H,
	int filter_W,
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type sum;
	__declspec(align(32)) float q[8];
	
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
	const float* cur_filter_slice_ptr;
	const float* cur_filter_row_ptr;
	const float* cur_filter_pix_ptr;
	const float* cur_filter_c_ptr;

	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w, out_c, kh, kw, kc;
	int rest_c;

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
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
				{
					//the full channels
					sum = zq_mm_setzero_ps();
					for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
						kc < filter_C - zq_mm_align_size;
						kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
					{
						for (kh = 0, cur_in_row_ptr = cur_in_c_ptr, cur_filter_row_ptr = cur_filter_c_ptr;
							kh < filter_H;
							kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
						{
							for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
								kw < filter_W;
								kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_pixelStep)
							{
								sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
							}
						}
					}
					zq_mm_store_ps(q, sum);
					*out_c_ptr = zq_final_sum_q;

					//the rest channels
					sum = zq_mm_setzero_ps();
					for (kh = 0, cur_in_row_ptr = cur_in_c_ptr, cur_filter_row_ptr = cur_filter_c_ptr;
						kh < filter_H;
						kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
					{
						for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
							kw < filter_W;
							kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_pixelStep)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
						}
					}
					zq_mm_store_ps(q, sum);
					for (rest_c = 0; kc < filter_C; kc++, rest_c++)
						*out_c_ptr += q[rest_c];
				}
			}
		}
	}

}

void zq_cnn_conv_no_padding_32f_general_omp(
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* filters_data,
	int filter_N,
	int filter_H,
	int filter_W,
	int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	int stride_H,
	int stride_W,
	int dilation_H,
	int dilation_W,
	float* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int thread_count
)
{
	int stride_H_mul_in_WidthStep = stride_H*in_widthStep;
	int stride_W_mul_in_pixStep = stride_W*in_pixelStep;
	int dilate_H_mul_in_widthStep = dilation_H*in_widthStep;
	int dilate_W_mul_in_pixStep = dilation_W*in_pixelStep;
	int out_n, out_h, out_w;
	int rest_c;

	int out_HW = out_H*out_W;
	int out_NHW = out_N*out_HW;
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
		zq_mm_type sum;
		__declspec(align(32)) float q[8];
		int out_c, kh, kw, kc;
		float* out_c_ptr;
		const float* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
		const float* cur_filter_slice_ptr, *cur_filter_row_ptr, *cur_filter_pix_ptr, *cur_filter_c_ptr;
		const float* in_pix_ptr = in_tensor4D_data + in_offsets[idx];
		float* out_pix_ptr = out_tensor4D_data + out_offsets[idx];
		for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
			out_c < out_C;
			out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
		{
			//the full channels
			sum = zq_mm_setzero_ps();
			for (kc = 0, cur_in_c_ptr = in_pix_ptr, cur_filter_c_ptr = cur_filter_slice_ptr;
				kc < filter_C - zq_mm_align_size;
				kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, cur_filter_c_ptr += zq_mm_align_size)
			{
				for (kh = 0, cur_in_row_ptr = cur_in_c_ptr, cur_filter_row_ptr = cur_filter_c_ptr;
					kh < filter_H;
					kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
				{
					for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
						kw < filter_W;
						kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_pixelStep)
					{
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
					}
				}
			}
			zq_mm_store_ps(q, sum);
			*out_c_ptr = zq_final_sum_q;

			//the rest channels
			sum = zq_mm_setzero_ps();
			for (kh = 0, cur_in_row_ptr = cur_in_c_ptr, cur_filter_row_ptr = cur_filter_c_ptr;
				kh < filter_H;
				kh++, cur_in_row_ptr += dilate_H_mul_in_widthStep, cur_filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, cur_filter_pix_ptr = cur_filter_row_ptr;
					kw < filter_W;
					kw++, cur_in_pix_ptr += dilate_W_mul_in_pixStep, cur_filter_pix_ptr += filter_pixelStep)
				{
					sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_pix_ptr), sum);
				}
			}
			zq_mm_store_ps(q, sum);
			for (rest_c = 0; kc < filter_C; kc++, rest_c++)
				*out_c_ptr += q[rest_c];
		}
	}
	free(in_offsets);
	free(out_offsets);
}
