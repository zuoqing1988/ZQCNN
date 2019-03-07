/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align(
	float* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const float* mean_data,
	const float* var_data,
	const float* slope_data,
	const float* bias_data,
	const float eps
)
{
	float* a, *b;
	int c;
	a = (float*)_aligned_malloc(in_C, (zq_mm_align_size << 2));
	b = (float*)_aligned_malloc(in_C, (zq_mm_align_size << 2));
	for (c = 0; c < in_C; c++)
	{
		b[c] = slope_data[c] / sqrt(__max(var_data[c]+eps,FLOAT_EPS_FOR_DIV));
		a[c] = bias_data[c] - mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const float*)b, (const float*)a);

	_aligned_free(a);
	_aligned_free(b);
}



/*
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_32f_mean_var_align(
	float* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const float* mean_data,
	const float* var_data,
	const float eps
)
{
	float* a, *b;
	int c;
	a = (float*)_aligned_malloc(in_C, (zq_mm_align_size << 2));
	b = (float*)_aligned_malloc(in_C, (zq_mm_align_size << 2));
	for (c = 0; c < in_C; c++)
	{
		b[c] = 1.0f / sqrt(__max(var_data[c]+eps, FLOAT_EPS_FOR_DIV));
		a[c] = -mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const float*)b, (const float*)a);

	_aligned_free(a);
	_aligned_free(b);
}



void zq_cnn_scale_32f_align(
	float* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const float* scale_data,
	const float* bias_data
)
{
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	register zq_mm_type scale_vec, bias_vec;


	if (bias_data != NULL)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						bias_vec = zq_mm_load_ps(bias_data + c);
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_mul_ps(zq_mm_load_ps(c_ptr), scale_vec), bias_vec));
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), scale_vec));
					}
				}
			}
		}
	}
}

/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
OR
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_32f_b_a_align(
	float* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const float* b_data,
	const float* a_data
)
{
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	register zq_mm_type a_vec, b_vec;


	for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
			{
				for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
				{
					a_vec = zq_mm_load_ps(a_data + c);
					b_vec = zq_mm_load_ps(b_data + c);
					zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), b_vec, a_vec));
				}
			}
		}
	}
}

