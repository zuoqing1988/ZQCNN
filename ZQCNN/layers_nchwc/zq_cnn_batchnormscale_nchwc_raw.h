/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnormscale_mean_var_scale_bias_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type* slope_data,
	const zq_base_type* bias_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	int ceil_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	a = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	for (c = 0; c < ceil_C; c++)
	{
		b[c] = slope_data[c] / sqrt(__max(var_data[c] + eps, FLOAT_EPS_FOR_DIV));
		a[c] = bias_data[c] - mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_b_a_nchwc(in_data, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



/*
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_mean_var_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	int ceil_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	a = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	for (c = 0; c < ceil_C; c++)
	{
		b[c] = 1.0f / sqrt(__max(var_data[c] + eps, FLOAT_EPS_FOR_DIV));
		a[c] = -mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_b_a_nchwc(in_data, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



void zq_cnn_scale_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* scale_data,
	const zq_base_type* bias_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type scale_vec, bias_vec;


	if (bias_data != NULL)
	{
		for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						bias_vec = zq_mm_load_ps(bias_data + c);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(pix_ptr), scale_vec, bias_vec));
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						zq_mm_store_ps(pix_ptr, zq_mm_mul_ps(zq_mm_load_ps(pix_ptr), scale_vec));
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
void zq_cnn_batchnorm_b_a_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* b_data,
	const zq_base_type* a_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type a_vec, b_vec;
	
	for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
	{
		for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
		{
			a_vec = zq_mm_load_ps(a_data + c);
			b_vec = zq_mm_load_ps(b_data + c);
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(pix_ptr), b_vec, a_vec));
				}
			}
		}
	}
}

