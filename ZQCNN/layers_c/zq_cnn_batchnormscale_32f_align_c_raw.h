/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type* slope_data,
	const zq_base_type* bias_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	a = (zq_base_type*)_aligned_malloc(in_C*sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(in_C*sizeof(zq_base_type), (zq_mm_align_size << 2));
	for (c = 0; c < in_C; c++)
	{
		b[c] = slope_data[c] / sqrt(__max(var_data[c]+eps,FLOAT_EPS_FOR_DIV));
		a[c] = bias_data[c] - mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



/*
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_32f_mean_var_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	a = (zq_base_type*)_aligned_malloc(in_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(in_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	for (c = 0; c < in_C; c++)
	{
		b[c] = 1.0f / sqrt(__max(var_data[c]+eps, FLOAT_EPS_FOR_DIV));
		a[c] = -mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



void zq_cnn_scale_32f_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* scale_data,
	const zq_base_type* bias_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
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
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* b_data,
	const zq_base_type* a_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const zq_base_type* a_ptr, *b_ptr;
	register zq_mm_type a_vec0, a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7;
	register zq_mm_type b_vec0, b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7;
	register zq_mm_type c_vec0, c_vec1, c_vec2, c_vec3, c_vec4, c_vec5, c_vec6, c_vec7;

	if (in_C % (zq_mm_align_size8) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr;
						c < in_C;
						c += zq_mm_align_size8, c_ptr += zq_mm_align_size8, a_ptr += zq_mm_align_size8, b_ptr += zq_mm_align_size8)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						a_vec2 = zq_mm_load_ps(a_ptr + zq_mm_align_size2);
						a_vec3 = zq_mm_load_ps(a_ptr + zq_mm_align_size3);
						a_vec4 = zq_mm_load_ps(a_ptr + zq_mm_align_size4);
						a_vec5 = zq_mm_load_ps(a_ptr + zq_mm_align_size5);
						a_vec6 = zq_mm_load_ps(a_ptr + zq_mm_align_size6);
						a_vec7 = zq_mm_load_ps(a_ptr + zq_mm_align_size7);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						b_vec2 = zq_mm_load_ps(b_ptr + zq_mm_align_size2);
						b_vec3 = zq_mm_load_ps(b_ptr + zq_mm_align_size3);
						b_vec4 = zq_mm_load_ps(b_ptr + zq_mm_align_size4);
						b_vec5 = zq_mm_load_ps(b_ptr + zq_mm_align_size5);
						b_vec6 = zq_mm_load_ps(b_ptr + zq_mm_align_size6);
						b_vec7 = zq_mm_load_ps(b_ptr + zq_mm_align_size7);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec2 = zq_mm_load_ps(c_ptr + zq_mm_align_size2);
						c_vec3 = zq_mm_load_ps(c_ptr + zq_mm_align_size3);
						c_vec4 = zq_mm_load_ps(c_ptr + zq_mm_align_size4);
						c_vec5 = zq_mm_load_ps(c_ptr + zq_mm_align_size5);
						c_vec6 = zq_mm_load_ps(c_ptr + zq_mm_align_size6);
						c_vec7 = zq_mm_load_ps(c_ptr + zq_mm_align_size7);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						c_vec2 = zq_mm_fmadd_ps(c_vec2, b_vec2, a_vec2);
						c_vec3 = zq_mm_fmadd_ps(c_vec3, b_vec3, a_vec3);
						c_vec4 = zq_mm_fmadd_ps(c_vec4, b_vec4, a_vec4);
						c_vec5 = zq_mm_fmadd_ps(c_vec5, b_vec5, a_vec5);
						c_vec6 = zq_mm_fmadd_ps(c_vec6, b_vec6, a_vec6);
						c_vec7 = zq_mm_fmadd_ps(c_vec7, b_vec7, a_vec7);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
						zq_mm_store_ps(c_ptr + zq_mm_align_size2, c_vec2);
						zq_mm_store_ps(c_ptr + zq_mm_align_size3, c_vec3);
						zq_mm_store_ps(c_ptr + zq_mm_align_size4, c_vec4);
						zq_mm_store_ps(c_ptr + zq_mm_align_size5, c_vec5);
						zq_mm_store_ps(c_ptr + zq_mm_align_size6, c_vec6);
						zq_mm_store_ps(c_ptr + zq_mm_align_size7, c_vec7);
					}
				}
			}
		}
	}
	else if (in_C % (zq_mm_align_size4) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr; 
						c < in_C; 
						c += zq_mm_align_size4, c_ptr += zq_mm_align_size4, a_ptr += zq_mm_align_size4, b_ptr += zq_mm_align_size4)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						a_vec2 = zq_mm_load_ps(a_ptr + zq_mm_align_size2);
						a_vec3 = zq_mm_load_ps(a_ptr + zq_mm_align_size3);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						b_vec2 = zq_mm_load_ps(b_ptr + zq_mm_align_size2);
						b_vec3 = zq_mm_load_ps(b_ptr + zq_mm_align_size3);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec2 = zq_mm_load_ps(c_ptr + zq_mm_align_size2);
						c_vec3 = zq_mm_load_ps(c_ptr + zq_mm_align_size3);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						c_vec2 = zq_mm_fmadd_ps(c_vec2, b_vec2, a_vec2);
						c_vec3 = zq_mm_fmadd_ps(c_vec3, b_vec3, a_vec3);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
						zq_mm_store_ps(c_ptr + zq_mm_align_size2, c_vec2);
						zq_mm_store_ps(c_ptr + zq_mm_align_size3, c_vec3);
					}
				}
			}
		}
	}
	else if (in_C % (zq_mm_align_size2) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr;
						c < in_C;
						c += zq_mm_align_size2, c_ptr += zq_mm_align_size2, a_ptr += zq_mm_align_size2, b_ptr += zq_mm_align_size2)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
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
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr;
						c < in_C;
						c += zq_mm_align_size, c_ptr += zq_mm_align_size, a_ptr += zq_mm_align_size, b_ptr += zq_mm_align_size)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						b_vec0 = zq_mm_load_ps(b_ptr);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						zq_mm_store_ps(c_ptr, c_vec0);
					}
				}
			}
		}
	}
}

