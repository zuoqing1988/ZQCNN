/* it is safe to use out_tensor4D_data = in_tensor4D_data */
void zq_cnn_lrn_across_channels_32f_align(
	int local_size,
	float alpha,
	float beta,
	float k,								// k must be odd number
	const float* in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	float* out_tensor4D_data,
	int out_pixStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
	int n, h, w, c, pad_size, len;
	float* square_buf, *accumulate_buf, *local_sum_buf, *square_ptr,*acc_ptr, *local_ptr;
	float alpha_div_local_size = alpha / (float)local_size;
	register zq_mm_type data_v, sum_v,pow_v;
	register zq_mm_type minus_beta_v = zq_mm_set1_ps(-beta);
	register zq_mm_type alpha_div_local_size_v = zq_mm_set1_ps(alpha_div_local_size);
	register zq_mm_type k_v = zq_mm_set1_ps(k);

	pad_size = local_size / 2 + zq_mm_align_size - 1;
	pad_size = pad_size - pad_size%zq_mm_align_size;
	len = C + (pad_size << 1);
	square_buf = (float*)_aligned_malloc(sizeof(float)*len,zq_mm_align_size*sizeof(float));
	accumulate_buf = (float*)_aligned_malloc(sizeof(float)*(len + 1),zq_mm_align_size*sizeof(float));
	local_sum_buf = (float*)_aligned_malloc(sizeof(float)*C, zq_mm_align_size * sizeof(float));

	accumulate_buf[0] = 0;
	for (c = 0; c < pad_size; c++)
	{
		square_buf[c] = 0;
		square_buf[len - 1 - c] = 0;
		accumulate_buf[c + 1] = 0;
	}


	for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		n < N;
		n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			h < H;
			h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
		{
			for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				w < W;
				w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixStep)
			{
				//compute x^2
				for (c = 0, in_c_ptr = in_pix_ptr,square_ptr = square_buf+pad_size; c < C; 
					c+=zq_mm_align_size, in_c_ptr+=zq_mm_align_size,square_ptr+=zq_mm_align_size)
				{
					data_v = zq_mm_load_ps(in_c_ptr);
					zq_mm_store_ps(square_ptr, zq_mm_mul_ps(data_v, data_v));
				}
				//compute accumulate
				for (c = pad_size; c < len; c++)
					accumulate_buf[c + 1] = accumulate_buf[c] + square_buf[c];

				//compute local sum
				for (c = 0, acc_ptr = accumulate_buf+pad_size-(local_size/2); c < C; c++,acc_ptr++)
				{
					local_sum_buf[c] = acc_ptr[local_size] - acc_ptr[0];
				}
				for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr, local_ptr = local_sum_buf; 
					c < C; 
					c+=zq_mm_align_size, in_c_ptr+=zq_mm_align_size, out_c_ptr+=zq_mm_align_size,local_ptr+=zq_mm_align_size)
				{
					sum_v = zq_mm_load_ps(local_ptr);
					data_v = zq_mm_load_ps(in_c_ptr);
					sum_v = zq_mm_fmadd_ps(sum_v, alpha_div_local_size_v, k_v);
					pow_v = zq_mm_exp_ps(zq_mm_mul_ps(minus_beta_v, zq_mm_log_ps(sum_v)));
					data_v = zq_mm_mul_ps(data_v, pow_v);
					zq_mm_store_ps(out_c_ptr, data_v);
				}
			}
		}
	}

	_aligned_free(square_buf);
	_aligned_free(accumulate_buf);
	_aligned_free(local_sum_buf);
}