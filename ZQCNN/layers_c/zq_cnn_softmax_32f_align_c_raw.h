
void zq_cnn_softmax_32f_align_C(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep
)
{
	// value = exp( value - global max value )
	// sum all value
	// value = value / sum
	register zq_mm_type val,tmp, sum;
	zq_base_type max_val, tmp_val,sum_val;
	zq_base_type result[zq_mm_align_size << 2];
	zq_base_type* q = (zq_base_type*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr+= in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				//compute max_val
				val = zq_mm_set1_ps(-FLT_MAX);
				for (c = 0, c_ptr = pix_ptr; c < in_C - zq_mm_align_size; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
					val = zq_mm_max_ps(val, zq_mm_load_ps(c_ptr));
				zq_mm_store_ps(q, val);
				max_val = zq_final_max_q;
				for (; c < in_C; c++, c_ptr++)
					max_val = __max(max_val, *c_ptr);
				
				//compute sum
#if 0
				sum = zq_mm_set1_ps(0);
				for (c = 0, c_ptr = pix_ptr; c < in_C - zq_mm_align_size; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
				{
					tmp = zq_mm_exp_ps(zq_mm_sub_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(max_val)));
					sum = zq_mm_add_ps(sum, tmp);
					zq_mm_store_ps(c_ptr, tmp);
				}
				sum_val = zq_final_sum_q;
				for (; c < in_C; c++, c_ptr++)
				{
					tmp_val = exp(*c_ptr - max_val);
					sum_val += tmp_val;
					*c_ptr = tmp_val;
				}
#else
				sum_val = 0;
				for (c = 0, c_ptr = pix_ptr; c < in_C; c++, c_ptr++)
				{
					tmp_val = exp(*c_ptr - max_val);
					sum_val += tmp_val;
					*c_ptr = tmp_val;
				}
#endif
				
				//divide
				sum_val = 1.0f / sum_val;
				tmp = zq_mm_set1_ps(sum_val);
				for (c = 0, c_ptr = pix_ptr; c < in_C - zq_mm_align_size; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), tmp));
				}
				for (; c < in_C; c++, c_ptr++)
					*c_ptr *= sum_val;
				
			}
		}
	}
}
