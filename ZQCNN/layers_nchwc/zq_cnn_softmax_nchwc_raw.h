
void zq_cnn_softmax_nchwc_C(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep
)
{
	// value = exp( value - global max value )
	// sum all value
	// value = value / sum
	register zq_mm_type val, tmp;
	zq_base_type max_val, tmp_val, sum_val;
	zq_base_type result[zq_mm_align_size << 2];
	zq_base_type* q = (zq_base_type*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);
	int n, h, w, c, i;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
	{
		for (h = 0, row_ptr = im_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
			{
				//compute max_val
				val = zq_mm_set1_ps(-FLT_MAX);
				for (c = 0, slice_ptr = pix_ptr; c < in_C - zq_mm_align_size; c += zq_mm_align_size, slice_ptr += in_sliceStep)
					val = zq_mm_max_ps(val, zq_mm_load_ps(slice_ptr));
				zq_mm_store_ps(q, val);
				max_val = zq_final_max_q;
				for (; c < in_C; c++, slice_ptr++)
					max_val = __max(max_val, *slice_ptr);

				//compute sum

				sum_val = 0;
				for (c = 0, slice_ptr = pix_ptr; c < in_C-zq_mm_align_size; c+=zq_mm_align_size, slice_ptr+=in_sliceStep)
				{
					for (i = 0; i < zq_mm_align_size; i++)
					{
						tmp_val = exp(slice_ptr[i] - max_val);
						sum_val += tmp_val;
						slice_ptr[i] = tmp_val;
					}
				}
				for (; c < in_C; c++, slice_ptr++)
				{
					tmp_val = exp(*slice_ptr - max_val);
					sum_val += tmp_val;
					*slice_ptr = tmp_val;
				}

				//divide
				sum_val = 1.0f / sum_val;
				tmp = zq_mm_set1_ps(sum_val);
				for (c = 0, slice_ptr = pix_ptr; c < in_C - zq_mm_align_size; c += zq_mm_align_size, slice_ptr += in_sliceStep)
				{
					zq_mm_store_ps(slice_ptr, zq_mm_mul_ps(zq_mm_load_ps(slice_ptr), tmp));
				}
				for (; c < in_C; c++, slice_ptr++)
					*slice_ptr *= sum_val;

			}
		}
	}
}
