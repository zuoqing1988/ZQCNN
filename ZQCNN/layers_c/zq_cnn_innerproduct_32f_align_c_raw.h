void zq_cnn_innerproduct_32f_align(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* filters_data,
	int filter_N,
	//int filter_H, // must be in_H
	//int filter_W, // must be in_W
	//int filter_C, // must be in_C
	int filter_pixelStep,
	int filter_widthStep,
	int filter_sliceStep,
	zq_base_type* out_tensor4D_data,
	//int out_N,	// must be in_N
	//int out_H,	// must be 1
	//int out_W,	// must be 1
	//int out_C,	// must be filter_N
	//int out_pixelStep,
	//int out_widthStep,
	int out_sliceStep
)
{
	register zq_mm_type sum_vec;
	zq_base_type result[zq_mm_align_size << 2];
	zq_base_type* q = (zq_base_type*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);
	int out_n, out_c, kh, kw, kc;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* filter_slice_ptr, *filter_row_ptr, *filter_pix_ptr, *filter_c_ptr;
	zq_base_type* out_slice_ptr, *out_c_ptr;
	const zq_base_type* cur_in_row_ptr, *cur_in_pix_ptr, *cur_in_c_ptr;
	zq_base_type sum;
	
	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < in_N; out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_c = 0, out_c_ptr = out_slice_ptr, filter_slice_ptr = filters_data;
			out_c < filter_N;
			out_c++, out_c_ptr++, filter_slice_ptr += filter_sliceStep)
		{
			sum_vec = zq_mm_setzero_ps();
			sum = 0;
			for (kh = 0, cur_in_row_ptr = in_slice_ptr, filter_row_ptr = filter_slice_ptr;
				kh < in_H;
				kh++, cur_in_row_ptr += in_widthStep, filter_row_ptr += filter_widthStep)
			{
				for (kw = 0, cur_in_pix_ptr = cur_in_row_ptr, filter_pix_ptr = filter_row_ptr;
					kw < in_W;
					kw++, cur_in_pix_ptr += in_pixelStep, filter_pix_ptr += filter_pixelStep)
				{
					for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, filter_c_ptr = filter_pix_ptr;
						kc < in_C - zq_mm_align_size;
						kc += zq_mm_align_size, cur_in_c_ptr += zq_mm_align_size, filter_c_ptr += zq_mm_align_size)
					{
						sum_vec = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(filter_c_ptr), sum_vec);
					}
					
					for (; kc < in_C; kc++)
						sum += (*(cur_in_c_ptr++))*(*(filter_c_ptr++));
				}
			}
			zq_mm_store_ps(q, sum_vec);
			sum += zq_final_sum_q;
			*out_c_ptr = sum;
		}
	}
}

void zq_cnn_innerproduct_32f_align_noborder(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_HWC,
	const zq_base_type* filters_data,
	int filter_N,
	zq_base_type* out_tensor4D_data,
	int out_sliceStep
)
{
	register zq_mm_type sum_vec;
	zq_base_type result[zq_mm_align_size << 2];
	zq_base_type* q = (zq_base_type*)(((long long)result + (zq_mm_align_size << 2) - 1) & zq_mm_bitor_longlong);
	int out_n, out_c, in_hwc;
	const zq_base_type* in_slice_ptr;
	const zq_base_type* filter_slice_ptr, *filter_c_ptr;
	zq_base_type* out_slice_ptr, *out_c_ptr;
	const zq_base_type* cur_in_c_ptr;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < in_N; out_n++, in_slice_ptr += in_HWC, out_slice_ptr += out_sliceStep)
	{
		for (out_c = 0, out_c_ptr = out_slice_ptr, filter_slice_ptr = filters_data;
			out_c < filter_N;
			out_c++, out_c_ptr++, filter_slice_ptr += in_HWC)
		{
			
			sum_vec = zq_mm_setzero_ps();
			for (in_hwc = 0, cur_in_c_ptr = in_slice_ptr, filter_c_ptr = filter_slice_ptr; 
				in_hwc < in_HWC; 
				in_hwc += zq_mm_align_size, cur_in_c_ptr+=zq_mm_align_size, filter_c_ptr+=zq_mm_align_size)
			{
				sum_vec = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(filter_c_ptr), sum_vec);
			}
			
			zq_mm_store_ps(q, sum_vec);
			*out_c_ptr = zq_final_sum_q;
		}
	}
}
