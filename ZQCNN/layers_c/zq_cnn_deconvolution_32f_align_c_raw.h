
void zq_cnn_deconv_with_padding_32f_general(
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
	zq_base_type* out_tensor4D_data,
	int out_N,	// must be in_N
	int out_H,	// must be (in_H - filter_H)/stride_H + 1
	int out_W,	// must be (in_W - filter_W)/stride_W + 1
	int out_C,	// must be filter_N
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep,
	int pad_top,
	int pad_bottom,
	int pad_left,
	int pad_right
)
{
	register zq_mm_type sum;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

	const zq_base_type* in_slice_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr;

	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_filter_slice_ptr;
	const zq_base_type* cur_filter_pix_ptr;

	int out_n, out_h, out_w, out_c, kh, kw, kc;
	int rest_c;
	int need_in_h_idx, need_in_w_idx, real_in_h_idx, real_in_w_idx;
	int begin_kh, end_kh, begin_kw, end_kw;

	for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		out_n < out_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (out_h = 0, out_row_ptr = out_slice_ptr;
			out_h < out_H;
			out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr;
				out_w < out_W;
				out_w++, out_pix_ptr += out_pixelStep)
			{
				need_in_h_idx = out_h - pad_top;
				need_in_w_idx = out_w - pad_left;
				if (need_in_h_idx >= 0)
					begin_kh = (stride_H - need_in_h_idx%stride_H) % stride_H;
				else
					begin_kh = 0 - need_in_h_idx;
				if (need_in_w_idx >= 0)
					begin_kw = (stride_W - need_in_w_idx%stride_W) % stride_W;
				else
					begin_kw = 0 - need_in_w_idx;
				end_kh = __min(filter_H, in_H*stride_H - need_in_h_idx + 1);
				end_kw = __min(filter_W, in_W*stride_W - need_in_w_idx + 1);
				for (out_c = 0, out_c_ptr = out_pix_ptr, cur_filter_slice_ptr = filters_data;
					out_c < out_C;
					out_c++, cur_filter_slice_ptr += filter_sliceStep, out_c_ptr++)
				{
					//the full channels
					sum = zq_mm_setzero_ps();
					for (kc = 0;
						kc < filter_C - zq_mm_align_size;
						kc += zq_mm_align_size)
					{
						for (kh = begin_kh;
							kh < end_kh;
							kh += stride_H)
						{
							for (kw = begin_kw;
								kw < end_kw;
								kw += stride_W)
							{
								real_in_h_idx = (need_in_h_idx + kh) / stride_H;
								real_in_w_idx = (need_in_w_idx + kw) / stride_W;
								cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
								cur_filter_pix_ptr = cur_filter_slice_ptr + kh*filter_widthStep + kw*filter_pixelStep;
								sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + kc), zq_mm_load_ps(cur_filter_pix_ptr + kc), sum);
							}
						}
					}
					zq_mm_store_ps(q, sum);
					*out_c_ptr = zq_final_sum_q;

					//the rest channels
					sum = zq_mm_setzero_ps();
					for (kh = begin_kh;
						kh < end_kh;
						kh += stride_H)
					{
						for (kw = begin_kw;
							kw < end_kw;
							kw += stride_W)
						{
							real_in_h_idx = (need_in_h_idx + kh) / stride_H;
							real_in_w_idx = (need_in_w_idx + kw) / stride_W;
							cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
							cur_filter_pix_ptr = cur_filter_slice_ptr + kh*filter_widthStep + kw*filter_pixelStep;
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr+kc), zq_mm_load_ps(cur_filter_pix_ptr+kc), sum);
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
