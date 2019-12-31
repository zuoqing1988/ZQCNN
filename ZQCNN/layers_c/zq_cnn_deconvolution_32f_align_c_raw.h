
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

void zq_cnn_deconv_with_padding_32f_k2s2(
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
	int pad_top, //pad_top must <= 1
	int pad_bottom,
	int pad_left, //pad_left must <= 1
	int pad_right
)
{
	register zq_mm_type sum,sum00,sum01,sum10,sum11;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];

	const zq_base_type* in_slice_ptr;
	zq_base_type* out_slice_ptr;
	zq_base_type* out_row_ptr;
	zq_base_type* out_pix_ptr;
	zq_base_type* out_c_ptr00, *out_c_ptr01, *out_c_ptr10, *out_c_ptr11;

	const zq_base_type* cur_in_pix_ptr;
	const zq_base_type* cur_in_c_ptr;
	const zq_base_type* cur_filter_slice_ptr00, *cur_filter_slice_ptr01, *cur_filter_slice_ptr10, *cur_filter_slice_ptr11;
	const zq_base_type* cur_filter_pix_ptr00, *cur_filter_pix_ptr01, *cur_filter_pix_ptr10, *cur_filter_pix_ptr11;
	const zq_base_type* cur_filter_c_ptr00, *cur_filter_c_ptr01, *cur_filter_c_ptr10, *cur_filter_c_ptr11;

	int out_n, out_h, out_w, out_c, kh, kw, kc;
	int rest_c;
	int need_in_h_idx, need_in_w_idx, real_in_h_idx, real_in_w_idx;
	int begin_kh, begin_kw;
	int out_pixelStep2 = out_pixelStep << 1;
	int out_widthStep2 = out_widthStep << 1;

	if (filter_C % zq_mm_align_size8 == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			need_in_h_idx = -pad_top;
			kh = need_in_h_idx & 1;
			real_in_h_idx = (need_in_h_idx + kh) >> 1;
			for (out_h = 0, out_row_ptr = out_slice_ptr;
				out_h < out_H - 1;
				out_h += 2, out_row_ptr += out_widthStep2, real_in_h_idx++)
			{
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				cur_filter_pix_ptr10 = filters_data + (1 - kh)*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr11 = filters_data + (1 - kh)*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					cur_filter_slice_ptr11 = cur_filter_pix_ptr11;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr01 = out_pix_ptr + out_pixelStep,
						out_c_ptr10 = out_pix_ptr + out_widthStep, out_c_ptr11 = out_pix_ptr + out_widthStep + out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++, out_c_ptr10++, out_c_ptr11++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep,
						cur_filter_slice_ptr10 += filter_sliceStep, cur_filter_slice_ptr11 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum01 = zq_mm_setzero_ps();
						sum10 = zq_mm_setzero_ps();
						sum11 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr, 
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr01 = cur_filter_slice_ptr01,
							cur_filter_c_ptr10 = cur_filter_slice_ptr10,
							cur_filter_c_ptr11 = cur_filter_slice_ptr11;
							kc < filter_C;
							kc += zq_mm_align_size8, cur_in_c_ptr += zq_mm_align_size8, 
							cur_filter_c_ptr00 += zq_mm_align_size8,
							cur_filter_c_ptr01 += zq_mm_align_size8,
							cur_filter_c_ptr10 += zq_mm_align_size8,
							cur_filter_c_ptr11 += zq_mm_align_size8)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr01), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr10), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr11), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size2), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size2), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size2), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size3), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size3), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size3), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size4), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size4), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size4), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size4), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size5), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size5), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size5), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size5), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size6), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size6), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size6), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size6), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size7), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size7), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size7), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size7), sum11);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum01);
						*out_c_ptr01 = zq_final_sum_q;
						zq_mm_store_ps(q, sum10);
						*out_c_ptr10 = zq_final_sum_q;
						zq_mm_store_ps(q, sum11);
						*out_c_ptr11 = zq_final_sum_q;
					}
				}
				for (;
					out_w < out_W;
					out_w ++, out_pix_ptr += out_pixelStep,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr,
						out_c_ptr10 = out_pix_ptr + out_widthStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr10++,
						cur_filter_slice_ptr00 += filter_sliceStep,
						cur_filter_slice_ptr10 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum10 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr10 = cur_filter_slice_ptr10;
							kc < filter_C;
							kc += zq_mm_align_size8, cur_in_c_ptr += zq_mm_align_size8,
							cur_filter_c_ptr00 += zq_mm_align_size8,
							cur_filter_c_ptr10 += zq_mm_align_size8)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr10), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size2), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size3), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size4), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size4), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size5), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size5), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size6), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size6), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size7), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size7), sum10);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum10);
						*out_c_ptr10 = zq_final_sum_q;
					}
				}
			}

			for (;
				out_h < out_H;
				out_h ++)
			{
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr01 = out_pix_ptr + out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum01 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr01 = cur_filter_slice_ptr01;
							kc < filter_C;
							kc += zq_mm_align_size8, cur_in_c_ptr += zq_mm_align_size8,
							cur_filter_c_ptr00 += zq_mm_align_size8,
							cur_filter_c_ptr01 += zq_mm_align_size8)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr01), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size2), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size3), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size4), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size4), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size5), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size5), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size6), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size6), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size7), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size7), sum01);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum01);
						*out_c_ptr01 = zq_final_sum_q;
					}
				}
				for (;
					out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr00++,
						cur_filter_slice_ptr00 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00;
							kc < filter_C;
							kc += zq_mm_align_size8, cur_in_c_ptr += zq_mm_align_size8,
							cur_filter_c_ptr00 += zq_mm_align_size8)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size4), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size4), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size5), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size5), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size6), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size6), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size7), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size7), sum00);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
					}
				}
			}
		}
	}
	else if (filter_C %zq_mm_align_size4 == 0)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			need_in_h_idx = -pad_top;
			kh = need_in_h_idx & 1;
			real_in_h_idx = (need_in_h_idx + kh) >> 1;
			for (out_h = 0, out_row_ptr = out_slice_ptr;
				out_h < out_H - 1;
				out_h += 2, out_row_ptr += out_widthStep2, real_in_h_idx++)
			{
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				cur_filter_pix_ptr10 = filters_data + (1 - kh)*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr11 = filters_data + (1 - kh)*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					cur_filter_slice_ptr11 = cur_filter_pix_ptr11;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr01 = out_pix_ptr + out_pixelStep,
						out_c_ptr10 = out_pix_ptr + out_widthStep, out_c_ptr11 = out_pix_ptr + out_widthStep + out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++, out_c_ptr10++, out_c_ptr11++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep,
						cur_filter_slice_ptr10 += filter_sliceStep, cur_filter_slice_ptr11 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum01 = zq_mm_setzero_ps();
						sum10 = zq_mm_setzero_ps();
						sum11 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr01 = cur_filter_slice_ptr01,
							cur_filter_c_ptr10 = cur_filter_slice_ptr10,
							cur_filter_c_ptr11 = cur_filter_slice_ptr11;
							kc < filter_C;
							kc += zq_mm_align_size4, cur_in_c_ptr += zq_mm_align_size4,
							cur_filter_c_ptr00 += zq_mm_align_size4,
							cur_filter_c_ptr01 += zq_mm_align_size4,
							cur_filter_c_ptr10 += zq_mm_align_size4,
							cur_filter_c_ptr11 += zq_mm_align_size4)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr01), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr10), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr11), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size2), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size2), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size2), sum11);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size3), sum01);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size3), sum10);
							sum11 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr11 + zq_mm_align_size3), sum11);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum01);
						*out_c_ptr01 = zq_final_sum_q;
						zq_mm_store_ps(q, sum10);
						*out_c_ptr10 = zq_final_sum_q;
						zq_mm_store_ps(q, sum11);
						*out_c_ptr11 = zq_final_sum_q;
					}
				}
				for (;
					out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr,
						out_c_ptr10 = out_pix_ptr + out_widthStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr10++,
						cur_filter_slice_ptr00 += filter_sliceStep,
						cur_filter_slice_ptr10 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum10 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr10 = cur_filter_slice_ptr10;
							kc < filter_C;
							kc += zq_mm_align_size4, cur_in_c_ptr += zq_mm_align_size4,
							cur_filter_c_ptr00 += zq_mm_align_size4,
							cur_filter_c_ptr10 += zq_mm_align_size4)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr10), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size2), sum10);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum10 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr10 + zq_mm_align_size3), sum10);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum10);
						*out_c_ptr10 = zq_final_sum_q;
					}
				}
			}

			for (;
				out_h < out_H;
				out_h++)
			{
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr01 = out_pix_ptr + out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						sum01 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00,
							cur_filter_c_ptr01 = cur_filter_slice_ptr01;
							kc < filter_C;
							kc += zq_mm_align_size4, cur_in_c_ptr += zq_mm_align_size4,
							cur_filter_c_ptr00 += zq_mm_align_size4,
							cur_filter_c_ptr01 += zq_mm_align_size4)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr01), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size2), sum01);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
							sum01 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr01 + zq_mm_align_size3), sum01);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
						zq_mm_store_ps(q, sum01);
						*out_c_ptr01 = zq_final_sum_q;
					}
				}
				for (;
					out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr;
						out_c < out_C;
						out_c++, out_c_ptr00++,
						cur_filter_slice_ptr00 += filter_sliceStep)
					{
						sum00 = zq_mm_setzero_ps();
						for (kc = 0, cur_in_c_ptr = cur_in_pix_ptr,
							cur_filter_c_ptr00 = cur_filter_slice_ptr00;
							kc < filter_C;
							kc += zq_mm_align_size4, cur_in_c_ptr += zq_mm_align_size4,
							cur_filter_c_ptr00 += zq_mm_align_size4)
						{
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr), zq_mm_load_ps(cur_filter_c_ptr00), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size2), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size2), sum00);
							sum00 = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_c_ptr + zq_mm_align_size3), zq_mm_load_ps(cur_filter_c_ptr00 + zq_mm_align_size3), sum00);
						}
						zq_mm_store_ps(q, sum00);
						*out_c_ptr00 = zq_final_sum_q;
					}
				}
			}
		}
	}
	else if (filter_C == zq_mm_align_size2)
	{
		for (out_n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			out_n < out_N;
			out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			need_in_h_idx = -pad_top;
			kh = need_in_h_idx & 1;
			real_in_h_idx = (need_in_h_idx + kh) >> 1;
			for (out_h = 0, out_row_ptr = out_slice_ptr;
				out_h < out_H - 1;
				out_h+=2, out_row_ptr += out_widthStep2, real_in_h_idx++)
			{	
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				cur_filter_pix_ptr10 = filters_data + (1 - kh)*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr11 = filters_data + (1 - kh)*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2,
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					cur_filter_slice_ptr11 = cur_filter_pix_ptr11;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr01 = out_pix_ptr + out_pixelStep,
						out_c_ptr10 = out_pix_ptr + out_widthStep, out_c_ptr11 = out_pix_ptr + out_widthStep+out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++,out_c_ptr10++, out_c_ptr11++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep,
						cur_filter_slice_ptr10 += filter_sliceStep, cur_filter_slice_ptr11 += filter_sliceStep)
					{
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr00));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr00 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr00 = zq_final_sum_q;
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr01));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr01 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr01 = zq_final_sum_q;
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr10));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr10 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr10 = zq_final_sum_q;
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr11));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr11 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr11 = zq_final_sum_q;
					}
				}

				for (; out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr10 = cur_filter_pix_ptr10;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, out_c_ptr10 = out_pix_ptr + out_widthStep;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr10 += filter_sliceStep, 
						out_c_ptr00++, out_c_ptr10++)
					{
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr00));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr00 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr00 = zq_final_sum_q;
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr10));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr10 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr10 = zq_final_sum_q;
					}
				}
			}

			for (;
				out_h < out_H;
				out_h++)
			{
				need_in_w_idx = -pad_left;
				kw = need_in_w_idx & 1;
				real_in_w_idx = (need_in_w_idx + kw) >> 1;
				cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
				cur_filter_pix_ptr00 = filters_data + kh*filter_widthStep + kw*filter_pixelStep;
				cur_filter_pix_ptr01 = filters_data + kh*filter_widthStep + (1 - kw)*filter_pixelStep;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W - 1;
					out_w += 2, out_pix_ptr += out_pixelStep2, 
					cur_in_pix_ptr += in_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					cur_filter_slice_ptr01 = cur_filter_pix_ptr01;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr,out_c_ptr01 = out_pix_ptr+out_pixelStep;
						out_c < out_C;
						out_c++, out_c_ptr00++, out_c_ptr01++,
						cur_filter_slice_ptr00 += filter_sliceStep, cur_filter_slice_ptr01 += filter_sliceStep)
					{
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr00));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr00 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr00 = zq_final_sum_q;
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr01));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr01 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr01 = zq_final_sum_q;
					}
				}

				for (;out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep)
				{
					cur_filter_slice_ptr00 = cur_filter_pix_ptr00;
					
					for (out_c = 0, out_c_ptr00 = out_pix_ptr;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr00 += filter_sliceStep, out_c_ptr00++)
					{
						sum = zq_mm_mul_ps(zq_mm_load_ps(cur_in_pix_ptr), zq_mm_load_ps(cur_filter_slice_ptr00));
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + zq_mm_align_size), zq_mm_load_ps(cur_filter_slice_ptr00 + zq_mm_align_size), sum);
						zq_mm_store_ps(q, sum);
						*out_c_ptr00 = zq_final_sum_q;
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
			need_in_h_idx = -pad_top;
			for (out_h = 0, out_row_ptr = out_slice_ptr;
				out_h < out_H;
				out_h++, out_row_ptr += out_widthStep, need_in_h_idx++)
			{
				need_in_w_idx = -pad_left;
				for (out_w = 0, out_pix_ptr = out_row_ptr;
					out_w < out_W;
					out_w++, out_pix_ptr += out_pixelStep, need_in_w_idx++)
				{
					begin_kh = need_in_h_idx & 1;
					begin_kw = need_in_w_idx & 1;
					kh = begin_kh;
					kw = begin_kw;
					real_in_h_idx = (need_in_h_idx + kh) >> 1;
					real_in_w_idx = (need_in_w_idx + kw) >> 1;
					cur_in_pix_ptr = in_slice_ptr + real_in_h_idx*in_widthStep + real_in_w_idx*in_pixelStep;
					for (out_c = 0, out_c_ptr00 = out_pix_ptr, cur_filter_slice_ptr00 = filters_data;
						out_c < out_C;
						out_c++, cur_filter_slice_ptr00 += filter_sliceStep, out_c_ptr00++)
					{
						cur_filter_pix_ptr00 = cur_filter_slice_ptr00 + kh*filter_widthStep + kw*filter_pixelStep;
						//the full channels
						sum = zq_mm_setzero_ps();
						for (kc = 0;
							kc < filter_C - zq_mm_align_size;
							kc += zq_mm_align_size)
						{
							sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + kc), zq_mm_load_ps(cur_filter_pix_ptr00 + kc), sum);
						}
						zq_mm_store_ps(q, sum);
						*out_c_ptr00 = zq_final_sum_q;

						//the rest channels
						sum = zq_mm_setzero_ps();
						sum = zq_mm_fmadd_ps(zq_mm_load_ps(cur_in_pix_ptr + kc), zq_mm_load_ps(cur_filter_pix_ptr00 + kc), sum);
						zq_mm_store_ps(q, sum);
						for (rest_c = 0; kc < filter_C; kc++, rest_c++)
							*out_c_ptr00 += q[rest_c];
					}
				}
			}
		}
	}
}
