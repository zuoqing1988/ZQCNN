if (matrix_B_rows%zq_mm_align_size == 0)
{
#if EXPAND_CHANNEL == 0
	matrix_C_row_ptr = matrix_C;
	if (out_W % 4 == 0)
	{
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep, matrix_C_row_ptr += out_H*out_W*matrix_B_cols)
		{
			for (kc = 0, matrix_C_col_ptr = matrix_C_row_ptr, out_slice_ptr = out_im_ptr;
				kc < out_C;
				kc += zq_mm_align_size, matrix_C_col_ptr += zq_mm_align_size, out_slice_ptr += out_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + kc);
#endif
				cur_matrix_C_row_ptr = matrix_C_col_ptr;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w+=4, out_pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
						a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
						a2 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols2);
						a3 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v, b3, c3);
#endif
						zq_mm_store_ps(out_pix_ptr, a0);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, a2);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, a3);
						cur_matrix_C_row_ptr += matrix_B_cols4;
					}
				}
			}
		}
	}
	else if (out_W % 4 == 1)
	{
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep, matrix_C_row_ptr += out_H*out_W*matrix_B_cols)
		{
			for (kc = 0, matrix_C_col_ptr = matrix_C_row_ptr, out_slice_ptr = out_im_ptr;
				kc < out_C;
				kc += zq_mm_align_size, matrix_C_col_ptr += zq_mm_align_size, out_slice_ptr += out_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + kc);
#endif
				cur_matrix_C_row_ptr = matrix_C_col_ptr;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W-1; out_w += 4, out_pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
						a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
						a2 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols2);
						a3 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v, b3, c3);
#endif
						zq_mm_store_ps(out_pix_ptr, a0);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, a2);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, a3);
						cur_matrix_C_row_ptr += matrix_B_cols4;
					}
					a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
#endif
#if WITH_PRELU
					b0 = zq_mm_min_ps(zero_v, a0);
					c0 = zq_mm_max_ps(zero_v, a0);
					a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
#endif
					zq_mm_store_ps(out_pix_ptr, a0);
					cur_matrix_C_row_ptr += matrix_B_cols;
				}
			}
		}
	}
	else if (out_W % 4 == 2)
	{
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep, matrix_C_row_ptr += out_H*out_W*matrix_B_cols)
		{
			for (kc = 0, matrix_C_col_ptr = matrix_C_row_ptr, out_slice_ptr = out_im_ptr;
				kc < out_C;
				kc += zq_mm_align_size, matrix_C_col_ptr += zq_mm_align_size, out_slice_ptr += out_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + kc);
#endif
				cur_matrix_C_row_ptr = matrix_C_col_ptr;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W-2; out_w += 4, out_pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
						a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
						a2 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols2);
						a3 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v, b3, c3);
#endif
						zq_mm_store_ps(out_pix_ptr, a0);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, a2);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, a3);
						cur_matrix_C_row_ptr += matrix_B_cols4;
					}
					a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
					a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
#endif
#if WITH_PRELU
					b0 = zq_mm_min_ps(zero_v, a0);
					b1 = zq_mm_min_ps(zero_v, a1);
					c0 = zq_mm_max_ps(zero_v, a0);
					c1 = zq_mm_max_ps(zero_v, a1);
					a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
					a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
#endif
					zq_mm_store_ps(out_pix_ptr, a0);
					zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
					cur_matrix_C_row_ptr += matrix_B_cols2;
				}
			}
		}
	}
	else //if (out_W % 4 == 3)
	{
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep, matrix_C_row_ptr += out_H*out_W*matrix_B_cols)
		{
			for (kc = 0, matrix_C_col_ptr = matrix_C_row_ptr, out_slice_ptr = out_im_ptr;
				kc < out_C;
				kc += zq_mm_align_size, matrix_C_col_ptr += zq_mm_align_size, out_slice_ptr += out_sliceStep)
			{
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + kc);
#endif
				cur_matrix_C_row_ptr = matrix_C_col_ptr;
				for (out_h = 0, out_row_ptr = out_slice_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
				{
					for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W-3; out_w += 4, out_pix_ptr += zq_mm_align_size4)
					{
						a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
						a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
						a2 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols2);
						a3 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
						a1 = zq_mm_add_ps(a1, bias_v);
						a2 = zq_mm_add_ps(a2, bias_v);
						a3 = zq_mm_add_ps(a3, bias_v);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v, b3, c3);
#endif
						zq_mm_store_ps(out_pix_ptr, a0);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, a2);
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, a3);
						cur_matrix_C_row_ptr += matrix_B_cols4;
					}
					a0 = zq_mm_load_ps(cur_matrix_C_row_ptr);
					a1 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols);
					a2 = zq_mm_load_ps(cur_matrix_C_row_ptr + matrix_B_cols2);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
					a1 = zq_mm_add_ps(a1, bias_v);
					a2 = zq_mm_add_ps(a2, bias_v);
#endif
#if WITH_PRELU
					b0 = zq_mm_min_ps(zero_v, a0);
					b1 = zq_mm_min_ps(zero_v, a1);
					b2 = zq_mm_min_ps(zero_v, a2);
					c0 = zq_mm_max_ps(zero_v, a0);
					c1 = zq_mm_max_ps(zero_v, a1);
					c2 = zq_mm_max_ps(zero_v, a2);
					a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
					a1 = zq_mm_fmadd_ps(slope_v, b1, c1);
					a2 = zq_mm_fmadd_ps(slope_v, b2, c2);
#endif
					zq_mm_store_ps(out_pix_ptr, a0);
					zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, a1);
					zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, a2);
					cur_matrix_C_row_ptr += matrix_B_cols3;
				}
			}
		}
	}
#else
	if (out_C % zq_mm_align_size8 == 0)
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
				{
					out_slice_ptr0 = out_pix_ptr;
					out_slice_ptr1 = out_pix_ptr + out_sliceStep;
					out_slice_ptr2 = out_pix_ptr + out_sliceStep2;
					out_slice_ptr3 = out_pix_ptr + out_sliceStep3;
					out_slice_ptr4 = out_pix_ptr + out_sliceStep4;
					out_slice_ptr5 = out_pix_ptr + out_sliceStep5;
					out_slice_ptr6 = out_pix_ptr + out_sliceStep6;
					out_slice_ptr7 = out_pix_ptr + out_sliceStep7;
#if WITH_BIAS
					bias_ptr = bias;
#endif
#if WITH_PRELU
					slope_ptr = slope;
#endif
					matrix_C_col_ptr = matrix_C_row_ptr;
					for (kc = 0; kc < out_C; kc += zq_mm_align_size8,
						out_slice_ptr0 += out_sliceStep8,
						out_slice_ptr1 += out_sliceStep8,
						out_slice_ptr2 += out_sliceStep8,
						out_slice_ptr3 += out_sliceStep8,
						out_slice_ptr4 += out_sliceStep8,
						out_slice_ptr5 += out_sliceStep8,
						out_slice_ptr6 += out_sliceStep8,
						out_slice_ptr7 += out_sliceStep8,
						matrix_C_col_ptr += zq_mm_align_size8)
					{
#if WITH_BIAS
						bias_v0 = zq_mm_load_ps(bias_ptr);
						bias_v1 = zq_mm_load_ps(bias_ptr + zq_mm_align_size);
						bias_v2 = zq_mm_load_ps(bias_ptr + zq_mm_align_size2);
						bias_v3 = zq_mm_load_ps(bias_ptr + zq_mm_align_size3);
						bias_v4 = zq_mm_load_ps(bias_ptr + zq_mm_align_size4);
						bias_v5 = zq_mm_load_ps(bias_ptr + zq_mm_align_size5);
						bias_v6 = zq_mm_load_ps(bias_ptr + zq_mm_align_size6);
						bias_v7 = zq_mm_load_ps(bias_ptr + zq_mm_align_size7);
#endif
#if WITH_PRELU
						slope_v0 = zq_mm_load_ps(slope_ptr);
						slope_v1 = zq_mm_load_ps(slope_ptr + zq_mm_align_size);
						slope_v2 = zq_mm_load_ps(slope_ptr + zq_mm_align_size2);
						slope_v3 = zq_mm_load_ps(slope_ptr + zq_mm_align_size3);
						slope_v4 = zq_mm_load_ps(slope_ptr + zq_mm_align_size4);
						slope_v5 = zq_mm_load_ps(slope_ptr + zq_mm_align_size5);
						slope_v6 = zq_mm_load_ps(slope_ptr + zq_mm_align_size6);
						slope_v7 = zq_mm_load_ps(slope_ptr + zq_mm_align_size7);
#endif
						a0 = zq_mm_load_ps(matrix_C_col_ptr);
						a1 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size3);
						a4 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size4);
						a5 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size5);
						a6 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size6);
						a7 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size7);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v0);
						a1 = zq_mm_add_ps(a1, bias_v1);
						a2 = zq_mm_add_ps(a2, bias_v2);
						a3 = zq_mm_add_ps(a3, bias_v3);
						a4 = zq_mm_add_ps(a4, bias_v4);
						a5 = zq_mm_add_ps(a5, bias_v5);
						a6 = zq_mm_add_ps(a6, bias_v6);
						a7 = zq_mm_add_ps(a7, bias_v7);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						b4 = zq_mm_min_ps(zero_v, a4);
						b5 = zq_mm_min_ps(zero_v, a5);
						b6 = zq_mm_min_ps(zero_v, a6);
						b7 = zq_mm_min_ps(zero_v, a7);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						c4 = zq_mm_max_ps(zero_v, a4);
						c5 = zq_mm_max_ps(zero_v, a5);
						c6 = zq_mm_max_ps(zero_v, a6);
						c7 = zq_mm_max_ps(zero_v, a7);
						a0 = zq_mm_fmadd_ps(slope_v0, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v1, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v2, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v3, b3, c3);
						a4 = zq_mm_fmadd_ps(slope_v4, b4, c4);
						a5 = zq_mm_fmadd_ps(slope_v5, b5, c5);
						a6 = zq_mm_fmadd_ps(slope_v6, b6, c6);
						a7 = zq_mm_fmadd_ps(slope_v7, b7, c7);
#endif
						zq_mm_store_ps(out_slice_ptr0, a0);
						zq_mm_store_ps(out_slice_ptr1, a1);
						zq_mm_store_ps(out_slice_ptr2, a2);
						zq_mm_store_ps(out_slice_ptr3, a3);
						zq_mm_store_ps(out_slice_ptr4, a4);
						zq_mm_store_ps(out_slice_ptr5, a5);
						zq_mm_store_ps(out_slice_ptr6, a6);
						zq_mm_store_ps(out_slice_ptr7, a7);
#if WITH_BIAS
						bias_ptr += zq_mm_align_size8;
#endif
#if WITH_PRELU
						slope_ptr += zq_mm_align_size8;
#endif
					}
					matrix_C_row_ptr += matrix_B_cols;
				}
			}
		}
	}
	else if (out_C % zq_mm_align_size4 == 0)
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
				{
					out_slice_ptr0 = out_pix_ptr;
					out_slice_ptr1 = out_pix_ptr + out_sliceStep;
					out_slice_ptr2 = out_pix_ptr + out_sliceStep2;
					out_slice_ptr3 = out_pix_ptr + out_sliceStep3;
#if WITH_BIAS
					bias_ptr = bias;
#endif
#if WITH_PRELU
					slope_ptr = slope;
#endif
					matrix_C_col_ptr = matrix_C_row_ptr;
					for (kc = 0; kc < out_C; kc += zq_mm_align_size4,
						out_slice_ptr0 += out_sliceStep4,
						out_slice_ptr1 += out_sliceStep4,
						out_slice_ptr2 += out_sliceStep4,
						out_slice_ptr3 += out_sliceStep4,
						matrix_C_col_ptr += zq_mm_align_size4)
					{
#if WITH_BIAS
						bias_v0 = zq_mm_load_ps(bias_ptr);
						bias_v1 = zq_mm_load_ps(bias_ptr + zq_mm_align_size);
						bias_v2 = zq_mm_load_ps(bias_ptr + zq_mm_align_size2);
						bias_v3 = zq_mm_load_ps(bias_ptr + zq_mm_align_size3);
#endif
#if WITH_PRELU
						slope_v0 = zq_mm_load_ps(slope_ptr);
						slope_v1 = zq_mm_load_ps(slope_ptr + zq_mm_align_size);
						slope_v2 = zq_mm_load_ps(slope_ptr + zq_mm_align_size2);
						slope_v3 = zq_mm_load_ps(slope_ptr + zq_mm_align_size3);
#endif
						a0 = zq_mm_load_ps(matrix_C_col_ptr);
						a1 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size2);
						a3 = zq_mm_load_ps(matrix_C_col_ptr + zq_mm_align_size3);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v0);
						a1 = zq_mm_add_ps(a1, bias_v1);
						a2 = zq_mm_add_ps(a2, bias_v2);
						a3 = zq_mm_add_ps(a3, bias_v3);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						b1 = zq_mm_min_ps(zero_v, a1);
						b2 = zq_mm_min_ps(zero_v, a2);
						b3 = zq_mm_min_ps(zero_v, a3);
						c0 = zq_mm_max_ps(zero_v, a0);
						c1 = zq_mm_max_ps(zero_v, a1);
						c2 = zq_mm_max_ps(zero_v, a2);
						c3 = zq_mm_max_ps(zero_v, a3);
						a0 = zq_mm_fmadd_ps(slope_v0, b0, c0);
						a1 = zq_mm_fmadd_ps(slope_v1, b1, c1);
						a2 = zq_mm_fmadd_ps(slope_v2, b2, c2);
						a3 = zq_mm_fmadd_ps(slope_v3, b3, c3);
#endif
						zq_mm_store_ps(out_slice_ptr0, a0);
						zq_mm_store_ps(out_slice_ptr1, a1);
						zq_mm_store_ps(out_slice_ptr2, a2);
						zq_mm_store_ps(out_slice_ptr3, a3);
#if WITH_BIAS
						bias_ptr += zq_mm_align_size4;
#endif
#if WITH_PRELU
						slope_ptr += zq_mm_align_size4;
#endif
					}
					matrix_C_row_ptr += matrix_B_cols;
				}
			}
		}
	}
	else
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
			{
				for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
				{
					for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
					{
#if WITH_BIAS
						bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
						slope_v = zq_mm_load_ps(slope + kc);
#endif
						a0 = zq_mm_load_ps(matrix_C_row_ptr + kc);
#if WITH_BIAS
						a0 = zq_mm_add_ps(a0, bias_v);
#endif
#if WITH_PRELU
						b0 = zq_mm_min_ps(zero_v, a0);
						c0 = zq_mm_max_ps(zero_v, a0);
						a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
#endif
						zq_mm_store_ps(out_slice_ptr, a0);
					}
					matrix_C_row_ptr += matrix_B_cols;
				}
			}
		}
	}
#endif //EXPAND_CHANNEL == 0
}
else
{
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (out_h = 0, out_row_ptr = out_im_ptr; out_h < out_H; out_h++, out_row_ptr += out_widthStep)
		{
			for (out_w = 0, out_pix_ptr = out_row_ptr; out_w < out_W; out_w++, out_pix_ptr += zq_mm_align_size)
			{
				for (kc = 0, out_slice_ptr = out_pix_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
				{
#if WITH_BIAS
					bias_v = zq_mm_load_ps(bias + kc);
#endif
#if WITH_PRELU
					slope_v = zq_mm_load_ps(slope + kc);
#endif
					a0 = zq_mm_loadu_ps(matrix_C_row_ptr + kc);
#if WITH_BIAS
					a0 = zq_mm_add_ps(a0, bias_v);
#endif
#if WITH_PRELU
					b0 = zq_mm_min_ps(zero_v, a0);
					c0 = zq_mm_max_ps(zero_v, a0);
					a0 = zq_mm_fmadd_ps(slope_v, b0, c0);
#endif
					zq_mm_store_ps(out_slice_ptr, a0);
				}
				for (i = 0; i + kc < out_C; i++)
				{
					val = matrix_C_row_ptr[i + kc];
#if WITH_BIAS
					val += bias[i + kc];
#endif
#if WITH_PRELU
					if (val < 0)
						val *= slope[i + kc];
#endif
					out_slice_ptr[i] = val;
				}
				matrix_C_row_ptr += matrix_B_cols;
			}
		}
	}
}