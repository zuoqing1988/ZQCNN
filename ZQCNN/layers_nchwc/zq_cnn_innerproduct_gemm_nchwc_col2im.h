if (matrix_B_rows%zq_mm_align_size == 0)
{
	if (out_C % zq_mm_align_size8 == 0)
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			out_slice_ptr0 = out_im_ptr;
			out_slice_ptr1 = out_im_ptr + out_sliceStep;
			out_slice_ptr2 = out_im_ptr + out_sliceStep2;
			out_slice_ptr3 = out_im_ptr + out_sliceStep3;
			out_slice_ptr4 = out_im_ptr + out_sliceStep4;
			out_slice_ptr5 = out_im_ptr + out_sliceStep5;
			out_slice_ptr6 = out_im_ptr + out_sliceStep6;
			out_slice_ptr7 = out_im_ptr + out_sliceStep7;
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
	else if (out_C % zq_mm_align_size4 == 0)
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			out_slice_ptr0 = out_im_ptr;
			out_slice_ptr1 = out_im_ptr + out_sliceStep;
			out_slice_ptr2 = out_im_ptr + out_sliceStep2;
			out_slice_ptr3 = out_im_ptr + out_sliceStep3;
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
	else
	{
		matrix_C_row_ptr = matrix_C;
		for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
		{
			out_slice_ptr0 = out_im_ptr;
			out_slice_ptr1 = out_im_ptr + out_sliceStep;
			out_slice_ptr2 = out_im_ptr + out_sliceStep2;
			out_slice_ptr3 = out_im_ptr + out_sliceStep3;
#if WITH_BIAS
			bias_ptr = bias;
#endif
#if WITH_PRELU
			slope_ptr = slope;
#endif
			matrix_C_col_ptr = matrix_C_row_ptr;
			for (kc = 0; kc < out_C; kc += zq_mm_align_size,
				out_slice_ptr0 += out_sliceStep,
				matrix_C_col_ptr += zq_mm_align_size)
			{
#if WITH_BIAS
				bias_v0 = zq_mm_load_ps(bias_ptr);
#endif
#if WITH_PRELU
				slope_v0 = zq_mm_load_ps(slope_ptr);
#endif
				a0 = zq_mm_load_ps(matrix_C_col_ptr);
#if WITH_BIAS
				a0 = zq_mm_add_ps(a0, bias_v0);
#endif
#if WITH_PRELU
				b0 = zq_mm_min_ps(zero_v, a0);
				c0 = zq_mm_max_ps(zero_v, a0);
				a0 = zq_mm_fmadd_ps(slope_v0, b0, c0);
#endif
				zq_mm_store_ps(out_slice_ptr0, a0);
#if WITH_BIAS
				bias_ptr += zq_mm_align_size;
#endif
#if WITH_PRELU
				slope_ptr += zq_mm_align_size;
#endif
			}
			matrix_C_row_ptr += matrix_B_cols;
		}
	}
}
else
{
	matrix_C_row_ptr = matrix_C;
	for (out_n = 0, out_im_ptr = out_tensor4D_data; out_n < out_N; out_n++, out_im_ptr += out_imStep)
	{
		for (kc = 0, out_slice_ptr = out_im_ptr; kc < out_C - zq_mm_align_size; kc += zq_mm_align_size, out_slice_ptr += out_sliceStep)
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