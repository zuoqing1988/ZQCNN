#define op4x4_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_mul_ps(a0, b0);\
c01 = zq_mm_mul_ps(a0, b1);\
c02 = zq_mm_mul_ps(a0, b2);\
c03 = zq_mm_mul_ps(a0, b3);\
c10 = zq_mm_mul_ps(a1, b0);\
c11 = zq_mm_mul_ps(a1, b1);\
c12 = zq_mm_mul_ps(a1, b2);\
c13 = zq_mm_mul_ps(a1, b3);\
c20 = zq_mm_mul_ps(a2, b0);\
c21 = zq_mm_mul_ps(a2, b1);\
c22 = zq_mm_mul_ps(a2, b2);\
c23 = zq_mm_mul_ps(a2, b3);\
c30 = zq_mm_mul_ps(a3, b0);\
c31 = zq_mm_mul_ps(a3, b1);\
c32 = zq_mm_mul_ps(a3, b2);\
c33 = zq_mm_mul_ps(a3, b3);\
src_ptr0 += zq_mm_align_size4;\
src_ptr1 += zq_mm_align_size4

#define op1x4_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_mul_ps(a0, b0);\
c01 = zq_mm_mul_ps(a0, b1);\
c02 = zq_mm_mul_ps(a0, b2);\
c03 = zq_mm_mul_ps(a0, b3);\
src_ptr0 += zq_mm_align_size;\
src_ptr1 += zq_mm_align_size4

#define op4x4_1 \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_fmadd_ps(a0, b0, c00);\
c01 = zq_mm_fmadd_ps(a0, b1, c01);\
c02 = zq_mm_fmadd_ps(a0, b2, c02);\
c03 = zq_mm_fmadd_ps(a0, b3, c03);\
c10 = zq_mm_fmadd_ps(a1, b0, c10);\
c11 = zq_mm_fmadd_ps(a1, b1, c11);\
c12 = zq_mm_fmadd_ps(a1, b2, c12);\
c13 = zq_mm_fmadd_ps(a1, b3, c13);\
c20 = zq_mm_fmadd_ps(a2, b0, c20);\
c21 = zq_mm_fmadd_ps(a2, b1, c21);\
c22 = zq_mm_fmadd_ps(a2, b2, c22);\
c23 = zq_mm_fmadd_ps(a2, b3, c23);\
c30 = zq_mm_fmadd_ps(a3, b0, c30);\
c31 = zq_mm_fmadd_ps(a3, b1, c31);\
c32 = zq_mm_fmadd_ps(a3, b2, c32);\
c33 = zq_mm_fmadd_ps(a3, b3, c33);\
src_ptr0 += zq_mm_align_size4;\
src_ptr1 += zq_mm_align_size4

#define op1x4_1 \
a0 = zq_mm_load_ps(src_ptr0);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_fmadd_ps(a0, b0, c00);\
c01 = zq_mm_fmadd_ps(a0, b1, c01);\
c02 = zq_mm_fmadd_ps(a0, b2, c02);\
c03 = zq_mm_fmadd_ps(a0, b3, c03);\
src_ptr0 += zq_mm_align_size;\
src_ptr1 += zq_mm_align_size4

#define op4x4_2_first \
op4x4_1_first;\
op4x4_1

#define op4x4_2 \
op4x4_1;\
op4x4_1

#define op4x4_4_first \
op4x4_2_first;\
op4x4_2

#define op4x4_4 \
op4x4_2;\
op4x4_2

#define op4x4_8_first \
op4x4_4_first;\
op4x4_4

#define op4x4_8 \
op4x4_4;\
op4x4_4

#define op4x4_16_first \
op4x4_8_first;\
op4x4_8

#define op4x4_16 \
op4x4_8;\
op4x4_8

#define op1x4_2_first \
op1x4_1_first;\
op1x4_1

#define op1x4_2 \
op1x4_1;\
op1x4_1

#define op1x4_4_first \
op1x4_2_first;\
op1x4_2

#define op1x4_4 \
op1x4_2;\
op1x4_2

#define op1x4_8_first \
op1x4_4_first;\
op1x4_4

#define op1x4_8 \
op1x4_4;\
op1x4_4

#define op1x4_16_first \
op1x4_8_first;\
op1x4_8

#define op1x4_16 \
op1x4_8;\
op1x4_8

#if __ARM_NEON && __ARM_NEON_ARMV8
#define store4x4 \
dst_ptr0[0] = vaddvq_f32(c00);\
dst_ptr0[1] = vaddvq_f32(c01);\
dst_ptr0[2] = vaddvq_f32(c02);\
dst_ptr0[3] = vaddvq_f32(c03);\
dst_ptr1[0] = vaddvq_f32(c10);\
dst_ptr1[1] = vaddvq_f32(c11);\
dst_ptr1[2] = vaddvq_f32(c12);\
dst_ptr1[3] = vaddvq_f32(c13);\
dst_ptr2[0] = vaddvq_f32(c20);\
dst_ptr2[1] = vaddvq_f32(c21);\
dst_ptr2[2] = vaddvq_f32(c22);\
dst_ptr2[3] = vaddvq_f32(c23);\
dst_ptr3[0] = vaddvq_f32(c30);\
dst_ptr3[1] = vaddvq_f32(c31);\
dst_ptr3[2] = vaddvq_f32(c32);\
dst_ptr3[3] = vaddvq_f32(c33)

#define store1x4 \
dst_ptr0[0] = vaddvq_f32(c00);\
dst_ptr0[1] = vaddvq_f32(c01);\
dst_ptr0[2] = vaddvq_f32(c02);\
dst_ptr0[3] = vaddvq_f32(c03)

#else

#define store4x4 \
zq_mm_store_ps(q, c00);\
dst_ptr0[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c01);\
dst_ptr0[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c02);\
dst_ptr0[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c03);\
dst_ptr0[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c10);\
dst_ptr1[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c11);\
dst_ptr1[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c12);\
dst_ptr1[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c13);\
dst_ptr1[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c20);\
dst_ptr2[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c21);\
dst_ptr2[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c22);\
dst_ptr2[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c23);\
dst_ptr2[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c30);\
dst_ptr3[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c31);\
dst_ptr3[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c32);\
dst_ptr3[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c33);\
dst_ptr3[3] = zq_final_sum_q

#define store1x4 \
zq_mm_store_ps(q, c00);\
dst_ptr0[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c01);\
dst_ptr0[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c02);\
dst_ptr0[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c03);\
dst_ptr0[3] = zq_final_sum_q
#endif

/*zq_mm_align_size must be 4*/
void zq_cnn_innerproduct_gemm_nchwc_packed4(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int div4_size = in_N >> 2;
	int paddedC = (in_C + 3) >> 2 << 2;
	int HWpaddedC = in_H*in_W*paddedC;
	int packed_A_num = div4_size + (in_N - (div4_size << 2));
	int packed_A_step = HWpaddedC * 4;
	int packed_B_step = HWpaddedC * 4;
	int packed_B_num = (out_C + 3) >> 2;
	int i, ii, h, w, c, out_c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1;
	const zq_base_type* slice_ptr0, *slice_ptr1, *slice_ptr2, *slice_ptr3;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03;
	register zq_mm_type c10, c11, c12, c13;
	register zq_mm_type c20, c21, c22, c23;
	register zq_mm_type c30, c31, c32, c33;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[16];
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < div4_size; i++)
	{
		dst_ptr = A_buffer + packed_A_step*i;
		ii = i * 4;
		slice_ptr0 = in_data + ii*in_imStep;
		ii++;
		slice_ptr1 = in_data + ii*in_imStep;
		ii++;
		slice_ptr2 = in_data + ii*in_imStep;
		ii++;
		slice_ptr3 = in_data + ii*in_imStep;
		
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			row_ptr1 = slice_ptr1;
			row_ptr2 = slice_ptr2;
			row_ptr3 = slice_ptr3;
			for (h = 0; h < in_H; h++)
			{
				pix_ptr0 = row_ptr0;
				pix_ptr1 = row_ptr1;
				pix_ptr2 = row_ptr2;
				pix_ptr3 = row_ptr3;
				for (w = 0; w < in_W; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
					dst_ptr += zq_mm_align_size4;
					pix_ptr0 += zq_mm_align_size;
					pix_ptr1 += zq_mm_align_size;
					pix_ptr2 += zq_mm_align_size;
					pix_ptr3 += zq_mm_align_size;
				}
				row_ptr0 += in_widthStep;
				row_ptr1 += in_widthStep;
				row_ptr2 += in_widthStep;
				row_ptr3 += in_widthStep;
			}
			
			slice_ptr0 += in_sliceStep;
			slice_ptr1 += in_sliceStep;
			slice_ptr2 += in_sliceStep;
			slice_ptr3 += in_sliceStep;
		}
	}

	for (i = 0; i < in_N - div4_size * 4; i++)
	{
		dst_ptr = A_buffer + packed_A_step*(i + div4_size);
		ii = div4_size * 4 + i;
		slice_ptr0 = in_data + ii*in_imStep;
		
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			for (h = 0; h < in_H; h++)
			{
				pix_ptr0 = row_ptr0;
				for (w = 0; w < in_W; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					dst_ptr += zq_mm_align_size;
					pix_ptr0 += zq_mm_align_size;
				}
				row_ptr0 += in_widthStep;
			}

			slice_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < div4_size; i++)
	{
		ii = i * 4;
		dst_ptr0 = out_data + ii*out_imStep;
		ii++;
		dst_ptr1 = out_data + ii*out_imStep;
		ii++;
		dst_ptr2 = out_data + ii*out_imStep;
		ii++;
		dst_ptr3 = out_data + ii*out_imStep;
		if (paddedC % zq_mm_align_size16 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);

				op4x4_16_first;
				c = zq_mm_align_size16;
				for (; c < HWpaddedC; c += zq_mm_align_size16)
				{
					op4x4_16;
				}
				store4x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
				a1 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr1));
				a2 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr2));
				a3 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr3));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
				a1 = zq_mm_fmadd_ps(slope_v, c01, c11);
				a2 = zq_mm_fmadd_ps(slope_v, c02, c12);
				a3 = zq_mm_fmadd_ps(slope_v, c03, c13);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
				zq_mm_store_ps(dst_ptr1, a1);
				zq_mm_store_ps(dst_ptr2, a2);
				zq_mm_store_ps(dst_ptr3, a3);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				a1 = zq_mm_load_ps(dst_ptr1);
				a2 = zq_mm_load_ps(dst_ptr2);
				a3 = zq_mm_load_ps(dst_ptr3);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
				zq_mm_store_ps(dst_ptr1, zq_mm_fmadd_ps(slope_v, c01, c11));
				zq_mm_store_ps(dst_ptr2, zq_mm_fmadd_ps(slope_v, c02, c12));
				zq_mm_store_ps(dst_ptr3, zq_mm_fmadd_ps(slope_v, c03, c13));
#endif
#endif
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op4x4_8_first;
				c = zq_mm_align_size8;
				for (; c < HWpaddedC; c += zq_mm_align_size8)
				{
					op4x4_8;
				}
				store4x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
				a1 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr1));
				a2 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr2));
				a3 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr3));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
				a1 = zq_mm_fmadd_ps(slope_v, c01, c11);
				a2 = zq_mm_fmadd_ps(slope_v, c02, c12);
				a3 = zq_mm_fmadd_ps(slope_v, c03, c13);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
				zq_mm_store_ps(dst_ptr1, a1);
				zq_mm_store_ps(dst_ptr2, a2);
				zq_mm_store_ps(dst_ptr3, a3);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				a1 = zq_mm_load_ps(dst_ptr1);
				a2 = zq_mm_load_ps(dst_ptr2);
				a3 = zq_mm_load_ps(dst_ptr3);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
				zq_mm_store_ps(dst_ptr1, zq_mm_fmadd_ps(slope_v, c01, c11));
				zq_mm_store_ps(dst_ptr2, zq_mm_fmadd_ps(slope_v, c02, c12));
				zq_mm_store_ps(dst_ptr3, zq_mm_fmadd_ps(slope_v, c03, c13));
#endif
#endif
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op4x4_4_first;
				c = zq_mm_align_size4;
				for (; c < HWpaddedC; c += zq_mm_align_size4)
				{
					op4x4_4;
				}
				store4x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
				a1 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr1));
				a2 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr2));
				a3 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr3));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
				a1 = zq_mm_fmadd_ps(slope_v, c01, c11);
				a2 = zq_mm_fmadd_ps(slope_v, c02, c12);
				a3 = zq_mm_fmadd_ps(slope_v, c03, c13);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
				zq_mm_store_ps(dst_ptr1, a1);
				zq_mm_store_ps(dst_ptr2, a2);
				zq_mm_store_ps(dst_ptr3, a3);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				a1 = zq_mm_load_ps(dst_ptr1);
				a2 = zq_mm_load_ps(dst_ptr2);
				a3 = zq_mm_load_ps(dst_ptr3);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
				zq_mm_store_ps(dst_ptr1, zq_mm_fmadd_ps(slope_v, c01, c11));
				zq_mm_store_ps(dst_ptr2, zq_mm_fmadd_ps(slope_v, c02, c12));
				zq_mm_store_ps(dst_ptr3, zq_mm_fmadd_ps(slope_v, c03, c13));
#endif
#endif
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op4x4_2_first;
				c = zq_mm_align_size2;
				for (; c < HWpaddedC; c += zq_mm_align_size2)
				{
					op4x4_2;
				}
				store4x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
				a1 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr1));
				a2 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr2));
				a3 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr3));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
				a1 = zq_mm_fmadd_ps(slope_v, c01, c11);
				a2 = zq_mm_fmadd_ps(slope_v, c02, c12);
				a3 = zq_mm_fmadd_ps(slope_v, c03, c13);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
				zq_mm_store_ps(dst_ptr1, a1);
				zq_mm_store_ps(dst_ptr2, a2);
				zq_mm_store_ps(dst_ptr3, a3);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				a1 = zq_mm_load_ps(dst_ptr1);
				a2 = zq_mm_load_ps(dst_ptr2);
				a3 = zq_mm_load_ps(dst_ptr3);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
				zq_mm_store_ps(dst_ptr1, zq_mm_fmadd_ps(slope_v, c01, c11));
				zq_mm_store_ps(dst_ptr2, zq_mm_fmadd_ps(slope_v, c02, c12));
				zq_mm_store_ps(dst_ptr3, zq_mm_fmadd_ps(slope_v, c03, c13));
#endif
#endif
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op4x4_1_first;
				c = zq_mm_align_size;
				for (; c < HWpaddedC; c += zq_mm_align_size)
				{
					op4x4_1;
				}
				store4x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
				a1 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr1));
				a2 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr2));
				a3 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr3));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
				a1 = zq_mm_fmadd_ps(slope_v, c01, c11);
				a2 = zq_mm_fmadd_ps(slope_v, c02, c12);
				a3 = zq_mm_fmadd_ps(slope_v, c03, c13);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
				zq_mm_store_ps(dst_ptr1, a1);
				zq_mm_store_ps(dst_ptr2, a2);
				zq_mm_store_ps(dst_ptr3, a3);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				a1 = zq_mm_load_ps(dst_ptr1);
				a2 = zq_mm_load_ps(dst_ptr2);
				a3 = zq_mm_load_ps(dst_ptr3);
				c00 = zq_mm_min_ps(a0, zero_v);
				c01 = zq_mm_min_ps(a1, zero_v);
				c02 = zq_mm_min_ps(a2, zero_v);
				c03 = zq_mm_min_ps(a3, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				c11 = zq_mm_max_ps(a1, zero_v);
				c12 = zq_mm_max_ps(a2, zero_v);
				c13 = zq_mm_max_ps(a3, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
				zq_mm_store_ps(dst_ptr1, zq_mm_fmadd_ps(slope_v, c01, c11));
				zq_mm_store_ps(dst_ptr2, zq_mm_fmadd_ps(slope_v, c02, c12));
				zq_mm_store_ps(dst_ptr3, zq_mm_fmadd_ps(slope_v, c03, c13));
#endif
#endif
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
	}

	//rest 
	for (i = 0; i < in_N - div4_size * 4; i++)
	{
		ii = (div4_size << 2) + i;
		dst_ptr0 = out_data + ii*out_imStep;
		if (paddedC % zq_mm_align_size16 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op1x4_16_first;
				c = zq_mm_align_size16;
				for (; c < HWpaddedC; c += zq_mm_align_size16)
				{
					op1x4_16;
				}
				store1x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op1x4_8_first;
				c = zq_mm_align_size8;
				for (; c < HWpaddedC; c += zq_mm_align_size8)
				{
					op1x4_8;
				}
				store1x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op1x4_4_first;
				c = zq_mm_align_size4;
				for (; c < HWpaddedC; c += zq_mm_align_size4)
				{
					op1x4_4;
				}
				store1x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op1x4_2_first;
				c = zq_mm_align_size2;
				for (; c < HWpaddedC; c += zq_mm_align_size2)
				{
					op1x4_2;
				}
				store1x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
				dst_ptr0 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op1x4_1_first;
				c = zq_mm_align_size;
				for (; c < HWpaddedC; c += zq_mm_align_size)
				{
					op1x4_1;
				}
				store1x4;
#if WITH_BIAS
				bias_v = zq_mm_load_ps(bias + out_c);
				a0 = zq_mm_add_ps(bias_v, zq_mm_load_ps(dst_ptr0));
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				a0 = zq_mm_fmadd_ps(slope_v, c00, c10);
#endif
				zq_mm_store_ps(dst_ptr0, a0);
#else
#if WITH_PRELU
				slope_v = zq_mm_load_ps(slope + out_c);
				a0 = zq_mm_load_ps(dst_ptr0);
				c00 = zq_mm_min_ps(a0, zero_v);
				c10 = zq_mm_max_ps(a0, zero_v);
				zq_mm_store_ps(dst_ptr0, zq_mm_fmadd_ps(slope_v, c00, c10));
#endif
#endif
				dst_ptr0 += out_sliceStep;
			}
		}
	}

	/*  */
}

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_innerproduct_gemm_nchwc_packed8_other(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int B_alignN = (out_C + 3) >> 2 << 2;
	int B_div8_num = B_alignN >> 3;
	int B_div4_num = (B_alignN - (B_div8_num << 3)) >> 2;
	int paddedC = (in_C + 3) >> 2 << 2;
	int HWpaddedC = in_H*in_W*paddedC;
	int A_div4_num = in_N >> 2;
	int A_rest_num = in_N - (A_div4_num << 2);
	int packed_A_num = A_div4_num + A_rest_num;
	int packed_A_step = HWpaddedC * 4;
	int packed_B_step = HWpaddedC * 8;
	int i, ii, h, w, c, out_c, j ,k;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1;
	const zq_base_type* slice_ptr0, *slice_ptr1, *slice_ptr2, *slice_ptr3;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1;
	register zq_mm_type c00, c01, c10, c11, c20, c21, c30, c31;
#if WITH_BIAS
	register zq_mm_type bias_v0,bias_v1;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v0,slope_v1;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	//printf("out_C = %d, B_div8_num = %d, B_div4_num = %d\n", out_C, B_div8_num, B_div4_num);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < A_div4_num; i++)
	{
		dst_ptr = A_buffer + packed_A_step*i;
		ii = i * 4;
		slice_ptr0 = in_data + ii*in_imStep;
		ii++;
		slice_ptr1 = in_data + ii*in_imStep;
		ii++;
		slice_ptr2 = in_data + ii*in_imStep;
		ii++;
		slice_ptr3 = in_data + ii*in_imStep;

		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			row_ptr1 = slice_ptr1;
			row_ptr2 = slice_ptr2;
			row_ptr3 = slice_ptr3;
			for (h = 0; h < in_H; h++)
			{
				pix_ptr0 = row_ptr0;
				pix_ptr1 = row_ptr1;
				pix_ptr2 = row_ptr2;
				pix_ptr3 = row_ptr3;
				for (w = 0; w < in_W; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
					zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
					dst_ptr += zq_mm_align_size4;
					pix_ptr0 += zq_mm_align_size;
					pix_ptr1 += zq_mm_align_size;
					pix_ptr2 += zq_mm_align_size;
					pix_ptr3 += zq_mm_align_size;
				}
				row_ptr0 += in_widthStep;
				row_ptr1 += in_widthStep;
				row_ptr2 += in_widthStep;
				row_ptr3 += in_widthStep;
			}

			slice_ptr0 += in_sliceStep;
			slice_ptr1 += in_sliceStep;
			slice_ptr2 += in_sliceStep;
			slice_ptr3 += in_sliceStep;
		}
	}

	for (i = 0; i < in_N - A_div4_num * 4; i++)
	{
		dst_ptr = A_buffer + packed_A_step*(i + A_div4_num);
		ii = A_div4_num * 4 + i;
		slice_ptr0 = in_data + ii*in_imStep;

		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			row_ptr0 = slice_ptr0;
			for (h = 0; h < in_H; h++)
			{
				pix_ptr0 = row_ptr0;
				for (w = 0; w < in_W; w++)
				{
					zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
					dst_ptr += zq_mm_align_size;
					pix_ptr0 += zq_mm_align_size;
				}
				row_ptr0 += in_widthStep;
			}

			slice_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4;
		dst_ptr0 = out_data + ii*out_imStep;
		ii++;
		dst_ptr1 = out_data + ii*out_imStep;
		ii++;
		dst_ptr2 = out_data + ii*out_imStep;
		ii++;
		dst_ptr3 = out_data + ii*out_imStep;
		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*i;
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);
			a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);
			a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
			c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);
			c11 = vfmaq_laneq_f32(bias_v1, b1, a1, 0);
			c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);
			c21 = vfmaq_laneq_f32(bias_v1, b1, a2, 0);
			c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);
			c31 = vfmaq_laneq_f32(bias_v1, b1, a3, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
			c10 = vmulq_laneq_f32(b0, a1, 0);
			c11 = vmulq_laneq_f32(b1, a1, 0);
			c20 = vmulq_laneq_f32(b0, a2, 0);
			c21 = vmulq_laneq_f32(b1, a2, 0);
			c30 = vmulq_laneq_f32(b0, a3, 0);
			c31 = vmulq_laneq_f32(b1, a3, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 1);
			c11 = vfmaq_laneq_f32(c11, b1, a1, 1);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 1);
			c21 = vfmaq_laneq_f32(c21, b1, a2, 1);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 1);
			c31 = vfmaq_laneq_f32(c31, b1, a3, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 2);
			c11 = vfmaq_laneq_f32(c11, b1, a1, 2);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 2);
			c21 = vfmaq_laneq_f32(c21, b1, a2, 2);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 2);
			c31 = vfmaq_laneq_f32(c31, b1, a3, 2);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 3);
			c11 = vfmaq_laneq_f32(c11, b1, a1, 3);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 3);
			c21 = vfmaq_laneq_f32(c21, b1, a2, 3);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 3);
			c31 = vfmaq_laneq_f32(c31, b1, a3, 3);
			for (c = 4; c < HWpaddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size4;
				a0 = zq_mm_load_ps(src_ptr0);
				a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);
				a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);
				a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 0);
				c11 = vfmaq_laneq_f32(c11, b1, a1, 0);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 0);
				c21 = vfmaq_laneq_f32(c21, b1, a2, 0);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 0);
				c31 = vfmaq_laneq_f32(c31, b1, a3, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 1);
				c11 = vfmaq_laneq_f32(c11, b1, a1, 1);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 1);
				c21 = vfmaq_laneq_f32(c21, b1, a2, 1);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 1);
				c31 = vfmaq_laneq_f32(c31, b1, a3, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 2);
				c11 = vfmaq_laneq_f32(c11, b1, a1, 2);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 2);
				c21 = vfmaq_laneq_f32(c21, b1, a2, 2);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 2);
				c31 = vfmaq_laneq_f32(c31, b1, a3, 2);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 3);
				c11 = vfmaq_laneq_f32(c11, b1, a1, 3);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 3);
				c21 = vfmaq_laneq_f32(c21, b1, a2, 3);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 3);
				c31 = vfmaq_laneq_f32(c31, b1, a3, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
			c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			zq_mm_store_ps(dst_ptr1, c11);
			zq_mm_store_ps(dst_ptr2, c21);
			zq_mm_store_ps(dst_ptr3, c31);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}
		
		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*i;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);
			a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);
			a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);
			c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);
			c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c10 = vmulq_laneq_f32(b0, a1, 0);
			c20 = vmulq_laneq_f32(b0, a2, 0);
			c30 = vmulq_laneq_f32(b0, a3, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 1);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 1);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 2);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 2);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 2);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			c10 = vfmaq_laneq_f32(c10, b0, a1, 3);
			c20 = vfmaq_laneq_f32(c20, b0, a2, 3);
			c30 = vfmaq_laneq_f32(c30, b0, a3, 3);
			for (c = 4; c < HWpaddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size4;
				a0 = zq_mm_load_ps(src_ptr0);
				a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);
				a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);
				a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 0);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 0);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 1);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 1);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 2);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 2);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 2);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
				c10 = vfmaq_laneq_f32(c10, b0, a1, 3);
				c20 = vfmaq_laneq_f32(c20, b0, a2, 3);
				c30 = vfmaq_laneq_f32(c30, b0, a3, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}
	}

	//rest 
	for (i = 0; i < in_N - A_div4_num * 4; i++)
	{
		ii = (A_div4_num << 2) + i;
		dst_ptr0 = out_data + ii*out_imStep;
		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(A_div4_num + i);
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			for (c = 4; c < HWpaddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			dst_ptr0 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(A_div4_num + i);
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			for (c = 4; c < HWpaddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
		}
	}
}

#endif

#undef op4x4_1_first
#undef op4x4_1
#undef op4x4_2_first
#undef op4x4_2
#undef op4x4_4_first
#undef op4x4_4
#undef op4x4_8_first
#undef op4x4_8
#undef op4x4_16_first
#undef op4x4_16
#undef store4x4

#undef op1x4_1_first
#undef op1x4_1
#undef op1x4_2_first
#undef op1x4_2
#undef op1x4_4_first
#undef op1x4_4
#undef op1x4_8_first
#undef op1x4_8
#undef op1x4_16_first
#undef op1x4_16
#undef store1x4