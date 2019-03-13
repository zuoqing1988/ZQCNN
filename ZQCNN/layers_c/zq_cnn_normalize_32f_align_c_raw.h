#define op_sum_0_4 \
	a0 = zq_mm_load_ps(c_ptr);\
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_2);\
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_3);\
	sum_v = zq_mm_fmadd_ps(a0, a0, sum_v);\
	sum_v = zq_mm_fmadd_ps(a1, a1, sum_v);\
	sum_v = zq_mm_fmadd_ps(a2, a2, sum_v);\
	sum_v = zq_mm_fmadd_ps(a3, a3, sum_v);\
	c_ptr += zq_mm_align_size_mul_4

#define op_sum_0_8 \
	op_sum_0_4;\
	op_sum_0_4

#define op_sum_0_16 \
	op_sum_0_8;\
	op_sum_0_8

#define op_sum_0_32 \
	op_sum_0_16;\
	op_sum_0_16

#define op_sum_0_64 \
	op_sum_0_32;\
	op_sum_0_32

#define op_mul_0_4 \
	a0 = zq_mm_load_ps(c_ptr); \
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size); \
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_2); \
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_3); \
	b0 = zq_mm_load_ps(scale_c_ptr); \
	b1 = zq_mm_load_ps(scale_c_ptr+zq_mm_align_size); \
	b2 = zq_mm_load_ps(scale_c_ptr+zq_mm_align_size_mul_2); \
	b3 = zq_mm_load_ps(scale_c_ptr+zq_mm_align_size_mul_3); \
	c0 = zq_mm_mul_ps(a0, b0); \
	c1 = zq_mm_mul_ps(a1, b1); \
	c2 = zq_mm_mul_ps(a2, b2); \
	c3 = zq_mm_mul_ps(a3, b3); \
	zq_mm_store_ps(c_ptr, zq_mm_mul_ps(c0, sum_v)); \
	zq_mm_store_ps(c_ptr+zq_mm_align_size, zq_mm_mul_ps(c1, sum_v)); \
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_2, zq_mm_mul_ps(c2, sum_v)); \
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_3, zq_mm_mul_ps(c3, sum_v)); \
	c_ptr += zq_mm_align_size_mul_4;\
	scale_c_ptr += zq_mm_align_size_mul_4

#define op_mul_0_8 \
	op_mul_0_4;\
	op_mul_0_4

#define op_mul_0_16 \
	op_mul_0_8;\
	op_mul_0_8

#define op_mul_0_32 \
	op_mul_0_16;\
	op_mul_0_16

#define op_mul_0_64 \
	op_mul_0_32;\
	op_mul_0_32

#define op_mul_0_4_share \
	a0 = zq_mm_load_ps(c_ptr);\
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_2);\
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_3);\
	b0 = zq_mm_mul_ps(a0, sum_mul_scale_v);\
	b1 = zq_mm_mul_ps(a1, sum_mul_scale_v);\
	b2 = zq_mm_mul_ps(a2, sum_mul_scale_v);\
	b3 = zq_mm_mul_ps(a3, sum_mul_scale_v);\
	zq_mm_store_ps(c_ptr, b0);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size, b1);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_2, b2);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_3, b3);\
	c_ptr += zq_mm_align_size_mul_4

#define op_mul_0_8_share \
	op_mul_0_4_share;\
	op_mul_0_4_share

#define op_mul_0_16_share \
	op_mul_0_8_share;\
	op_mul_0_8_share

#define op_mul_0_32_share \
	op_mul_0_16_share;\
	op_mul_0_16_share

#define op_mul_0_64_share \
	op_mul_0_32_share;\
	op_mul_0_32_share

void zq_cnn_normalize_not_across_spatial_32f_align(
	int channel_shared,
	zq_base_type* in_tensor4D_data,	// in & out
	const zq_base_type* scale_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type eps
)
{
	register zq_mm_type sum_v, sum_mul_scale_v;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];
	zq_base_type sum;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const zq_base_type *scale_c_ptr;
#if !__ARM_NEON
	if (in_C%zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					sum_v = zq_mm_setzero_ps();
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
					{
						op_sum_0_32;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q + eps);

					if (channel_shared)
					{
						sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_mul_0_32_share;
						}
					}
					else
					{
						sum_v = zq_mm_set1_ps(sum);
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_mul_0_32;
						}
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_16 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					sum_v = zq_mm_setzero_ps();
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
					{
						op_sum_0_16;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q+eps);

					if (channel_shared)
					{
						sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_mul_0_16_share;
						}
					}
					else
					{
						sum_v = zq_mm_set1_ps(sum);
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_mul_0_16;
						}
					}
				}
			}
		}
	}
	else
#endif
		if (in_C%zq_mm_align_size_mul_8 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					sum_v = zq_mm_setzero_ps();
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
					{
						op_sum_0_8;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q+eps);

					if (channel_shared)
					{
						sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_mul_0_8_share;
						}
					}
					else
					{
						sum_v = zq_mm_set1_ps(sum);
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_mul_0_8;
						}
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					sum_v = zq_mm_setzero_ps();
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
					{
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr + c), zq_mm_load_ps(c_ptr + c), sum_v);
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q+eps);

					if (channel_shared)
					{
						sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), sum_mul_scale_v));
						}
					}
					else
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data + c))));
						}
					}
				}
			}
		}
	}
}


void zq_cnn_normalize_across_spatial_32f_align(
	int channel_shared,
	zq_base_type* in_tensor4D_data,	// in & out
	const zq_base_type* scale_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type eps
)
{
	register zq_mm_type sum_v, sum_mul_scale_v;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c0, c1, c2, c3;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[8];
	zq_base_type sum;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const zq_base_type *scale_c_ptr;
#if !__ARM_NEON
	if (in_C%zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			sum_v = zq_mm_setzero_ps();
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
					{
						op_sum_0_32;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_mul_0_32_share;
						}
					}
				}
			}
			else
			{
				sum_v = zq_mm_set1_ps(sum);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_mul_0_32;
						}
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_16 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			sum_v = zq_mm_setzero_ps();
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
					{
						op_sum_0_16;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_mul_0_16_share;
						}
					}
				}
			}
			else
			{
				sum_v = zq_mm_set1_ps(sum);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_mul_0_16;
						}
					}
				}
			}
		}
	}
	else
#endif
		if (in_C%zq_mm_align_size_mul_8 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			sum_v = zq_mm_setzero_ps();
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
					{
						op_sum_0_8;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_mul_0_8_share;
						}
					}
				}
			}
			else
			{
				sum_v = zq_mm_set1_ps(sum);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_mul_0_8;
						}
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			sum_v = zq_mm_setzero_ps();
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
					{
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr + c), zq_mm_load_ps(c_ptr + c), sum_v);
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				sum_mul_scale_v = zq_mm_set1_ps(sum*scale_data[0]);
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), sum_mul_scale_v));
						}
					}
				}
			}
			else
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data + c))));
						}
					}
				}
			}
		}
	}
}

#undef op_sum_0_4
#undef op_sum_0_8
#undef op_sum_0_16
#undef op_sum_0_32
#undef op_sum_0_64
#undef op_mul_0_4
#undef op_mul_0_8
#undef op_mul_0_16
#undef op_mul_0_32
#undef op_mul_0_64
#undef op_mul_0_4_share
#undef op_mul_0_8_share
#undef op_mul_0_16_share
#undef op_mul_0_32_share
#undef op_mul_0_64_share