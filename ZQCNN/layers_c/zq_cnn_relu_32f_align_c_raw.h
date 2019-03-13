
#define op_0_4 \
	a0 = zq_mm_load_ps(c_ptr);\
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_2);\
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_3);\
	zq_mm_store_ps(c_ptr, zq_mm_max_ps(zero_v, a0));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size, zq_mm_max_ps(zero_v, a1));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_2, zq_mm_max_ps(zero_v, a2));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_3, zq_mm_max_ps(zero_v, a3));\
	c_ptr += zq_mm_align_size_mul_4

#define op_0_4_slope \
	a0 = zq_mm_load_ps(c_ptr); \
	a1 = zq_mm_load_ps(c_ptr + zq_mm_align_size); \
	a2 = zq_mm_load_ps(c_ptr + zq_mm_align_size_mul_2); \
	a3 = zq_mm_load_ps(c_ptr + zq_mm_align_size_mul_3); \
	c0 = zq_mm_min_ps(zero_v, a0); \
	c1 = zq_mm_min_ps(zero_v, a1); \
	c2 = zq_mm_min_ps(zero_v, a2); \
	c3 = zq_mm_min_ps(zero_v, a3); \
	d0 = zq_mm_max_ps(zero_v, a0); \
	d1 = zq_mm_max_ps(zero_v, a1); \
	d2 = zq_mm_max_ps(zero_v, a2); \
	d3 = zq_mm_max_ps(zero_v, a3); \
	zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, c0, d0)); \
	zq_mm_store_ps(c_ptr + zq_mm_align_size, zq_mm_fmadd_ps(slope_v, c1, d1)); \
	zq_mm_store_ps(c_ptr + zq_mm_align_size_mul_2, zq_mm_fmadd_ps(slope_v, c2, d2)); \
	zq_mm_store_ps(c_ptr + zq_mm_align_size_mul_3, zq_mm_fmadd_ps(slope_v, c3, d3)); \
	c_ptr += zq_mm_align_size_mul_4

#define op_0_8 \
	op_0_4;\
	op_0_4

#define op_0_8_slope \
	op_0_4_slope;\
	op_0_4_slope

#define op_0_16 \
	op_0_8;\
	op_0_8

#define op_0_16_slope \
	op_0_8_slope;\
	op_0_8_slope

#define op_0_32 \
	op_0_16;\
	op_0_16

#define op_0_32_slope \
	op_0_16_slope;\
	op_0_16_slope

#define op_0_64 \
	op_0_32;\
	op_0_32

#define op_0_64_slope \
	op_0_32_slope;\
	op_0_32_slope

/*
y = max(0,x)
*/
void zq_cnn_relu_32f_align(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	zq_base_type slope
)
{

	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type c0, c1, c2, c3;
	register zq_mm_type d0, d1, d2, d3;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
	register zq_mm_type slope_v = zq_mm_set1_ps(slope);
	
	if (slope == 0)
	{
#if !__ARM_NEON
		if (in_C % zq_mm_align_size_mul_32 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_0_32;
						}
					}
				}
			}
		}
		else if (in_C % zq_mm_align_size_mul_16 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_0_16;
						}
					}
				}
			}
		}
		else
#endif
			if (in_C % zq_mm_align_size_mul_8 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_0_8;

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
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
	}
	else
	{
#if !__ARM_NEON
		if (in_C % zq_mm_align_size_mul_32 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							op_0_32_slope;
						}
					}
				}
			}
		}
		else if (in_C % zq_mm_align_size_mul_16 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							op_0_16_slope;
						}
					}
				}
			}
		}
		else
#endif
			if (in_C % zq_mm_align_size_mul_8 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							op_0_8_slope;

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
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							a0 = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), a0), zq_mm_max_ps(zq_mm_setzero_ps(), a0)));
							c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
	}
}

#undef op_0_4
#undef op_0_8
#undef op_0_16
#undef op_0_32
#undef op_0_64
#undef op_0_4_slope
#undef op_0_8_slope
#undef op_0_16_slope
#undef op_0_32_slope
#undef op_0_64_slope