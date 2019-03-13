
#define op_0_4 \
	a0 = zq_mm_load_ps(in_c_ptr);\
	a1 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size_mul_2);\
	a3 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size_mul_3);\
	b0 = zq_mm_operation_ps(a0, scalar_v);\
	b1 = zq_mm_operation_ps(a1, scalar_v);\
	b2 = zq_mm_operation_ps(a2, scalar_v);\
	b3 = zq_mm_operation_ps(a3, scalar_v);\
	zq_mm_store_ps(out_c_ptr, b0);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, b1);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size_mul_2, b2);\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size_mul_3, b3);\
	in_c_ptr += zq_mm_align_size_mul_4;\
	out_c_ptr += zq_mm_align_size_mul_4

#define op_0_4_inplace \
	a0 = zq_mm_load_ps(c_ptr);\
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_2);\
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size_mul_3);\
	b0 = zq_mm_operation_ps(a0, scalar_v);\
	b1 = zq_mm_operation_ps(a1, scalar_v);\
	b2 = zq_mm_operation_ps(a2, scalar_v);\
	b3 = zq_mm_operation_ps(a3, scalar_v);\
	zq_mm_store_ps(c_ptr, b0);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size, b1);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_2, b2);\
	zq_mm_store_ps(c_ptr+zq_mm_align_size_mul_3, b3);\
	c_ptr += zq_mm_align_size_mul_4

#define op_0_8 \
	op_0_4;\
	op_0_4

#define op_0_8_inplace \
	op_0_4_inplace;\
	op_0_4_inplace

#define op_0_16 \
	op_0_8;\
	op_0_8

#define op_0_16_inplace \
	op_0_8_inplace;\
	op_0_8_inplace

#define op_0_32 \
	op_0_16;\
	op_0_16

#define op_0_32_inplace \
	op_0_16_inplace;\
	op_0_16_inplace

#define op_0_64 \
	op_0_32;\
	op_0_32

#define op_0_64_inplace \
	op_0_32_inplace;\
	op_0_32_inplace

void zq_cnn_scalaroperation_32f_align(
	zq_base_type scalar,
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	register zq_mm_type scalar_v = zq_mm_set1_ps(scalar);
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	int n, h, w, c;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
#if !__ARM_NEON
	if (in_C%zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data; 
			n < in_N; 
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr; 
				h < in_H; 
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr; 
					w < in_W; 
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr; 
						c < in_C; 
						c += zq_mm_align_size_mul_32)
					{
						op_0_32;
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_16 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
						c < in_C;
						c += zq_mm_align_size_mul_16)
					{
						op_0_16;
					}
				}
			}
		}
	}
	else
#endif
		if (in_C%zq_mm_align_size_mul_8 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
						c < in_C;
						c += zq_mm_align_size_mul_8)
					{
						op_0_8;
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < in_H;
				h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < in_W;
					w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
						c < in_C;
						c += zq_mm_align_size)
					{
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
					}
				}
			}
		}
	}
}

void zq_cnn_scalaroperation_inplace_32f_align(
	zq_base_type scalar,
	zq_base_type* in_tensor4D_data,	
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep
)
{
	register zq_mm_type scalar_v = zq_mm_set1_ps(scalar);
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
#if !__ARM_NEON
	if (in_C%zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
					{
						op_0_32_inplace;
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
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
					{
						op_0_16_inplace;
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
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
					{
						op_0_8_inplace;
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
					for (c = 0, c_ptr = pix_ptr; c < in_C;
						c += zq_mm_align_size, c_ptr += zq_mm_align_size)
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
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
#undef op_0_4_inplace
#undef op_0_8_inplace
#undef op_0_16_inplace
#undef op_0_32_inplace
#undef op_0_64_inplace