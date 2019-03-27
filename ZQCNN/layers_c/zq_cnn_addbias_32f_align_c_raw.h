#define op_0_4 \
	a0 = zq_mm_load_ps(c_ptr);\
	a1 = zq_mm_load_ps(c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(c_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(c_ptr+zq_mm_align_size3);\
	b0 = zq_mm_load_ps(bias_ptr);\
	b1 = zq_mm_load_ps(bias_ptr+zq_mm_align_size);\
	b2 = zq_mm_load_ps(bias_ptr+zq_mm_align_size2);\
	b3 = zq_mm_load_ps(bias_ptr+zq_mm_align_size3);\
	zq_mm_store_ps(c_ptr, zq_mm_add_ps(a0,b0));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size, zq_mm_add_ps(a1,b1));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size2, zq_mm_add_ps(a2,b2));\
	zq_mm_store_ps(c_ptr+zq_mm_align_size3, zq_mm_add_ps(a3,b3));\
	c_ptr += zq_mm_align_size4;\
	bias_ptr += zq_mm_align_size4

#define op_0_8 \
	op_0_4;\
	op_0_4

#define op_0_16 \
	op_0_8;\
	op_0_8

#define op_0_32 \
	op_0_16;\
	op_0_16

#define op_0_64 \
	op_0_32;\
	op_0_32

void zq_cnn_addbias_32f_align(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* bias_data
)
{
	//zq_mm_type bias_v;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const zq_base_type* bias_ptr;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;

#if 1
#if !__ARM_NEON
	if (in_C%zq_mm_align_size32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size32)
					{
						op_0_32;
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size16 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size16)
					{
						op_0_16;
					}
				}
			}
		}
	}
	else
#endif
		if (in_C%zq_mm_align_size8 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size8)
					{
						op_0_8;
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size4 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C; c += zq_mm_align_size4)
					{
						op_0_4;
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
					for (c = 0, c_ptr = pix_ptr, bias_ptr = bias_data; c < in_C;
						c += zq_mm_align_size, c_ptr += zq_mm_align_size, bias_ptr += zq_mm_align_size)
						zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(bias_ptr)));
				}
			}
		}
	}

#else
	for (c = 0, c_ptr = in_tensor4D_data; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
	{
		bias_v = zq_mm_load_ps(bias_data + c);
		for (n = 0, slice_ptr = c_ptr; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					zq_mm_store_ps(pix_ptr, zq_mm_add_ps(zq_mm_load_ps(pix_ptr), bias_v));
				}
			}
		}
	}
#endif

}

#undef op_0_4
#undef op_0_8
#undef op_0_16
#undef op_0_32
#undef op_0_64