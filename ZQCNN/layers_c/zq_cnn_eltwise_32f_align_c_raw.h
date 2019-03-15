#define op_sum_0_4_first \
	a0 = zq_mm_load_ps(in_c_ptr);\
	b0 = zq_mm_load_ps(in1_c_ptr);\
	a1 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(in1_c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(in1_c_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(in1_c_ptr+zq_mm_align_size3);\
	zq_mm_store_ps(out_c_ptr, zq_mm_add_ps(a0, b0));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, zq_mm_add_ps(a1, b1));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, zq_mm_add_ps(a2, b2));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, zq_mm_add_ps(a3, b3));\
	in_c_ptr += zq_mm_align_size4;\
	in1_c_ptr += zq_mm_align_size4;\
	out_c_ptr += zq_mm_align_size4

#define op_sum_0_4 \
	a0 = zq_mm_load_ps(in_c_ptr);\
	b0 = zq_mm_load_ps(out_c_ptr);\
	a1 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(in_c_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(out_c_ptr+zq_mm_align_size3);\
	zq_mm_store_ps(out_c_ptr, zq_mm_add_ps(a0, b0));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size, zq_mm_add_ps(a1, b1));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size2, zq_mm_add_ps(a2, b2));\
	zq_mm_store_ps(out_c_ptr+zq_mm_align_size3, zq_mm_add_ps(a3, b3));\
	in_c_ptr += zq_mm_align_size4;\
	out_c_ptr += zq_mm_align_size4
	
#define op_sum_0_8 \
	op_sum_0_4;\
	op_sum_0_4

#define op_sum_0_8_first \
	op_sum_0_4_first;\
	op_sum_0_4_first

#define op_sum_0_16 \
	op_sum_0_8;\
	op_sum_0_8

#define op_sum_0_16_first \
	op_sum_0_8_first;\
	op_sum_0_8_first

#define op_sum_0_32 \
	op_sum_0_16;\
	op_sum_0_16

#define op_sum_0_32_first \
	op_sum_0_16_first;\
	op_sum_0_16_first

#define op_sum_0_64 \
	op_sum_0_32;\
	op_sum_0_32

#define op_sum_0_64_first \
	op_sum_0_32_first;\
	op_sum_0_32_first

#define op_weight_sum_0_1_first \
	zq_mm_store_ps(out_c_ptr,zq_mm_add_ps(zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_set1_ps(weight[0])),zq_mm_mul_ps(zq_mm_load_ps(in1_c_ptr), zq_mm_set1_ps(weight[1]))));\
	in_c_ptr += zq_mm_align_size;\
	in1_c_ptr += zq_mm_align_size;\
	out_c_ptr += zq_mm_align_size

#define op_weight_sum_0_1 \
	zq_mm_store_ps(out_c_ptr,zq_mm_add_ps(zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_set1_ps(weight[tensor_id])),zq_mm_load_ps(out_c_ptr)));\
	in_c_ptr += zq_mm_align_size;\
	out_c_ptr += zq_mm_align_size

#define op_weight_sum_0_2_first \
	op_weight_sum_0_1_first;\
	op_weight_sum_0_1_first

#define op_weight_sum_0_2 \
	op_weight_sum_0_1;\
	op_weight_sum_0_1

#define op_weight_sum_0_4_first \
	op_weight_sum_0_2_first;\
	op_weight_sum_0_2_first

#define op_weight_sum_0_4 \
	op_weight_sum_0_2;\
	op_weight_sum_0_2

#define op_weight_sum_0_8_first \
	op_weight_sum_0_4_first;\
	op_weight_sum_0_4_first

#define op_weight_sum_0_8 \
	op_weight_sum_0_4;\
	op_weight_sum_0_4

#define op_weight_sum_0_16_first \
	op_weight_sum_0_8_first;\
	op_weight_sum_0_8_first

#define op_weight_sum_0_16 \
	op_weight_sum_0_8;\
	op_weight_sum_0_8

#define op_weight_sum_0_32_first \
	op_weight_sum_0_16_first;\
	op_weight_sum_0_16_first

#define op_weight_sum_0_32 \
	op_weight_sum_0_16;\
	op_weight_sum_0_16

#define op_weight_sum_0_64_first \
	op_weight_sum_0_32_first;\
	op_weight_sum_0_32_first

#define op_weight_sum_0_64 \
	op_weight_sum_0_32;\
	op_weight_sum_0_32

void zq_cnn_eltwise_sum_32f_align(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_pixelStep,
	const int* in_widthStep,
	const int* in_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_c_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;

	if (C%zq_mm_align_size32 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C; c += zq_mm_align_size32)
					{
						op_sum_0_32_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr; c < C; c += zq_mm_align_size32)
						{
							op_sum_0_32;
						}
					}
				}
			}
		}
	}
	else if (C%zq_mm_align_size16 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C; c += zq_mm_align_size16)
					{
						op_sum_0_16_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr; c < C; c += zq_mm_align_size16)
						{
							op_sum_0_16;
						}
					}
				}
			}
		}
	}
	else if (C%zq_mm_align_size8 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C; c += zq_mm_align_size8)
					{
						op_sum_0_8_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;	c < C;	c += zq_mm_align_size8)
						{
							op_sum_0_8;
						}
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;
						c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, in1_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_c_ptr, zq_mm_add_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(in1_c_ptr)));
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
						{
							zq_mm_store_ps(out_c_ptr, zq_mm_add_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(out_c_ptr)));
						}
					}
				}
			}
		}
	}
	
}


void zq_cnn_eltwise_sum_with_weight_32f_align(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	const zq_base_type* weight,
	int N,
	int H,
	int W,
	int C,
	const int* in_pixelStep,
	const int* in_widthStep,
	const int* in_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_c_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;

	if (C%zq_mm_align_size32 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;	c += zq_mm_align_size32)
					{
						op_weight_sum_0_32_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;	c += zq_mm_align_size32)
						{
							op_weight_sum_0_32;
						}
					}
				}
			}
		}
	}
	else if (C%zq_mm_align_size16 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;	c += zq_mm_align_size16)
					{
						op_weight_sum_0_16_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;	c += zq_mm_align_size16)
						{
							op_weight_sum_0_16;
						}
					}
				}
			}
		}
	}
	else if (C%zq_mm_align_size8 == 0)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;	c += zq_mm_align_size8)
					{
						op_weight_sum_0_8_first;
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;	c += zq_mm_align_size8)
						{
							op_weight_sum_0_8;
						}
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;
						c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, in1_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_c_ptr,
							zq_mm_add_ps(
								zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_set1_ps(weight[0])),
								zq_mm_mul_ps(zq_mm_load_ps(in1_c_ptr), zq_mm_set1_ps(weight[1]))
							)
						);
					}
				}
			}
		}
		for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
		{
			for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
				n < N;
				n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
					{
						for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
							c < C;
							c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
						{
							zq_mm_store_ps(out_c_ptr,
								zq_mm_add_ps(
									zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_set1_ps(weight[tensor_id])),
									zq_mm_load_ps(out_c_ptr)
								)
							);
						}
					}
				}
			}
		}
	}
}

void zq_cnn_eltwise_mul_32f_align(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_pixelStep,
	const int* in_widthStep,
	const int* in_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_c_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
	for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
		n < N;
		n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
	{
		for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
			h < H;
			h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
		{
			for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
				w < W;
				w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
			{
				for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
					c < C;
					c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, in1_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(in1_c_ptr)));
				}
			}
		}
	}
	for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;
						c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_c_ptr, zq_mm_mul_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(out_c_ptr)));
					}
				}
			}
		}
	}
}


void zq_cnn_eltwise_max_32f_align(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_pixelStep,
	const int* in_widthStep,
	const int* in_sliceStep,
	zq_base_type* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_c_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
	for (n = 0, in_slice_ptr = in_tensor4D_data[0], in1_slice_ptr = in_tensor4D_data[1], out_slice_ptr = out_tensor4D_data;
		n < N;
		n++, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
	{
		for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
			h < H;
			h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
		{
			for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
				w < W;
				w++, in_pix_ptr += in_pixelStep[0], in1_pix_ptr += in_pixelStep[1], out_pix_ptr += out_pixelStep)
			{
				for (c = 0, in_c_ptr = in_pix_ptr, in1_c_ptr = in1_pix_ptr, out_c_ptr = out_pix_ptr;
					c < C;
					c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, in1_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_c_ptr, zq_mm_max_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(in1_c_ptr)));
				}
			}
		}
	}
	for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
			n < N;
			n++, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += in_pixelStep[tensor_id], out_pix_ptr += out_pixelStep)
				{
					for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr;
						c < C;
						c += zq_mm_align_size, in_c_ptr += zq_mm_align_size, out_c_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_c_ptr, zq_mm_max_ps(zq_mm_load_ps(in_c_ptr), zq_mm_load_ps(out_c_ptr)));
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
#undef op_sum_0_4_first
#undef op_sum_0_8_first
#undef op_sum_0_16_first
#undef op_sum_0_32_first
#undef op_sum_0_64_first
#undef op_weight_sum_0_1
#undef op_weight_sum_0_2
#undef op_weight_sum_0_4
#undef op_weight_sum_0_8
#undef op_weight_sum_0_16
#undef op_weight_sum_0_32
#undef op_weight_sum_0_64
#undef op_weight_sum_0_1_first
#undef op_weight_sum_0_2_first
#undef op_weight_sum_0_4_first
#undef op_weight_sum_0_8_first
#undef op_weight_sum_0_16_first
#undef op_weight_sum_0_32_first
#undef op_weight_sum_0_64_first