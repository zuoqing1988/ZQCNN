#define op_sum_0_4_first \
	a0 = zq_mm_load_ps(in_pix_ptr);\
	b0 = zq_mm_load_ps(in1_pix_ptr);\
	a1 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(in1_pix_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(in1_pix_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(in1_pix_ptr+zq_mm_align_size3);\
	zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size, zq_mm_add_ps(a1, b1));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size2, zq_mm_add_ps(a2, b2));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size3, zq_mm_add_ps(a3, b3));\
	in_pix_ptr += zq_mm_align_size4;\
	in1_pix_ptr += zq_mm_align_size4;\
	out_pix_ptr += zq_mm_align_size4

#define op_sum_0_4 \
	a0 = zq_mm_load_ps(in_pix_ptr);\
	b0 = zq_mm_load_ps(out_pix_ptr);\
	a1 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size);\
	b1 = zq_mm_load_ps(out_pix_ptr+zq_mm_align_size);\
	a2 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size2);\
	b2 = zq_mm_load_ps(out_pix_ptr+zq_mm_align_size2);\
	a3 = zq_mm_load_ps(in_pix_ptr+zq_mm_align_size3);\
	b3 = zq_mm_load_ps(out_pix_ptr+zq_mm_align_size3);\
	zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size, zq_mm_add_ps(a1, b1));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size2, zq_mm_add_ps(a2, b2));\
	zq_mm_store_ps(out_pix_ptr+zq_mm_align_size3, zq_mm_add_ps(a3, b3));\
	in_pix_ptr += zq_mm_align_size4;\
	out_pix_ptr += zq_mm_align_size4

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

void zq_cnn_eltwise_sum_nchwc(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_widthStep,
	const int* in_sliceStep,
	const int* in_imStep,
	zq_base_type* out_tensor4D_data,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_im_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_im_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	register zq_mm_type a0, a1, a2, a3;
	register zq_mm_type b0, b1, b2, b3;
	int HWC_align = H*W*((C + zq_mm_align_size - 1) / zq_mm_align_size);
	int compactImageStep = HWC_align*zq_mm_align_size;
	int no_padding = compactImageStep == out_imStep;
	int handled = 1;

	for (tensor_id = 0; tensor_id < in_tensor_num; tensor_id++)
	{
		no_padding = no_padding && compactImageStep == in_imStep[tensor_id];
	}
	if (no_padding)
	{
		if (HWC_align % 64 == 0)
		{
			handled = 1;
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_pix_ptr = in_im_ptr, in1_pix_ptr = in1_im_ptr, out_pix_ptr = out_im_ptr;
					c < HWC_align; c += 64)
				{
					op_sum_0_64_first;
				}
			}

			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_pix_ptr = in_im_ptr, out_pix_ptr = out_im_ptr;
						c < HWC_align; c += 64)
					{
						op_sum_0_64;
					}
				}
			}
		}
		else if (HWC_align % 32 == 0)
		{
			handled = 1;
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_pix_ptr = in_im_ptr, in1_pix_ptr = in1_im_ptr, out_pix_ptr = out_im_ptr;
					c < HWC_align; c += 32)
				{
					op_sum_0_32_first;
				}
			}

			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_pix_ptr = in_im_ptr, out_pix_ptr = out_im_ptr;
						c < HWC_align; c += 32)
					{
						op_sum_0_32;
					}
				}
			}
		}
		else if (HWC_align % 16 == 0)
		{
			handled = 1;
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_pix_ptr = in_im_ptr, in1_pix_ptr = in1_im_ptr, out_pix_ptr = out_im_ptr;
					c < HWC_align; c += 16)
				{
					op_sum_0_16_first;
				}
			}

			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_pix_ptr = in_im_ptr, out_pix_ptr = out_im_ptr;
						c < HWC_align; c += 16)
					{
						op_sum_0_16;
					}
				}
			}
		}
		else if (HWC_align % 8 == 0)
		{
			handled = 1;
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_pix_ptr = in_im_ptr, in1_pix_ptr = in1_im_ptr, out_pix_ptr = out_im_ptr;
					c < HWC_align; c += 8)
				{
					op_sum_0_8_first;
				}
			}

			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_pix_ptr = in_im_ptr, out_pix_ptr = out_im_ptr;
						c < HWC_align; c += 8)
					{
						op_sum_0_8;
					}
				}
			}
		}
		else if (HWC_align % 4 == 0)
		{
			handled = 1;
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_pix_ptr = in_im_ptr, in1_pix_ptr = in1_im_ptr, out_pix_ptr = out_im_ptr;
					c < HWC_align; c += 4)
				{
					op_sum_0_4_first;
				}
			}

			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_pix_ptr = in_im_ptr, out_pix_ptr = out_im_ptr;
						c < HWC_align; c += 4)
					{
						op_sum_0_4;
					}
				}
			}
		}
	}
	
	if(handled == 0)
	{
		if (W % 4 == 0)
		{
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
					c < C;
					c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
							w < W;
							w += 4, in_pix_ptr += zq_mm_align_size4, in1_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
						{
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(in1_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
							a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
							b2 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size2);
							a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
							b3 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size3);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
						}
					}
				}
			}
			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
						c < C;
						c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
					{
						for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
							h < H;
							h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
						{
							for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
								w < W;
								w += 4, in_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
							{
								a0 = zq_mm_load_ps(in_pix_ptr);
								b0 = zq_mm_load_ps(out_pix_ptr);
								a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
								b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
								a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
								b2 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size2);
								a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
								b3 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size3);
								zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
							}
						}
					}
				}
			}
		}
		else if (W % 4 == 1)
		{
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
					c < C;
					c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
							w < W - 1;
							w += 4, in_pix_ptr += zq_mm_align_size4, in1_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
						{
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(in1_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
							a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
							b2 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size2);
							a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
							b3 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size3);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
						}
						a0 = zq_mm_load_ps(in_pix_ptr);
						b0 = zq_mm_load_ps(in1_pix_ptr);
						zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
					}
				}
			}
			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
						c < C;
						c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
					{
						for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
							h < H;
							h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
						{
							for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
								w < W - 1;
								w += 4, in_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
							{
								a0 = zq_mm_load_ps(in_pix_ptr);
								b0 = zq_mm_load_ps(out_pix_ptr);
								a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
								b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
								a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
								b2 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size2);
								a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
								b3 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size3);
								zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
							}
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(out_pix_ptr);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
						}
					}
				}
			}
		}
		else if (W % 4 == 2)
		{
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
					c < C;
					c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
							w < W - 2;
							w += 4, in_pix_ptr += zq_mm_align_size4, in1_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
						{
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(in1_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
							a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
							b2 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size2);
							a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
							b3 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size3);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
						}
						a0 = zq_mm_load_ps(in_pix_ptr);
						b0 = zq_mm_load_ps(in1_pix_ptr);
						a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
						b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
						zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
					}
				}
			}
			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
						c < C;
						c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
					{
						for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
							h < H;
							h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
						{
							for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
								w < W - 2;
								w += 4, in_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
							{
								a0 = zq_mm_load_ps(in_pix_ptr);
								b0 = zq_mm_load_ps(out_pix_ptr);
								a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
								b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
								a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
								b2 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size2);
								a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
								b3 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size3);
								zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
							}
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(out_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
						}
					}
				}
			}
		}
		else// if (W % 4 == 3)
		{
			for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
				n < N;
				n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
			{
				for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
					c < C;
					c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
				{
					for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
						h < H;
						h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
					{
						for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
							w < W - 3;
							w += 4, in_pix_ptr += zq_mm_align_size4, in1_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
						{
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(in1_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
							a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
							b2 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size2);
							a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
							b3 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size3);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
						}
						a0 = zq_mm_load_ps(in_pix_ptr);
						b0 = zq_mm_load_ps(in1_pix_ptr);
						a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
						b1 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size);
						a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
						b2 = zq_mm_load_ps(in1_pix_ptr + zq_mm_align_size2);
						zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
						zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
					}
				}
			}
			for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
			{
				for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_slice_ptr = out_tensor4D_data;
					n < N;
					n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
				{
					for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
						c < C;
						c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
					{
						for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
							h < H;
							h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
						{
							for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
								w < W - 3;
								w += 4, in_pix_ptr += zq_mm_align_size4, out_pix_ptr += zq_mm_align_size4)
							{
								a0 = zq_mm_load_ps(in_pix_ptr);
								b0 = zq_mm_load_ps(out_pix_ptr);
								a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
								b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
								a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
								b2 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size2);
								a3 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size3);
								b3 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size3);
								zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
								zq_mm_store_ps(out_pix_ptr + zq_mm_align_size3, zq_mm_add_ps(a3, b3));
							}
							a0 = zq_mm_load_ps(in_pix_ptr);
							b0 = zq_mm_load_ps(out_pix_ptr);
							a1 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size);
							b1 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size);
							a2 = zq_mm_load_ps(in_pix_ptr + zq_mm_align_size2);
							b2 = zq_mm_load_ps(out_pix_ptr + zq_mm_align_size2);
							zq_mm_store_ps(out_pix_ptr, zq_mm_add_ps(a0, b0));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size, zq_mm_add_ps(a1, b1));
							zq_mm_store_ps(out_pix_ptr + zq_mm_align_size2, zq_mm_add_ps(a2, b2));
						}
					}
				}
			}
		}
	}
}

void zq_cnn_eltwise_sum_with_weight_nchwc(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	const zq_base_type* weight,
	int N,
	int H,
	int W,
	int C,
	const int* in_widthStep,
	const int* in_sliceStep,
	const int* in_imStep,
	zq_base_type* out_tensor4D_data,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_im_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_im_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	register zq_mm_type weight0_v = zq_mm_set1_ps(weight[0]);
	register zq_mm_type weight1_v = zq_mm_set1_ps(weight[1]);

	for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
		n < N;
		n++, in_im_ptr += in_imStep[0], in1_slice_ptr += in_imStep[1], out_slice_ptr += out_imStep)
	{
		for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
			c < C;
			c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += zq_mm_align_size, in1_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_pix_ptr,
						zq_mm_add_ps(
							zq_mm_mul_ps(zq_mm_load_ps(in_pix_ptr), weight0_v),
							zq_mm_mul_ps(zq_mm_load_ps(in1_pix_ptr), weight1_v)
						)
					);
				}
			}
		}
	}
	for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
	{
		weight0_v = zq_mm_set1_ps(weight[tensor_id]);
		for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_im_ptr = out_tensor4D_data;
			n < N;
			n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
		{

			for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
				c < C;
				c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_pix_ptr,
							zq_mm_fmadd_ps(zq_mm_load_ps(in_pix_ptr), weight0_v, zq_mm_load_ps(out_pix_ptr))
						);
					}
				}
			}
		}
	}
}

void zq_cnn_eltwise_mul_nchwc(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_widthStep,
	const int* in_sliceStep,
	const int* in_imStep,
	zq_base_type* out_tensor4D_data,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_im_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_im_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
		n < N;
		n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
	{
		for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
			c < C;
			c += zq_mm_align_size, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += zq_mm_align_size, in1_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
				{

					zq_mm_store_ps(out_pix_ptr, zq_mm_mul_ps(zq_mm_load_ps(in_pix_ptr), zq_mm_load_ps(in1_pix_ptr)));
				}
			}
		}
	}
	for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
	{
		for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_im_ptr = out_tensor4D_data;
			n < N;
			n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
		{
			for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
				c < C;
				c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
					{

						zq_mm_store_ps(out_pix_ptr, zq_mm_mul_ps(zq_mm_load_ps(in_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
					}
				}
			}
		}
	}
}


void zq_cnn_eltwise_max_nchwc(
	int in_tensor_num,	//must be >=2
	const zq_base_type** in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	const int* in_widthStep,
	const int* in_sliceStep,
	const int* in_imStep,
	zq_base_type* out_tensor4D_data,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep
)
{
	int n, h, w, c, tensor_id;
	const zq_base_type* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_im_ptr;
	const zq_base_type* in1_slice_ptr, *in1_row_ptr, *in1_pix_ptr, *in1_im_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_im_ptr;
	for (n = 0, in_im_ptr = in_tensor4D_data[0], in1_im_ptr = in_tensor4D_data[1], out_im_ptr = out_tensor4D_data;
		n < N;
		n++, in_im_ptr += in_imStep[0], in1_im_ptr += in_imStep[1], out_im_ptr += out_imStep)
	{
		for (c = 0, in_slice_ptr = in_im_ptr, in1_slice_ptr = in1_im_ptr, out_slice_ptr = out_im_ptr;
			c < C;
			c += zq_mm_align_size, in_slice_ptr += in_sliceStep[0], in1_slice_ptr += in_sliceStep[1], out_slice_ptr += out_sliceStep)
		{
			for (h = 0, in_row_ptr = in_slice_ptr, in1_row_ptr = in1_slice_ptr, out_row_ptr = out_slice_ptr;
				h < H;
				h++, in_row_ptr += in_widthStep[0], in1_row_ptr += in_widthStep[1], out_row_ptr += out_widthStep)
			{
				for (w = 0, in_pix_ptr = in_row_ptr, in1_pix_ptr = in1_row_ptr, out_pix_ptr = out_row_ptr;
					w < W;
					w++, in_pix_ptr += zq_mm_align_size, in1_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(out_pix_ptr, zq_mm_max_ps(zq_mm_load_ps(in_pix_ptr), zq_mm_load_ps(in1_pix_ptr)));
				}
			}
		}
	}
	for (tensor_id = 2; tensor_id < in_tensor_num; tensor_id++)
	{
		for (n = 0, in_im_ptr = in_tensor4D_data[tensor_id], out_im_ptr = out_tensor4D_data;
			n < N;
			n++, in_im_ptr += in_imStep[tensor_id], out_im_ptr += out_imStep)
		{
			for (c = 0, in_slice_ptr = in_im_ptr, out_slice_ptr = out_im_ptr;
				c < C;
				c += zq_mm_align_size, in_slice_ptr += in_sliceStep[tensor_id], out_slice_ptr += out_sliceStep)
			{
				for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
					h < H;
					h++, in_row_ptr += in_widthStep[tensor_id], out_row_ptr += out_widthStep)
				{
					for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
						w < W;
						w++, in_pix_ptr += zq_mm_align_size, out_pix_ptr += zq_mm_align_size)
					{
						zq_mm_store_ps(out_pix_ptr, zq_mm_max_ps(zq_mm_load_ps(in_pix_ptr), zq_mm_load_ps(out_pix_ptr)));
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