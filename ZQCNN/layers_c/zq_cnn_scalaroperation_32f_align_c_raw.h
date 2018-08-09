
void zq_cnn_scalaroperation_32f_align(
	float scalar,
	const float* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	float* out_tensor4D_data,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	zq_mm_type scalar_v = zq_mm_set1_ps(scalar);
	int n, h, w, c;
	const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;

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
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
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
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;

					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_8 == 0)
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
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;
						zq_mm_store_ps(out_c_ptr, zq_mm_operation_ps(zq_mm_load_ps(in_c_ptr), scalar_v));
						in_c_ptr += zq_mm_align_size; out_c_ptr += zq_mm_align_size;

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
	float scalar,
	float* in_tensor4D_data,	
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep
)
{
	zq_mm_type scalar_v = zq_mm_set1_ps(scalar);
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

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
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;

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
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;

					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_8 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
					{
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;
						zq_mm_store_ps(c_ptr, zq_mm_operation_ps(zq_mm_load_ps(c_ptr), scalar_v));
						c_ptr += zq_mm_align_size;

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