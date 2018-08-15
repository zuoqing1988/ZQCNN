void zq_cnn_normalize_not_across_spatial_32f_align(
	int channel_shared,
	float* in_tensor4D_data,	// in & out
	const float* scale_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float eps
)
{
	zq_mm_type sum_v;
	__declspec(align(32)) float q[8];
	float sum;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr, *scale_c_ptr;

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
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q + eps);

					if (channel_shared)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
						}
					}
					else
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_32)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
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
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q+eps);

					if (channel_shared)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
						}
					}
					else
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_16)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
						}
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
					sum_v = zq_mm_setzero_ps();
					for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
					{
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
					zq_mm_store_ps(q, sum_v);
					sum = 1.0f / sqrt(zq_final_sum_q+eps);

					if (channel_shared)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
						}
					}
					else
					{
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_8)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
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
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), zq_mm_set1_ps(sum*scale_data[0])));
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
	float* in_tensor4D_data,	// in & out
	const float* scale_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float eps
)
{
	zq_mm_type sum_v;
	__declspec(align(32)) float q[8];
	float sum;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr, *scale_c_ptr;
	
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
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_32)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
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
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_32)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
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
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_16)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
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
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_16)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
	}
	else if (in_C%zq_mm_align_size_mul_8 == 0)
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
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
						sum_v = zq_mm_fmadd_ps(zq_mm_load_ps(c_ptr), zq_mm_load_ps(c_ptr), sum_v);
						c_ptr += zq_mm_align_size;
					}
				}
			}
			zq_mm_store_ps(q, sum_v);
			sum = 1.0f / sqrt(zq_final_sum_q+eps);

			if (channel_shared)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_set1_ps(sum*scale_data[0])));
							c_ptr += zq_mm_align_size;
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
						for (c = 0, c_ptr = pix_ptr, scale_c_ptr = scale_data; c < in_C; c += zq_mm_align_size_mul_8)
						{
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_mul_ps(zq_mm_load_ps(c_ptr), zq_mm_mul_ps(zq_mm_set1_ps(sum), zq_mm_load_ps(scale_data))));
							c_ptr += zq_mm_align_size; scale_c_ptr += zq_mm_align_size;
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
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							zq_mm_store_ps(c_ptr + c, zq_mm_mul_ps(zq_mm_load_ps(c_ptr + c), zq_mm_set1_ps(sum*scale_data[0])));
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

