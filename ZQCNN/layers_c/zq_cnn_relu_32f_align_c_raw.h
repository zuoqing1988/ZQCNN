
/*
y = max(0,x)
*/
void zq_cnn_relu_32f_align(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	float slope
)
{

	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	if (slope == 0)
	{
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
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
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
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
		else if (in_C % zq_mm_align_size_mul_8 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
							c_ptr += zq_mm_align_size;
							zq_mm_store_ps(c_ptr, zq_mm_max_ps(zq_mm_setzero_ps(), zq_mm_load_ps(c_ptr)));
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
		register zq_mm_type slope_v = zq_mm_set1_ps(slope);
		register zq_mm_type value_v;
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
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size; 
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
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
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
		else if (in_C % zq_mm_align_size_mul_8 == 0)
		{
			for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size_mul_8)
						{
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
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
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size)
						{
							value_v = zq_mm_load_ps(c_ptr);
							zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
							c_ptr += zq_mm_align_size;
						}
					}
				}
			}
		}
	}
}