
/*
y = max(0,x)+a*min(0,x)
*/
void zq_cnn_prelu_32f_align(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* slope_data
)
{
	zq_mm_type value_v;
	zq_mm_type slope_v;
	int n, h, w, c;
	const float *slope_c_ptr;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;

	if (in_C % zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_32)
					{	
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_16)
					{
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_8)
					{
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size)
					{
						value_v = zq_mm_load_ps(c_ptr);
						slope_v = zq_mm_load_ps(slope_c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
					}
				}
			}
		}
	}
}

/*
y = max(0,x)+a*min(0,x)
*/
void zq_cnn_prelu_32f_align_omp(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* slope_data,
	int thread_count
)
{
	int n, h, w;
	float* slice_ptr, *row_ptr, *pix_ptr;
	int NHW = in_N*in_H*in_W;
	int chunk_size = (NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(sizeof(int)*NHW);
	int idx = 0;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				in_offsets[idx] = n*in_sliceStep + h*in_widthStep + w*in_pixelStep;
				idx++;
			}
		}
	}

	if (in_C % zq_mm_align_size_mul_32 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type value_v;
			zq_mm_type slope_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_32)
			{
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else if (in_C % zq_mm_align_size_mul_16 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type value_v;
			zq_mm_type slope_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_16)
			{
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else if (in_C % zq_mm_align_size_mul_8 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type value_v;
			zq_mm_type slope_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_8)
			{
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type value_v;
			zq_mm_type slope_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size)
			{
				value_v = zq_mm_load_ps(c_ptr);
				slope_v = zq_mm_load_ps(slope_c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_fmadd_ps(slope_v, zq_mm_min_ps(zq_mm_setzero_ps(), value_v), zq_mm_max_ps(zq_mm_setzero_ps(), value_v)));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	free(in_offsets);
}

void zq_cnn_prelu_32f_align_sure_slope_lessthan1(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* slope_data
)
{
	zq_mm_type data_v;
	int n, h, w, c;
	float* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const float* slope_c_ptr;

#if 1
	if (in_C % zq_mm_align_size_mul_32 == 0)
	{
		for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					for (c = 0, c_ptr = pix_ptr,slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_32)
					{
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr,slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_16)
					{
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr,slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_8)
					{
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
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
					for (c = 0, c_ptr = pix_ptr,slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size)
					{
						data_v = zq_mm_load_ps(c_ptr);
						zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
						c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
					}
				}
			}
		}
	}
	
#else

	for (c = 0, c_ptr = in_tensor4D_data; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
	{
		slope_v = zq_mm_load_ps(slope_data + c);

		for (n = 0, slice_ptr = c_ptr; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
				{
					data_v = zq_mm_load_ps(pix_ptr);
					data_v = zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, slope_v));
					zq_mm_store_ps(pix_ptr, data_v);
				}
			}
		}
	}
#endif
}

void zq_cnn_prelu_32f_align_sure_slope_lessthan1_omp(
	float* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	const float* slope_data,
	int thread_count
)
{
	int n, h, w;
	float* slice_ptr, *row_ptr, *pix_ptr;
	int NHW = in_N*in_H*in_W;
	int chunk_size = (NHW + thread_count - 1) / thread_count;
	int* in_offsets = (int*)malloc(sizeof(int)*NHW);
	int idx = 0;
	for (n = 0, slice_ptr = in_tensor4D_data; n < in_N; n++, slice_ptr += in_sliceStep)
	{
		for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
		{
			for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixelStep)
			{
				in_offsets[idx] = n*in_sliceStep + h*in_widthStep + w*in_pixelStep;
				idx++;
			}
		}
	}


	if (in_C % zq_mm_align_size_mul_32 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type data_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_32)
			{
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else if (in_C % zq_mm_align_size_mul_16 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type data_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_16)
			{
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else if (in_C % zq_mm_align_size_mul_8 == 0)
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type data_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size_mul_8)
			{
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	else
	{
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_count)
		for (idx = 0; idx < NHW; idx++)
		{
			zq_mm_type data_v;
			int c;
			float* pix_ptr = in_tensor4D_data + in_offsets[idx];
			float* c_ptr;
			const float* slope_c_ptr;
			for (c = 0, c_ptr = pix_ptr, slope_c_ptr = slope_data; c < in_C; c += zq_mm_align_size)
			{
				data_v = zq_mm_load_ps(c_ptr);
				zq_mm_store_ps(c_ptr, zq_mm_max_ps(data_v, zq_mm_mul_ps(data_v, zq_mm_load_ps(slope_c_ptr))));
				c_ptr += zq_mm_align_size; slope_c_ptr += zq_mm_align_size;
			}
		}
	}
	free(in_offsets);
}