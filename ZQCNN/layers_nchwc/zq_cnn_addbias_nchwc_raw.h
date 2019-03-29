void zq_cnn_addbias_nchwc(
	zq_base_type* in_tensor4D_data,	// in & out
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* bias_data
)
{
	zq_mm_type bias_v;
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;

	for (n = 0, im_ptr = in_tensor4D_data; n < in_N; n++, im_ptr += in_imStep)
	{
		for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
		{
			bias_v = zq_mm_load_ps(bias_data+c);
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(pix_ptr, zq_mm_add_ps(zq_mm_load_ps(pix_ptr), bias_v));
				}
			}
		}
	}
}
