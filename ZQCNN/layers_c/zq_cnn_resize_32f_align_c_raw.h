void zq_cnn_resize_with_safeborder(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	int in_off_x,
	int in_off_y,
	int in_rect_width,
	int in_rect_height,
	zq_base_type* out_tensor4D_data,
	int out_H,
	int out_W,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int* x0 = (int*)malloc(sizeof(int)*(out_W));
	int* x1 = (int*)malloc(sizeof(int)*(out_W));
	zq_base_type* sx = (zq_base_type*)malloc(sizeof(zq_base_type)*(out_W));
	zq_base_type src_H = in_rect_height;
	zq_base_type src_W = in_rect_width;
	zq_base_type w_step = 1.0f / (zq_base_type)out_W*src_W;
	zq_base_type h_step = 1.0f / (zq_base_type)out_H*src_H;
	zq_base_type coord_y_ini = 0.5f * h_step - 0.5f + (zq_base_type)in_off_y;
	zq_base_type coord_x_ini = 0.5f*w_step - 0.5f + (zq_base_type)in_off_x;
	zq_base_type x0_f, y0_f;
	int y0, y1;
	zq_base_type sy;
	zq_base_type coord_x, coord_y;
	const zq_base_type* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	int n, h, w, c, cur_x0, cur_x1;
	zq_mm_type cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;


	/*********** compute the map and weight begin ************/
	// coord_x
	coord_x = coord_x_ini;
	for (w = 0; w < out_W; w++, coord_x += w_step)
	{
		x0_f = floor(coord_x);
		x0[w] = (int)x0_f;
		x1[w] = x0[w] + 1;
		sx[w] = coord_x - x0_f;
		x0[w] *= in_pixelStep;
		x1[w] *= in_pixelStep;
	}

	/*********** compute the map and weight end ************/



	if (in_C <= zq_mm_align_size)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr, sum);	
				}
			}
		}
	}
	else if (in_C <= (zq_mm_align_size<<1)) //*2
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
				}
			}
		}
	}
	else if (in_C <= ((zq_mm_align_size << 1) + 1)) //*3
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
				}
			}
		}
	}
	else if (in_C <= (zq_mm_align_size << 2)) //*4
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
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

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c += zq_mm_align_size, cur_x0 += zq_mm_align_size, cur_x1 += zq_mm_align_size)
					{
						v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
						dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
						result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
						v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
						dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
						result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
						dy = zq_mm_sub_ps(result1, result0);
						sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
						zq_mm_store_ps(out_pix_ptr + c, sum);
					}
				}
			}
		}
	}

	free(x0);
	free(x1);
	free(sx);
}


void zq_cnn_resize_without_safeborder(
	const zq_base_type* in_tensor4D_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	int in_off_x,
	int in_off_y,
	int in_rect_width,
	int in_rect_height,
	zq_base_type* out_tensor4D_data,
	int out_H,
	int out_W,
	int out_pixelStep,
	int out_widthStep,
	int out_sliceStep
)
{
	int* x0 = (int*)malloc(sizeof(int)*(out_W));
	int* x1 = (int*)malloc(sizeof(int)*(out_W));
	zq_base_type* sx = (zq_base_type*)malloc(sizeof(zq_base_type)*(out_W));
	zq_base_type src_H = in_rect_height;
	zq_base_type src_W = in_rect_width;
	zq_base_type w_step = 1.0f / (zq_base_type)out_W*src_W;
	zq_base_type h_step = 1.0f / (zq_base_type)out_H*src_H;
	zq_base_type coord_y_ini = 0.5f * h_step - 0.5f + (zq_base_type)in_off_y;
	zq_base_type coord_x_ini = 0.5f*w_step - 0.5f + (zq_base_type)in_off_x;
	zq_base_type x0_f, y0_f;
	int y0, y1;
	zq_base_type sy;
	zq_base_type coord_x, coord_y;
	const zq_base_type* in_slice_ptr, *in_row0_ptr, *in_row1_ptr;
	zq_base_type* out_slice_ptr, *out_row_ptr, *out_pix_ptr;
	int n, h, w, c, cur_x0, cur_x1;
	zq_mm_type cur_sx, v00, dx0, result0, v10, dx1, result1, dy, sum;


	/*********** compute the map and weight begin ************/
	// coord_x
	coord_x = coord_x_ini;
	for (w = 0; w < out_W; w++, coord_x += w_step)
	{
		x0_f = floor(coord_x);
		x0[w] = (int)x0_f;
		x1[w] = x0[w] + 1;
		sx[w] = coord_x - x0_f;
		x0[w] = __min(in_W - 1, __max(0, x0[w]));
		x1[w] = __min(in_W - 1, __max(0, x1[w]));
		x0[w] *= in_pixelStep;
		x1[w] *= in_pixelStep;
	}

	/*********** compute the map and weight end ************/



	if (in_C <= zq_mm_align_size)
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr, sum);
				}
			}
		}
	}
	else if (in_C <= (zq_mm_align_size << 1)) //*2
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
				}
			}
		}
	}
	else if (in_C <= ((zq_mm_align_size << 1) + 1)) //*3
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
				}
			}
		}
	}
	else if (in_C <= (zq_mm_align_size << 2)) //*4
	{
		for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
			n < in_N;
			n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
		{

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					cur_x0 = x0[w], cur_x1 = x1[w];
					c = 0, cur_x0 = x0[w], cur_x1 = x1[w];

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);

					c += zq_mm_align_size; cur_x0 += zq_mm_align_size; cur_x1 += zq_mm_align_size;

					v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
					dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
					result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
					v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
					dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
					result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
					dy = zq_mm_sub_ps(result1, result0);
					sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
					zq_mm_store_ps(out_pix_ptr + c, sum);
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

			coord_y = coord_y_ini;
			for (h = 0, out_row_ptr = out_slice_ptr;
				h < out_H;
				h++, coord_y += h_step, out_row_ptr += out_widthStep)
			{
				y0_f = floor(coord_y);
				y0 = (int)y0_f;
				y1 = y0 + 1;
				sy = coord_y - y0_f;
				y0 = __min(in_H, __max(0, y0));
				y1 = __min(in_H, __max(0, y1));

				in_row0_ptr = in_slice_ptr + y0*in_widthStep;
				in_row1_ptr = in_slice_ptr + y1*in_widthStep;

				for (w = 0, out_pix_ptr = out_row_ptr; w < out_W; w++, out_pix_ptr += out_pixelStep)
				{
					cur_sx = zq_mm_set1_ps(sx[w]);
					for (c = 0, cur_x0 = x0[w], cur_x1 = x1[w]; c < in_C; c += zq_mm_align_size, cur_x0 += zq_mm_align_size, cur_x1 += zq_mm_align_size)
					{
						v00 = zq_mm_load_ps(in_row0_ptr + cur_x0);
						dx0 = zq_mm_sub_ps(zq_mm_load_ps(in_row0_ptr + cur_x1), v00);
						result0 = zq_mm_add_ps(v00, zq_mm_mul_ps(dx0, cur_sx));
						v10 = zq_mm_load_ps(in_row1_ptr + cur_x0);
						dx1 = zq_mm_sub_ps(zq_mm_load_ps(in_row1_ptr + cur_x1), v10);
						result1 = zq_mm_add_ps(v10, zq_mm_mul_ps(dx1, cur_sx));
						dy = zq_mm_sub_ps(result1, result0);
						sum = zq_mm_add_ps(result0, zq_mm_mul_ps(dy, zq_mm_set1_ps(sy)));
						zq_mm_store_ps(out_pix_ptr + c, sum);
					}
				}
			}
		}
	}

	free(x0);
	free(x1);
	free(sx);
}