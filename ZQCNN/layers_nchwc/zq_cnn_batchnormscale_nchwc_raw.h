/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnormscale_mean_var_scale_bias_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type* slope_data,
	const zq_base_type* bias_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	int ceil_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	a = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	if (a == NULL || b == NULL)
	{
		/* 审计 2026-10-02（附录 AY）：_aligned_malloc 失败时原来直接解引用，
		   而 in_C / ceil_C 来自模型文件（不可信输入）—— 一个巨大的通道数
		   就能让分配失败。MSVC /analyze 的 C6011「取消对 NULL 指针 a/b 的引用」
		   报的正是这 4 对（共 24 处，去重后 8 个分配点）。
		   函数返回 void，失败时只能释放兄弟再返回；正常路径行为一字不变。 */
		if (a) _aligned_free(a);
		if (b) _aligned_free(b);
		return;
	}
	/* 审计修复 2026-10-02（附录 AY.2）：上界原来用 ceil_C，
	   而 slope_data / var_data / mean_data / bias_data 是**每通道一个 float**
	   的模型参数，长度只有 in_C。in_C 不是 zq_mm_align_size 的倍数时，
	   四个数组各被多读 zq_mm_align_size-1 个 float —— ASan 实测：
	     ERROR: AddressSanitizer: heap-buffer-overflow READ of size 4
	        #1 zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4  
	             zq_cnn_batchnormscale_nchwc_raw.h:40
	   in_C 来自模型文件（不可信输入），所以这是可被模型触发的堆越界读。
	   下面 a/b 仍然按 ceil_C 填满（主内核是整宽读），
	   但 [in_C, ceil_C) 那段**只能清零**——那里根本没有模型参数。 */
	for (c = 0; c < in_C; c++)
	{
		b[c] = slope_data[c] / (float)sqrt(__max(var_data[c] + eps, FLOAT_EPS_FOR_DIV));
		a[c] = bias_data[c] - mean_data[c] * b[c];
	}
	for (c = in_C; c < ceil_C; c++)
	{
		a[c] = 0;
		b[c] = 0;
	}

	zq_cnn_batchnorm_b_a_nchwc(in_data, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



/*
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_mean_var_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	int ceil_C = (in_C + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	a = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(ceil_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	if (a == NULL || b == NULL)
	{
		/* 审计 2026-10-02（附录 AY）：_aligned_malloc 失败时原来直接解引用，
		   而 in_C / ceil_C 来自模型文件（不可信输入）—— 一个巨大的通道数
		   就能让分配失败。MSVC /analyze 的 C6011「取消对 NULL 指针 a/b 的引用」
		   报的正是这 4 对（共 24 处，去重后 8 个分配点）。
		   函数返回 void，失败时只能释放兄弟再返回；正常路径行为一字不变。 */
		if (a) _aligned_free(a);
		if (b) _aligned_free(b);
		return;
	}
	/* 审计修复 2026-10-02（附录 AY.2）：上界原来用 ceil_C，
	   而 slope_data / var_data / mean_data / bias_data 是**每通道一个 float**
	   的模型参数，长度只有 in_C。in_C 不是 zq_mm_align_size 的倍数时，
	   四个数组各被多读 zq_mm_align_size-1 个 float —— ASan 实测：
	     ERROR: AddressSanitizer: heap-buffer-overflow READ of size 4
	        #1 zq_cnn_batchnormscale_mean_var_scale_bias_nchwc4  
	             zq_cnn_batchnormscale_nchwc_raw.h:40
	   in_C 来自模型文件（不可信输入），所以这是可被模型触发的堆越界读。
	   下面 a/b 仍然按 ceil_C 填满（主内核是整宽读），
	   但 [in_C, ceil_C) 那段**只能清零**——那里根本没有模型参数。 */
	for (c = 0; c < in_C; c++)
	{
		b[c] = 1.0f / (float)sqrt(__max(var_data[c] + eps, FLOAT_EPS_FOR_DIV));
		a[c] = -mean_data[c] * b[c];
	}
	for (c = in_C; c < ceil_C; c++)
	{
		a[c] = 0;
		b[c] = 0;
	}

	zq_cnn_batchnorm_b_a_nchwc(in_data, in_N, in_H, in_W, in_C, in_widthStep, in_sliceStep, in_imStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



void zq_cnn_scale_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* scale_data,
	const zq_base_type* bias_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type scale_vec, bias_vec;


	if (bias_data != NULL)
	{
		for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						bias_vec = zq_mm_load_ps(bias_data + c);
						zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(pix_ptr), scale_vec, bias_vec));
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
		{
			for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
			{
				for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
				{
					for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
					{
						scale_vec = zq_mm_load_ps(scale_data + c);
						zq_mm_store_ps(pix_ptr, zq_mm_mul_ps(zq_mm_load_ps(pix_ptr), scale_vec));
					}
				}
			}
		}
	}
}

/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
OR
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_b_a_nchwc(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* b_data,
	const zq_base_type* a_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *im_ptr;
	register zq_mm_type a_vec, b_vec;
	
	for (n = 0, im_ptr = in_data; n < in_N; n++, im_ptr += in_imStep)
	{
		for (c = 0, slice_ptr = im_ptr; c < in_C; c += zq_mm_align_size, slice_ptr += in_sliceStep)
		{
			a_vec = zq_mm_load_ps(a_data + c);
			b_vec = zq_mm_load_ps(b_data + c);
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += zq_mm_align_size)
				{
					zq_mm_store_ps(pix_ptr, zq_mm_fmadd_ps(zq_mm_load_ps(pix_ptr), b_vec, a_vec));
				}
			}
		}
	}
}

