/*
a = bias - slope * mean / sqrt(var+eps)
b = slope / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnormscale_32f_mean_var_scale_bias_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type* slope_data,
	const zq_base_type* bias_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	a = (zq_base_type*)_aligned_malloc(in_C*sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(in_C*sizeof(zq_base_type), (zq_mm_align_size << 2));
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
	for (c = 0; c < in_C; c++)
	{
		b[c] = (float)(slope_data[c] / sqrt(__max(var_data[c]+eps,FLOAT_EPS_FOR_DIV)));
		a[c] = bias_data[c] - mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



/*
a = - mean / sqrt(var+eps)
b = 1 / sqrt(var+eps)
value = b * value + a
*/
void zq_cnn_batchnorm_32f_mean_var_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* mean_data,
	const zq_base_type* var_data,
	const zq_base_type eps
)
{
	zq_base_type* a, *b;
	int c;
	a = (zq_base_type*)_aligned_malloc(in_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
	b = (zq_base_type*)_aligned_malloc(in_C * sizeof(zq_base_type), (zq_mm_align_size << 2));
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
	for (c = 0; c < in_C; c++)
	{
		b[c] = 1.0f / (float)sqrt(__max(var_data[c]+eps, FLOAT_EPS_FOR_DIV));
		a[c] = -mean_data[c] * b[c];
	}

	zq_cnn_batchnorm_32f_b_a_align(in_data, in_N, in_H, in_W, in_C, in_pixStep, in_widthStep, in_sliceStep, (const zq_base_type*)b, (const zq_base_type*)a);

	_aligned_free(a);
	_aligned_free(b);
}



void zq_cnn_scale_32f_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* scale_data,
	const zq_base_type* bias_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	register zq_mm_type scale_vec, bias_vec;


	if (bias_data != NULL)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					/* 审计修复 2026-10-05（audit_k3_20261001.md 附录 IB）
					 * -----------------------------------------------------
					 * `scale` / `bias` 两个张量都是 `ChangeSize(1,1,1,in_C,0,0)`
					 * —— **只有 in_C 个 float**。下面这一支原来无条件走整向量，
					 * 于是 in_C 不是 4/8 的倍数时最后一次
					 * `zq_mm_load_ps(scale_data + c)` / `(bias_data + c)`
					 * 读过 C-1。
					 *
					 * ASan 实测（tools/zq_scale_overread_check.cpp，默认构建 AVX2、align=8）：
					 *   ERROR: AddressSanitizer: heap-buffer-overflow
					 *          READ of size 32 ... in _mm256_load_ps
					 *          #1 zq_cnn_scale_32f_align256bit
					 *             ZQCNN/layers_c/zq_cnn_batchnormscale_32f_align_c_raw.h:127
					 *   0 bytes to the right of 12-byte region       <- scale 的 3 个 float
					 *
					 * 可达性：`ZQ_CNN_Layer_Scale` 的 `in_C` 来自 bottom blob 的通道数，
					 * 而 C 完全可以不是 4/8 的倍数。
					 *
					 * 下面 `else` 那一支（不带 bias）**早就修过**，
					 * 注释就写在旁边 —— 也就是说**只修了一条分支，另一条留着**。
					 * 现在两条用同一个判据：in_C 是整倍数才走向量，否则走标量。
					 * 数值结果**完全不变**（标量版本来就是同一组乘法加法）。
					 */
					if (in_C % zq_mm_align_size == 0)
					{
						for (c = 0, c_ptr = pix_ptr; c < in_C; c += zq_mm_align_size, c_ptr += zq_mm_align_size)
						{
							scale_vec = zq_mm_load_ps(scale_data + c);
							bias_vec = zq_mm_load_ps(bias_data + c);
							zq_mm_store_ps(c_ptr, zq_mm_add_ps(zq_mm_mul_ps(zq_mm_load_ps(c_ptr), scale_vec), bias_vec));
						}
					}
					else
					{
						for (c = 0; c < in_C; c++)
						{
							pix_ptr[c] = pix_ptr[c] * scale_data[c] + bias_data[c];
						}
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					/* in_C 不是 4/8 的倍数: 整向量读会越过 scale_data 的分配, 改走标量 */
					for (c = 0; c < in_C; c++)
					{
						pix_ptr[c] = pix_ptr[c] * scale_data[c];
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
void zq_cnn_batchnorm_32f_b_a_align(
	zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_pixStep,
	int in_widthStep,
	int in_sliceStep,
	const zq_base_type* b_data,
	const zq_base_type* a_data
)
{
	int n, h, w, c;
	zq_base_type* slice_ptr, *row_ptr, *pix_ptr, *c_ptr;
	const zq_base_type* a_ptr, *b_ptr;
	register zq_mm_type a_vec0, a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7;
	register zq_mm_type b_vec0, b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7;
	register zq_mm_type c_vec0, c_vec1, c_vec2, c_vec3, c_vec4, c_vec5, c_vec6, c_vec7;

	if (in_C % (zq_mm_align_size8) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr;
						c < in_C;
						c += zq_mm_align_size8, c_ptr += zq_mm_align_size8, a_ptr += zq_mm_align_size8, b_ptr += zq_mm_align_size8)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						a_vec2 = zq_mm_load_ps(a_ptr + zq_mm_align_size2);
						a_vec3 = zq_mm_load_ps(a_ptr + zq_mm_align_size3);
						a_vec4 = zq_mm_load_ps(a_ptr + zq_mm_align_size4);
						a_vec5 = zq_mm_load_ps(a_ptr + zq_mm_align_size5);
						a_vec6 = zq_mm_load_ps(a_ptr + zq_mm_align_size6);
						a_vec7 = zq_mm_load_ps(a_ptr + zq_mm_align_size7);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						b_vec2 = zq_mm_load_ps(b_ptr + zq_mm_align_size2);
						b_vec3 = zq_mm_load_ps(b_ptr + zq_mm_align_size3);
						b_vec4 = zq_mm_load_ps(b_ptr + zq_mm_align_size4);
						b_vec5 = zq_mm_load_ps(b_ptr + zq_mm_align_size5);
						b_vec6 = zq_mm_load_ps(b_ptr + zq_mm_align_size6);
						b_vec7 = zq_mm_load_ps(b_ptr + zq_mm_align_size7);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec2 = zq_mm_load_ps(c_ptr + zq_mm_align_size2);
						c_vec3 = zq_mm_load_ps(c_ptr + zq_mm_align_size3);
						c_vec4 = zq_mm_load_ps(c_ptr + zq_mm_align_size4);
						c_vec5 = zq_mm_load_ps(c_ptr + zq_mm_align_size5);
						c_vec6 = zq_mm_load_ps(c_ptr + zq_mm_align_size6);
						c_vec7 = zq_mm_load_ps(c_ptr + zq_mm_align_size7);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						c_vec2 = zq_mm_fmadd_ps(c_vec2, b_vec2, a_vec2);
						c_vec3 = zq_mm_fmadd_ps(c_vec3, b_vec3, a_vec3);
						c_vec4 = zq_mm_fmadd_ps(c_vec4, b_vec4, a_vec4);
						c_vec5 = zq_mm_fmadd_ps(c_vec5, b_vec5, a_vec5);
						c_vec6 = zq_mm_fmadd_ps(c_vec6, b_vec6, a_vec6);
						c_vec7 = zq_mm_fmadd_ps(c_vec7, b_vec7, a_vec7);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
						zq_mm_store_ps(c_ptr + zq_mm_align_size2, c_vec2);
						zq_mm_store_ps(c_ptr + zq_mm_align_size3, c_vec3);
						zq_mm_store_ps(c_ptr + zq_mm_align_size4, c_vec4);
						zq_mm_store_ps(c_ptr + zq_mm_align_size5, c_vec5);
						zq_mm_store_ps(c_ptr + zq_mm_align_size6, c_vec6);
						zq_mm_store_ps(c_ptr + zq_mm_align_size7, c_vec7);
					}
				}
			}
		}
	}
	else if (in_C % (zq_mm_align_size4) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr; 
						c < in_C; 
						c += zq_mm_align_size4, c_ptr += zq_mm_align_size4, a_ptr += zq_mm_align_size4, b_ptr += zq_mm_align_size4)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						a_vec2 = zq_mm_load_ps(a_ptr + zq_mm_align_size2);
						a_vec3 = zq_mm_load_ps(a_ptr + zq_mm_align_size3);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						b_vec2 = zq_mm_load_ps(b_ptr + zq_mm_align_size2);
						b_vec3 = zq_mm_load_ps(b_ptr + zq_mm_align_size3);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec2 = zq_mm_load_ps(c_ptr + zq_mm_align_size2);
						c_vec3 = zq_mm_load_ps(c_ptr + zq_mm_align_size3);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						c_vec2 = zq_mm_fmadd_ps(c_vec2, b_vec2, a_vec2);
						c_vec3 = zq_mm_fmadd_ps(c_vec3, b_vec3, a_vec3);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
						zq_mm_store_ps(c_ptr + zq_mm_align_size2, c_vec2);
						zq_mm_store_ps(c_ptr + zq_mm_align_size3, c_vec3);
					}
				}
			}
		}
	}
	else if (in_C % (zq_mm_align_size2) == 0)
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					for (c = 0, a_ptr = a_data, b_ptr = b_data, c_ptr = pix_ptr;
						c < in_C;
						c += zq_mm_align_size2, c_ptr += zq_mm_align_size2, a_ptr += zq_mm_align_size2, b_ptr += zq_mm_align_size2)
					{
						a_vec0 = zq_mm_load_ps(a_ptr);
						a_vec1 = zq_mm_load_ps(a_ptr + zq_mm_align_size);
						b_vec0 = zq_mm_load_ps(b_ptr);
						b_vec1 = zq_mm_load_ps(b_ptr + zq_mm_align_size);
						c_vec0 = zq_mm_load_ps(c_ptr);
						c_vec1 = zq_mm_load_ps(c_ptr + zq_mm_align_size);
						c_vec0 = zq_mm_fmadd_ps(c_vec0, b_vec0, a_vec0);
						c_vec1 = zq_mm_fmadd_ps(c_vec1, b_vec1, a_vec1);
						zq_mm_store_ps(c_ptr, c_vec0);
						zq_mm_store_ps(c_ptr + zq_mm_align_size, c_vec1);
					}
				}
			}
		}
	}
	else
	{
		for (n = 0, slice_ptr = in_data; n < in_N; n++, slice_ptr += in_sliceStep)
		{
			for (h = 0, row_ptr = slice_ptr; h < in_H; h++, row_ptr += in_widthStep)
			{
				for (w = 0, pix_ptr = row_ptr; w < in_W; w++, pix_ptr += in_pixStep)
				{
					/* in_C 不是 4/8 的倍数: 整向量读会越过 a/b 的分配(_aligned_malloc(2*4)),
					   这里改走标量, 只处理真实的 in_C 个元素 */
					for (c = 0; c < in_C; c++)
					{
						pix_ptr[c] = pix_ptr[c] * b_data[c] + a_data[c];
					}
				}
			}
		}
	}
}

