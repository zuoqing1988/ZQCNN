/* it is safe to use out_tensor4D_data = in_tensor4D_data */
void zq_cnn_lrn_across_channels_32f_align(
	int local_size,
	float alpha,
	float beta,
	float k,								// k must be odd number
	const float* in_tensor4D_data,
	int N,
	int H,
	int W,
	int C,
	int in_pixelStep,
	int in_widthStep,
	int in_sliceStep,
	float* out_tensor4D_data,
	int out_pixStep,
	int out_widthStep,
	int out_sliceStep
)
{
	const float* in_slice_ptr, *in_row_ptr, *in_pix_ptr, *in_c_ptr;
	float* out_slice_ptr, *out_row_ptr, *out_pix_ptr, *out_c_ptr;
	int n, h, w, c, pad_size, len;
	float* square_buf, *accumulate_buf, *local_sum_buf, *square_ptr,*acc_ptr, *local_ptr;
	float alpha_div_local_size = alpha / (float)local_size;
	register zq_mm_type data_v, sum_v,pow_v;
	register zq_mm_type minus_beta_v = zq_mm_set1_ps(-beta);
	register zq_mm_type alpha_div_local_size_v = zq_mm_set1_ps(alpha_div_local_size);
	register zq_mm_type k_v = zq_mm_set1_ps(k);

	// 审计修复 2026-10-02（audit_k3_20261001.md 附录 AX.2）
	// -------------------------------------------------------
	// 原来第二行是**向下**取整到 align 的倍数。当 local_size == 1 时
	// p0 = 0/2 + align - 1 = align-1 < align，于是 pad_size 变成 **0**，
	// len == C，而下面那个「c += align、每次 zq_mm_store_ps 写 align 个 float」
	// 的循环最后一下会写到 square_buf[ceil(C/align)*align - 1] ——
	// C % align != 0 时这**越过 len**。
	//
	// ASan 实测（tools/zq_lrn_check.cpp，默认构建是 AVX2、align=8）：
	//   C=1, local_size=1 -> square_buf 只有 4 字节，_mm256_store_ps 写 32 字节
	//   ERROR: AddressSanitizer: heap-buffer-overflow
	//          WRITE of size 32 at ... zq_cnn_lrn_32f_align_c_raw.h:64
	//
	// 可达性：ZQ_CNN_Forward_SSEUtils::LRN_across_channels 只校验
	// `local_size % 2 != 1`，local_size == 1 通过；local_size 来自模型文件
	// （.zqparams）里不可信的 `local_size=` 那一行。
	//
	// 修法：pad_size 至少取一个 align。这样 `ceil(C/align)*align <= C + pad_size`
	// 在 align=4/8/16 下都恒成立，而且**数值结果完全不变** —— 多出来的那些
	// pad 元素被下面的初始化循环填 0，而窗口 [pad-L/2, pad+L/2] 是随 pad
	// 整体平移的，累加的相对区间没变。
	pad_size = local_size / 2 + zq_mm_align_size - 1;
	pad_size = pad_size - pad_size%zq_mm_align_size;
	if (pad_size < zq_mm_align_size)
		pad_size = zq_mm_align_size;
	len = C + (pad_size << 1);
	square_buf = (float*)_aligned_malloc(sizeof(float)*len,zq_mm_align_size*sizeof(float));
	accumulate_buf = (float*)_aligned_malloc(sizeof(float)*(len + 1),zq_mm_align_size*sizeof(float));
	// 审计修复 2026-10-02（附录 AX.2）：local_sum_buf 原来只给 C 个 float，
	// 但下面 `zq_mm_load_ps(local_ptr)` / `zq_mm_store_ps(out_c_ptr)` 都以
	// **zq_mm_align_size** 为步长、每次读/写 align 个 float，C % align != 0 时
	// 最后一下会越过 C-1。ASan 实测：
	//   ERROR: AddressSanitizer: heap-buffer-overflow
	//          READ of size 32 at ... zq_cnn_lrn_32f_align_c_raw.h:102
	// 多给 align 个 float 就够（多出来的部分读到什么算什么，写出去的是
	// local_sum_buf 的补零值，落在 out 像素的对齐空隙里，与标量结果一致）。
	local_sum_buf = (float*)_aligned_malloc(sizeof(float)*(C + zq_mm_align_size), zq_mm_align_size * sizeof(float));

	// 审计修复 2026-10-05（附录 IX.6）：三次分配**一次都没查返回值**，
	// 紧接着的 `accumulate_buf[0] = 0;` 就是空指针解引用。
	// `C` / `local_size` 都来自模型文件（不可信输入），一个巨大的通道数
	// 或 local_size 就能让分配失败。
	if (square_buf == 0 || accumulate_buf == 0 || local_sum_buf == 0)
	{
		if (square_buf) _aligned_free(square_buf);
		if (accumulate_buf) _aligned_free(accumulate_buf);
		if (local_sum_buf) _aligned_free(local_sum_buf);
		return;
	}

	accumulate_buf[0] = 0;
	for (c = 0; c < pad_size; c++)
	{
		square_buf[c] = 0;
		square_buf[len - 1 - c] = 0;
		accumulate_buf[c + 1] = 0;
	}


	for (n = 0, in_slice_ptr = in_tensor4D_data, out_slice_ptr = out_tensor4D_data;
		n < N;
		n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		for (h = 0, in_row_ptr = in_slice_ptr, out_row_ptr = out_slice_ptr;
			h < H;
			h++, in_row_ptr += in_widthStep, out_row_ptr += out_widthStep)
		{
			for (w = 0, in_pix_ptr = in_row_ptr, out_pix_ptr = out_row_ptr;
				w < W;
				w++, in_pix_ptr += in_pixelStep, out_pix_ptr += out_pixStep)
			{
				//compute x^2
				for (c = 0, in_c_ptr = in_pix_ptr,square_ptr = square_buf+pad_size; c < C; 
					c+=zq_mm_align_size, in_c_ptr+=zq_mm_align_size,square_ptr+=zq_mm_align_size)
				{
					data_v = zq_mm_load_ps(in_c_ptr);
					zq_mm_store_ps(square_ptr, zq_mm_mul_ps(data_v, data_v));
				}
				//compute accumulate
				for (c = pad_size; c < len; c++)
					accumulate_buf[c + 1] = accumulate_buf[c] + square_buf[c];

				//compute local sum
				for (c = 0, acc_ptr = accumulate_buf+pad_size-(local_size/2); c < C; c++,acc_ptr++)
				{
					local_sum_buf[c] = acc_ptr[local_size] - acc_ptr[0];
				}
				for (c = 0, in_c_ptr = in_pix_ptr, out_c_ptr = out_pix_ptr, local_ptr = local_sum_buf; 
					c < C; 
					c+=zq_mm_align_size, in_c_ptr+=zq_mm_align_size, out_c_ptr+=zq_mm_align_size,local_ptr+=zq_mm_align_size)
				{
					sum_v = zq_mm_load_ps(local_ptr);
					data_v = zq_mm_load_ps(in_c_ptr);
					sum_v = zq_mm_fmadd_ps(sum_v, alpha_div_local_size_v, k_v);
					pow_v = zq_mm_exp_ps(zq_mm_mul_ps(minus_beta_v, zq_mm_log_ps(sum_v)));
					data_v = zq_mm_mul_ps(data_v, pow_v);
					zq_mm_store_ps(out_c_ptr, data_v);
				}
			}
		}
	}

	_aligned_free(square_buf);
	_aligned_free(accumulate_buf);
	_aligned_free(local_sum_buf);
}