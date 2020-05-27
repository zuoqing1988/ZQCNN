void zq_cnn_lstm_TF_32f_align(
	const zq_base_type* in_data,
	int in_N,
	int in_W,
	int in_C,
	int in_pixelStep,
	int in_sliceStep,
	const zq_base_type* xc_I_data,
	int xc_I_pixelStep,
	int xc_I_sliceStep,
	const zq_base_type* xc_F_data,
	int xc_F_pixelStep,
	int xc_F_sliceStep,
	const zq_base_type* xc_O_data,
	int xc_O_pixelStep,
	int xc_O_sliceStep,
	const zq_base_type* xc_G_data,
	int xc_G_pixelStep,
	int xc_G_sliceStep,
	const zq_base_type* hc_I_data,
	int hc_I_pixelStep,
	int hc_I_sliceStep,
	const zq_base_type* hc_F_data,
	int hc_F_pixelStep,
	int hc_F_sliceStep,
	const zq_base_type* hc_O_data,
	int hc_O_pixelStep,
	int hc_O_sliceStep,
	const zq_base_type* hc_G_data,
	int hc_G_pixelStep,
	int hc_G_sliceStep,
	const zq_base_type* b_I_data,
	const zq_base_type* b_F_data,
	const zq_base_type* b_O_data,
	const zq_base_type* b_G_data,
	zq_base_type* out_data,
	int out_pixelStep,
	int out_sliceStep,
	int hidden_dim,
	int is_fw,
	zq_base_type forget_bias,
	zq_base_type cell_clip,
	void** buffer,
	__int64* buffer_len)
{
	register zq_mm_type v_x, v_h, v_I, v_F, v_O, v_G;
	zq_base_type buffer_I[zq_mm_align_size];
	zq_base_type buffer_F[zq_mm_align_size];
	zq_base_type buffer_O[zq_mm_align_size];
	zq_base_type buffer_G[zq_mm_align_size];
	int out_n, t, q, i, ti;
	const zq_base_type* in_slice_ptr, *x, *weight_xc_I, *weight_xc_F, *weight_xc_O, *weight_xc_G;
	const zq_base_type* weight_hc_I, *weight_hc_F, *weight_hc_O, *weight_hc_G;
	zq_base_type* out_slice_ptr, *out_pixel_ptr;
	zq_base_type* h, *cell, *cs, *I, *F, *cs_prev, *ci, *co, *o;
	int need_buffer_size = (hidden_dim*sizeof(zq_base_type) + 31) / 32 * 32;
	int total_need_buffer_size = need_buffer_size * 9;
	if (buffer == 0)
	{
		h = _aligned_malloc(need_buffer_size, 32);
		cell = _aligned_malloc(need_buffer_size, 32);
		cs = _aligned_malloc(need_buffer_size, 32);
		I = _aligned_malloc(need_buffer_size, 32);
		F = _aligned_malloc(need_buffer_size, 32);
		cs_prev = _aligned_malloc(need_buffer_size, 32);
		ci = _aligned_malloc(need_buffer_size, 32);
		co = _aligned_malloc(need_buffer_size, 32);
		o = _aligned_malloc(need_buffer_size, 32);
	}
	else
	{
		if (*buffer_len < total_need_buffer_size)
		{
			_aligned_free(*buffer);
			*buffer = _aligned_malloc(total_need_buffer_size, 32);
			*buffer_len = total_need_buffer_size;
		}
		h = (zq_base_type*)(*buffer);
		cell = (zq_base_type*)((char*)(*buffer) + need_buffer_size);
		cs = (zq_base_type*)((char*)(*buffer) + need_buffer_size*2);
		I = (zq_base_type*)((char*)(*buffer) + need_buffer_size*3);
		F = (zq_base_type*)((char*)(*buffer) + need_buffer_size*4);
		cs_prev = (zq_base_type*)((char*)(*buffer) + need_buffer_size*5);
		ci = (zq_base_type*)((char*)(*buffer) + need_buffer_size*6);
		co = (zq_base_type*)((char*)(*buffer) + need_buffer_size*7);
		o = (zq_base_type*)((char*)(*buffer) + need_buffer_size*8);
	}
	
	for (out_n = 0, in_slice_ptr = in_data, out_slice_ptr = out_data;
		out_n < in_N;
		out_n++, in_slice_ptr += in_sliceStep, out_slice_ptr += out_sliceStep)
	{
		memset(h, 0, sizeof(zq_base_type)*hidden_dim);
		memset(cell, 0, sizeof(zq_base_type)*hidden_dim);
		for (t = 0; t < in_W; t++)
		{
			ti = is_fw ? t : in_W - 1 - t;
			x = in_slice_ptr + ti*in_pixelStep;
			for (q = 0; q < hidden_dim; q++)
			{
				weight_xc_I = xc_I_data + q*xc_I_pixelStep;
				weight_xc_F = xc_F_data + q*xc_F_pixelStep;
				weight_xc_O = xc_O_data + q*xc_O_pixelStep;
				weight_xc_G = xc_G_data + q*xc_G_pixelStep;
				weight_hc_I = hc_I_data + q*hc_I_pixelStep;
				weight_hc_F = hc_F_data + q*hc_F_pixelStep;
				weight_hc_O = hc_O_data + q*hc_O_pixelStep;
				weight_hc_G = hc_G_data + q*hc_G_pixelStep;
				I[q] = b_I_data[q];
				F[q] = b_F_data[q];
				ci[q] = b_G_data[q];
				o[q] = b_O_data[q];
				v_I = zq_mm_setzero_ps();
				v_F = zq_mm_setzero_ps();
				v_O = zq_mm_setzero_ps();
				v_G = zq_mm_setzero_ps();
				for (i = 0; i < in_C - zq_mm_align_size + 1; i+= zq_mm_align_size)
				{
					v_x = zq_mm_load_ps(x + i);
					v_I = zq_mm_fmadd_ps(zq_mm_load_ps(weight_xc_I + i), v_x, v_I);
					v_F = zq_mm_fmadd_ps(zq_mm_load_ps(weight_xc_F + i), v_x, v_F);
					v_O = zq_mm_fmadd_ps(zq_mm_load_ps(weight_xc_O + i), v_x, v_O);
					v_G = zq_mm_fmadd_ps(zq_mm_load_ps(weight_xc_G + i), v_x, v_G);
				}
				for (; i < in_C; i++)
				{
					I[q] += weight_xc_I[i] * x[i];
					F[q] += weight_xc_F[i] * x[i];
					o[q] += weight_xc_O[i] * x[i];
					ci[q] += weight_xc_G[i] * x[i];
				}
				for (i = 0; i < hidden_dim - zq_mm_align_size + 1; i+= zq_mm_align_size)
				{
					v_h = zq_mm_load_ps(h + i);
					v_I = zq_mm_fmadd_ps(zq_mm_load_ps(weight_hc_I + i), v_h, v_I);
					v_F = zq_mm_fmadd_ps(zq_mm_load_ps(weight_hc_F + i), v_h, v_F);
					v_O = zq_mm_fmadd_ps(zq_mm_load_ps(weight_hc_O + i), v_h, v_O);
					v_G = zq_mm_fmadd_ps(zq_mm_load_ps(weight_hc_G + i), v_h, v_G);
				}
				for (; i < hidden_dim; i++)
				{
					I[q] += weight_hc_I[i] * h[i];
					F[q] += weight_hc_F[i] * h[i];
					o[q] += weight_hc_O[i] * h[i];
					ci[q] += weight_hc_G[i] * h[i];
				}
				zq_mm_store_ps(buffer_I, v_I);
				zq_mm_store_ps(buffer_F, v_F);
				zq_mm_store_ps(buffer_O, v_O);
				zq_mm_store_ps(buffer_G, v_G);
				for (i = 0; i < zq_mm_align_size; i++)
				{
					I[q] += buffer_I[i];
					F[q] += buffer_F[i];
					o[q] += buffer_O[i];
					ci[q] += buffer_G[i];
				}
				F[q] += forget_bias; //forget_bias = 1.0f
			}

			////https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/contrib/rnn/ops/lstm_ops.cc
			////注意权重顺序：应该是i, ci, f, o，来源于以下链接
			////https://github.com/tensorflow/tensorflow/blob/722b96b22926dbc05881c35cb63fd342c6843112/tensorflow/core/kernels/rnn/lstm_ops_gpu.cu.cc
			/*python
			xh = [x, h_prev]
			[i, f, ci, o] = xh * w + b
			f = f + forget_bias

			if not use_peephole:
			wci = wcf = wco = 0

			i = sigmoid(cs_prev * wci + i)
			f = sigmoid(cs_prev * wcf + f)
			ci = tanh(ci)
			cs = ci.*i + cs_prev.*f
			cs = clip(cs, cell_clip)
			o = sigmoid(cs * wco + o)
			co = tanh(cs)
			h = co.*o
			*/
			out_pixel_ptr = out_slice_ptr + ti*out_pixelStep;
			for (q = 0; q < hidden_dim; q++)
			{
				cs_prev[q] = cell[q];
				I[q] = 1.f / (1.f + exp(-I[q]));				// i = sigmoid(cs_prev * wci + i)
				F[q] = 1.f / (1.f + exp(-F[q]));				// f = sigmoid(cs_prev * wcf + f)
				ci[q] = tanh(ci[q]);							// ci = tanh(ci)
				cs[q] = ci[q] * I[q] + cs_prev[q] * F[q];			// cs = ci.*i + cs_prev.*f	
				cs[q] = __min(cell_clip, __max(-cell_clip, cs[q]));				//cs = clip(cs, cell_clip)
				o[q] = 1.f / (1.f + exp(-o[q]));				// o = sigmoid(cs * wco + o)
				co[q] = tanh(cs[q]);							// co = tanh(cs)
				h[q] = co[q] * o[q];								// h = co.*o

				cell[q] = cs[q];
				out_pixel_ptr[q] = h[q];
			}

			// no cell output here
		}
	}
	if (buffer == 0)
	{
		_aligned_free(h);
		_aligned_free(cell);
		_aligned_free(cs);
		_aligned_free(I);
		_aligned_free(F);
		_aligned_free(ci);
		_aligned_free(co);
		_aligned_free(o);
	}
}