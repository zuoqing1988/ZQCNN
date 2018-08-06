#ifndef _ZQ_TEXTURE_OPTIMIZATION_H_
#define _ZQ_TEXTURE_OPTIMIZATION_H_
#pragma once

#include "ZQ_TextureSynthesisProbe.h"
#include "ZQ_DoubleImage.h"

namespace ZQ
{
	class ZQ_TextureOptimization
	{
		typedef ZQ_DImage<float> DImage;

		class Neighbor
		{
		public:
			int border_width;
			int center_x, center_y;
		};
	
	private:
		DImage data_src;
		DImage data_ctrl;
		DImage data_dst;

		int border_width;

		float search_probe;

	public:
		ZQ_TextureOptimization()
		{
			search_probe = 0.05;
		}

		~ZQ_TextureOptimization()
		{

		}
	public:
		void SetSource(const DImage& src){data_src = src;}

		void SetSearchProbe(float val) {search_probe = val;}

		bool SynthesisWithoutControl(const int border_width, const float reduce_factor, const int max_level, const float grad_weight, const int to_max_iter, const int sor_iter,
			DImage& output)
		{
			int dst_width = output.width();
			int dst_height = output.height();
			if (output.nchannels() != data_src.nchannels())
				return false;

			this->border_width = border_width;
			int cur_border_width = border_width;

			float* x = output.data();
			this->data_dst = output;


			for (int i = 0; i < max_level; i++)
			{
				if (i == 0)
				{
					_optimize_without_control_one_border_size(false, x, cur_border_width, to_max_iter, sor_iter, grad_weight);
				}
				else
				{
					_optimize_without_control_one_border_size(true, x, cur_border_width, to_max_iter, sor_iter, grad_weight);
				}
				cur_border_width *= reduce_factor;
				if (cur_border_width < 4)
					break;
			}
			return true;
		}

		bool SynthesisWithControl(const DImage& ctrl, int border_width, const float reduce_factor, const int max_level, const float grad_weight, const float ctrl_weight,
			const int to_max_iter, const int sor_iter, DImage& output)
		{

			int dst_width = ctrl.width();
			int dst_height = ctrl.height();
			int nChannels = ctrl.nchannels();
			if (nChannels != data_src.nchannels())
				return false;

			this->border_width = border_width;
			this->data_ctrl = ctrl;

			int cur_border_width = border_width;

			if (!output.matchDimension(dst_width, dst_height, nChannels))
				output.allocate(dst_width, dst_height, nChannels);

			this->data_dst = output;

			float* x = output.data();
			memcpy(x, ctrl.data(), sizeof(float)*dst_height*dst_width*nChannels);

			for (int i = 0; i < max_level; i++)
			{
				_optimize_with_control_one_border_size(x, cur_border_width, to_max_iter, sor_iter, grad_weight, ctrl_weight);

				cur_border_width *= reduce_factor;
				if (cur_border_width < 4)
					break;
			}
			return x;
		}


	private:

		void _optimize_without_control_one_border_size(bool has_init_value, float *x, int border_width, int to_max_iter, int sor_iter, float grad_weight)
		{
			this->border_width = border_width;
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();
			int nChannels = data_dst.nchannels();


			Neighbor* xps = 0;
			int num = 0;
			_select_xps(dst_width, dst_height, border_width, num, xps);
			Neighbor* zps = new Neighbor[num];
			Neighbor* zps_n1 = new Neighbor[num];

			if (!has_init_value)
				_randomNeighbor(num, xps, zps);
			else
				_search_neighbor_without_control(num, xps, x, zps);

			int it = 0;
			while (true)
			{
				_argmin_x_without_control(num, xps, zps, sor_iter, grad_weight, x);

				_search_neighbor_without_control(num, xps, x, zps_n1);

				bool check = true;
				if (it != 0)
				{
					for (int i = 0; i < num; i++)
					{
						if (zps_n1[i].center_x != zps[i].center_x || zps_n1[i].center_y != zps[i].center_y)
						{
							check = false;
							break;
						}
					}
				}
				else
					check = false;

				it++;

				if (check)
				{
					break;
				}
				else
				{
					Neighbor* tmp = zps;
					zps = zps_n1;
					zps_n1 = tmp;
				}
				if (it >= to_max_iter)
					break;

			}
			delete[]xps;
			delete[]zps;
			delete[]zps_n1;

		}

		void _optimize_with_control_one_border_size(float* x, int border_size, int to_max_iter, int sor_iter, float grad_weight, float ctrl_weight)
		{
			this->border_width = border_width;
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();

			Neighbor* xps = 0;
			int num = 0;
			_select_xps(dst_width, dst_height, border_width, num, xps);
			Neighbor* zps = new Neighbor[num];
			Neighbor* zps_n1 = new Neighbor[num];

			_search_neighbor_with_control(num, xps, x, ctrl_weight, zps);

			int it = 0;
			while (true)
			{
				_argmin_x_with_control(num, xps, zps, sor_iter, grad_weight, ctrl_weight, x);
				_search_neighbor_with_control(num, xps, x, ctrl_weight, zps_n1);

				bool check = true;
				if (it != 0)
				{
					for (int i = 0; i < num; i++)
					{
						if (zps_n1[i].center_x != zps[i].center_x || zps_n1[i].center_y != zps[i].center_y)
						{
							check = false;
							break;
						}
					}
				}
				else
					check = false;

				it++;
				if (check)
				{
					break;
				}
				else
				{
					Neighbor* tmp = zps;
					zps = zps_n1;
					zps_n1 = tmp;
				}
				if (it >= to_max_iter)
					break;

			}
			delete[]xps;
			delete[]zps;
			delete[]zps_n1;
		}

		void _select_xps(int dst_width, int dst_height, int border_width, int& num, Neighbor* & xps)
		{
			int shift_x = 0;
			int shift_y = 0;
			int interval = border_width + 1;
			int x_num = (dst_width - shift_x) / interval + 1;
			int y_num = (dst_height - shift_y) / interval + 1;
			num = x_num * y_num;
			xps = new Neighbor[num];

			int count = 0;
			for (int i = 0; i < x_num; i++)
			{
				for (int j = 0; j < y_num; j++)
				{
					xps[count].border_width = border_width;
					xps[count].center_x = i * interval + shift_x;
					xps[count].center_y = j * interval + shift_y;
					count++;
				}
			}
		}

		void _randomNeighbor(int num, const Neighbor* xps, Neighbor* zps)
		{
			int src_width = data_src.width();
			int src_height = data_src.height();
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();

			for (int i = 0; i < num; i++)
			{
				int xps_cx = xps[i].center_x;
				int xps_cy = xps[i].center_y;
				int border_width = xps[i].border_width;
				int x_range, y_range;
				int zps_cx, zps_cy;
				if (xps_cx < border_width)
				{
					x_range = src_width - border_width - xps_cx;
					zps_cx = rand() % x_range + xps_cx;
				}
				else if (xps_cx >= dst_width - border_width)
				{
					x_range = src_width - border_width - (dst_width - 1 - xps_cx);
					zps_cx = rand() % x_range + border_width;
				}
				else
				{
					x_range = src_width - 2 * border_width;
					zps_cx = rand() % x_range + border_width;
				}

				if (xps_cy < border_width)
				{
					y_range = src_height - border_width - xps_cy;
					zps_cy = rand() % y_range + xps_cy;
				}
				else if (xps_cy >= dst_height - border_width)
				{
					y_range = src_height - border_width - (dst_height - 1 - xps_cy);
					zps_cy = rand() % y_range + border_width;
				}
				else
				{
					y_range = src_height - 2 * border_width;
					zps_cy = rand() % y_range + border_width;
				}
				zps[i].border_width = border_width;
				zps[i].center_x = zps_cx;
				zps[i].center_y = zps_cy;
			}
		}

		void _search_neighbor_without_control(int num, const Neighbor* xps, const float* x, Neighbor* zps )
		{
			int src_width = data_src.width();
			int src_height = data_src.height();
			int nChannels = data_src.nchannels();
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();
			float* data_src_ptr = data_src.data();

			for (int nn = 0; nn < num; nn++)
			{
				int xps_cx = xps[nn].center_x;
				int xps_cy = xps[nn].center_y;
				int border_width = xps[nn].border_width;
				int x_range, y_range;
				int zps_cx, zps_cy;
				int off_x, off_y;
				if (xps_cx < border_width)
				{
					x_range = src_width - border_width - xps_cx;
					off_x = xps_cx;
				}
				else if (xps_cx >= dst_width - border_width)
				{
					x_range = src_width - border_width - (dst_width - 1 - xps_cx);
					off_x = border_width;
				}
				else
				{
					x_range = src_width - 2 * border_width;
					off_x = border_width;
				}

				if (xps_cy < border_width)
				{
					y_range = src_height - border_width - xps_cy;
					off_y = xps_cy;
				}
				else if (xps_cy >= dst_height - border_width)
				{
					y_range = src_height - border_width - (dst_height - 1 - xps_cy);
					off_y = border_width;
				}
				else
				{
					y_range = src_height - 2 * border_width;
					off_y = border_width;
				}

				ZQ_TextureSynthesisProbe probe(x_range, y_range, search_probe);

				probe.Reset();

				bool completed = false;
				float max_float_value = 1e16;
				float d, dmin = max_float_value;
				while (!completed)
				{
					int xx, yy;
					probe.NextRandom(xx, yy, completed);
					if (completed)
						break;
					zps_cx = xx + off_x;
					zps_cy = yy + off_y;

					d = 0;
					for (int i = __max(0, xps_cx - border_width); i <= __min(dst_width - 1, xps_cx + border_width); i++)
					{
						for (int j = __max(0, xps_cy - border_width); j <= __min(dst_height - 1, xps_cy + border_width); j++)
						{
							int zps_x = zps_cx + i - xps_cx;
							int zps_y = zps_cy + j - xps_cy;

							for (int c = 0; c < nChannels; c++)
							{
								float d_r = data_src_ptr[(zps_y*src_width + zps_x)*nChannels + c] - x[(j*dst_width + i)*nChannels + c];

								d += d_r*d_r;
							}
						}
					}
					if (d < dmin)
					{
						dmin = d;
						zps[nn].center_x = zps_cx;
						zps[nn].center_y = zps_cy;
						zps[nn].border_width = border_width;
					}
				}
			}
		}

		void _search_neighbor_with_control(int num, const Neighbor* xps, const float* x, float ctrl_weight, Neighbor* zps)
		{
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();
			int src_width = data_src.width();
			int src_height = data_src.height();
			int nChannels = data_src.nchannels();
			float* data_src_ptr = data_src.data();
			float* data_ctrl_ptr = data_ctrl.data();

			for (int nn = 0; nn < num; nn++)
			{
				int xps_cx = xps[nn].center_x;
				int xps_cy = xps[nn].center_y;
				int border_width = xps[nn].border_width;
				int x_range, y_range;
				int zps_cx, zps_cy;
				int off_x, off_y;
				if (xps_cx < border_width)
				{
					x_range = src_width - border_width - xps_cx;
					off_x = xps_cx;
				}
				else if (xps_cx >= dst_width - border_width)
				{
					x_range = src_width - border_width - (dst_width - 1 - xps_cx);
					off_x = border_width;
				}
				else
				{
					x_range = src_width - 2 * border_width;
					off_x = border_width;
				}

				if (xps_cy < border_width)
				{
					y_range = src_height - border_width - xps_cy;
					off_y = xps_cy;
				}
				else if (xps_cy >= dst_height - border_width)
				{
					y_range = src_height - border_width - (dst_height - 1 - xps_cy);
					off_y = border_width;
				}
				else
				{
					y_range = src_height - 2 * border_width;
					off_y = border_width;
				}

				ZQ_TextureSynthesisProbe probe(x_range, y_range, search_probe);

				probe.Reset();

				bool completed = false;
				float max_float_value = 1e16;
				float d, dmin = max_float_value;
				while (!completed)
				{
					int xx, yy;
					probe.NextRandom(xx, yy, completed);
					if (completed)
						break;
					zps_cx = xx + off_x;
					zps_cy = yy + off_y;

					d = 0;
					for (int i = __max(0, xps_cx - border_width); i <= __min(dst_width - 1, xps_cx + border_width); i++)
					{
						for (int j = __max(0, xps_cy - border_width); j <= __min(dst_height - 1, xps_cy + border_width); j++)
						{
							int zps_x = zps_cx + i - xps_cx;
							int zps_y = zps_cy + j - xps_cy;

							for (int c = 0; c < nChannels; c++)
							{
								float d_r = data_src_ptr[(zps_y*src_width + zps_x)*nChannels + c] - x[(j*dst_width + i)*nChannels + c];

								d += d_r*d_r;

								d_r = data_src_ptr[(zps_y*src_width + zps_x)*nChannels + c] - data_ctrl_ptr[(j*dst_width + i)*nChannels + c];
								d += ctrl_weight*d_r*d_r;
							}

						}
					}
					if (d < dmin)
					{
						dmin = d;
						zps[nn].center_x = zps_cx;
						zps[nn].center_y = zps_cy;
						zps[nn].border_width = border_width;
					}
				}
			}
		}

		void _argmin_x_without_control(int num, const Neighbor* xps, const Neighbor* zps, int sor_iter, float grad_weight,  float* x)
		{
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();
			int src_width = data_src.width();
			int src_height = data_src.height();
			int nChannels = data_src.nchannels();
			float* data_src_ptr = data_src.data();

			DImage src_dx, src_dy;
			data_src.dx(src_dx);
			data_src.dy(src_dy);

			DImage target_pixel(dst_width, dst_height, nChannels), target_grad_x(dst_width, dst_height, nChannels), tagret_grad_y(dst_width, dst_height, nChannels);
			DImage overlap_weight(dst_width, dst_height, 1);

			float*& target_pixel_ptr = target_pixel.data();
			float*& target_grad_x_ptr = target_grad_x.data();
			float*& target_grad_y_ptr = tagret_grad_y.data();
			float*& overlap_weight_ptr = overlap_weight.data();
			float*& src_dx_ptr = src_dx.data();
			float*& src_dy_ptr = src_dy.data();

			for (int nn = 0; nn < num; nn++)
			{
				int xps_cx = xps[nn].center_x;
				int xps_cy = xps[nn].center_y;
				int zps_cx = zps[nn].center_x;
				int zps_cy = zps[nn].center_y;

				int border_width = xps[nn].border_width;

				for (int j = __max(0, xps_cx - border_width); j <= __min(dst_width - 1, xps_cx + border_width); j++)
				{
					for (int i = __max(0, xps_cy - border_width); i <= __min(dst_height - 1, xps_cy + border_width); i++)
					{
						int z_x = j - xps_cx + zps_cx;
						int z_y = i - xps_cy + zps_cy;

						int xps_offset = i*dst_width + j;
						int zps_offset = z_y*src_width + z_x;

						overlap_weight_ptr[xps_offset] += 1;
						for (int c = 0; c < nChannels; c++)
						{
							target_pixel_ptr[xps_offset*nChannels + c] += data_src_ptr[zps_offset*nChannels + c];
							target_grad_x_ptr[xps_offset*nChannels + c] += src_dx_ptr[zps_offset*nChannels + c];
							target_grad_y_ptr[xps_offset*nChannels + c] += src_dy_ptr[zps_offset*nChannels + c];
						}
					}
				}
			}

			for (int ii = 0; ii < dst_width*dst_height; ii++)
			{
				for (int c = 0; c < nChannels; c++)
				{
					target_pixel_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
					target_grad_x_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
					target_grad_y_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
				}
			}

			DImage tmp_dx, tmp_dy, divergence;
			target_grad_x.dx(tmp_dx);
			tagret_grad_y.dy(tmp_dy);
			divergence.Add(tmp_dx, tmp_dy);
			float*& divergence_ptr = divergence.data();

			for (int sor_it = 0; sor_it < sor_iter; sor_it++)
			{
				for (int h = 0; h < dst_height; h++)
				{
					for (int w = 0; w < dst_width; w++)
					{
						int offset = h*dst_width + w;
						for (int c = 0; c < nChannels; c++)
						{
							float coeff = 0, sigma = 0;
							if (h > 0)
							{
								coeff += 1;
								sigma += x[(offset - dst_width)*nChannels + c];
							}
							if (h < dst_height - 1)
							{
								coeff += 1;
								sigma += x[(offset + dst_width)*nChannels + c];
							}
							if (w > 0)
							{
								coeff += 1;
								sigma += x[(offset - 1)*nChannels + c];
							}
							if (w < dst_width - 1)
							{
								coeff += 1;
								sigma += x[(offset + 1)*nChannels + c];
							}
							sigma += divergence_ptr[offset*nChannels + c];
							coeff *= grad_weight;
							sigma *= grad_weight;

							coeff += 1;
							sigma += target_pixel_ptr[offset*nChannels + c];

							x[offset*nChannels + c] = sigma / coeff;
						}
					}
				}
			}
		}

		void _argmin_x_with_control(int num, const Neighbor* xps, const Neighbor* zps, int sor_iter, float grad_weight, float ctrl_weight,  float* x)
		{
			int dst_width = data_dst.width();
			int dst_height = data_dst.height();
			int src_width = data_src.width();
			int src_height = data_src.height();
			int nChannels = data_src.nchannels();
			float* data_src_ptr = data_src.data();
			float* data_ctrl_ptr = data_ctrl.data();

			DImage src_dx, src_dy;
			data_src.dx(src_dx);
			data_src.dy(src_dy);

			DImage target_pixel(dst_width, dst_height, nChannels), target_grad_x(dst_width, dst_height, nChannels), tagret_grad_y(dst_width, dst_height, nChannels);
			DImage overlap_weight(dst_width, dst_height, 1);

			float*& target_pixel_ptr = target_pixel.data();
			float*& target_grad_x_ptr = target_grad_x.data();
			float*& target_grad_y_ptr = tagret_grad_y.data();
			float*& overlap_weight_ptr = overlap_weight.data();
			float*& src_dx_ptr = src_dx.data();
			float*& src_dy_ptr = src_dy.data();

			for (int nn = 0; nn < num; nn++)
			{
				int xps_cx = xps[nn].center_x;
				int xps_cy = xps[nn].center_y;
				int zps_cx = zps[nn].center_x;
				int zps_cy = zps[nn].center_y;

				int border_width = xps[nn].border_width;

				for (int j = __max(0, xps_cx - border_width); j <= __min(dst_width - 1, xps_cx + border_width); j++)
				{
					for (int i = __max(0, xps_cy - border_width); i <= __min(dst_height - 1, xps_cy + border_width); i++)
					{
						int z_x = j - xps_cx + zps_cx;
						int z_y = i - xps_cy + zps_cy;

						int xps_offset = i*dst_width + j;
						int zps_offset = z_y*src_width + z_x;

						overlap_weight_ptr[xps_offset] += 1;
						for (int c = 0; c < nChannels; c++)
						{
							target_pixel_ptr[xps_offset*nChannels + c] += data_src_ptr[zps_offset*nChannels + c];
							target_grad_x_ptr[xps_offset*nChannels + c] += src_dx_ptr[zps_offset*nChannels + c];
							target_grad_y_ptr[xps_offset*nChannels + c] += src_dy_ptr[zps_offset*nChannels + c];
						}
					}
				}
			}

			for (int ii = 0; ii < dst_width*dst_height; ii++)
			{
				for (int c = 0; c < nChannels; c++)
				{
					target_pixel_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
					target_grad_x_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
					target_grad_y_ptr[ii*nChannels + c] /= overlap_weight_ptr[ii];
				}
			}

			DImage tmp_dx, tmp_dy, divergence;
			target_grad_x.dx(tmp_dx);
			tagret_grad_y.dy(tmp_dy);
			divergence.Add(tmp_dx, tmp_dy);
			float*& divergence_ptr = divergence.data();

			for (int sor_it = 0; sor_it < sor_iter; sor_it++)
			{
				for (int h = 0; h < dst_height; h++)
				{
					for (int w = 0; w < dst_width; w++)
					{
						int offset = h*dst_width + w;
						for (int c = 0; c < nChannels; c++)
						{
							float coeff = 0, sigma = 0;
							if (h > 0)
							{
								coeff += 1;
								sigma += x[(offset - dst_width)*nChannels + c];
							}
							if (h < dst_height - 1)
							{
								coeff += 1;
								sigma += x[(offset + dst_width)*nChannels + c];
							}
							if (w > 0)
							{
								coeff += 1;
								sigma += x[(offset - 1)*nChannels + c];
							}
							if (w < dst_width - 1)
							{
								coeff += 1;
								sigma += x[(offset + 1)*nChannels + c];
							}
							sigma += divergence_ptr[offset*nChannels + c];
							coeff *= grad_weight;
							sigma *= grad_weight;

							coeff += 1 + ctrl_weight;
							sigma += target_pixel_ptr[offset*nChannels + c] + ctrl_weight*data_ctrl_ptr[offset*nChannels + c];

							x[offset*nChannels + c] = sigma / coeff;
						}
					}
				}
			}
		}
	};
}


#endif