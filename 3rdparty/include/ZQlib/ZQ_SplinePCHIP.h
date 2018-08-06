#ifndef _ZQ_SPLINE_PCHIP_H_
#define _ZQ_SPLINE_PCHIP_H_
#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

namespace ZQ
{
	class ZQ_SplinePCHIP
	{
	private:
		std::vector<double> m_x, m_y;            // x,y coordinates of points
		// interpolation parameters
		// f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
		std::vector<double> m_a, m_b, m_c;        // spline coefficients
	public:
		void SetPoints(const std::vector<double>& x, const std::vector<double>& y)
		{
			assert(x.size() == y.size());
			assert(x.size() > 2);
			m_x = x;
			m_y = y;
			int   n = x.size();
			// TODO: maybe sort x and y, rather than returning an error
			for (int i = 0; i < n - 1; i++)
			{
				assert(m_x[i] < m_x[i + 1]);
			}

			std::vector<double> h(n - 1);
			std::vector<double> del(n - 1);
			std::vector<double> slopes;
			for (int i = 0; i < n - 1; i++)
			{
				h[i] = x[i + 1] - x[i];
				del[i] = y[i + 1] - y[i];
			}
			_pchipslopes(x, y, del, slopes);

			//slove
			m_a.resize(n - 1);
			m_b.resize(n - 1);
			m_c.resize(n - 1);
			for (int i = 0; i < n - 1; i++)
			{
				m_c[i] = slopes[i];
				m_a[i] = (slopes[i + 1] +m_c[i])/ (h[i] * h[i]) - 2.0*del[i] / (h[i] * h[i] * h[i]);
				m_b[i] = del[i]/(h[i]*h[i])-m_c[i]/h[i] - m_a[i] * h[i];
			}
		}

		double operator() (double x) const
		{
			size_t n = m_x.size();
			// find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
			std::vector<double>::const_iterator it;
			it = std::lower_bound(m_x.begin(), m_x.end(), x);
			int idx = __min(__max(int(it - m_x.begin()) - 1, 0), n - 2);

			double h = x - m_x[idx];
			double interpol;
			if (x < m_x[0])
			{
				// extrapolation to the left
				interpol = ((m_a[0] * h + m_b[0])*h + m_c[0])*h + m_y[0];
			}
			else if (x > m_x[n - 1])
			{
				// extrapolation to the right
				interpol = ((m_a[n - 2] * h + m_b[n - 2])*h + m_c[n - 2])*h + m_y[n - 2];
			}
			else
			{
				// interpolation
				interpol = ((m_a[idx] * h + m_b[idx])*h + m_c[idx])*h + m_y[idx];
			}
			return interpol;
		}

	private:
		void _pchipslopes(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& del,
			std::vector<double>& d)
		{
			/*
			%PCHIPSLOPES  Derivative values for shape-preserving Piecewise Cubic Hermite
			% Interpolation.
			% d = pchipslopes(x,y,del) computes the first derivatives, d(k) = P'(x(k)).

			%  Special case n=2, use linear interpolation.
			*/
			int n = x.size();
			if (n == 2)
			{
				d.resize(n);
				for (int i = 0; i < n; i++)
					d[i] = del[0];
				return;
			}

			/*
			%  Slopes at interior points.
			%  d(k) = weighted average of del(k-1) and del(k) when they have the same sign.
			%  d(k) = 0 when del(k-1) and del(k) have opposites signs or either is zero.
			*/

			std::vector<double> h(n - 1);
			for (int k = 0; k < n - 1; k++)
				h[k] = x[k + 1] - x[k];

			d.resize(n);
			for (int k = 0; k < n - 2; k++)
			{
				if ((del[k] > 0 && del[k+1] > 0) || (del[k] < 0 && del[k+1] < 0))
				{
					double hs = h[k] + h[k+1];
					double w1 = (h[k] + hs) / (3 * hs);
					double w2 = (hs + h[k+1]) / (3 * hs);
					double dmax = __max(fabs(del[k]), fabs(del[k+1]));
					double dmin = __min(fabs(del[k]), fabs(del[k+1]));
					d[k+1] = dmin / (w1*del[k] / dmax + w2*del[k+1] / dmax);
				}
				else
				{
					d[k+1] = 0;
				}
			}

			/*
			%  Slopes at end points.
			%  Set d(1) and d(n) via non-centered, shape-preserving three-point formulae.
			*/

			d[0] = ((2 * h[0] + h[1])*del[1] - h[0] * del[1]) / (h[0] + h[1]);
			if ((d[0] > 0 && del[0] <= 0) || (d[0] < 0 && del[0] >= 0))
			{
				d[0] = 0;
			}
			else if (((del[0] > 0 && del[1] <= 0) || (del[0] < 0 && del[1] >= 0)) && (fabs(d[0]) > fabs(3.0*del[0])))
			{
				d[0] = 3 * del[0];
			}

			d[n - 1] = ((2 * h[n - 2] + h[n - 3])*del[n - 2] - h[n - 2] * del[n - 3]) / (h[n - 2] + h[n - 3]);
			if ((d[n - 1] > 0 || del[n - 2] <= 0) || (d[n - 1] < 0 || del[n - 2] >= 0))
			{
				d[n - 1] = 0;
			}
			else if (((del[n - 2] > 0 && del[n - 3] <= 0) || (del[n - 2] < 0 && del[n - 3] >= 0)) && (fabs(d[n - 1]) > fabs(3.0*del[n - 2])))
			{
				d[n - 1] = 3 * del[n - 2];
			}
		}
	};
}
#endif
