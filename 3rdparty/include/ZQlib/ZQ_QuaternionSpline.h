#ifndef _ZQ_QUATERNION_SPLINE_H_
#define _ZQ_QUATERNION_SPLINE_H_
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include "ZQ_Quaternion.h"

namespace ZQ
{
	/*ZQ_QuaternionSpline is modified (by Zuo Qing) from qspline.c (written by J. J. McEnnan, April, 2003)
	
		J. J. McEnnan, April, 2003.

		COPYRIGHT (C) 2003 by James McEnnan

		This program is free software; you can redistribute it and/or modify
		it under the terms of the GNU General Public License as published by
		the Free Software Foundation; either version 2 of the License, or
		(at your option) any later version.

		This program is distributed in the hope that it will be useful,
		but WITHOUT ANY WARRANTY; without even the implied warranty of
		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
		GNU General Public License for more details.

		You should have received a copy of the GNU General Public License along
		with this program; if not, write to the Free Software Foundation, Inc.,
		51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA..
	*/
	class ZQ_QuaternionSpline
	{
	public:
		ZQ_QuaternionSpline()
		{ 
			ns = 0; t = 0; q = 0; 
		}
		~ZQ_QuaternionSpline()
		{
			_clear();
		}

	private:
		int ns;
		double* t;
		double* q;
	public:
		template<class T>
		bool SetPoints(const std::vector<T>& input_t, const std::vector<ZQ_Quaternion<T>>& input_quat, int n_segments = 1000)
		{
			int n = input_t.size();
			if (n != input_quat.size() || n < 4 || n_segments < 2)
				return false;
	
			double wi[3] = { 0, 0,0 }, wf[3] = { 0, 0,0 };
			double* x = (double*)malloc(sizeof(double)*n);
			double* y = (double*)malloc(sizeof(double)*n * 4);
			double* tmp_t = (double*)malloc(sizeof(double)*n_segments);
			double* tmp_q = (double*)malloc(sizeof(double)*n_segments * 4);
			double* omega = (double*)malloc(sizeof(double)*n_segments * 3);
			double* alpha = (double*)malloc(sizeof(double)*n_segments * 3);
			if (x == 0 || y == 0 || tmp_t == 0 || tmp_q == 0 || omega == 0 || alpha == 0)
			{
				if (x) free(x);
				if (y) free(y);
				if (tmp_t) free(tmp_t);
				if (tmp_q) free(tmp_q);
				if (omega) free(omega);
				if (alpha) free(alpha);
				return false;
			}
			for (int i = 0; i < n; i++)
			{
				x[i] = input_t[i];
				y[i * 4 + 0] = input_quat[i].x;
				y[i * 4 + 1] = input_quat[i].y;
				y[i * 4 + 2] = input_quat[i].z;
				y[i * 4 + 3] = input_quat[i].w;
			}

			/* NOTE: as q and -q map to the same rotation, we reorder the quaternion to obtain better curves */ 
			for (int i = 1; i < n; i++)
			{
				double dot = y[i * 4 + 0] * y[i * 4 - 4] + y[i * 4 + 1] * y[i * 4 - 3] + y[i * 4 + 2] * y[i * 4 - 2] + y[i * 4 + 3] * y[i * 4 - 1];
				if (dot < 0)
				{
					y[i * 4 + 0] = -y[i * 4 + 0];
					y[i * 4 + 1] = -y[i * 4 + 1];
					y[i * 4 + 2] = -y[i * 4 + 2];
					y[i * 4 + 3] = -y[i * 4 + 3];
				}
			}

			int maxit = __min(1000,__max(10,n_segments));
			double tol = 1e-6;
			int ret = qspline(n, n_segments, maxit, tol, wi, wf, x, y, tmp_t, tmp_q, omega, alpha);
			if (ret < 2)
			{
				if (x) free(x);
				if (y) free(y);
				if (tmp_t) free(tmp_t);
				if (tmp_q) free(tmp_q);
				if (omega) free(omega);
				if (alpha) free(alpha);
				return false;
			}

			_clear();
			ns = n_segments;
			t = tmp_t;
			q = tmp_q;
			free(x);
			free(y);
			free(omega);
			free(alpha);
			return true;
		}

		template<class T>
		ZQ_Quaternion<T> operator() (T x) const
		{
			if (ns < 2)
			{
				return ZQ_Quaternion<T>(0, 0, 0, 1);
			}
			else
			{
				if (x <= t[0])
					return ZQ_Quaternion<T>(q[0], q[1], q[2], q[3]);
				else if (x >= t[ns - 1])
					return ZQ_Quaternion<T>(q[ns * 4 - 4], q[ns * 4 - 3], q[ns * 4 - 2], q[ns * 4 - 1]);
				else
				{
					double ds = (t[ns - 1] - t[0]) / (ns - 1);
					double off = (x - t[0]) / ds;
					int i = __min(ns - 2, __max(0, off));
					int j = i + 1;
					double weight = off - i;
					if (off == i)
						return ZQ_Quaternion<T>(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);

					ZQ_Quaternion<double> q1(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
					ZQ_Quaternion<double> q2(q[j * 4], q[j * 4 + 1], q[j * 4 + 2], q[j * 4 + 3]);
					ZQ_Quaternion<double> q3 = ZQ_Quaternion<double>::Slerp(q1, q2, weight);
					return ZQ_Quaternion<T>(q3.x, q3.y, q3.z, q3.w);
				}
			}
		}

	private:
		void _clear()
		{
			ns = 0;
			if (t)
			{
				free(t);
				t = 0;
			}
			if (q)
			{
				free(q);
				q = 0;
			}
		}
	public:
		/*
		purpose

		Subroutine qspline produces a quaternion spline interpolation of sparse data.
		The method is based on a third-order polynomial expansion of the
		rotation angle vector.

		variable     i/o     description
		--------     ---     -----------
		n             i      number of input points (n >= 4).
		ns            i      number of output points (ns >= 2).
		maxit         i      maximum number of iterations.
		tol           i      convergence tolerance (rad/sec) for iteration termination.
		wi            i      initial angular rate vector.
		wf            i      final angular rate vector.
		x             i      pointer to input vector of time values. n*1
		y             i      pointer to input vector of quaternion values. n*4 
		t             o      pointer to output vector of time values. ns*1
		q             o      pointer to output array of interpolated quaternion values. ns*4
		omega         o      pointer to output array of interpolated angular rate values (rad/sec). ns*3
		alpha         o      pointer to output array of interpolated angular acceleration values (rad/sec^2). ns*3

		return value

		ns >= 2 -> normal return, no error
		-1      -> insufficient input/output data (n < 4 or ns < 2)
		-2      -> independent variable is not monotonic increasing
		-3      -> memory allocation failure
		*/
		static int qspline(int n, int ns, int maxit, double tol, const double* wi, const double* wf, const double* x, const double* y, double* t, double* q, double* omega, double* alpha)
		{
			/* error checking. */

			if (n < 4)
			{
				fprintf(stderr, "qspline: insufficient input data.\n");
				return -1;
			}
			if (ns < 2)
			{
				fprintf(stderr, "qspline: too few output points.\n");
				return -1;
			}

			double dx = (x[n - 1] - x[0]) / (ns - 1);

			double* w = (double*)malloc(sizeof(double)*n * 3);
			if (w == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				return -3;
			}

			double* wprev = (double*)malloc(sizeof(double)*n * 3);
			if (wprev == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				return -3;
			}

			double* a = (double*)malloc(sizeof(double)*(n - 1));
			if (a == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				free(wprev);
				return -3;
			}

			double* b = (double*)malloc(sizeof(double)*(n - 1));
			if (b == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				free(wprev);
				free(a);
				return -3;
			}

			double* c = (double*)malloc(sizeof(double)*(n - 1));
			if (c == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");

				free(w);
				free(wprev);
				free(a);
				free(b);
				return -3;
			}

			double* h = (double*)malloc(sizeof(double)*(n - 1));
			if (h == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				free(wprev);
				free(c);
				free(b);
				free(a);
				return -3;
			}

			double* dtheta = (double*)malloc(sizeof(double)*(n - 1));
			if (dtheta == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				free(wprev);
				free(c);
				free(b);
				free(a);
				free(dtheta);
				return -3;
			}

			double* e = (double*)malloc(sizeof(double)*(n - 1) * 3);
			if (e == 0)
			{
				fprintf(stderr, "qspline: memory allocation failure.\n");
				free(w);
				free(wprev);
				free(e);
				free(dtheta);
				free(c);
				free(b);
				free(a);
				return -3;
			}

			for (int i = 0; i < n; i++)
			{
				for (int j = 0; j < 3; j++)
					w[i * 3 + j] = 0.0;
			}

			for (int i = 0; i < n - 1; i++)
			{
				h[i] = x[i + 1] - x[i];

				if (h[i] <= 0.0)
				{
					fprintf(stderr, "qspline: x is not monotonic.\n");
					_freeall(h, a, b, c, dtheta, e, w, wprev);
					return -2;
				}
			}

			/* compute spline coefficients. */

			for (int i = 0; i < n - 1; i++)
				dtheta[i] = _getang(y + i * 4, y + (i + 1) * 4, e + i * 3);

			_rates(n, maxit, tol, wi, wf, h, a, b, c, dtheta, e, w, wprev);

			/* interpolate and output results. */

			int idx = 0;
			double xi = x[0];

			double dum1[3], dum2[3];
			double tmp_a[3][3], tmp_b[3][3], tmp_c[2][3], tmp_d[3];
			_slew3_init(h[0], dtheta[0], e, w, dum1, w+3, dum2, tmp_a, tmp_b, tmp_c, tmp_d);

			for (int j = 0; j < ns; j++)
			{
				while (xi >= x[idx + 1] && idx < n - 2)
				{
					idx++;
					_slew3_init(h[idx], dtheta[idx], e + idx * 3, w + idx * 3, dum1, w + (idx + 1) * 3, dum2, tmp_a, tmp_b, tmp_c, tmp_d);
				}

				t[j] = xi;
				_slew3(xi - x[idx], h[idx], y + idx * 4, q + j * 4, omega + j * 3, alpha + j * 3, dum1, tmp_a, tmp_b, tmp_c, tmp_d);

				xi += dx;
			}
		
			_freeall(h, a, b, c, dtheta, e, w, wprev);

			return ns;
		}

	private:
		static void _freeall(double*& h, double*& a, double*& b, double*& c, double*& dtheta, double*& e, double*& w, double*& wprev)
		{
			if (h != 0)
			{
				free(h);
				h = 0;
			}
			if (a != 0)
			{
				free(a);
				a = 0;
			}
			if (b != 0)
			{
				free(b);
				b = 0;
			}
			if (c != 0)
			{
				free(c);
				c = 0;
			}
			if (dtheta != 0)
			{
				free(dtheta);
				dtheta = 0;
			}
			if (e != 0)
			{
				free(e);
				e = 0;
			}
			if (w != 0)
			{
				free(w);
				w = 0;
			}
			if (wprev != 0)
			{
				free(wprev);
				wprev = 0;
			}
		}

		/*
		purpose

		Subroutine getang computes the slew angle and axis between the input initial and
		final states.

		calling sequence

		variable     i/o     description
		--------     ---     -----------
		qi            i      initial attitude quaternion.
		qf            i      final attitude quaternion.
		e             o      unit vector along slew eigen-axis.

		return value
		slew angle in radians
		*/
		static double _getang(const double* qi, const double* qf, double* e)
		{
			double temp[3] = 
			{
				qi[3] * qf[0] - qi[0] * qf[3] - qi[1] * qf[2] + qi[2] * qf[1],
				qi[3] * qf[1] - qi[1] * qf[3] - qi[2] * qf[0] + qi[0] * qf[2],
				qi[3] * qf[2] - qi[2] * qf[3] - qi[0] * qf[1] + qi[1] * qf[0]
			};

			double ca = qi[0] * qf[0] + qi[1] * qf[1] + qi[2] * qf[2] + qi[3] * qf[3];
			double sa = _unvec(temp, e);
			double dtheta = 2.0*atan2(sa, ca);
			return dtheta;
		}

		/*
		purpose
		subroutine rates computes intermediate angular rates for interpolation.

		variable     i/o     description
		--------     ---     -----------
		n             i      number of input data points.
		maxit         i      maximum number of iterations.
		tol           i      convergence tolerance (rad/sec) for iteration termination.
		wi            i      initial angular rate vector.
		wf            i      final angular rate vector.
		h             i      pointer to vector of time interval values.
		a             i/o    pointer to intermediate work space.
		b             i/o    pointer to intermediate work space.
		c             i/o    pointer to intermediate work space.
		dtheta        i      pointer to vector of rotation angles.
		e             i      pointer to array of rotation axis vectors.
		w             o      pointer to output intermediate angular rate values.
		wprev         o      pointer to previous intermediate angular rate values.
		*/
		static void _rates(int n, int maxit, double tol, const double* wi, const double* wf, const double* h, double* a, double* b, double* c,
			const double* dtheta, const double* e, double* w, double* wprev)
		{
			
		

			int iter = 0;
			double dw;
			do        /* start iteration loop. */
			{

				for (int i = 1; i < n - 1; i++)
				{
					for (int j = 0; j < 3; j++)
						wprev[i * 3 + j] = w[i * 3 + j];
				}

				/* set up the tridiagonal matrix. d initially holds the RHS vector array;
				it is then overlaid with the calculated angular rate vector array. */
				double tmp1[3], tmp2[3];
				for (int i = 1; i < n - 1; i++)
				{
					a[i] = 2.0 / h[i - 1];
					b[i] = 4.0 / h[i - 1] + 4.0 / h[i];
					c[i] = 2.0 / h[i];

					_rf(e + (i - 1) * 3, dtheta[i - 1], wprev + i * 3, tmp1);

					for (int j = 0; j < 3; j++)
					{
						w[i * 3 + j] = 6.0*(dtheta[i - 1] * e[(i - 1) * 3 + j] / (h[i - 1] * h[i - 1]) + dtheta[i] * e[i * 3 + j] / (h[i] * h[i])) - tmp1[j];
					}
				}

				_bd(e, dtheta[0], 1, wi, tmp1);
				_bd(e + (n - 2) * 3, dtheta[n - 2], 0, wf, tmp2);

				for (int j = 0; j < 3; j++)
				{
					w[1 * 3 + j] -= a[1] * tmp1[j];
					w[(n - 2) * 3 + j] -= c[n - 2] * tmp2[j];
				}

				/* reduce the matrix to upper triangular form. */

				for (int i = 1; i < n - 2; i++)
				{
					b[i + 1] -= c[i] * a[i + 1] / b[i];

					for (int j = 0; j < 3; j++)
					{
						_bd(e + i * 3, dtheta[i], 1, w + i * 3, tmp1);
						w[(i + 1) * 3 + j] -= tmp1[j] * a[i + 1] / b[i];
					}
				}

				/* solve using back substitution. */

				for (int j = 0; j < 3; j++)
					w[(n - 2) * 3 + j] /= b[n - 2];

				for (int i = n - 3; i > 0; i--)
				{
					_bd(e + i * 3, dtheta[i], 0, w + (i + 1) * 3, tmp1);

					for (int j = 0; j < 3; j++)
						w[i * 3 + j] = (w[i * 3 + j] - c[i] * tmp1[j]) / b[i];
				}

				dw = 0.0;

				for (int i = 1; i < n - 1; i++)
				{
					for (int j = 0; j < 3; j++)
						dw += (w[i * 3 + j] - wprev[i * 3 + j])*(w[i * 3 + j] - wprev[i * 3 + j]);
				}

				dw = sqrt(dw);
			} while (iter++ < maxit && dw > tol);

			/* solve for end conditions. */

			for (int j = 0; j < 3; j++)
			{
				w[j] = wi[j];
				w[(n - 1) * 3 + j] = wf[j];
			}
		}


		/*
		purpose

		Subroutine bd performs the transformation between the coefficient vector and
		the angular rate vector.

		variable     i/o     description
		--------     ---     -----------
		e             i      unit vector along slew eigen-axis.
		dtheta        i      slew angle (rad).
		flag          i      flag determining direction of transformation.
		= 0 -> compute coefficient vector from
		angular rate vector
		= 1 -> compute angular rate vector from
		coefficient vector

		xin           i      input vector.
		xout          o      output vector.

		return value

		0 -> no error
		-1 -> transformation direction incorrectly specified.
		*/
		static int _bd(const double* e, double dtheta, int flag, const double* xin, double* xout)
		{
			const double EPS = 1e-6;
			if (dtheta > EPS)
			{
				double b0, b1, b2, temp1[3], temp2[3];
				double ca = cos(dtheta);
				double sa = sin(dtheta);

				if (flag == 0)
				{
					b1 = 0.5*dtheta*sa / (1.0 - ca);
					b2 = 0.5*dtheta;
				}
				else if (flag == 1)
				{
					b1 = sa / dtheta;
					b2 = (ca - 1.0) / dtheta;
				}
				else
					return -1;

				b0 = xin[0] * e[0] + xin[1] * e[1] + xin[2] * e[2];

				_crossp(e, xin, temp2);
				_crossp(temp2, e, temp1);

				for (int i = 0; i < 3; i++)
					xout[i] = b0*e[i] + b1*temp1[i] + b2*temp2[i];
			}
			else
			{
				for (int i = 0; i < 3; i++)
					xout[i] = xin[i];
			}

			return 0;
		}

		/*
		purpose

		Subroutine rf computes the non-linear rate contributions to the final
		angular acceleration.

		variable     i/o     description
		--------     ---     -----------
		e             i      unit vector along slew eigen-axis.
		dtheta        i      slew angle (rad).
		win           i      input final angular rate vector.
		rhs           o      output vector containing non-linear rate contributions to the final acceleration.
		*/
		static void _rf(const double* e, double dtheta, const double* win, double* rhs)
		{
			const double EPS = 1e-6;
			if (dtheta > EPS)
			{
				double ca = cos(dtheta);
				double sa = sin(dtheta);

				double tmp1[3], tmp2[3];
				_crossp(e, win, tmp2);
				_crossp(tmp2, e, tmp1);

				double dot = win[0] * e[0] + win[1] * e[1] + win[2] * e[2];
				double mag = win[0] * win[0] + win[1] * win[1] + win[2] * win[2];
				double c1 = (1.0 - ca);
				double r0 = 0.5*(mag - dot*dot)*(dtheta - sa) / c1;
				double r1 = dot*(dtheta*sa - 2.0*c1) / (dtheta*c1);
				for (int i = 0; i < 3; i++)
					rhs[i] = r0*e[i] + r1*tmp1[i];
			}
			else
			{
				for (int i = 0; i < 3; i++)
					rhs[i] = 0.0;
			}
		}


		/*
		purpose

		Subroutine slew3_init computes the coefficients for a third-order polynomial
		interpolation function describing a slew between the input initial and
		final states.

		variable     i/o     description
		--------     ---     -----------
		dt            i      slew time (sec).
		dtheta        i      slew angle (rad).
		e             i      unit vector along slew eigen-axis.
		wi            i      initial body angular rate (rad/sec).
		ai            i      initial body angular acceleration (rad/sec^2)
		(included for compatibility only).
		wf            i      final body angular rate (rad/sec).
		af            i      final body angular acceleration (rad/sec^2) 
		(included for compatibility only).

		a,b,c,d		  o		 for later use in _slew3().
		*/
		static void _slew3_init(double dt, double dtheta, const double* e, const double* wi, const double* ai, const double* wf, const double* af,
			double a[3][3], double b[3][3], double c[2][3], double d[3])
		{
			if (dt <= 0.0)
				return;

			double sa = sin(dtheta);
			double ca = cos(dtheta);
			const double EPS = 1e-6;

			double bvec[3];

			/* final angular rate terms. */
			if (dtheta > EPS)
			{
				double c1 = 0.5*sa*dtheta / (1.0 - ca);
				double c2 = 0.5*dtheta;
				double b0 = e[0] * wf[0] + e[1] * wf[1] + e[2] * wf[2];

				double bvec1[3], bvec2[3];
				_crossp(e, wf, bvec2);
				_crossp(bvec2, e, bvec1);

				for (int i = 0; i < 3; i++)
					bvec[i] = b0*e[i] + c1*bvec1[i] + c2*bvec2[i];
			}
			else
			{
				for (int i = 0; i < 3; i++)
					bvec[i] = wf[i];
			}

			/* compute coefficients. */
			for (int i = 0; i < 3; i++)
			{
				b[0][i] = wi[i];
				a[2][i] = e[i] * dtheta;
				b[2][i] = bvec[i];

				a[0][i] = b[0][i] * dt;
				a[1][i] = (b[2][i] * dt - 3.0*a[2][i]);

				b[1][i] = (2.0*a[0][i] + 2.0*a[1][i]) / dt;
				c[0][i] = (2.0*b[0][i] + b[1][i]) / dt;
				c[1][i] = (b[1][i] + 2.0*b[2][i]) / dt;

				d[i] = (c[0][i] + c[1][i]) / dt;
			}
		}


		/*
		purpose

		Subroutine slew3 computes the quaternion, body angular rate, acceleration and
		jerk as a function of time corresponding to a third-order polynomial
		interpolation function describing a slew between initial and final states.

		variable     i/o     description
		--------     ---     -----------
		t             i      current time (seconds from start).
		dt            i      slew time (sec).
		qi            i      initial attitude quaternion.
		q             o      current attitude quaternion.
		omega         o      current body angular rate (rad/sec).
		alpha         o      current body angular acceleration (rad/sec^2).
		jerk          o      current body angular jerk (rad/sec^3).
		a,b,c,d	      i	     acquired in _slew3_init().
		*/
		static void _slew3(double t, double dt, const double* qi, double* q, double* omega, double* alpha, double* jerk,
			const double a[3][3], const double b[3][3], const double c[2][3], const double d[3])
		{
			if (dt <= 0.0)
				return;

			double u[3], x1[2];
			double x = t / dt;
			x1[0] = x - 1.0;
			for (int i = 1; i < 2; i++)
				x1[i] = x1[i - 1] * x1[0];

			double th0[3], th1[3], th2[3], th3[3];
			for (int i = 0; i < 3; i++)
			{
				th0[i] = ((x*a[2][i] + x1[0] * a[1][i])*x + x1[1] * a[0][i])*x;
				th1[i] = (x*b[2][i] + x1[0] * b[1][i])*x + x1[1] * b[0][i];
				th2[i] = x*c[1][i] + x1[0] * c[0][i];
				th3[i] = d[i];
			}

			double ang = _unvec(th0, u);
			double ca = cos(0.5*ang);
			double sa = sin(0.5*ang);

			q[0] = ca*qi[0] + sa*(u[2] * qi[1] - u[1] * qi[2] + u[0] * qi[3]);
			q[1] = ca*qi[1] + sa*(-u[2] * qi[0] + u[0] * qi[2] + u[1] * qi[3]);
			q[2] = ca*qi[2] + sa*(u[1] * qi[0] - u[0] * qi[1] + u[2] * qi[3]);
			q[3] = ca*qi[3] + sa*(-u[0] * qi[0] - u[1] * qi[1] - u[2] * qi[2]);

			ca = cos(ang);
			sa = sin(ang);

			const double EPS = 1e-6;
			if (ang > EPS)
			{
				double tmp1[3];
				/* compute angular rate vector. */
				_crossp(u, th1, tmp1);

				double w[3];
				for (int i = 0; i < 3; i++)
					w[i] = tmp1[i] / ang;

				double udot[3];
				_crossp(w, u, udot);

				double thd1 = u[0] * th1[0] + u[1] * th1[1] + u[2] * th1[2];

				for (int i = 0; i < 3; i++)
					omega[i] = thd1*u[i] + sa*udot[i] - (1.0 - ca)*w[i];

				/* compute angular acceleration vector. */

				double thd2 = udot[0] * th1[0] + udot[1] * th1[1] + udot[2] * th1[2] + u[0] * th2[0] + u[1] * th2[1] + u[2] * th2[2];

				_crossp(u, th2, tmp1);

				double wd1[3];
				for (int i = 0; i < 3; i++)
					wd1[i] = (tmp1[i] - 2.0*thd1*w[i]) / ang;

				double wd1xu[3];
				_crossp(wd1, u, wd1xu);

				double tmp0[3];
				for (int i = 0; i < 3; i++)
					tmp0[i] = thd1*u[i] - w[i];

				_crossp(omega, tmp0, tmp1);

				for (int i = 0; i < 3; i++)
					alpha[i] = thd2*u[i] + sa*wd1xu[i] - (1.0 - ca)*wd1[i] + thd1*udot[i] + tmp1[i];

				/* compute angular jerk vector. */

				double w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];

				double thd3 = wd1xu[0] * th1[0] + wd1xu[1] * th1[1] + wd1xu[2] * th1[2] -
					w2*(u[0] * th1[0] + u[1] * th1[1] + u[2] * th1[2]) +
					2.0*(udot[0] * th2[0] + udot[1] * th2[1] + udot[2] * th2[2]) +
					u[0] * th3[0] + u[1] * th3[1] + u[2] * th3[2];

				_crossp(th1, th2, tmp1);

				for (int i = 0; i < 3; i++)
					tmp1[i] /= ang;

				double tmp2[3];
				_crossp(u, th3, tmp2);

				double td2 = (th1[0] * th1[0] + th1[1] * th1[1] + th1[2] * th1[2]) / ang;
				double ut2 = u[0] * th2[0] + u[1] * th2[1] + u[2] * th2[2];
				double wwd = w[0] * wd1[0] + w[1] * wd1[1] + w[2] * wd1[2];

				double wd2[3];
				for (int i = 0; i < 3; i++)
					wd2[i] = (tmp1[i] + tmp2[i] - 2.0*(td2 + ut2)*w[i] - 4.0*thd1*wd1[i]) / ang;

				double wd2xu[3];
				_crossp(wd2, u, wd2xu);


				for (int i = 0; i < 3; i++)
					tmp2[i] = thd2*u[i] + thd1*udot[i] - wd1[i];

				_crossp(omega, tmp2, tmp1);
				_crossp(alpha, tmp0, tmp2);

				for (int i = 0; i < 3; i++)
					jerk[i] = thd3*u[i] + sa*wd2xu[i] - (1.0 - ca)*wd2[i] +
					2.0*thd2*udot[i] + thd1*((1.0 + ca)*wd1xu[i] - w2*u[i] - sa*wd1[i]) -
					wwd*sa*u[i] + tmp1[i] + tmp2[i];
			}
			else
			{
				double tmp[3];
				_crossp(th1, th2, tmp);
				for (int i = 0; i < 3; i++)
				{
					omega[i] = th1[i];
					alpha[i] = th2[i];
					jerk[i] = th3[i] - 0.5*tmp[i];
				}
			}
		}

		/*
		purpose

		subroutine unvec unitizes a vector and computes its magnitude.

		variable     i/o     description
		--------     ---     -----------
		a             i      input vector.
		au            o      output unit vector.

		return value
		magnitude of vector a.
		*/
		static double _unvec(const double* a, double* au)
		{
			double amag = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

			if (amag > 0.0)
			{
				au[0] = a[0] / amag;
				au[1] = a[1] / amag;
				au[2] = a[2] / amag;
			}
			else
			{
				au[0] = 0.0;
				au[1] = 0.0;
				au[2] = 0.0;
			}
			return amag;
		}

		/*
		purpose

		subroutine crossp computes the vector cross product b x c.

		variable     i/o     description
		--------     ---     -----------
		b             i      input vector.
		c             i      input vector.
		a             o      output vector = b x c.
		*/
		static void _crossp(const double* b, const double* c, double* a)
		{
			a[0] = b[1] * c[2] - b[2] * c[1];
			a[1] = b[2] * c[0] - b[0] * c[2];
			a[2] = b[0] * c[1] - b[1] * c[0];
		}

	};
}

#endif