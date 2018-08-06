#ifndef _ZQ_PIV_SIMULATOR_H_
#define _ZQ_PIV_SIMULATOR_H_
#pragma once

#include "ZQ_PIVMovingObject.h"
#include "ZQ_PoissonSolver.h"
#include "ZQ_DoubleImage.h"
#include <vector>

namespace ZQ
{
	class ZQ_PIVSimulator
	{
	public:
		typedef ZQ_PIVMovingObject::BaseType BaseType;
		typedef ZQ_DImage<BaseType> DImage;
	public:
		ZQ_PIVSimulator(int width, int height, ZQ_PIVMovingObject* mvobj = 0)
		{
			this->width = width;
			this->height = height;

			if (mvobj == 0)
			{
				has_occupy = false;
			}
			else
			{
				this->mvobj = mvobj;
				has_occupy = true;
			}
		}

		~ZQ_PIVSimulator() {}

	private:

		int width, height;

		std::vector<BaseType> par_x;
		std::vector<BaseType> par_y;
		std::vector<BaseType> par_intensity;
		std::vector<BaseType> par_radius;

		DImage regularU;
		DImage regularV;

		ZQ_PIVMovingObject* mvobj;

		bool has_occupy;
		DImage global_texture;
		DImage global_occupy;
		DImage global_u;
		DImage global_v;
		DImage global_mac_u;
		DImage global_mac_v;

	public:

		bool RandomInit(int particle_num, DImage& u, DImage& v, DImage& par_mask)
		{
			if (u.matchDimension(width, height, 1) == false)
				return false;
			if (v.matchDimension(width, height, 1) == false)
				return false;

			regularU.copyData(u);
			regularV.copyData(v);

			_Make_Divergence_Free();

			const double par_max_radius = 3;
			const double par_min_radius = 1.5;
			const double par_max_intensity = 0.8;
			const double par_min_intensity = 0.3;

			_Clear_Particles();

			for (int p = 0; p < particle_num; p++)
			{
				double x = rand() % 10001 / 10000.0*width;
				double y = rand() % 10001 / 10000.0*height;
				int ix = x + 0.5;
				int iy = y + 0.5;
				ix = __max(0, __min(width - 1, ix));
				iy = __max(0, __min(height - 1, iy));
				if (par_mask.data()[iy*width + ix] < 0.5)
				{
					par_x.push_back(x);
					par_y.push_back(y);
					par_intensity.push_back(rand() % 10001 / 10000.0*(par_max_intensity - par_min_intensity) + par_min_intensity);
					par_radius.push_back(rand() % 10001 / 10000.0*(par_max_radius - par_min_radius) + par_min_radius);
				}
			}
			return true;
		}

		void RunOneFrame(float dt, bool use_period_coord = false, bool advect_par = true)
		{
			bool* occupy = new bool[width*height];
			memset(occupy, 0, sizeof(bool)*width*height);
			if (has_occupy)
			{
				for (int i = 0; i < width*height; i++)
				{
					occupy[i] = global_occupy.data()[i] > 0.5;
				}
			}
			if (advect_par)
			{
				_Advect_Particles(dt, occupy, use_period_coord);
				if (use_period_coord)
				{
					_PeroidParticleCoord();
				}
			}
			_Advect_Velocity(dt, occupy, use_period_coord);
			if (has_occupy)
				mvobj->UpdateOneFrame();
			_Make_Divergence_Free();
			delete[]occupy;
		}

		void ExportParticleImage(DImage& img)
		{
			if (img.matchDimension(width, height, 1) == false)
				img.allocate(width, height, 1);

			for (int p = 0; p < par_x.size(); p++)
				DrawOneParticle(img, par_x[p], par_y[p], par_intensity[p], par_radius[p], false);

			if (has_occupy)
			{
				for (int i = 0; i < height*width; i++)
				{
					if (global_occupy.data()[i] > 0.5)
					{
						img.data()[i] = global_texture.data()[i];
					}
				}
			}
		}

		void ExportVelocity(DImage& u, DImage& v)
		{
			u.copyData(regularU);
			v.copyData(regularV);
		}

	private:

		void _Clear_Particles()
		{
			par_intensity.clear();
			par_radius.clear();
			par_x.clear();
			par_y.clear();
		}

		void _Sample_Velocity_Bicubic(float normalized_x, float normalized_y, BaseType& vx, BaseType& vy, bool use_period_coord)
		{
			float rx = normalized_x*width;
			float ry = normalized_y*height;

			float u_coordx = rx - 0.5;
			float u_coordy = ry - 0.5;
			float v_coordx = rx - 0.5;
			float v_coordy = ry - 0.5;

			ZQ_ImageProcessing::BicubicInterpolate(regularU.data(), width, height, 1, u_coordx, u_coordy, &vx, use_period_coord);
			ZQ_ImageProcessing::BicubicInterpolate(regularV.data(), width, height, 1, v_coordx, v_coordy, &vy, use_period_coord);
		}

		void _RK4(float x, float y, float dt, BaseType& out_x, BaseType& out_y, bool use_period_coord)
		{
			BaseType u1, v1;
			BaseType u2, v2;
			BaseType u3, v3;
			BaseType u4, v4;

			// k1
			_Sample_Velocity_Bicubic((float)x / width, (float)y / height, u1, v1, use_period_coord);


			//k2
			_Sample_Velocity_Bicubic((x + 0.5f*u1*dt) / width, (y + 0.5f*v1*dt) / height, u2, v2, use_period_coord);

			//k3
			_Sample_Velocity_Bicubic((x + 0.5f*u2*dt) / width, (y + 0.5f*v2*dt) / height, u3, v3, use_period_coord);

			//k4
			_Sample_Velocity_Bicubic((x + u3*dt) / width, (y + v3*dt) / height, u4, v4, use_period_coord);

			out_x = x + dt / 6.0f*(u1 + 2 * u2 + 2 * u3 + u4);
			out_y = y + dt / 6.0f*(v1 + 2 * v2 + 2 * v3 + v4);
		}

		void _Simple_Advect(float x, float y, float dt, BaseType& out_x, BaseType& out_y, bool use_period_coord)
		{
			BaseType u, v;

			_Sample_Velocity_Bicubic((double)x / width, (double)y / height, u, v, use_period_coord);

			out_x = x + dt*u;
			out_y = y + dt*v;
		}

		void _Advect_Particles(float dt, bool* occupy, bool use_period_coord)
		{
			int par_num = par_x.size();
			for (int p = 0; p < par_num; p++)
			{
				float x = par_x[p];
				float y = par_y[p];
				BaseType tmpx, tmpy;

				//_RK4(x,y,dt,tmpx,tmpy);
				_Simple_Advect(x, y, dt, tmpx, tmpy, use_period_coord);

				par_x[p] = tmpx;
				par_y[p] = tmpy;
			}

			if (occupy)
			{
				std::vector<BaseType> p_x;
				std::vector<BaseType> p_y;
				std::vector<BaseType> p_intensity;
				std::vector<BaseType> p_radius;

				for (int p = 0; p < par_num; p++)
				{
					int ix = par_x[p] - floor(par_x[p] / width)*width;
					int iy = par_y[p] - floor(par_y[p] / height)*height;

					if (!occupy[iy*width + ix])
					{
						p_x.push_back(par_x[p]);
						p_y.push_back(par_y[p]);
						p_intensity.push_back(par_intensity[p]);
						p_radius.push_back(par_radius[p]);
					}
				}
				par_x = p_x;
				par_y = p_y;
				par_intensity = p_intensity;
				par_radius = p_radius;
			}
		}

		void _PeroidParticleCoord()
		{
			for (int i = 0; i < par_x.size(); i++)
			{
				float x = par_x[i];
				float y = par_y[i];
				x -= floor(x / width)*width;
				y -= floor(y / height)*height;
				par_x[i] = x;
				par_y[i] = y;
			}
		}

		void _Advect_Velocity(float dt, bool* occupy, bool use_period_coord)
		{
			if (occupy)
			{
				BaseType* uData = new BaseType[width*height];
				BaseType* vData = new BaseType[width*height];
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (occupy[i*width + j])
							continue;
						float x = j + 0.5;
						float y = i + 0.5;
						BaseType tmpx, tmpy;
						BaseType tmpu, tmpv;
						_RK4(x, y, -dt, tmpx, tmpy, use_period_coord);
						_Sample_Velocity_Bicubic(tmpx / width, tmpy / height, tmpu, tmpv, use_period_coord);
						uData[i*width + j] = tmpu;
						vData[i*width + j] = tmpv;
					}
				}
				memcpy(regularU.data(), uData, sizeof(BaseType)*width*height);
				memcpy(regularV.data(), vData, sizeof(BaseType)*width*height);
				delete[]uData;
				delete[]vData;
			}
			else
			{
				BaseType* uData = new BaseType[width*height];
				BaseType* vData = new BaseType[width*height];
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						float x = j + 0.5;
						float y = i + 0.5;
						BaseType tmpx, tmpy;
						BaseType tmpu, tmpv;
						_RK4(x, y, -dt, tmpx, tmpy, use_period_coord);
						_Sample_Velocity_Bicubic(tmpx / width, tmpy / height, tmpu, tmpv, use_period_coord);
						uData[i*width + j] = tmpu;
						vData[i*width + j] = tmpv;
					}
				}
				memcpy(regularU.data(), uData, sizeof(BaseType)*width*height);
				memcpy(regularV.data(), vData, sizeof(BaseType)*width*height);
				delete[]uData;
				delete[]vData;

			}
		}

		void _Make_Divergence_Free()
		{
			int datatype = sizeof(BaseType) == sizeof(double) ? ZQ_DOUBLE : ZQ_FLOAT;
			int nSORIterations = __max(100, __min(5000, sqrt((double)width*height)*5.0));
			if (!has_occupy)
				ZQ_PoissonSolver::SolveOpenPoissonSOR(regularU.data(), regularV.data(), width, height, nSORIterations, datatype);
			else
			{
				mvobj->ExportToGlobal(width, height, global_occupy, global_texture, global_u, global_v, global_mac_u, global_mac_v);
				bool* cur_occupy = new bool[width*height];
				for (int i = 0; i < width*height; i++)
					cur_occupy[i] = global_occupy.data()[i] > 0.5;
				DImage cur_mac_u(width + 1, height), cur_mac_v(width, height + 1);
				ZQ_PoissonSolver::RegularGridtoMAC(width, height, regularU.data(), regularV.data(), cur_mac_u.data(), cur_mac_v.data(), datatype);
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (global_occupy.data()[i*width + j] > 0.5)
						{
							cur_mac_u.data()[i*(width + 1) + j] = global_mac_u.data()[i*(width + 1) + j];
							cur_mac_u.data()[i*(width + 1) + j + 1] = global_mac_u.data()[i*(width + 1) + j + 1];
							cur_mac_v.data()[i*width + j] = global_mac_v.data()[i*width + j];
							cur_mac_v.data()[(i + 1)*width + j] = global_mac_v.data()[(i + 1)*width + j];
						}
					}
				}

				ZQ_PoissonSolver::SolveOpenPoissonSOR_MACGrid(cur_mac_u.data(), cur_mac_v.data(), width, height, cur_occupy, nSORIterations, false);
				ZQ_PoissonSolver::MACtoRegularGrid(width, height, cur_mac_u.data(), cur_mac_v.data(), regularU.data(), regularV.data());

				for (int i = 0; i < height*width; i++)
				{
					if (cur_occupy[i])
					{
						regularU.data()[i] = global_u.data()[i];
						regularV.data()[i] = global_v.data()[i];
					}
				}
				delete[]cur_occupy;
			}
		}

	public:
		static double GaussianKernel(double distance, double radius)
		{
			if (distance > radius)
				return 0;
			else
				return exp(-4.5*(distance*distance) / (radius*radius));
		}

		static bool DrawOneParticle(DImage& img, float x, float y, BaseType intensity, float radius, bool use_period_coord = false)
		{
			if (!use_period_coord)
			{
				BaseType* pData = img.data();
				int width = img.width();
				int height = img.height();
				int nChannels = img.nchannels();
				if (nChannels != 1)
					return false;

				for (int i = __max(0, floor(y - radius)); i < __min(height - 1, ceil(y + radius)); i++)
				{
					for (int j = __max(0, floor(x - radius)); j < __min(width - 1, ceil(x + radius)); j++)
					{
						double dis = sqrt((j - x)*(j - x) + (i - y)*(i - y));
						BaseType old_value = pData[i*width + j];
						BaseType new_value = intensity*GaussianKernel(dis, radius);

						pData[i*width + j] = fabs(old_value) > fabs(new_value) ? old_value : new_value;
					}
				}
				return true;
			}
			else
			{
				BaseType* pData = img.data();
				int width = img.width();
				int height = img.height();
				int nChannels = img.nchannels();
				if (nChannels != 1)
					return false;

				for (int i = floor(y - radius); i < ceil(y + radius); i++)
				{
					for (int j = floor(x - radius); j < ceil(x + radius); j++)
					{
						double dis = sqrt((j - x)*(j - x) + (i - y)*(i - y));
						int ii = i - floor((double)i / height)*height;
						int jj = j - floor((double)j / width)*width;
						BaseType old_value = pData[ii*width + jj];
						BaseType new_value = intensity*GaussianKernel(dis, radius);

						pData[ii*width + jj] = fabs(old_value) > fabs(new_value) ? old_value : new_value;
					}
				}
				return true;
			}
		}
	};
}
#endif