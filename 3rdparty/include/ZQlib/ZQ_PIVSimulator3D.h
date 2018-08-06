#ifndef _ZQ_PIV_SIMULATOR3D_H_
#define _ZQ_PIV_SIMULATOR3D_H_
#pragma once

#include "ZQ_PIVMovingObject3D.h"
#include "ZQ_PoissonSolver3D.h"
#include <vector>

namespace ZQ
{
	class ZQ_PIVSimulator3D
	{
	public:
		typedef ZQ_PIVMovingObject3D::BaseType BaseType;
		typedef ZQ_DImage3D<BaseType> DImage3D;
	public:
		ZQ_PIVSimulator3D(int width, int height, int depth, ZQ_PIVMovingObject3D* mvobj = 0)
		{
			this->width = width;
			this->height = height;
			this->depth = depth;

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

		~ZQ_PIVSimulator3D() {}

	private:

		int width, height, depth;

		std::vector<BaseType> par_x;
		std::vector<BaseType> par_y;
		std::vector<BaseType> par_z;
		std::vector<BaseType> par_intensity;
		std::vector<BaseType> par_radius;

		DImage3D regularU;
		DImage3D regularV;
		DImage3D regularW;

		ZQ_PIVMovingObject3D* mvobj;

		bool has_occupy;
		DImage3D global_texture;
		DImage3D global_occupy;
		DImage3D global_u;
		DImage3D global_v;
		DImage3D global_w;
		DImage3D global_mac_u;
		DImage3D global_mac_v;
		DImage3D global_mac_w;

	public:

		bool RandomInit(int particle_num, DImage3D& u, DImage3D& v, DImage3D& w, DImage3D& par_mask,
			const float max_radius = 6.0, const float min_radius = 3.0, const float max_density = 0.8, const float min_density = 0.6)
		{

			if (u.matchDimension(width, height, depth, 1) == false)
				return false;
			if (v.matchDimension(width, height, depth, 1) == false)
				return false;
			if (w.matchDimension(width, height, depth, 1) == false)
				return false;

			regularU.copyData(u);
			regularV.copyData(v);
			regularW.copyData(w);

			_Make_Divergence_Free();

			double par_max_radius = max_radius;
			double par_min_radius = min_radius;
			double par_max_intensity = max_density;
			double par_min_intensity = min_density;

			_Clear_Particles();

			for (int p = 0; p < particle_num; p++)
			{
				double x = rand() % 10001 / 10000.0*width;
				double y = rand() % 10001 / 10000.0*height;
				double z = rand() % 10001 / 10000.0*depth;
				int ix = x + 0.5;
				int iy = y + 0.5;
				int iz = z + 0.5;
				ix = __max(0, __min(width - 1, ix));
				iy = __max(0, __min(height - 1, iy));
				iz = __max(0, __min(depth - 1, iz));
				if (par_mask.data()[iz*height*width + iy*width + ix] < 0.5)
				{
					par_x.push_back((BaseType)x);
					par_y.push_back((BaseType)y);
					par_z.push_back((BaseType)z);

					par_intensity.push_back(rand() % 10001 / 10000.0*(par_max_intensity - par_min_intensity) + par_min_intensity);
					par_radius.push_back(rand() % 10001 / 10000.0*(par_max_radius - par_min_radius) + par_min_radius);
				}
			}
			return true;
		}

		void RunOneFrame(float dt, bool use_period_coord = false, bool advect_par = true)
		{
			bool* occupy = new bool[width*height*depth];
			memset(occupy, 0, sizeof(bool)*width*height*depth);

			if (has_occupy)
			{
				for (int i = 0; i < width*height*depth; i++)
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

		void ExportParticleImage(DImage3D& img)
		{
			if (img.matchDimension(width, height, depth, 1) == false)
				img.allocate(width, height, depth, 1);

			for (int p = 0; p < par_x.size(); p++)
				DrawOneParticle(img, par_x[p], par_y[p], par_z[p], par_intensity[p], par_radius[p], false);

			if (has_occupy)
			{
				for (int i = 0; i < depth*height*width; i++)
				{
					if (global_occupy.data()[i] > 0.5)
					{
						img.data()[i] = global_texture.data()[i];
					}
				}
			}
		}

		void ExportVelocity(DImage3D& u, DImage3D& v, DImage3D& w)
		{
			u.copyData(regularU);
			v.copyData(regularV);
			w.copyData(regularW);
		}

	private:

		void _Clear_Particles()
		{
			par_intensity.clear();
			par_radius.clear();
			par_x.clear();
			par_y.clear();
			par_z.clear();
		}

		void _Sample_Velocity_Tricubic(float normalized_x, float normalized_y, float normalized_z, BaseType& vx, BaseType& vy, BaseType& vz, bool use_period_coord)
		{
			float rx = normalized_x*width;
			float ry = normalized_y*height;
			float rz = normalized_z*depth;

			float u_coordx = rx - 0.5;
			float u_coordy = ry - 0.5;
			float u_coordz = rz - 0.5;
			float v_coordx = rx - 0.5;
			float v_coordy = ry - 0.5;
			float v_coordz = rz - 0.5;
			float w_coordx = rx - 0.5;
			float w_coordy = ry - 0.5;
			float w_coordz = rz - 0.5;

			ZQ_ImageProcessing3D::TricubicInterpolate(regularU.data(), width, height, depth, 1, u_coordx, u_coordy, u_coordz, &vx, use_period_coord);
			ZQ_ImageProcessing3D::TricubicInterpolate(regularV.data(), width, height, depth, 1, v_coordx, v_coordy, v_coordz, &vy, use_period_coord);
			ZQ_ImageProcessing3D::TricubicInterpolate(regularW.data(), width, height, depth, 1, w_coordx, w_coordy, w_coordz, &vz, use_period_coord);
		}

		void _RK4(float x, float y, float z, float dt, BaseType& out_x, BaseType& out_y, BaseType& out_z, bool use_period_coord)
		{
			BaseType u1, v1, w1;
			BaseType u2, v2, w2;
			BaseType u3, v3, w3;
			BaseType u4, v4, w4;

			// k1
			_Sample_Velocity_Tricubic((float)x / width, (float)y / height, (float)z / depth, u1, v1, w1, use_period_coord);
			
			//k2
			_Sample_Velocity_Tricubic((x + 0.5f*u1*dt) / width, (y + 0.5f*v1*dt) / height, (z + 0.5f*w1*dt) / depth, u2, v2, w2, use_period_coord);

			//k3
			_Sample_Velocity_Tricubic((x + 0.5f*u2*dt) / width, (y + 0.5f*v2*dt) / height, (z + 0.5f*w2*dt) / depth, u3, v3, w3, use_period_coord);

			//k4
			_Sample_Velocity_Tricubic((x + u3*dt) / width, (y + v3*dt) / height, (z + w3*dt) / depth, u4, v4, w4, use_period_coord);

			out_x = x + dt / 6.0f*(u1 + 2 * u2 + 2 * u3 + u4);
			out_y = y + dt / 6.0f*(v1 + 2 * v2 + 2 * v3 + v4);
			out_z = z + dt / 6.0f*(w1 + 2 * w2 + 2 * w3 + w4);
		}

		void _Simple_Advect(float x, float y, float z, float dt, BaseType& out_x, BaseType& out_y, BaseType& out_z, bool use_period_coord)
		{
			BaseType u, v, w;

			_Sample_Velocity_Tricubic((double)x / width, (double)y / height, (double)z / depth, u, v, w, use_period_coord);

			out_x = x + dt*u;
			out_y = y + dt*v;
			out_z = z + dt*w;
		}


		void _Advect_Particles(float dt, bool* occupy, bool use_period_coord)
		{
			int par_num = par_x.size();
			for (int p = 0; p < par_num; p++)
			{
				float x = par_x[p];
				float y = par_y[p];
				float z = par_z[p];
				BaseType tmpx, tmpy, tmpz;

				//_RK4(x,y,dt,tmpx,tmpy);
				_Simple_Advect(x, y, z, dt, tmpx, tmpy, tmpz, use_period_coord);

				par_x[p] = tmpx;
				par_y[p] = tmpy;
				par_z[p] = tmpz;
			}

			if (occupy)
			{
				std::vector<BaseType> p_x;
				std::vector<BaseType> p_y;
				std::vector<BaseType> p_z;
				std::vector<BaseType> p_intensity;
				std::vector<BaseType> p_radius;

				for (int p = 0; p < par_num; p++)
				{
					int ix = par_x[p] - floor(par_x[p] / width)*width;
					int iy = par_y[p] - floor(par_y[p] / height)*height;
					int iz = par_z[p] - floor(par_z[p] / depth)*depth;

					if (!occupy[iz*height*width + iy*width + ix])
					{
						p_x.push_back(par_x[p]);
						p_y.push_back(par_y[p]);
						p_z.push_back(par_z[p]);
						p_intensity.push_back(par_intensity[p]);
						p_radius.push_back(par_radius[p]);
					}
				}
				par_x = p_x;
				par_y = p_y;
				par_z = p_z;
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
				float z = par_z[i];
				x -= floor(x / width)*width;
				y -= floor(y / height)*height;
				z -= floor(z / depth)*depth;
				par_x[i] = x;
				par_y[i] = y;
				par_z[i] = z;
			}
		}

		void _Advect_Velocity(float dt, bool* occupy, bool use_period_coord)
		{

			if (occupy)
			{
				BaseType* uData = new BaseType[width*height*depth];
				BaseType* vData = new BaseType[width*height*depth];
				BaseType* wData = new BaseType[width*height*depth];

				for (int k = 0; k < depth; k++)
				{
					for (int j = 0; j < height; j++)
					{
						for (int i = 0; i < width; i++)
						{
							if (occupy[k*height*width + j*width + i])
								continue;
							float x = i + 0.5;
							float y = j + 0.5;
							float z = k + 0.5;
							BaseType tmpx, tmpy, tmpz;
							BaseType tmpu, tmpv, tmpw;
							_RK4(x, y, z, -dt, tmpx, tmpy, tmpz, use_period_coord);
							_Sample_Velocity_Tricubic(tmpx / width, tmpy / height, tmpz / depth, tmpu, tmpv, tmpw, use_period_coord);
							uData[k*height*width + j*width + i] = tmpu;
							vData[k*height*width + j*width + i] = tmpv;
							wData[k*height*width + j*width + i] = tmpw;
						}
					}
				}

				memcpy(regularU.data(), uData, sizeof(BaseType)*width*height*depth);
				memcpy(regularV.data(), vData, sizeof(BaseType)*width*height*depth);
				memcpy(regularW.data(), wData, sizeof(BaseType)*width*height*depth);
				delete[]uData;
				delete[]vData;
				delete[]wData;
			}
			else
			{
				BaseType* uData = new BaseType[width*height*depth];
				BaseType* vData = new BaseType[width*height*depth];
				BaseType* wData = new BaseType[width*height*depth];

				for (int k = 0; k < depth; k++)
				{
					for (int j = 0; j < height; j++)
					{
						for (int i = 0; i < width; i++)
						{
							float x = i + 0.5;
							float y = j + 0.5;
							float z = k + 0.5;
							BaseType tmpx, tmpy, tmpz;
							BaseType tmpu, tmpv, tmpw;
							_RK4(x, y, z, -dt, tmpx, tmpy, tmpz, use_period_coord);
							_Sample_Velocity_Tricubic(tmpx / width, tmpy / height, tmpz / depth, tmpu, tmpv, tmpw, use_period_coord);
							uData[k*height*width + j*width + i] = tmpu;
							vData[k*height*width + j*width + i] = tmpv;
							wData[k*height*width + j*width + i] = tmpw;
						}
					}
				}

				memcpy(regularU.data(), uData, sizeof(BaseType)*width*height*depth);
				memcpy(regularV.data(), vData, sizeof(BaseType)*width*height*depth);
				memcpy(regularW.data(), wData, sizeof(BaseType)*width*height*depth);
				delete[]uData;
				delete[]vData;
				delete[]wData;
			}
		}

		void _Make_Divergence_Free()
		{
			int nSORIterations = __max(100, __min(5000, sqrt((double)width*height*depth)*5.0));

			if (!has_occupy)
				ZQ_PoissonSolver3D::SolveOpenPoissonSOR(regularU.data(), regularV.data(), regularW.data(), width, height, depth, nSORIterations);
			else
			{
				mvobj->ExportToGlobal(width, height, depth, global_occupy, global_texture, global_u, global_v, global_w, global_mac_u, global_mac_v, global_mac_w);

				bool* cur_occupy = new bool[width*height*depth];
				for (int i = 0; i < width*height*depth; i++)
					cur_occupy[i] = global_occupy.data()[i] > 0.5;

				DImage3D cur_mac_u(width + 1, height, depth), cur_mac_v(width, height + 1, depth), cur_mac_w(width, height, depth + 1);
				ZQ_PoissonSolver3D::RegularGridtoMAC(width, height, depth, regularU.data(), regularV.data(), regularW.data(), cur_mac_u.data(), cur_mac_v.data(), cur_mac_w.data(),false);

				for (int k = 0; k < depth; k++)
				{
					for (int j = 0; j < height; j++)
					{
						for (int i = 0; i < width; i++)
						{
							if (global_occupy.data()[k*height*width + j*width + i] > 0.5)
							{
								cur_mac_u.data()[k*height*(width + 1) + j*(width + 1) + i] = global_mac_u.data()[k*height*(width + 1) + j*(width + 1) + i];
								cur_mac_u.data()[k*height*(width + 1) + j*(width + 1) + i + 1] = global_mac_u.data()[k*height*(width + 1) + j*(width + 1) + i + 1];
								cur_mac_v.data()[k*(height + 1)*width + j*width + i] = global_mac_v.data()[k*(height + 1)*width + j*width + i];
								cur_mac_v.data()[k*(height + 1)*width + (j + 1)*width + i] = global_mac_v.data()[k*(height + 1)*width + (j + 1)*width + i];
								cur_mac_w.data()[k*height*width + j*width + i] = global_mac_w.data()[k*height*width + j*width + i];
								cur_mac_w.data()[(k + 1)*height*width + j*width + i] = global_mac_w.data()[(k + 1)*height*width + j*width + i];
							}
						}
					}
				}

				ZQ_PoissonSolver3D::SolveOpenPoissonSOR_MACGrid(cur_mac_u.data(), cur_mac_v.data(), cur_mac_w.data(), width, height, depth, cur_occupy, nSORIterations, false);

				ZQ_PoissonSolver3D::MACtoRegularGrid(width, height, depth, cur_mac_u.data(), cur_mac_v.data(), cur_mac_w.data(), regularU.data(), regularV.data(), regularW.data());

				for (int i = 0; i < depth*height*width; i++)
				{
					if (cur_occupy[i])
					{
						regularU.data()[i] = global_u.data()[i];
						regularV.data()[i] = global_v.data()[i];
						regularW.data()[i] = global_w.data()[i];
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
				//return exp(-4.5*(distance*distance)/(radius*radius));
				return exp(-0.5*(distance*distance) / (radius*radius));
		}

		static bool DrawOneParticle(DImage3D& img, float x, float y, float z, BaseType intensity, float radius, bool use_period_coord = false)
		{
			if (!use_period_coord)
			{
				BaseType* pData = img.data();
				int width = img.width();
				int height = img.height();
				int depth = img.depth();
				int nChannels = img.nchannels();
				if (nChannels != 1)
					return false;

				for (int k = __max(0, floor(z - radius)); k < __min(depth - 1, ceil(z + radius)); k++)
				{
					for (int j = __max(0, floor(y - radius)); j < __min(height - 1, ceil(y + radius)); j++)
					{
						for (int i = __max(0, floor(x - radius)); i < __min(width - 1, ceil(x + radius)); i++)
						{
							double dis = sqrt((k - z)*(k - z) + (j - y)*(j - y) + (i - x)*(i - x));
							BaseType old_value = pData[k*height*width + j*width + i];
							BaseType new_value = intensity*GaussianKernel(dis, radius);

							pData[k*height*width + j*width + i] = fabs(old_value) > fabs(new_value) ? old_value : new_value;
						}
					}
				}

				return true;

			}
			else
			{
				BaseType* pData = img.data();
				int width = img.width();
				int height = img.height();
				int depth = img.depth();
				int nChannels = img.nchannels();
				if (nChannels != 1)
					return false;

				for (int k = floor(z - radius); k < ceil(z + radius); k++)
				{
					for (int j = floor(y - radius); j < ceil(y + radius); j++)
					{
						for (int i = floor(x - radius); i < ceil(x + radius); i++)
						{
							double dis = sqrt((k - z)*(k - z) + (j - y)*(j - y) + (i - x)*(i - x));
							int kk = k - floor((double)k / depth)*depth;
							int jj = j - floor((double)j / height)*height;
							int ii = i - floor((double)i / width)*width;
							BaseType old_value = pData[kk*height*width + jj*width + ii];
							BaseType new_value = intensity*GaussianKernel(dis, radius);

							pData[kk*height*width + jj*width + ii] = fabs(old_value) > fabs(new_value) ? old_value : new_value;
						}
					}
				}

				return true;

			}
		}
	};
}
#endif