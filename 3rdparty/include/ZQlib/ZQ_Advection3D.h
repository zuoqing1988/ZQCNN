#ifndef _ZQ_ADVECTION_3D_H_
#define _ZQ_ADVECTION_3D_H_
#pragma once

#include "ZQ_ImageProcessing3D.h"
#include "ZQ_Vec3D.h"
#include <vector>

namespace ZQ
{
	namespace ZQ_Advection3D
	{
		/* in_pos and out_pos are real coordinates with regarding voxel_len, NOT image coordinates,
		make sure voxel_xlen , voxel_ylen, substeps are valid,
		if occupy == NULL, it means no obstacles in the scene
		*/
		template<class T>  
		void ZQ_BacktraceAdvection(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos);

		template<class T>  
		void ZQ_ForwardAdvection(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos);


		template<class T>
		void ZQ_BacktraceAdvection_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_ForwardAdvection_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_BacktraceAdvectionOnePositionOneStep(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_ForwardAdvectionOnePositionOneStep(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_BacktraceAdvectionOnePositionOneStep_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_ForwardAdvectionOnePositionOneStep_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const T* in_pos, T* out_pos);

		template<class T>
		void ZQ_BacktraceStreamlineRK4(const T* flow, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels);

		template<class T>
		void ZQ_ForwardStreamlineRK4(const T* flow, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels);

		template<class T>
		void ZQ_BacktraceStreamlineRK4_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels);

		template<class T>
		void ZQ_ForwardStreamlineRK4_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels);

		/*********************************************************************************/
		/********************************** definitions **********************************/
		/*********************************************************************************/

		template<class T>
		void ZQ_BacktraceAdvection(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos)
		{
			memcpy(out_pos,in_pos,sizeof(T)*3*nPts);

			float substepTime = time/substeps;

			for(int i = 0;i < nPts;i++)
			{

				T last_pos[3] = {in_pos[i*3],in_pos[i*3+1],in_pos[i*3+2]};

				float cur_x = last_pos[0]/voxel_xlen;
				float cur_y = last_pos[1]/voxel_ylen;
				float cur_z = last_pos[2]/voxel_zlen;
				int ix = __min(width-1,__max(0,(int)cur_x));
				int iy = __min(height-1,__max(0,(int)cur_y));
				int iz = __min(depth-1,__max(0,(int)cur_z));

				if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					continue;

				for(int ss = 0;ss < substeps;ss++)
				{
					ZQ_BacktraceAdvectionOnePositionOneStep(u,v,w,occupy,width,height,depth,voxel_xlen,voxel_ylen,voxel_zlen,substepTime,last_pos,out_pos+i*3);
					cur_x = out_pos[i*3+0]/voxel_xlen;
					cur_y = out_pos[i*3+1]/voxel_ylen;
					cur_z = out_pos[i*3+2]/voxel_zlen;
					ix = __min(width-1,__max(0,(int)cur_x));
					iy = __min(height-1,__max(0,(int)cur_y));
					iz = __min(depth-1,__max(0,(int)cur_z));
					if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					{
						break;
					}
					else
					{
						last_pos[0] = out_pos[i*3+0];
						last_pos[1] = out_pos[i*3+1];
						last_pos[2] = out_pos[i*3+2];
					}
				}
				out_pos[i*3+0] = last_pos[0];
				out_pos[i*3+1] = last_pos[1];
				out_pos[i*3+2] = last_pos[2];
			}
		}

		template<class T>
		void ZQ_ForwardAdvection(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos)
		{
			memcpy(out_pos,in_pos,sizeof(T)*3*nPts);

			float substepTime = time/substeps;

			for(int i = 0;i < nPts;i++)
			{
				T last_pos[3] = {in_pos[i*3],in_pos[i*3+1],in_pos[i*3+2]};

				float cur_x = last_pos[0]/voxel_xlen;
				float cur_y = last_pos[1]/voxel_ylen;
				float cur_z = last_pos[2]/voxel_zlen;
				int ix = __min(width-1,__max(0,(int)cur_x));
				int iy = __min(height-1,__max(0,(int)cur_y));
				int iz = __min(depth-1,__max(0,(int)cur_z));

				if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					continue;

				for(int ss = 0;ss < substeps;ss++)
				{
					ZQ_ForwardAdvectionOnePositionOneStep(u,v,w,occupy,width,height,depth,voxel_xlen,voxel_ylen,voxel_zlen,substepTime,last_pos,out_pos+i*3);
					cur_x = out_pos[i*3+0]/voxel_xlen;
					cur_y = out_pos[i*3+1]/voxel_ylen;
					cur_z = out_pos[i*3+2]/voxel_zlen;
					ix = __min(width-1,__max(0,(int)cur_x));
					iy = __min(height-1,__max(0,(int)cur_y));
					iz = __min(depth-1,__max(0,(int)cur_z));
					if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					{
						break;
					}
					else
					{
						last_pos[0] = out_pos[i*3+0];
						last_pos[1] = out_pos[i*3+1];
						last_pos[2] = out_pos[i*3+2];
					}
				}
				out_pos[i*3+0] = last_pos[0];
				out_pos[i*3+1] = last_pos[1];
				out_pos[i*3+2] = last_pos[2];
			}
		}

		template<class T>
		void ZQ_BacktraceAdvection_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos)
		{
			memcpy(out_pos,in_pos,sizeof(T)*3*nPts);

			float substepTime = time/substeps;

			for(int i = 0;i < nPts;i++)
			{
				T last_pos[3] = {in_pos[i*3],in_pos[i*3+1],in_pos[i*3+2]};

				float cur_x = last_pos[0]/voxel_xlen;
				float cur_y = last_pos[1]/voxel_ylen;
				float cur_z = last_pos[2]/voxel_zlen;
				int ix = __min(width-1,__max(0,(int)cur_x));
				int iy = __min(height-1,__max(0,(int)cur_y));
				int iz = __min(depth-1,__max(0,(int)cur_z));

				if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					continue;

				for(int ss = 0;ss < substeps;ss++)
				{
					ZQ_BacktraceAdvectionOnePositionOneStep_MACGrid(mac_u,mac_v,mac_w,occupy,width,height,depth,voxel_xlen,voxel_ylen,voxel_zlen,substepTime,last_pos,out_pos+i*3);
					cur_x = out_pos[i*3+0]/voxel_xlen;
					cur_y = out_pos[i*3+1]/voxel_ylen;
					cur_z = out_pos[i*3+2]/voxel_zlen;
					ix = __min(width-1,__max(0,(int)cur_x));
					iy = __min(height-1,__max(0,(int)cur_y));
					iz = __min(depth-1,__max(0,(int)cur_z));
					if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					{
						break;
					}
					else
					{
						last_pos[0] = out_pos[i*3+0];
						last_pos[1] = out_pos[i*3+1];
						last_pos[2] = out_pos[i*3+2];
					}
				}
				out_pos[i*3+0] = last_pos[0];
				out_pos[i*3+1] = last_pos[1];
				out_pos[i*3+2] = last_pos[2];
			}
		}


		template<class T>
		void ZQ_ForwardAdvection_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const int substeps, const int nPts, const T* in_pos, T* out_pos)
		{
			memcpy(out_pos,in_pos,sizeof(T)*3*nPts);

			float substepTime = time/substeps;

			for(int i = 0;i < nPts;i++)
			{
				T last_pos[3] = {in_pos[i*3],in_pos[i*3+1],in_pos[i*3+2]};

				float cur_x = last_pos[0]/voxel_xlen;
				float cur_y = last_pos[1]/voxel_ylen;
				float cur_z = last_pos[2]/voxel_zlen;
				int ix = __min(width-1,__max(0,(int)cur_x));
				int iy = __min(height-1,__max(0,(int)cur_y));
				int iz = __min(depth-1,__max(0,(int)cur_z));

				if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					continue;

				for(int ss = 0;ss < substeps;ss++)
				{
					ZQ_ForwardAdvectionOnePositionOneStep_MACGrid(mac_u,mac_v,mac_w,occupy,width,height,depth,voxel_xlen,voxel_ylen,voxel_zlen,substepTime,last_pos,out_pos+i*3);
					cur_x = out_pos[i*3+0]/voxel_xlen;
					cur_y = out_pos[i*3+1]/voxel_ylen;
					cur_z = out_pos[i*3+2]/voxel_zlen;
					ix = __min(width-1,__max(0,(int)cur_x));
					iy = __min(height-1,__max(0,(int)cur_y));
					iz = __min(depth-1,__max(0,(int)cur_z));
					if(cur_x < 0 || cur_x > width || cur_y < 0 || cur_y > height || cur_z < 0 || cur_z > depth || (occupy != 0 && occupy[iz*height*width+iy*width+ix]))
					{
						break;
					}
					else
					{
						last_pos[0] = out_pos[i*3+0];
						last_pos[1] = out_pos[i*3+1];
						last_pos[2] = out_pos[i*3+2];
					}
				}
				out_pos[i*3+0] = last_pos[0];
				out_pos[i*3+1] = last_pos[1];
				out_pos[i*3+2] = last_pos[2];
			}
		}

		template<class T>
		void ZQ_BacktraceAdvectionOnePositionOneStep(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen,
			const float time, const T* in_pos, T* out_pos)
		{
			float cur_x = in_pos[0]/voxel_xlen;
			float cur_y = in_pos[1]/voxel_ylen;
			float cur_z = in_pos[2]/voxel_zlen;

			T cur_u = 0, cur_v = 0, cur_w = 0;
			ZQ_ImageProcessing3D::TrilinearInterpolate(u,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_u,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(v,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_v,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(w,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_w,false);

			out_pos[0] = in_pos[0] - cur_u*time;
			out_pos[1] = in_pos[1] - cur_v*time;
			out_pos[2] = in_pos[2] - cur_w*time;
		}

		template<class T>
		void ZQ_ForwardAdvectionOnePositionOneStep(const T* u, const T* v, const T* w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float time, const T* in_pos, T* out_pos)
		{
			float cur_x = in_pos[0]/voxel_xlen;
			float cur_y = in_pos[1]/voxel_ylen;
			float cur_z = in_pos[2]/voxel_zlen;

			T cur_u = 0, cur_v = 0, cur_w = 0;
			ZQ_ImageProcessing3D::TrilinearInterpolate(u,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_u,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(v,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_v,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(w,width,height,depth,1,cur_x-0.5f,cur_y-0.5f,cur_z-0.5f,&cur_w,false);

			out_pos[0] = in_pos[0] + cur_u*time;
			out_pos[1] = in_pos[1] + cur_v*time;
			out_pos[2] = in_pos[2] + cur_w*time;
		}

		template<class T>
		void ZQ_BacktraceAdvectionOnePositionOneStep_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen,  const float voxel_zlen, 
			const float time, const T* in_pos, T* out_pos)
		{
			float cur_x = in_pos[0]/voxel_xlen;
			float cur_y = in_pos[1]/voxel_ylen;
			float cur_z = in_pos[2]/voxel_zlen;

			T cur_u = 0, cur_v = 0, cur_w = 0;
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,cur_x,cur_y-0.5f,cur_z-0.5f,&cur_u,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,cur_x-0.5f,cur_y,cur_z-0.5f,&cur_v,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,cur_x-0.5f,cur_y-0.5f,cur_z,&cur_w,false);

			out_pos[0] = in_pos[0] - cur_u*time;
			out_pos[1] = in_pos[1] - cur_v*time;
			out_pos[2] = in_pos[2] - cur_w*time;
		}

		template<class T>
		void ZQ_ForwardAdvectionOnePositionOneStep_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const bool* occupy, const int width, const int height, const int depth, float voxel_xlen, float voxel_ylen, float voxel_zlen, 
			const float time, const T* in_pos, T* out_pos)
		{
			float cur_x = in_pos[0]/voxel_xlen;
			float cur_y = in_pos[1]/voxel_ylen;
			float cur_z = in_pos[2]/voxel_zlen;

			T cur_u = 0, cur_v = 0, cur_w = 0;
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,cur_x,cur_y-0.5f,cur_z-0.5f,&cur_u,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,cur_x-0.5f,cur_y,cur_z-0.5f,&cur_v,false);
			ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,cur_x-0.5f,cur_y-0.5f,cur_z,&cur_w,false);

			out_pos[0] = in_pos[0] + cur_u*time;
			out_pos[1] = in_pos[1] + cur_v*time;
			out_pos[2] = in_pos[2] + cur_w*time;
		}

		template<class T>
		void ZQ_BacktraceStreamlineRK4(const T* flow, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels)
		{
			ZQ_Vec3D k1,k2,k3,k4;
			ZQ_Vec3D p1,p2,p3,p4;

			T result[3];
			memset(result,0,sizeof(T)*3);

			points.clear();
			vels.clear();

			int cur_step = 0;
			ZQ_Vec3D cur_pos = start_pos;
			
			while(cur_step < max_steps)
			{
				p1 = cur_pos;
				float coord_x = p1.x/voxel_xlen-0.5;
				float coord_y = p1.y/voxel_ylen-0.5;
				float coord_z = p1.z/voxel_zlen-0.5;

				if(cur_step != 0 && (coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1))
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k1.x = result[0];
				k1.y = result[1];
				k1.z = result[2];

				points.push_back(p1);
				vels.push_back(k1);

				p2 = cur_pos - k1*0.5*step_time;
				coord_x = p2.x/voxel_xlen-0.5;
				coord_y = p2.y/voxel_ylen-0.5;
				coord_z = p2.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k2.x = result[0];
				k2.y = result[1];
				k2.z = result[2];

				p3 = cur_pos - k2*0.5*step_time;
				coord_x = p3.x/voxel_xlen-0.5;
				coord_y = p3.y/voxel_ylen-0.5;
				coord_z = p3.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k3.x = result[0];
				k3.y = result[1];
				k3.z = result[2];

				p4 = cur_pos - k3*1.0*step_time;
				coord_x = p4.x/voxel_xlen-0.5;
				coord_y = p4.y/voxel_ylen-0.5;
				coord_z = p4.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k4.x = result[0];
				k4.y = result[1];
				k4.z = result[2];

				ZQ_Vec3D dir = (k1+k2*2.0+k3*2.0+k4)*(1.0/6.0);

				if(dir*dir*step_time*step_time < step_len*step_len*1e-8)
					break;

				dir *= 1.0/sqrt(dir*dir);
				dir *= step_len;

				cur_pos = cur_pos - dir;

				cur_step++;
			}
		}

		template<class T>
		void ZQ_ForwardStreamlineRK4(const T* flow, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels)
		{
			ZQ_Vec3D k1,k2,k3,k4;
			ZQ_Vec3D p1,p2,p3,p4;

			T result[3];
			memset(result,0,sizeof(T)*3);

			points.clear();

			int cur_step = 0;
			ZQ_Vec3D cur_pos = start_pos;

			while(cur_step < max_steps)
			{
				p1 = cur_pos;
				float coord_x = p1.x/voxel_xlen-0.5;
				float coord_y = p1.y/voxel_ylen-0.5;
				float coord_z = p1.z/voxel_zlen-0.5;

				if(cur_step != 0 && (coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1))
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k1.x = result[0];
				k1.y = result[1];
				k1.z = result[2];

				points.push_back(p1);
				vels.push_back(k1);

				p2 = cur_pos + k1*0.5*step_time;
				coord_x = p2.x/voxel_xlen-0.5;
				coord_y = p2.y/voxel_ylen-0.5;
				coord_z = p2.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k2.x = result[0];
				k2.y = result[1];
				k2.z = result[2];

				p3 = cur_pos + k2*0.5*step_time;
				coord_x = p3.x/voxel_xlen-0.5;
				coord_y = p3.y/voxel_ylen-0.5;
				coord_z = p3.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k3.x = result[0];
				k3.y = result[1];
				k3.z = result[2];

				p4 = cur_pos + k3*1.0*step_time;
				coord_x = p4.x/voxel_xlen-0.5;
				coord_y = p4.y/voxel_ylen-0.5;
				coord_z = p4.z/voxel_zlen-0.5;
				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(flow,width,height,depth,3,coord_x,coord_y,coord_z,result,false);
				k4.x = result[0];
				k4.y = result[1];
				k4.z = result[2];

				ZQ_Vec3D dir = (k1+k2*2.0+k3*2.0+k4)*(1.0/6.0);

				if(dir*dir*step_time*step_time < step_len*step_len*1e-8)
					break;

				dir *= 1.0/sqrt(dir*dir);
				dir *= step_len;

				cur_pos = cur_pos + dir;

				cur_step++;
			}
		}

		template<class T>
		void ZQ_BacktraceStreamlineRK4_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels)
		{
			ZQ_Vec3D k1,k2,k3,k4;
			ZQ_Vec3D p1,p2,p3,p4;

			T result[3];
			memset(result,0,sizeof(T)*3);

			points.clear();

			int cur_step = 0;
			ZQ_Vec3D cur_pos = start_pos;

			while(cur_step < max_steps)
			{
				p1 = cur_pos;
				float coord_x = p1.x/voxel_xlen;
				float coord_y = p1.y/voxel_ylen;
				float coord_z = p1.z/voxel_zlen;

				if(cur_step != 0 && (coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1))
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k1.x = result[0];
				k1.y = result[1];
				k1.z = result[2];

				points.push_back(p1);
				vels.push_back(k1);

				p2 = cur_pos - k1*0.5*step_time;
				coord_x = p2.x/voxel_xlen;
				coord_y = p2.y/voxel_ylen;
				coord_z = p2.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k2.x = result[0];
				k2.y = result[1];
				k2.z = result[2];

				p3 = cur_pos - k2*0.5*step_time;
				coord_x = p3.x/voxel_xlen;
				coord_y = p3.y/voxel_ylen;
				coord_z = p3.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k3.x = result[0];
				k3.y = result[1];
				k3.z = result[2];

				p4 = cur_pos - k3*1.0*step_time;
				coord_x = p4.x/voxel_xlen;
				coord_y = p4.y/voxel_ylen;
				coord_z = p4.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k4.x = result[0];
				k4.y = result[1];
				k4.z = result[2];

				ZQ_Vec3D dir = (k1+k2*2.0+k3*2.0+k4)*(1.0/6.0);

				if(dir*dir*step_time*step_time < step_len*step_len*1e-8)
					break;

				dir *= 1.0/sqrt(dir*dir);
				dir *= step_len;

				cur_pos = cur_pos - dir;

				cur_step++;
			}
		}

		template<class T>
		void ZQ_ForwardStreamlineRK4_MACGrid(const T* mac_u, const T* mac_v, const T* mac_w, const int width, const int height, const int depth, const float voxel_xlen, const float voxel_ylen, const float voxel_zlen, 
			const float step_time, const float step_len, const int max_steps, const ZQ_Vec3D& start_pos, std::vector<ZQ_Vec3D>& points, std::vector<ZQ_Vec3D>& vels)
		{
			ZQ_Vec3D k1,k2,k3,k4;
			ZQ_Vec3D p1,p2,p3,p4;

			T result[3];
			memset(result,0,sizeof(T)*3);

			points.clear();

			int cur_step = 0;
			ZQ_Vec3D cur_pos = start_pos;

			while(cur_step < max_steps)
			{
				p1 = cur_pos;
				float coord_x = p1.x/voxel_xlen;
				float coord_y = p1.y/voxel_ylen;
				float coord_z = p1.z/voxel_zlen;

				if(cur_step != 0 && (coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1))
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k1.x = result[0];
				k1.y = result[1];
				k1.z = result[2];

				points.push_back(p1);
				vels.push_back(k1);

				p2 = cur_pos + k1*0.5*step_time;
				coord_x = p2.x/voxel_xlen;
				coord_y = p2.y/voxel_ylen;
				coord_z = p2.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k2.x = result[0];
				k2.y = result[1];
				k2.z = result[2];

				p3 = cur_pos + k2*0.5*step_time;
				coord_x = p3.x/voxel_xlen;
				coord_y = p3.y/voxel_ylen;
				coord_z = p3.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k3.x = result[0];
				k3.y = result[1];
				k3.z = result[2];

				p4 = cur_pos + k3*1.0*step_time;
				coord_x = p4.x/voxel_xlen;
				coord_y = p4.y/voxel_ylen;
				coord_z = p4.z/voxel_zlen;

				if(coord_x < 0 || coord_x > width-1 || coord_y < 0 || coord_y > height-1 || coord_z < 0 || coord_z > depth-1)
					break;
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_u,width+1,height,depth,1,coord_x,coord_y-0.5f,coord_z-0.5f,result+0,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_v,width,height+1,depth,1,coord_x-0.5f,coord_y,coord_z-0.5f,result+1,false);
				ZQ_ImageProcessing3D::TrilinearInterpolate(mac_w,width,height,depth+1,1,coord_x-0.5f,coord_y-0.5f,coord_z,result+2,false);
				k4.x = result[0];
				k4.y = result[1];
				k4.z = result[2];

				ZQ_Vec3D dir = (k1+k2*2.0+k3*2.0+k4)*(1.0/6.0);

				if(dir*dir*step_time*step_time < step_len*step_len*1e-8)
					break;

				dir *= 1.0/sqrt(dir*dir);
				dir *= step_len;

				cur_pos = cur_pos + dir;

				cur_step++;
			}
		}
	}
}

#endif