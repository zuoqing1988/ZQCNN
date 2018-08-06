#ifndef _ZQ_CPU_RAYCASTING_H_
#define _ZQ_CPU_RAYCASTING_H_
#pragma once

#include "ZQ_Ray3D.h"
#include "ZQ_Vec3D.h"
#include "ZQ_MathBase.h"
#include "ZQ_ImageProcessing3D.h"

namespace ZQ
{
 	class ZQ_CPURayCasting
	{
	public:
		enum ColorFormat{COLOR_RGB,COLOR_RGBA,COLOR_BGR,COLOR_BGRA};

		class IntersectRes
		{
		public:
			int flag;
			float tnear;
			float tfar;
		};
		static IntersectRes IntersectBox(const ZQ_Ray3D r, const ZQ_Vec3D& boxmin, const ZQ_Vec3D& boxmax)
		{
			IntersectRes res = { 0, 0, 0 };
			const float maxfloat = 1e16;
			const float minfloat = 1.0f / maxfloat;

			float largest_tmin, smallest_tmax;

			// compute intersection of ray with all six bbox planes
			ZQ_Vec3D invR, tmin, tmax;

			invR.x = fabs(r.dir.x) > minfloat ? (1.0f / r.dir.x) : (1.0f / minfloat*(r.dir.x>0 ? 1 : -1));
			invR.y = fabs(r.dir.y) > minfloat ? (1.0f / r.dir.y) : (1.0f / minfloat*(r.dir.y>0 ? 1 : -1));
			invR.z = fabs(r.dir.z) > minfloat ? (1.0f / r.dir.z) : (1.0f / minfloat*(r.dir.z>0 ? 1 : -1));


			tmin.x = (invR.x >= 0) ? (invR.x * (boxmin.x - r.origin.x)) : (invR.x * (boxmax.x - r.origin.x));
			tmax.x = (invR.x >= 0) ? (invR.x * (boxmax.x - r.origin.x)) : (invR.x * (boxmin.x - r.origin.x));

			tmin.y = (invR.y >= 0) ? (invR.y * (boxmin.y - r.origin.y)) : (invR.y * (boxmax.y - r.origin.y));
			tmax.y = (invR.y >= 0) ? (invR.y * (boxmax.y - r.origin.y)) : (invR.y * (boxmin.y - r.origin.y));

			tmin.z = (invR.z >= 0) ? (invR.z * (boxmin.z - r.origin.z)) : (invR.z * (boxmax.z - r.origin.z));
			tmax.z = (invR.z >= 0) ? (invR.z * (boxmax.z - r.origin.z)) : (invR.z * (boxmin.z - r.origin.z));

			// find the largest tmin and the smallest tmax
			largest_tmin = __max(__max(tmin.x, tmin.y), __max(tmin.x, tmin.z));
			smallest_tmax = __min(__min(tmax.x, tmax.y), __min(tmax.x, tmax.z));

			res.tnear = largest_tmin;
			res.tfar = smallest_tmax;
			res.flag = (smallest_tmax > largest_tmin) ? 1 : 0;
			return res;
		}

		static float MiddleValue(float x, float y, float z)
		{
			if (x <= y && y <= z)
				return y;
			else if (x <= z && z <= y)
				return z;
			else if (y <= x && x <= z)
				return x;
			else if (y <= z && z <= x)
				return z;
			else if (z <= x && x <= y)
				return x;
			else
				return y;
		}

	public:
		ZQ_CPURayCasting(bool zAxis_in)
		{
			/*window's width and height, focus is the center*/
			width = 400;
			height = 400;
			center_x = 200;
			center_y = 200;
			focal_x = 800;
			focal_y = 800;

			xsize = 0;
			ysize = 0;
			zsize = 0;
			data = 0;
			densityScale = 1.0f;
			opacityScale = 0.0f;
			opacityThreshold = 0.98f;
			this->zAxis_in = zAxis_in;
		}

		~ZQ_CPURayCasting(){}

	private:
		/*window's width and height, focus is the center*/
		unsigned int width;
		unsigned int height;
		float center_x;
		float center_y;
		float focal_x;
		float focal_y;

		float view_matrix[16];
		float world_matrix[16];

		/*bounding box: min and max, and size = max - min, in local coordinate system*/
		ZQ_Vec3D boundingBoxMin;
		ZQ_Vec3D boundingBoxMax;
		bool zAxis_in;

	private:
		unsigned int xsize;
		unsigned int ysize;
		unsigned int zsize;
		const float *data;
		float densityScale;
		float opacityScale;
		float opacityThreshold;

	public:
		void SetWindowSize(unsigned int width, unsigned int height)
		{
			this->width = width;
			this->height = height;
		}

		void SetInnerPara(float cx, float cy, float fx, float fy)
		{
			this->center_x = cx;
			this->center_y = cy;
			this->focal_x = fx;
			this->focal_y = fy;
		}

		void SetViewMatrix(const float* view_mat)
		{
			memcpy(this->view_matrix, view_mat, sizeof(float) * 16);
		}

		void SetWorldMatrix(const float* world_mat)
		{
			memcpy(this->world_matrix, world_mat, sizeof(float) * 16);
		}

		void SetVolumeData(const float* data, unsigned int xsize, unsigned int ysize, unsigned int zsize)
		{
			this->xsize = xsize;
			this->ysize = ysize;
			this->zsize = zsize;
			this->data = data;
		}

		void SetVolumeDensityScale(float densityScale = 1.0f){ this->densityScale = densityScale;}

		void SetOpacityScale(float opacityScale = 1.0f){ this->opacityScale = opacityScale;}

		void SetOpacityThreshold(float opacityThreshold = 0.98){ this->opacityThreshold = opacityThreshold;}

		void SetVolumeBoundingBox(const ZQ_Vec3D& boxmin, const ZQ_Vec3D& boxmax)
		{
			this->boundingBoxMin = boxmin;
			this->boundingBoxMax = boxmax;
		}

		bool RenderToBuffer(float* buffer, int max_steps,ColorFormat fmt = COLOR_RGBA)
		{
			float modelview[16], c_invMv[16];
			ZQ_MathBase::MatrixMul(view_matrix, world_matrix, 4, 4, 4, modelview);
			if (!ZQ_MathBase::MatrixInverse(modelview, 4, c_invMv))
			{
				return false;
			}

			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					float r, g, b, a;
					_castRay(c_invMv, w, h, max_steps, r, g, b, a);
					switch (fmt)
					{
					case COLOR_BGR:
						buffer[offset * 3 + 0] = b;
						buffer[offset * 3 + 1] = g;
						buffer[offset * 3 + 2] = r;
						break;
					case COLOR_BGRA:
						buffer[offset * 4 + 0] = b;
						buffer[offset * 4 + 1] = g;
						buffer[offset * 4 + 2] = r;
						buffer[offset * 4 + 3] = a;
						break;
					case COLOR_RGB:
						buffer[offset * 3 + 0] = r;
						buffer[offset * 3 + 1] = g;
						buffer[offset * 3 + 2] = b;
						break;
					case COLOR_RGBA:
						buffer[offset * 4 + 0] = r;
						buffer[offset * 4 + 1] = g;
						buffer[offset * 4 + 2] = b;
						buffer[offset * 4 + 3] = a;
						break;
					default:
						return false;
					}
				}
			}
			return true;
		}

	private:
		
		/*return 0 correct*/
		/*return 1 not hit
		/*return 2 hit tfar < 0
		/*return 3 hit tnear < 0
		/*return 4 out of steps
		/**/
		int _castRay(const float* c_invMv, unsigned int w, unsigned h, unsigned int max_step, float& r, float& g, float& b, float& a)
		{
			r = g = b = a = 0.0f;
			ZQ_Vec3D boxSize = boundingBoxMax - boundingBoxMin;
			int max_dim = __max(__max(xsize, ysize), zsize);
			int maxSteps = __min(max_step, max_dim*1.732);
			float tstep = __max(__max(boxSize.x, boxSize.y), boxSize.z) / maxSteps;

			float u = w - center_x;
			float v = (height - 1 - h) - center_y;

			ZQ_Ray3D eyeRay;
			ZQ_Vec3D eyepos(c_invMv[3], c_invMv[7], c_invMv[11]);

			eyeRay.origin = eyepos;
			eyeRay.dir.x = u*focal_x / focal_y;
			eyeRay.dir.y = v;
			eyeRay.dir.z = focal_y * (zAxis_in ? 1.0f : -1.0f);
			eyeRay.dir.Normalized();

			float old_eyedir4[4] = { eyeRay.dir.x, eyeRay.dir.y, eyeRay.dir.z, 0.0f };
			float eyedir4[4];

			ZQ_MathBase::MatrixMul(c_invMv, old_eyedir4, 4, 4, 1, eyedir4);

			eyeRay.dir.x = eyedir4[0]; eyeRay.dir.y = eyedir4[1]; eyeRay.dir.z = eyedir4[2];

			// find intersection with box
			IntersectRes hit = IntersectBox(eyeRay, boundingBoxMin, boundingBoxMax);

			if (hit.flag == 0)
			{
				return 1;
			}

			if (hit.tfar < 0.0f)
			{
				return 2;
			}
			if (hit.tnear < 0.0f) {
				hit.tnear = 0.0f;     // clamp to near plane

				return 3;
			}

			// march along ray from front to back, accumulating color
			float t = hit.tnear;

			ZQ_Vec3D pos = eyeRay.origin + eyeRay.dir * hit.tnear;
			ZQ_Vec3D step = eyeRay.dir * tstep;

			float sum = 0;
			float opacity = 0;
			for (int istep = 0; istep < maxSteps; istep++)
			{
				float m_coord[3] =
				{
					(pos.x - boundingBoxMin.x) / boxSize.x*xsize - 0.5,
					(pos.y - boundingBoxMin.y) / boxSize.y*ysize - 0.5,
					(pos.z - boundingBoxMin.z) / boxSize.z*zsize - 0.5
				};


				float volumeElmt = 0;

				ZQ_ImageProcessing3D::TrilinearInterpolate(data, xsize, ysize, zsize, 1, m_coord[0], m_coord[1], m_coord[2], &volumeElmt, false);

				sum += (1 - opacity)*volumeElmt*densityScale*tstep;
				opacity = 1 - (1 - opacity)*exp(-volumeElmt*opacityScale*tstep);



				if (opacity > opacityThreshold)
				{
					r = sum;
					g = sum;
					b = sum;
					a = opacity;
					return 0;
				}

				t += tstep;

				if (t > hit.tfar)
				{
					r = sum;
					g = sum;
					b = sum;
					a = opacity;
					return 0;
				}

				pos += step;
			}

			r = sum;
			g = sum;
			b = sum;
			a = opacity;
			return 4;
		}
	};
}


#endif