#ifndef _ZQ_PIV_MOVING_OBJECT3D_H_
#define _ZQ_PIV_MOVING_OBJECT3D_H_
#pragma once

#include "ZQ_DoubleImage3D.h"

namespace ZQ
{
	class ZQ_PIVMovingObject3D
	{
	public:
		typedef float BaseType;
		typedef ZQ_DImage3D<BaseType> DImage3D;
		enum MOVOBJ_TYPE
		{
			ZQ_PIV_MOVOB_RECT_STATIC = 0,
			ZQ_PIV_MOVOB_RECT_UPDOWN = 1,
			ZQ_PIV_MOVOB_CIRCLE_STATIC = 2,
			ZQ_PIV_MOVOB_CIRCLE_CIRCULAR = 3,
			ZQ_PIV_MOVOB_CYLINDER_STATIC = 4,
			ZQ_PIV_MOVOB_CYLINDER_CIRCULAR = 5
		};

	public:
		ZQ_PIVMovingObject3D(int width, int height, int depth, int type = 0, const char* texturefile = 0)
		{
			this->width = width;
			this->height = height;
			this->depth = depth;

			velx = 0;
			vely = 0;
			velz = 0;
			posx = 0;
			posy = 0;
			posz = 0;
			cx = width*0.5;
			cy = height*0.5;
			cz = depth*0.5;

			occupy.allocate(width, height, depth);
			switch (type)
			{
			case ZQ_PIV_MOVOB_RECT_STATIC: case ZQ_PIV_MOVOB_RECT_UPDOWN:
				for (int i = 0; i < width*height*depth; i++)
					occupy.data()[i] = 1;
				break;

			case ZQ_PIV_MOVOB_CIRCLE_STATIC: case ZQ_PIV_MOVOB_CIRCLE_CIRCULAR:
				for (int k = 0; k < depth; k++)
				{
					for (int j = 0; j < height; j++)
					{
						for (int i = 0; i < width; i++)
						{
							double dis = sqrt((k - depth*0.5)*(k - depth*0.5) + (j - height*0.5)*(j - height*0.5) + (i - width*0.5)*(i - width*0.5));
							if (dis <= __min(depth*0.5, __min(width*0.5, height*0.5)))
							{
								occupy.data()[k*height*width + j*width + i] = 1.0;
							}
						}
					}
				}

				break;
			case ZQ_PIV_MOVOB_CYLINDER_STATIC: case ZQ_PIV_MOVOB_CYLINDER_CIRCULAR:
				for (int k = 0; k < depth; k++)
				{
					for (int j = 0; j < height; j++)
					{
						for (int i = 0; i < width; i++)
						{
							double dis = sqrt((j - height*0.5)*(j - height*0.5) + (i - width*0.5)*(i - width*0.5));
							if (dis <= __min(width*0.5, height*0.5))
							{
								occupy.data()[k*height*width + j*width + i] = 1.0;
							}
						}
					}
				}

				break;

			}

			if (texturefile == 0 || texture.loadImage(texturefile) == false)
			{
				has_texture = false;
			}
			else
			{
				has_texture = true;
				texture.imresize(width, height, depth);
				int nChannels = texture.nchannels();
				if (nChannels != 1)
				{
					DImage3D part1, part2;
					texture.separate(1, part1, part2);
					texture.copyData(part1);
				}
			}

			this->type = type;
			framecount = 0;
			UpdateOneFrame();
		}

		~ZQ_PIVMovingObject3D() {}

	private:
		int cx, cy, cz;
		int width, height, depth;
		float posx, posy, posz;

		float velx, vely, velz;

		DImage3D occupy;

		bool has_texture;
		DImage3D texture;

		int type;
		int framecount;

	public:

		void ExportToGlobal(int global_width, int global_height, int global_depth, DImage3D& global_occupy, DImage3D& global_texture, DImage3D& u, DImage3D& v, DImage3D& w,
			DImage3D& mac_u, DImage3D& mac_v, DImage3D& mac_w)
		{
			int center_x = global_width / 2;
			int center_y = global_height / 2;
			int center_z = global_depth / 2;

			int shift_x = -cx + center_x + posx + 0.5;
			int shift_y = -cy + center_y + posy + 0.5;
			int shift_z = -cz + center_z + posz + 0.5;

			global_occupy.allocate(global_width, global_height, global_depth);
			global_texture.allocate(global_width, global_height, global_depth);
			u.allocate(global_width, global_height, global_depth);
			v.allocate(global_width, global_height, global_depth);
			w.allocate(global_width, global_height, global_depth);
			mac_u.allocate(global_width + 1, global_height, global_depth);
			mac_v.allocate(global_width, global_height + 1, global_depth);
			mac_w.allocate(global_width, global_height, global_depth + 1);

			for (int k = 0; k < depth; k++)
			{
				for (int j = 0; j < height; j++)
				{
					for (int i = 0; i < width; i++)
					{
						int real_i = i + shift_x;
						int real_j = j + shift_y;
						int real_k = k + shift_z;
						if (real_k >= 0 && real_k < global_depth && real_j >= 0 && real_j < global_height && real_i >= 0 && real_i < global_width)
						{
							global_occupy.data()[real_k*global_height*global_width + real_j*global_width + real_i] = occupy.data()[k*height*width + j*width + i];

							if (has_texture)
								global_texture.data()[real_k*global_height*global_width + real_j*global_width + real_i] = texture.data()[k*height*width + j*width + i];
							else
								global_texture.data()[real_k*global_height*global_width + real_j*global_width + real_i] = 0.8;

							u.data()[real_k*global_height*global_width + real_j*global_width + real_i] = velx;
							v.data()[real_k*global_height*global_width + real_j*global_width + real_i] = vely;
							w.data()[real_k*global_height*global_width + real_j*global_width + real_i] = velz;

							mac_u.data()[real_k*global_height*(global_width + 1) + real_j*(global_width + 1) + real_i] = velx;
							mac_u.data()[real_k*global_height*(global_width + 1) + real_j*(global_width + 1) + real_i + 1] = velx;
							mac_v.data()[real_k*(global_height + 1)*global_width + real_j*global_width + real_i] = vely;
							mac_v.data()[real_k*(global_height + 1)*global_width + (real_j + 1)*global_width + real_i] = vely;
							mac_w.data()[real_k*global_height*global_width + real_j*global_width + real_i] = velz;
							mac_w.data()[(real_k + 1)*global_height*global_width + real_j*global_width + real_i] = velz;
						}
					}
				}
			}
		}

		void UpdateOneFrame()
		{
			const double m_pi = atan(1.0) * 4;

			double A = 64;
			double T = 80;
			double omega = 2 * m_pi / T;
			double theta = 0;

			switch (type)
			{
			case ZQ_PIV_MOVOB_RECT_STATIC:
				posx = 0;
				posy = 0;
				posz = 0;
				velx = 0;
				vely = 0;
				velz = 0;
				break;
			case ZQ_PIV_MOVOB_RECT_UPDOWN:

				theta = (framecount++)*omega;

				posx = 0;
				posy = A*sin(theta);
				posz = 0;

				velx = 0;
				vely = A*omega*cos(theta);
				velz = 0;

				break;
			case ZQ_PIV_MOVOB_CIRCLE_STATIC: case ZQ_PIV_MOVOB_CYLINDER_STATIC:
				posx = 0;
				posy = 0;
				posz = 0;
				velx = 0;
				vely = 0;
				velz = 0;
				break;
			case ZQ_PIV_MOVOB_CIRCLE_CIRCULAR: case ZQ_PIV_MOVOB_CYLINDER_CIRCULAR:

				theta = (framecount++)*omega;

				posx = A*cos(theta);
				posy = A*sin(theta);
				posz = 0;

				velx = -A*omega*sin(theta);
				vely = A*omega*cos(theta);
				velz = 0;
				break;
			default:
				break;
			}
		}
	};
}

#endif