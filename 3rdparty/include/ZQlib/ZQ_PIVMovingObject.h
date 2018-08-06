#ifndef _ZQ_PIV_MOVING_OBJECT_H_
#define _ZQ_PIV_MOVING_OBJECT_H_
#pragma once

#include "ZQ_DoubleImage.h"

namespace ZQ
{
	class ZQ_PIVMovingObject
	{
	public:
		typedef float BaseType;
		typedef ZQ_DImage<BaseType> DImage;
		enum MOVOBJ_TYPE
		{
			ZQ_PIV_MOVOB_RECT_STATIC = 0,
			ZQ_PIV_MOVOB_RECT_UPDOWN = 1,
			ZQ_PIV_MOVOB_CIRCLE_STATIC = 2,
			ZQ_PIV_MOVOB_CIRCLE_CIRCULAR = 3
		};

	public:
		ZQ_PIVMovingObject(int width, int height, int type, const char* texturefile = 0)
		{
			this->width = width;
			this->height = height;

			posx = 0;
			posy = 0;
			cx = width*0.5;
			cy = height*0.5;

			occupy.allocate(width, height);
			switch (type)
			{
			case ZQ_PIV_MOVOB_RECT_STATIC: case ZQ_PIV_MOVOB_RECT_UPDOWN:
				for (int i = 0; i < width*height; i++)
					occupy.data()[i] = 1;
				break;

			case ZQ_PIV_MOVOB_CIRCLE_STATIC: case ZQ_PIV_MOVOB_CIRCLE_CIRCULAR:
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						double dis = sqrt((i - height*0.5)*(i - height*0.5) + (j - width*0.5)*(j - width*0.5));
						if (dis <= __min(width*0.5, height*0.5))
						{
							occupy.data()[i*width + j] = 1.0;
						}
					}
				}
			}

			if (texture.loadImage(texturefile) == false)
			{
				has_texture = false;
			}
			else
			{
				has_texture = true;
				texture.imresize(width, height);
				int nChannels = texture.nchannels();
				if (nChannels != 1)
				{
					DImage part1, part2;
					texture.separate(1, part1, part2);
					texture.copyData(part1);
				}
			}

			this->type = type;
			framecount = 0;
			UpdateOneFrame();
		}

		~ZQ_PIVMovingObject() {}

	private:
		int cx, cy;
		int width, height;
		float posx, posy;
		float velx, vely;
		DImage occupy;
		bool has_texture;
		DImage texture;
		int type;
		int framecount;
	public:

		void ExportToGlobal(int global_width, int global_height, DImage& global_occupy, DImage& global_texture, DImage& u, DImage& v, DImage& mac_u, DImage& mac_v)
		{
			int center_x = global_width / 2;
			int center_y = global_height / 2;
			int shift_x = -cx + center_x + posx + 0.5;
			int shift_y = -cy + center_y + posy + 0.5;
			global_occupy.allocate(global_width, global_height);
			global_texture.allocate(global_width, global_height);
			u.allocate(global_width, global_height);
			v.allocate(global_width, global_height);
			mac_u.allocate(global_width + 1, global_height);
			mac_v.allocate(global_width, global_height + 1);

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					int real_i = i + shift_y;
					int real_j = j + shift_x;
					if (real_i >= 0 && real_i < global_height && real_j >= 0 && real_j < global_width)
					{
						global_occupy.data()[real_i*global_width + real_j] = occupy.data()[i*width + j];
						if (has_texture)
							global_texture.data()[real_i*global_width + real_j] = texture.data()[i*width + j];
						else
							global_texture.data()[real_i*global_width + real_j] = 0.8;
						u.data()[real_i*global_width + real_j] = velx;
						v.data()[real_i*global_width + real_j] = vely;

						mac_u.data()[real_i*(global_width + 1) + real_j] = velx;
						mac_u.data()[real_i*(global_width + 1) + real_j + 1] = velx;
						mac_v.data()[real_i*global_width + real_j] = vely;
						mac_v.data()[(real_i + 1)*global_width + real_j] = vely;
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
				velx = 0;
				vely = 0;
				break;
			case ZQ_PIV_MOVOB_RECT_UPDOWN:
				theta = (framecount++)*omega;
				posx = 0;
				posy = A*sin(theta);
				velx = 0;
				vely = A*omega*cos(theta);
				break;
			case ZQ_PIV_MOVOB_CIRCLE_STATIC:
				posx = 0;
				posy = 0;
				velx = 0;
				vely = 0;
				break;
			case ZQ_PIV_MOVOB_CIRCLE_CIRCULAR:
				theta = (framecount++)*omega;
				posx = A*cos(theta);
				posy = A*sin(theta);
				velx = -A*omega*sin(theta);
				vely = A*omega*cos(theta);
				break;
			default:
				break;
			}
		}
	};
}
#endif