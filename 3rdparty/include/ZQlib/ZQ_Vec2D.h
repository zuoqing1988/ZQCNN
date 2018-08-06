#ifndef _ZQ_VEC_2D_H_
#define _ZQ_VEC_2D_H_
#pragma once

#include <math.h>

namespace ZQ
{
	class ZQ_Vec2D
	{
	public:
		float x,y;

	public:

		ZQ_Vec2D()
		{
			x = y = 0;
		}

		ZQ_Vec2D(float x, float y)
		{
			this->x = x;
			this->y = y;
		}

		~ZQ_Vec2D()
		{
		}

		ZQ_Vec2D operator +(const ZQ_Vec2D &v) const
		{
			return ZQ_Vec2D(x + v.x, y + v.y);
		}

		void operator +=(const ZQ_Vec2D& v)
		{
			x += v.x;
			y += v.y;
		}

		ZQ_Vec2D operator -(const ZQ_Vec2D &v) const
		{
			return ZQ_Vec2D(x - v.x, y - v.y);
		}

		void operator -=(const ZQ_Vec2D& v)
		{
			x -= v.x;
			y -= v.y;
		}

		float operator *(const ZQ_Vec2D& v) const
		{
			return x*v.x + y*v.y;
		}

		ZQ_Vec2D operator *(float scale) const
		{
			return ZQ_Vec2D(x*scale, y*scale);
		}

		void operator *=(float scale)
		{
			x *= scale;
			y *= scale;
		}

		bool operator <(const ZQ_Vec2D &v) const
		{
			return x < v.x && y < v.y;
		}

		bool operator <=(const ZQ_Vec2D& v) const
		{
			return x <= v.x && y <= v.y;
		}

		bool operator >(const ZQ_Vec2D& v) const
		{
			return x > v.x && y > v.y;
		}

		bool operator >=(const ZQ_Vec2D& v) const
		{
			return x >= v.x && y >= v.y;
		}

		bool operator ==(const ZQ_Vec2D& v) const
		{
			return x == v.x && y == v.y;
		}

		float DotProduct(const ZQ_Vec2D& v) const
		{
			return (*this)*v;
		}


		float Length() const
		{
			return sqrt(x*x + y*y);
		}

		bool Normalized()
		{
			float length = sqrt(x*x + y*y);
			if (length == 0)
			{
				return false;
			}
			else
			{
				x /= length;
				y /= length;
				return true;
			}
		}
	};
}


#endif