#ifndef _ZQ_VEC_3D_H_
#define _ZQ_VEC_3D_H_
#pragma once

#include <math.h>

namespace ZQ
{
	class ZQ_Vec3D
	{
	public:
		float x,y,z;

	public:

		ZQ_Vec3D()
		{
			x = y = z = 0;
		}

		ZQ_Vec3D(float x, float y, float z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		~ZQ_Vec3D()
		{
		}

		ZQ_Vec3D operator +(const ZQ_Vec3D &v) const
		{
			return ZQ_Vec3D(x + v.x, y + v.y, z + v.z);
		}

		void operator +=(const ZQ_Vec3D& v)
		{
			x += v.x;
			y += v.y;
			z += v.z;
		}

		ZQ_Vec3D operator -(const ZQ_Vec3D &v) const
		{
			return ZQ_Vec3D(x - v.x, y - v.y, z - v.z);
		}

		void operator -=(const ZQ_Vec3D& v)
		{
			x -= v.x;
			y -= v.y;
			z -= v.z;
		}

		float operator *(const ZQ_Vec3D& v) const
		{
			return x*v.x + y*v.y + z*v.z;
		}

		ZQ_Vec3D operator *(float scale) const
		{
			return ZQ_Vec3D(x*scale, y*scale, z*scale);
		}

		void operator *=(float scale)
		{
			x *= scale;
			y *= scale;
			z *= scale;
		}

		bool operator <(const ZQ_Vec3D &v) const
		{
			return x < v.x && y < v.y && z < v.z;
		}

		bool operator <=(const ZQ_Vec3D& v) const
		{
			return x <= v.x && y <= v.y && z <= v.z;
		}

		bool operator >(const ZQ_Vec3D& v) const
		{
			return x > v.x && y > v.y && z > v.z;
		}

		bool operator >=(const ZQ_Vec3D& v) const
		{
			return x >= v.x && y >= v.y && z >= v.z;
		}

		bool operator ==(const ZQ_Vec3D& v) const
		{
			return x == v.x && y == v.y && z == v.z;
		}

		float DotProduct(const ZQ_Vec3D& v) const
		{
			return (*this)*v;
		}

		ZQ_Vec3D CrossProduct(const ZQ_Vec3D& v) const
		{
			return ZQ_Vec3D(this->y * v.z - this->z * v.y,
				this->z * v.x - this->x * v.z,
				this->x * v.y - this->y * v.x);
		}

		float Length() const
		{
			return sqrt(x*x + y*y + z*z);
		}

		bool Normalized()
		{
			float length = sqrt(x*x + y*y + z*z);
			if (length == 0)
			{
				return false;
			}
			else
			{
				x /= length;
				y /= length;
				z /= length;
				return true;
			}
		}
	};

}


#endif