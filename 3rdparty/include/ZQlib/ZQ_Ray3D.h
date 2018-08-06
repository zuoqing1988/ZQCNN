#ifndef _ZQ_RAY_3D_H_
#define _ZQ_RAY_3D_H_
#pragma once

#include "ZQ_Vec3D.h"
#include "ZQ_Matrix.h"
#include "ZQ_SVD.h"

namespace ZQ
{
	class ZQ_Ray3D
	{
	public:
		ZQ_Ray3D():origin(0,0,0),dir(1,0,0){}
		~ZQ_Ray3D(){}
	public:
		ZQ_Vec3D origin;
		ZQ_Vec3D dir;

		static bool RayCross(const ZQ_Ray3D& ray1, const ZQ_Ray3D& ray2, float& depth1, float& depth2, ZQ_Vec3D& crossPos)
		{
			if(ray1.dir.Length() == 0)
				return false;
			if(ray2.dir.Length() == 0)
				return false;

			ZQ_Matrix<float> matA(3,2);
			ZQ_Matrix<float> Ainv(2,3);
			ZQ_Matrix<float> matB(3,1);

			matA.SetData(0,0,ray1.dir.x);
			matA.SetData(0,1,-ray2.dir.x);
			matA.SetData(1,0,ray1.dir.y);
			matA.SetData(1,1,-ray2.dir.y);
			matA.SetData(2,0,ray1.dir.z);
			matA.SetData(2,1,-ray2.dir.z);

			matB.SetData(0,0,ray2.origin.x-ray1.origin.x);
			matB.SetData(1,0,ray2.origin.y-ray1.origin.y);
			matB.SetData(2,0,ray2.origin.z-ray1.origin.z);

			ZQ_SVD::Invert(matA,Ainv);

			ZQ_Matrix<float> matX = Ainv*matB;

			bool flag;
			depth1 = matX.GetData(0,0,flag);
			depth2 = matX.GetData(1,0,flag);

			crossPos = (ray1.origin + ray1.dir*depth1 + ray2.origin + ray2.dir*depth2)*0.5;

			return true;
		}
	};
}

#endif