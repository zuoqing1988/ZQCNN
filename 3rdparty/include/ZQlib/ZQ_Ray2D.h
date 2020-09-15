#ifndef _ZQ_RAY_2D_H_
#define _ZQ_RAY_2D_H_
#pragma once

#include "ZQ_Vec2D.h"
#include "ZQ_Matrix.h"
#include "ZQ_SVD.h"

namespace ZQ
{
	class ZQ_Ray2D
	{
	public:
		ZQ_Ray2D() :origin(0, 0), dir(1, 0) {}
		~ZQ_Ray2D() {}
	public:
		ZQ_Vec2D origin;
		ZQ_Vec2D dir;

		static bool RayCross(const ZQ_Ray2D& ray1, const ZQ_Ray2D& ray2, float& depth1, float& depth2, ZQ_Vec2D& crossPos)
		{
			if (ray1.dir.Length() == 0)
				return false;
			if (ray2.dir.Length() == 0)
				return false;

			ZQ_Matrix<float> matA(2, 2);
			ZQ_Matrix<float> Ainv(2, 2);
			ZQ_Matrix<float> matB(2, 1);

			matA.SetData(0, 0, ray1.dir.x);
			matA.SetData(0, 1, -ray2.dir.x);
			matA.SetData(1, 0, ray1.dir.y);
			matA.SetData(1, 1, -ray2.dir.y);
			
			matB.SetData(0, 0, ray2.origin.x - ray1.origin.x);
			matB.SetData(1, 0, ray2.origin.y - ray1.origin.y);

			ZQ_SVD::Invert(matA, Ainv);

			ZQ_Matrix<float> matX = Ainv*matB;

			bool flag;
			depth1 = matX.GetData(0, 0, flag);
			depth2 = matX.GetData(1, 0, flag);

			crossPos = (ray1.origin + ray1.dir*depth1 + ray2.origin + ray2.dir*depth2)*0.5;

			return true;
		}
	};
}

#endif
