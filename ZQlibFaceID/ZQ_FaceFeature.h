#ifndef _ZQ_FACE_FEATURE_H_
#define _ZQ_FACE_FEATURE_H_
#pragma once
#include <malloc.h>
#include <string.h>
namespace ZQ
{
	class ZQ_FaceFeature
	{
	public:
		int length;
		float* pData;
		
		ZQ_FaceFeature()
		{
			length = 0;
			pData = 0;
		}
		ZQ_FaceFeature(const ZQ_FaceFeature& other)
		{
			length = other.length;

			if (length > 0)
			{
				pData = (float*)malloc(sizeof(float)*length);
				memcpy(pData, other.pData, sizeof(float)*length);
			}
			else
				pData = 0;

		}

		~ZQ_FaceFeature()
		{
			if (pData)
				delete[]pData;
			pData = 0;
			length = 0;
		}


		void CopyData(const ZQ_FaceFeature& other)
		{
			if (length != other.length)
			{
				length = other.length;
				if (pData != 0)
					delete[]pData;
				pData = (float*)malloc(sizeof(float)*length);
			}
			if (length > 0)
				memcpy(pData, other.pData, sizeof(float)*length);
			else
				pData = 0;
		}

		ZQ_FaceFeature& operator=(const ZQ_FaceFeature& other)
		{
			CopyData(other);
			return *this;
		}

		void ChangeSize(int dst_len)
		{
			if (length != dst_len)
			{
				if (pData)
				{
					free(pData);
					pData = 0;
				}
				if (dst_len > 0)
				{
					pData = (float*)malloc(sizeof(float)*dst_len);
					length = dst_len;
				}
				else
				{
					pData = 0;
					length = 0;
				}
			}
		}
	};
}

#endif