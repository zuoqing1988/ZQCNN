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
				free(pData);
			pData = 0;
			length = 0;
		}


		void CopyData(const ZQ_FaceFeature& other)
		{
			if (length < 0)
			{
				length = 0;
				if (pData != 0)
					free(pData);
				pData = 0;
				return;
			}
			if (length != other.length)
			{
				length = other.length;
				if (pData != 0)
					free(pData);
				pData = (float*)malloc(sizeof(float)*length);
				if (pData == 0)
				{
					length = 0;
					return;
				}
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
					// malloc 失败时必须把 length 也清 0：同文件的 CopyData()
					// 已经是这么做的。不清的话会留下 pData==0 && length==dst_len
					// 的状态，调用方紧接着解引用 pData 就是空指针写。
					length = (pData != 0) ? dst_len : 0;
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