#ifndef _ZQ_FACE_GROUP_H_
#define _ZQ_FACE_GROUP_H_
#pragma once
#include "ZQ_FaceFeature.h"
#include "ZQ_CNN_BBox.h"
#include <vector>
#include <stdio.h>
namespace ZQ
{
	class ZQ_FaceGroup
	{
	public:
		virtual bool HasBox()const = 0;
		int GetFeatDim() const { return feat_dim; }
		const std::vector<ZQ_FaceFeature>& GetFeats()const { return face_feats; }
		const std::vector<ZQ_CNN_BBox>& GetBox() const { return face_boxes; }
		virtual bool LoadFromFile(FILE* in)
		{
#ifdef _WIN64
			long long pos = _ftelli64(in);
#else
			long pos = ftell(in);
#endif
			bool flag = true;

			if (flag)
			{
				flag = (1 == fread(&feat_dim, sizeof(int), 1, in));
				flag = flag && feat_dim >= 0 && feat_dim < 65535;
				if(!flag){
					printf("feat_dim = %d\n", feat_dim);
				}
			}
			if (flag)
			{
				flag = (1 == fread(&with_box, sizeof(bool), 1, in));
				if (!flag) {
					printf("feat_dim = %d\n", feat_dim);
				}
			}
			int num;
			if (flag)
			{
				flag = (1 == fread(&num, sizeof(int), 1, in));
				flag = flag && num >= 0;
				if (!flag) {
					printf("feat_dim = %d\n", feat_dim);
				}
				if (flag)
				{
					face_feats.resize(num);
				}
				for (int i = 0; i < num; i++)
				{
					face_feats[i].ChangeSize(feat_dim);
					flag = (feat_dim == fread(face_feats[i].pData, sizeof(float), feat_dim, in));
					if (!flag)
					{
						printf("feat_dim = %d\n", feat_dim);
						break;
					}
				}
				if (flag && with_box)
				{
					face_boxes.resize(num);
					for (int i = 0; i < num; i++)
					{
						flag = (1 == fread(&face_boxes[0]+i, sizeof(ZQ_CNN_BBox), 1, in));
						if (!flag)
						{
							printf("feat_dim = %d\n", feat_dim);
							break;
						}
					}
				}
			}

			if (!flag)
			{
#ifdef _WIN64
				_fseeki64(in, pos, SEEK_SET);
#else
				fseek(in, pos, SEEK_SET);
#endif
				return false;
			}

			return true;
				
		}

		virtual bool WriteToFile(FILE* out) const
		{
#ifdef _WIN64
			long long pos = _ftelli64(out);
#else
			long pos = ftell(in);
#endif
			bool flag = true;

			if (flag)
			{
				flag = (1 == fwrite(&feat_dim, sizeof(int), 1, out));
			}
			if (flag)
			{
				flag = (1 == fwrite(&with_box, sizeof(bool), 1, out));
			}
			int num = face_feats.size();
			if (flag)
			{
				flag = (1 == fwrite(&num, sizeof(int), 1, out));
				for (int i = 0; i < num; i++)
				{
					flag = (feat_dim == fwrite(face_feats[i].pData, sizeof(float), feat_dim, out));
					if (!flag)
						break;
				}
				if (flag && with_box)
				{
					for (int i = 0; i < num; i++)
					{
						flag = (1 == fwrite(&face_boxes[0]+i, sizeof(ZQ_CNN_BBox), 1, out));
						if (!flag)
							break;
					}
				}
			}

			if (!flag)
			{
#ifdef _WIN64
				_fseeki64(out, pos, SEEK_SET);
#else
				fseek(out, pos, SEEK_SET);
#endif
				return false;
			}
			return true;
		}
	public:
		std::vector<ZQ_FaceFeature> face_feats;
		std::vector<ZQ_CNN_BBox> face_boxes;
		int feat_dim;
	protected:
		bool with_box;	
	};

	class ZQ_FaceGroupWithBox : public ZQ_FaceGroup
	{
	public:
		ZQ_FaceGroupWithBox() { with_box = true; }
		bool HasBox()const { return with_box; }
	};
	class ZQ_FaceGroupWithoutBox : public ZQ_FaceGroup
	{
	public:
		ZQ_FaceGroupWithoutBox() { with_box = false; }
		bool HasBox() const { return with_box; }
	};
}
#endif
