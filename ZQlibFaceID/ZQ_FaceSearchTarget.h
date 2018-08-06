#ifndef _ZQ_FACE_SEARCH_TARGET_H_
#define _ZQ_FACE_SEARCH_TARGET_H_
#pragma once
#include "ZQ_FaceGroup.h"

namespace ZQ
{
	class ZQ_FaceSearchTarget
	{
	public:
		std::vector<ZQ_FaceGroupWithoutBox> targets;

	public:
		bool SaveToFile(const std::string& file) const
		{
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
			int num = targets.size();
			fwrite(&num, sizeof(int), 1, out);
			for (int i = 0; i < num; i++)
			{
				if (!targets[i].WriteToFile(out))
					return false;
			}
			fclose(out);
			return true;
		}

		bool LoadFromFile(const std::string& file)
		{
			targets.clear();
			FILE* in = 0;
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;

			int num;
			if (fread(&num, sizeof(int), 1, in) != 1 || num < 0)
			{
				fclose(in);
				return false;
			}
			targets.resize(num);
			for (int i = 0; i < num; i++)
			{
				if (!targets[i].LoadFromFile(in))
				{
					fclose(in);
					targets.clear();
					return false;
				}
			}
			fclose(in);
			return true;
		}
	};
}
#endif
