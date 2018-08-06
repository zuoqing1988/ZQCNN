#ifndef _ZQ_FACE_CONTAINER_FOR_VIDEO_H_
#define _ZQ_FACE_CONTAINER_FOR_VIDEO_H_
#pragma once

#include "ZQ_FaceGroup.h"

namespace ZQ
{
	class ZQ_FaceContainerForVideo
	{
	public:
		int skip;
		std::vector<ZQ_FaceGroupWithBox> frames;
	public:
		ZQ_FaceContainerForVideo()
		{
			skip = 1;
		}

		~ZQ_FaceContainerForVideo()
		{
			Clear();
		}

		void Clear()
		{
			skip = 1;
			frames.clear();
		}

		bool SaveToFile(const std::string& file) const
		{
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
			fwrite(&skip, sizeof(int), 1, out);
			int key_num = frames.size();
			fwrite(&key_num, sizeof(int), 1, out);
			for (int i = 0; i < key_num; i++)
			{
				if (!frames[i].WriteToFile(out))
					return false;
			}
			fclose(out);
			return true;
		}

		bool LoadFromFile(const std::string& file)
		{
			Clear();
			FILE* in = 0;
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;
			if (fread(&skip, sizeof(int), 1, in) != 1)
			{
				fclose(in);
				Clear();
				return false;
			}
			int key_num;
			if (fread(&key_num, sizeof(int), 1, in) != 1 || key_num < 0)
			{
				fclose(in);
				Clear();
				return false;
			}
			frames.resize(key_num);
			for (int i = 0; i < key_num; i++)
			{
				if (!frames[i].LoadFromFile(in))
				{
					fclose(in);
					Clear();
					return false;
				}
			}
			fclose(in);
			return true;
		}

	};
}

#endif