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
#if defined(_WIN32)
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
#else
			out = fopen(file.c_str(), "wb");
			if (out == NULL)
				return false;
#endif
			fwrite(&skip, sizeof(int), 1, out);
			int key_num = frames.size();
			fwrite(&key_num, sizeof(int), 1, out);
			for (int i = 0; i < key_num; i++)
			{
				if (!frames[i].WriteToFile(out))
				{
					// 原来直接 return false, out 没关 —— 文件句柄泄漏,
					// 而且此时 out 里还写着一份写了一半的容器。
					fclose(out);
					return false;
				}
			}
			fclose(out);
			return true;
		}

		bool LoadFromFile(const std::string& file)
		{
			Clear();
			FILE* in = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;
#else
			in = fopen(file.c_str(), "rb");
			if (in == NULL)
				return false;
#endif
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
			// key_num 来自不可信文件, 只挡负数不够: 0x7FFFFFFF 会让 frames.resize() 直接 OOM
			// (未捕获的 bad_alloc -> terminate)。用剩余文件长度做上界交叉校验:
			// 每个 frame 至少要能装下自己的长度字段(4 字节)
			__int64 rest_len = 0;
			{
				long cur = ftell(in);
				fseek(in, 0, SEEK_END);
				long end = ftell(in);
				if (cur >= 0 && end >= cur) rest_len = (__int64)end - cur;
				fseek(in, cur, SEEK_SET);
			}
			if (rest_len > 0 && (__int64)key_num * 4 > rest_len)
			{
				fclose(in);
				Clear();
				return false;
			}
			if (rest_len > 0 && (__int64)key_num * 4 > rest_len)
			{
				// 至少每个 frame 要能装下一个长度字段, 否则数量与文件大小对不上
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