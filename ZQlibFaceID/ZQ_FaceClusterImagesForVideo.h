#ifndef _ZQ_FACE_CLUSTER_IMAGES_FOR_VIDEO_H_
#define _ZQ_FACE_CLUSTER_IMAGES_FOR_VIDEO_H_
#pragma once

// 审计修复 2026-10-03（附录 EG）：这一行必须放在**最前面**。
// 本文件（以及下面 include 的 ZQ_MathBase.h）用了 __min / __max / __int64，
// 这三个在 ZQ_CNN_CompileConfig.h 里为非 MSVC 提供了可移植定义。
// 之前这里 include 的是 <opencv2\opencv.hpp>（**反斜杠**），加上 MSVC 专有写法，
// 整份头在 Linux/gcc 上**根本编不过** —— 而全仓没有任何 .cpp include 它，
// 所以这个缺陷在两边构建里都不会暴露。
#include "ZQ_CNN_CompileConfig.h"
#include "ZQ_JpegEncoder.h"
#include "ZQ_JpegDecoder.h"
#include <vector>
#include <opencv2/opencv.hpp>
namespace ZQ
{
	class ZQ_FaceClusterImagesForVideo
	{
	private:
		unsigned char* buffer;
		std::vector<int> offset;
		std::vector<int> length;

	public:
		ZQ_FaceClusterImagesForVideo()
		{
			buffer = 0;
		}
		~ZQ_FaceClusterImagesForVideo()
		{
			Clear();
		}

		int GetImageNum() const { return offset.size(); }

		bool FetchImage(int i, cv::Mat& image) const
		{
			if (i < 0 || i >= offset.size())
				return false;

			unsigned char* pDst = 0;
			int width, height, nChannels, widthStep;
			if (!ZQ_JpegDecoder::Decode(buffer + offset[i], length[i], ZQ_JpegCodecColorType::OUT_EXT_BGR,
				pDst, width, height, nChannels, widthStep, 4))
			{
				return false;
			}
			image = cv::Mat(cv::Size(width, height), CV_MAKETYPE(8, 3));
			int _widthstep = __min(widthStep, image.step[0]);
			for (int h = 0; h < height; h++)
			{
				memcpy(image.data + h*image.step[0], pDst + h*widthStep, _widthstep);
			}
			free(pDst);
			return true;
		}

		bool ConvertFromCVMat(const std::vector<cv::Mat>& images)
		{
			Clear();
			int image_num = images.size();
			for (int i = 0; i < image_num; i++)
			{
				if (images[i].empty() || images[i].channels() != 3)
					return false;
			}
			std::vector<unsigned char*> compressed_buffers;
			std::vector<unsigned long> compressed_length;
			

			for (int i = 0; i < image_num; i++)
			{
				unsigned char* pDst = 0;
				unsigned long dstLen = 0;
				if (!ZQ_JpegEncoder::Encode(images[i].data, images[i].cols, images[i].rows, 3, images[i].step[0],
					ZQ_JpegCodecColorType::IN_EXT_BGR, pDst, dstLen, 60))
				{
					for (int j = 0; j < i; j++)
					{
						free(compressed_buffers[j]);
						compressed_buffers[j] = 0;
					}
					// 原来漏了 return, 会把空指针压进容器
					return false;
				}
				compressed_buffers.push_back(pDst);
				compressed_length.push_back(dstLen);
			}

			offset.resize(image_num);
			length.resize(image_num);

			// 几十张高分辨率图累加起来 int 就能溢出成负，malloc((size_t)负数)
			// 会变成天文数字。这里用 __int64，与同文件 LoadFromFile 一致。
			__int64 _off = 0;
			for (int i = 0; i < image_num; i++)
			{
				offset[i] = (int)_off;
				length[i] = compressed_length[i];
				_off += length[i];
			}
			buffer = (unsigned char*)malloc((size_t)_off);
			if (buffer)
			{
				for (int i = 0; i < image_num; i++)
					memcpy(buffer + offset[i], compressed_buffers[i], length[i]);
			}

			for (int j = 0; j < image_num; j++)
			{
				free(compressed_buffers[j]);
				compressed_buffers[j] = 0;
			}

			if (buffer == 0)
			{
				Clear();
				return false;
			}
			else
			{
				return true;
			}
		}

		void Clear()
		{
			if (buffer)
			{
				free(buffer);
				buffer = 0;
			}
			offset.clear();
			length.clear();
		}

		bool SaveToFile(const std::string& file) const
		{
			int num = offset.size();
			FILE* out = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
#else
			out = fopen(file.c_str(), "wb");
			if (out == NULL)
				return false;
#endif
			fwrite(&num, sizeof(int), 1, out);
			if (num == 0)
			{
				fclose(out);
				return true;
			}

			fwrite(&offset[0], sizeof(int), num, out);
			fwrite(&length[0], sizeof(int), num, out);
			fwrite(buffer, sizeof(unsigned char), offset[num - 1] + length[num - 1], out);
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
			int num = 0;
			if (1 != fread(&num, sizeof(int), 1, in) || num < 0)
			{
				fclose(in);
				Clear();
				return false;
			}
			// **审计修复 2026-10-03（附录 EF）**：num 来自不可信文件，**只挡负数不够**。
			// 紧接着的 `offset.resize(num)` / `length.resize(num)` 会按它申请内存：
			// `num = 0x7FFFFFFF` 时每个 vector 要 8 GB。
			//   · 开着 overcommit 时 resize 成功，随后 fread 因文件没那么多字节而失败 ——
			//     只是瞬时占用 8 GB 虚拟地址；
			//   · **没开 overcommit / 有 cgroup 限额 / 32 位进程时抛 std::bad_alloc**，
			//     而这条加载路径**没有 catch**，异常一路冒到 std::terminate() -> abort。
			//     也就是说**一个 4 字节的文件就能让进程崩掉**。
			// 隔离复现（dbg_resize_oom.cpp）：
			//   不限内存      -> "resize 成功（虚拟地址空间够）"
			//   ulimit -v 1GB -> "抛出 std::bad_alloc"
			// 修法与同仓库 ZQ_FaceContainerForVideo.h:84-100 **完全一致**
			// （那里的 key_num 面对的是同一个问题，注释就是这么写的）——
			// 用**剩余文件长度**做上界交叉校验：每个条目至少要装下自己的
			// offset 与 length 两个 int（共 8 字节）。
			__int64 rest_len = 0;
			{
				long cur = ftell(in);
				fseek(in, 0, SEEK_END);
				long end = ftell(in);
				if (cur >= 0 && end >= cur) rest_len = (__int64)end - cur;
				fseek(in, cur, SEEK_SET);
			}
			if (rest_len > 0 && (__int64)num * 8 > rest_len)
			{
				fclose(in);
				Clear();
				return false;
			}
			if (num == 0)
			{
				fclose(in);
				return true;
			}

			offset.resize(num);
			length.resize(num);
			if (num != fread(&offset[0], sizeof(int), num, in)
				|| num != fread(&length[0], sizeof(int),num,in))
			{
				fclose(in);
				Clear();
				// 数量不符说明文件被截断, 返回 true 会让调用方拿到"加载成功但内容已清空"的容器
				return false;
			}

			// length 可以是恶意文件里的负数, 累加会污染 _off 并让后面的分配/读取越界
			__int64 _off = 0;
			for (int i = 0; i < num; i++)
			{
				if (length[i] <= 0 || offset[i] < 0 || (__int64)offset[i] != _off)
				{
					fclose(in);
					Clear();
					return false;
				}
				_off += length[i];
				if (_off > 0x7FFFFFFF)
				{
					fclose(in);
					Clear();
					return false;
				}
			}
			buffer = (unsigned char*)malloc(sizeof(unsigned char)*_off);
			if (buffer == 0)
			{
				fclose(in);
				Clear();
				return false;
			}
			if (_off != fread(buffer, sizeof(unsigned char), _off, in))
			{
				fclose(in);
				Clear();
				return false;
			}
			fclose(in);
			return true;
		}

	
	};
}
#endif
