#ifndef _ZQ_FACE_CLUSTER_IMAGES_FOR_VIDEO_H_
#define _ZQ_FACE_CLUSTER_IMAGES_FOR_VIDEO_H_
#pragma once

#include "ZQ_JpegEncoder.h"
#include "ZQ_JpegDecoder.h"
#include <vector>
#include <opencv2\opencv.hpp>
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
				}
				compressed_buffers.push_back(pDst);
				compressed_length.push_back(dstLen);
			}

			offset.resize(image_num);
			length.resize(image_num);

			int _off = 0;
			for (int i = 0; i < image_num; i++)
			{
				offset[i] = _off;
				length[i] = compressed_length[i];
				_off += length[i];
			}
			buffer = (unsigned char*)malloc(sizeof(unsigned char)*_off);
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
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
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
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;
			int num = 0;
			if (1 != fread(&num, sizeof(int), 1, in) || num < 0)
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
				return true;
			}

			int _off = 0;
			for (int i = 0; i < num; i++)
			{
				if (_off != offset[i])
				{
					fclose(in);
					Clear();
					return false;
				}
				_off += length[i];
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
