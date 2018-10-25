#ifndef _ZQ_FACE_RECOGNIZER_SPHERE_FACE_H_
#define _ZQ_FACE_RECOGNIZER_SPHERE_FACE_H_
#pragma once

#include "ZQ_FaceRecognizer.h"
#include "ZQ_FaceRecognizerUtils.h"
#include "ZQ_CNN_CompileConfig.h"
#include <immintrin.h>


#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX2
#define zq_mm256_fmadd_ps _mm256_fmadd_ps
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm256_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif

namespace ZQ
{
	class ZQ_FaceRecognizerSphereFace : public ZQ_FaceRecognizer
	{
	public:

		virtual bool Init(const std::string model_name,
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_blob_name = ""
		) = 0;

		virtual int GetFeatDim() const = 0;

		virtual int GetCropWidth() const { return 96; }
		
		virtual int GetCropHeight() const { return 112; }
		
		bool CropImage(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y, 
			unsigned char* crop_img, int crop_widthStep) const
		{
			float face5point[10] =
			{
				face5point_x[0],face5point_y[0],
				face5point_x[1],face5point_y[1],
				face5point_x[2],face5point_y[2],
				face5point_x[3],face5point_y[3],
				face5point_x[4],face5point_y[4]
			};

			cv::Mat input, crop;
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_BGR:case ZQ_PIXEL_FMT_RGB:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 3));
				for (int h = 0; h < in_height; h++)
					memcpy(input.data + input.step[0] * h, in_img + in_widthStep*h, in_width * 3);
				break;
			case ZQ_PIXEL_FMT_BGRX:case ZQ_PIXEL_FMT_RGBX: case ZQ_PIXEL_FMT_XBGR: case ZQ_PIXEL_FMT_XRGB:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 4));
				for (int h = 0; h < in_height; h++)
					memcpy(input.data + input.step[0] * h, in_img + in_widthStep*h, in_width * 4);
				break;
			case ZQ_PIXEL_FMT_GRAY:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 1));
				for (int h = 0; h < in_height; h++)
					memcpy(input.data + input.step[0] * h, in_img + in_widthStep*h, in_width * 1);
				break;
			default:
				return false;
				break;
			}
			
			if (GetCropHeight() == 160 && GetCropWidth() == 160)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_160x160<float>(input, face5point, crop))
					return false;
			}
			else if (GetCropHeight() == 112 && GetCropWidth() == 112)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_112x112<float>(input, face5point, crop))
					return false;
			}
			else if (GetCropHeight() == 112 && GetCropWidth() == 96)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_112x96<float>(input, face5point, crop))
					return false;
			}
			else
				return false;

			int crop_height = GetCropHeight();
			int crop_width = GetCropWidth();
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_BGR:case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < crop_height; h++)
					memcpy(crop_img + h*crop_widthStep, crop.data + h*crop.step[0], sizeof(unsigned char)*crop_width * 3);
				break;
			case ZQ_PIXEL_FMT_BGRX:case ZQ_PIXEL_FMT_RGBX: case ZQ_PIXEL_FMT_XBGR: case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < crop_height; h++)
					memcpy(crop_img + h*crop_widthStep, crop.data + h*crop.step[0], sizeof(unsigned char)*crop_width * 4);
				break;
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < crop_height; h++)
					memcpy(crop_img + h*crop_widthStep, crop.data + h*crop.step[0], sizeof(unsigned char)*crop_width * 1);
				break;
			default:
				return false;
				break;
			}
		
			return true;
		}

		bool ExtractFeature(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y, float* feat, bool normalize)
		{
			float face5point[10] =
			{
				face5point_x[0],face5point_y[0],
				face5point_x[1],face5point_y[1],
				face5point_x[2],face5point_y[2],
				face5point_x[3],face5point_y[3],
				face5point_x[4],face5point_y[4]
			};

			cv::Mat input, crop;
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_BGR:case ZQ_PIXEL_FMT_RGB:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 3), (void*)in_img, in_widthStep);

				break;
			case ZQ_PIXEL_FMT_BGRX:case ZQ_PIXEL_FMT_RGBX: case ZQ_PIXEL_FMT_XBGR: case ZQ_PIXEL_FMT_XRGB:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 4), (void*)in_img, in_widthStep);
				break;
			case ZQ_PIXEL_FMT_GRAY:
				input = cv::Mat(cv::Size(in_width, in_height), CV_MAKETYPE(8U, 1), (void*)in_img, in_widthStep);
				break;
			default:
				return false;
				break;
			}

			if (GetCropHeight() == 160 && GetCropWidth() == 160)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_160x160<float>(input, face5point, crop))
					return false;
			}
			else if (GetCropHeight() == 112 && GetCropWidth() == 112)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_112x112<float>(input, face5point, crop))
					return false;
			}
			else if (GetCropHeight() == 112 && GetCropWidth() == 96)
			{
				if (!ZQ_FaceRecognizerUtils::CropImage_112x96<float>(input, face5point, crop))
					return false;
			}
			else
				return false;

			return ExtractFeature(crop.data, crop.step[0], pixFmt, feat, normalize);
		}

		virtual bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat pixFmt, float* feat, bool normalize) = 0;

		float CalSimilarity(const float* feat1, const float* feat2) const
		{
			return CalSimilarity(GetFeatDim(), feat1, feat2);
		}

		static float CalSimilarity(int dim, const float* feat1, const float* feat2)
		{
			bool handled = false;
			if (dim == 128)
			{
				if (((long long)feat1) % 32 == 0 && ((long long)feat2) % 32 == 0)
					return _cal_similarity_avx_dim128(feat1, feat2);
			}
			else if (dim == 256)
			{
				if (((long long)feat1) % 32 == 0 && ((long long)feat2) % 32 == 0)
					return _cal_similarity_avx_dim256(feat1, feat2);
			}
			else if (dim == 512)
			{
				if (((long long)feat1) % 32 == 0 && ((long long)feat2) % 32 == 0)
					return _cal_similarity_avx_dim512(feat1, feat2);
			}
			
			if(!handled)
			{
				float dot = 0;
				for (int i = 0; i < dim; i++)
				{
					dot += feat1[i] * feat2[i];
				}
				return dot;
			}
			return 0;
		}

	public:

		static float _cal_similarity_avx_dim128(const float* pt1, const float* pt2)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			_declspec(align(32)) float q[8];
			__m256 sum_vec = _mm256_mul_ps(_mm256_load_ps(pt1), _mm256_load_ps(pt2));
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 8), _mm256_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 16), _mm256_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 24), _mm256_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 32), _mm256_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 40), _mm256_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 48), _mm256_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 56), _mm256_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 64), _mm256_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 72), _mm256_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 80), _mm256_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 88), _mm256_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 96), _mm256_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 104), _mm256_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 112), _mm256_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 120), _mm256_load_ps(pt2 + 120), sum_vec);
			_mm256_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
			return score;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			_declspec(align(32)) float q[4];
			__m128 sum_vec = _mm_mul_ps(_mm_load_ps(pt1), _mm_load_ps(pt2));
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 4), _mm_load_ps(pt2 + 4), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 8), _mm_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 12), _mm_load_ps(pt2 + 12), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 16), _mm_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 20), _mm_load_ps(pt2 + 20), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 24), _mm_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 28), _mm_load_ps(pt2 + 28), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 32), _mm_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 36), _mm_load_ps(pt2 + 36), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 40), _mm_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 44), _mm_load_ps(pt2 + 44), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 48), _mm_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 52), _mm_load_ps(pt2 + 52), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 56), _mm_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 60), _mm_load_ps(pt2 + 60), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 64), _mm_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 68), _mm_load_ps(pt2 + 68), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 72), _mm_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 76), _mm_load_ps(pt2 + 76), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 80), _mm_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 84), _mm_load_ps(pt2 + 84), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 88), _mm_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 92), _mm_load_ps(pt2 + 92), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 96), _mm_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 100), _mm_load_ps(pt2 + 100), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 104), _mm_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 108), _mm_load_ps(pt2 + 108), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 112), _mm_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 116), _mm_load_ps(pt2 + 116), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 120), _mm_load_ps(pt2 + 120), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 124), _mm_load_ps(pt2 + 124), sum_vec);
			_mm_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3];
			return score;
#else
			float score = 0.0f;
			for (int i = 0; i < 128; i++)
				score += pt1[i] * pt2[i];
			return score;
#endif
		}

		static float _cal_similarity_avx_dim256(const float* pt1, const float* pt2)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			_declspec(align(32)) float q[8];
			__m256 sum_vec = _mm256_mul_ps(_mm256_load_ps(pt1), _mm256_load_ps(pt2));
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 8), _mm256_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 16), _mm256_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 24), _mm256_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 32), _mm256_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 40), _mm256_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 48), _mm256_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 56), _mm256_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 64), _mm256_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 72), _mm256_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 80), _mm256_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 88), _mm256_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 96), _mm256_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 104), _mm256_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 112), _mm256_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 120), _mm256_load_ps(pt2 + 120), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 128), _mm256_load_ps(pt2 + 128), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 136), _mm256_load_ps(pt2 + 136), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 144), _mm256_load_ps(pt2 + 144), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 152), _mm256_load_ps(pt2 + 152), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 160), _mm256_load_ps(pt2 + 160), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 168), _mm256_load_ps(pt2 + 168), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 176), _mm256_load_ps(pt2 + 176), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 184), _mm256_load_ps(pt2 + 184), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 192), _mm256_load_ps(pt2 + 192), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 200), _mm256_load_ps(pt2 + 200), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 208), _mm256_load_ps(pt2 + 208), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 216), _mm256_load_ps(pt2 + 216), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 224), _mm256_load_ps(pt2 + 224), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 232), _mm256_load_ps(pt2 + 232), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 240), _mm256_load_ps(pt2 + 240), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 248), _mm256_load_ps(pt2 + 248), sum_vec);
			_mm256_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7]; 
			return score;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			_declspec(align(32)) float q[4];
			__m128 sum_vec = _mm_mul_ps(_mm_load_ps(pt1), _mm_load_ps(pt2));
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 4), _mm_load_ps(pt2 + 4), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 8), _mm_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 12), _mm_load_ps(pt2 + 12), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 16), _mm_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 20), _mm_load_ps(pt2 + 20), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 24), _mm_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 28), _mm_load_ps(pt2 + 28), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 32), _mm_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 36), _mm_load_ps(pt2 + 36), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 40), _mm_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 44), _mm_load_ps(pt2 + 44), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 48), _mm_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 52), _mm_load_ps(pt2 + 52), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 56), _mm_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 60), _mm_load_ps(pt2 + 60), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 64), _mm_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 68), _mm_load_ps(pt2 + 68), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 72), _mm_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 76), _mm_load_ps(pt2 + 76), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 80), _mm_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 84), _mm_load_ps(pt2 + 84), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 88), _mm_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 92), _mm_load_ps(pt2 + 92), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 96), _mm_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 100), _mm_load_ps(pt2 + 100), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 104), _mm_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 108), _mm_load_ps(pt2 + 108), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 112), _mm_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 116), _mm_load_ps(pt2 + 116), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 120), _mm_load_ps(pt2 + 120), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 124), _mm_load_ps(pt2 + 124), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 128), _mm_load_ps(pt2 + 128), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 132), _mm_load_ps(pt2 + 132), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 136), _mm_load_ps(pt2 + 136), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 140), _mm_load_ps(pt2 + 140), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 144), _mm_load_ps(pt2 + 144), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 148), _mm_load_ps(pt2 + 148), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 152), _mm_load_ps(pt2 + 152), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 156), _mm_load_ps(pt2 + 156), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 160), _mm_load_ps(pt2 + 160), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 164), _mm_load_ps(pt2 + 164), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 168), _mm_load_ps(pt2 + 168), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 172), _mm_load_ps(pt2 + 172), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 176), _mm_load_ps(pt2 + 176), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 180), _mm_load_ps(pt2 + 180), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 184), _mm_load_ps(pt2 + 184), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 188), _mm_load_ps(pt2 + 188), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 192), _mm_load_ps(pt2 + 192), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 196), _mm_load_ps(pt2 + 196), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 200), _mm_load_ps(pt2 + 200), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 204), _mm_load_ps(pt2 + 204), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 208), _mm_load_ps(pt2 + 208), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 212), _mm_load_ps(pt2 + 212), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 216), _mm_load_ps(pt2 + 216), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 220), _mm_load_ps(pt2 + 220), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 224), _mm_load_ps(pt2 + 224), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 228), _mm_load_ps(pt2 + 228), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 232), _mm_load_ps(pt2 + 232), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 236), _mm_load_ps(pt2 + 236), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 240), _mm_load_ps(pt2 + 240), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 244), _mm_load_ps(pt2 + 244), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 248), _mm_load_ps(pt2 + 248), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 252), _mm_load_ps(pt2 + 252), sum_vec);
			_mm_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3];
			return score;
#else
			float score = 0.0f;
			for (int i = 0; i < 256; i++)
				score += pt1[i] * pt2[i];
			return score;
#endif
		}

		static float _cal_similarity_avx_dim512(const float* pt1, const float* pt2)
		{
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
			_declspec(align(32)) float q[8];
			__m256 sum_vec = _mm256_mul_ps(_mm256_load_ps(pt1), _mm256_load_ps(pt2));
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 8), _mm256_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 16), _mm256_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 24), _mm256_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 32), _mm256_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 40), _mm256_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 48), _mm256_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 56), _mm256_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 64), _mm256_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 72), _mm256_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 80), _mm256_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 88), _mm256_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 96), _mm256_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 104), _mm256_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 112), _mm256_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 120), _mm256_load_ps(pt2 + 120), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 128), _mm256_load_ps(pt2 + 128), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 136), _mm256_load_ps(pt2 + 136), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 144), _mm256_load_ps(pt2 + 144), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 152), _mm256_load_ps(pt2 + 152), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 160), _mm256_load_ps(pt2 + 160), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 168), _mm256_load_ps(pt2 + 168), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 176), _mm256_load_ps(pt2 + 176), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 184), _mm256_load_ps(pt2 + 184), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 192), _mm256_load_ps(pt2 + 192), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 200), _mm256_load_ps(pt2 + 200), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 208), _mm256_load_ps(pt2 + 208), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 216), _mm256_load_ps(pt2 + 216), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 224), _mm256_load_ps(pt2 + 224), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 232), _mm256_load_ps(pt2 + 232), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 240), _mm256_load_ps(pt2 + 240), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 248), _mm256_load_ps(pt2 + 248), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 256), _mm256_load_ps(pt2 + 256), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 264), _mm256_load_ps(pt2 + 264), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 272), _mm256_load_ps(pt2 + 272), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 280), _mm256_load_ps(pt2 + 280), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 288), _mm256_load_ps(pt2 + 288), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 296), _mm256_load_ps(pt2 + 296), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 304), _mm256_load_ps(pt2 + 304), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 312), _mm256_load_ps(pt2 + 312), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 320), _mm256_load_ps(pt2 + 320), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 328), _mm256_load_ps(pt2 + 328), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 336), _mm256_load_ps(pt2 + 336), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 344), _mm256_load_ps(pt2 + 344), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 352), _mm256_load_ps(pt2 + 352), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 360), _mm256_load_ps(pt2 + 360), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 368), _mm256_load_ps(pt2 + 368), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 376), _mm256_load_ps(pt2 + 376), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 384), _mm256_load_ps(pt2 + 384), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 392), _mm256_load_ps(pt2 + 392), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 400), _mm256_load_ps(pt2 + 400), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 408), _mm256_load_ps(pt2 + 408), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 416), _mm256_load_ps(pt2 + 416), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 424), _mm256_load_ps(pt2 + 424), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 432), _mm256_load_ps(pt2 + 432), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 440), _mm256_load_ps(pt2 + 440), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 448), _mm256_load_ps(pt2 + 448), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 456), _mm256_load_ps(pt2 + 456), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 464), _mm256_load_ps(pt2 + 464), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 472), _mm256_load_ps(pt2 + 472), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 480), _mm256_load_ps(pt2 + 480), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 488), _mm256_load_ps(pt2 + 488), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 496), _mm256_load_ps(pt2 + 496), sum_vec);
			sum_vec = zq_mm256_fmadd_ps(_mm256_load_ps(pt1 + 504), _mm256_load_ps(pt2 + 504), sum_vec);
			_mm256_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
			return score;
#elif ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
			_declspec(align(32)) float q[4];
			__m128 sum_vec = _mm_mul_ps(_mm_load_ps(pt1), _mm_load_ps(pt2));
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 4), _mm_load_ps(pt2 + 4), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 8), _mm_load_ps(pt2 + 8), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 12), _mm_load_ps(pt2 + 12), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 16), _mm_load_ps(pt2 + 16), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 20), _mm_load_ps(pt2 + 20), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 24), _mm_load_ps(pt2 + 24), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 28), _mm_load_ps(pt2 + 28), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 32), _mm_load_ps(pt2 + 32), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 36), _mm_load_ps(pt2 + 36), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 40), _mm_load_ps(pt2 + 40), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 44), _mm_load_ps(pt2 + 44), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 48), _mm_load_ps(pt2 + 48), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 52), _mm_load_ps(pt2 + 52), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 56), _mm_load_ps(pt2 + 56), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 60), _mm_load_ps(pt2 + 60), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 64), _mm_load_ps(pt2 + 64), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 68), _mm_load_ps(pt2 + 68), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 72), _mm_load_ps(pt2 + 72), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 76), _mm_load_ps(pt2 + 76), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 80), _mm_load_ps(pt2 + 80), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 84), _mm_load_ps(pt2 + 84), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 88), _mm_load_ps(pt2 + 88), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 92), _mm_load_ps(pt2 + 92), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 96), _mm_load_ps(pt2 + 96), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 100), _mm_load_ps(pt2 + 100), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 104), _mm_load_ps(pt2 + 104), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 108), _mm_load_ps(pt2 + 108), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 112), _mm_load_ps(pt2 + 112), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 116), _mm_load_ps(pt2 + 116), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 120), _mm_load_ps(pt2 + 120), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 124), _mm_load_ps(pt2 + 124), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 128), _mm_load_ps(pt2 + 128), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 132), _mm_load_ps(pt2 + 132), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 136), _mm_load_ps(pt2 + 136), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 140), _mm_load_ps(pt2 + 140), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 144), _mm_load_ps(pt2 + 144), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 148), _mm_load_ps(pt2 + 148), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 152), _mm_load_ps(pt2 + 152), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 156), _mm_load_ps(pt2 + 156), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 160), _mm_load_ps(pt2 + 160), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 164), _mm_load_ps(pt2 + 164), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 168), _mm_load_ps(pt2 + 168), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 172), _mm_load_ps(pt2 + 172), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 176), _mm_load_ps(pt2 + 176), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 180), _mm_load_ps(pt2 + 180), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 184), _mm_load_ps(pt2 + 184), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 188), _mm_load_ps(pt2 + 188), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 192), _mm_load_ps(pt2 + 192), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 196), _mm_load_ps(pt2 + 196), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 200), _mm_load_ps(pt2 + 200), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 204), _mm_load_ps(pt2 + 204), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 208), _mm_load_ps(pt2 + 208), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 212), _mm_load_ps(pt2 + 212), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 216), _mm_load_ps(pt2 + 216), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 220), _mm_load_ps(pt2 + 220), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 224), _mm_load_ps(pt2 + 224), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 228), _mm_load_ps(pt2 + 228), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 232), _mm_load_ps(pt2 + 232), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 236), _mm_load_ps(pt2 + 236), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 240), _mm_load_ps(pt2 + 240), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 244), _mm_load_ps(pt2 + 244), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 248), _mm_load_ps(pt2 + 248), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 252), _mm_load_ps(pt2 + 252), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 256), _mm_load_ps(pt2 + 256), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 260), _mm_load_ps(pt2 + 260), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 264), _mm_load_ps(pt2 + 264), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 268), _mm_load_ps(pt2 + 268), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 272), _mm_load_ps(pt2 + 272), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 276), _mm_load_ps(pt2 + 276), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 280), _mm_load_ps(pt2 + 280), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 284), _mm_load_ps(pt2 + 284), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 288), _mm_load_ps(pt2 + 288), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 292), _mm_load_ps(pt2 + 292), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 296), _mm_load_ps(pt2 + 296), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 300), _mm_load_ps(pt2 + 300), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 304), _mm_load_ps(pt2 + 304), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 308), _mm_load_ps(pt2 + 308), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 312), _mm_load_ps(pt2 + 312), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 316), _mm_load_ps(pt2 + 316), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 320), _mm_load_ps(pt2 + 320), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 324), _mm_load_ps(pt2 + 324), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 328), _mm_load_ps(pt2 + 328), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 332), _mm_load_ps(pt2 + 332), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 336), _mm_load_ps(pt2 + 336), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 340), _mm_load_ps(pt2 + 340), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 344), _mm_load_ps(pt2 + 344), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 348), _mm_load_ps(pt2 + 348), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 352), _mm_load_ps(pt2 + 352), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 356), _mm_load_ps(pt2 + 356), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 360), _mm_load_ps(pt2 + 360), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 364), _mm_load_ps(pt2 + 364), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 368), _mm_load_ps(pt2 + 368), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 372), _mm_load_ps(pt2 + 372), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 376), _mm_load_ps(pt2 + 376), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 380), _mm_load_ps(pt2 + 380), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 384), _mm_load_ps(pt2 + 384), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 388), _mm_load_ps(pt2 + 388), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 392), _mm_load_ps(pt2 + 392), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 396), _mm_load_ps(pt2 + 396), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 400), _mm_load_ps(pt2 + 400), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 404), _mm_load_ps(pt2 + 404), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 408), _mm_load_ps(pt2 + 408), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 412), _mm_load_ps(pt2 + 412), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 416), _mm_load_ps(pt2 + 416), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 420), _mm_load_ps(pt2 + 420), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 424), _mm_load_ps(pt2 + 424), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 428), _mm_load_ps(pt2 + 428), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 432), _mm_load_ps(pt2 + 432), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 436), _mm_load_ps(pt2 + 436), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 440), _mm_load_ps(pt2 + 440), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 444), _mm_load_ps(pt2 + 444), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 448), _mm_load_ps(pt2 + 448), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 452), _mm_load_ps(pt2 + 452), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 456), _mm_load_ps(pt2 + 456), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 460), _mm_load_ps(pt2 + 460), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 464), _mm_load_ps(pt2 + 464), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 468), _mm_load_ps(pt2 + 468), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 472), _mm_load_ps(pt2 + 472), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 476), _mm_load_ps(pt2 + 476), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 480), _mm_load_ps(pt2 + 480), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 484), _mm_load_ps(pt2 + 484), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 488), _mm_load_ps(pt2 + 488), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 492), _mm_load_ps(pt2 + 492), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 496), _mm_load_ps(pt2 + 496), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 500), _mm_load_ps(pt2 + 500), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 504), _mm_load_ps(pt2 + 504), sum_vec);
			sum_vec = zq_mm_fmadd_ps(_mm_load_ps(pt1 + 508), _mm_load_ps(pt2 + 508), sum_vec);
			_mm_store_ps(q, sum_vec);
			float score = q[0] + q[1] + q[2] + q[3];
			return score;
#else
			float score = 0.0f;
			for (int i = 0; i < 512; i++)
				score += pt1[i] * pt2[i];
			return score;
#endif
		}
	};
}
#endif
