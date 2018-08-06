#ifndef _ZQ_FACE_RECOGNIZER_SEETAFACE_H_
#define _ZQ_FACE_RECOGNIZER_SEETAFACE_H_
#pragma once
#include "ZQ_FaceRecognizer.h"
#include "face_identification.h"
#include "ZQ_MathBase.h"
namespace ZQ
{
	class ZQ_FaceRecognizerSeetaFace : public ZQ_FaceRecognizer
	{
	public:
		ZQ_FaceRecognizerSeetaFace()
		{
			net = 0;
			width = 0;
			height = 0;
			feat_dim = 0;
		}

		~ZQ_FaceRecognizerSeetaFace()
		{
			if (net)
				delete net;
			net = 0;
		}

		virtual bool Init(const std::string model_name,
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_blob_name = ""
		)
		{
			if (net)
			{
				delete net;
				net = 0;
			}

			if (net == 0)
			{
				net = new seeta::FaceIdentification();
				if (0 == net->LoadModel(model_name.c_str()))
				{
					delete net;
					net = 0;
					return false;
				}
			}
			width = net->crop_width();
			height = net->crop_height();
			feat_dim = net->feature_size();
			return true;
		}

		virtual int GetFeatDim() const
		{
			return feat_dim;
		}

		int GetCropWidth() const 
		{ 
			return width;
		}

		int GetCropHeight() const 
		{
			return height;
		}

		virtual bool CropImage(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y,
			unsigned char* crop_img, int crop_widthStep) const
		{
			if (net == 0 || in_width <= 0 || in_height <= 0)
				return false;
		
			if (pixFmt == ZQ_PixelFormat::ZQ_PIXEL_FMT_RGB || ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR)
			{
				std::vector<unsigned char> buffer;
				const unsigned char* img_ptr = 0;
				if (in_width * 3 == in_widthStep)
				{
					img_ptr = in_img;
				}
				else
				{
					buffer.resize(in_width*in_height * 3);
					img_ptr = &buffer[0];
					for (int h = 0; h < in_height; h++)
					{
						memcpy(&buffer[0] + h*in_width * 3, in_img + h*in_widthStep, sizeof(unsigned char)*in_width * 3);
					}
				}

				std::vector<seeta::FacialLandmark> landmarks;
				landmarks.resize(5);
				for (int j = 0; j < 5; j++)
				{
					landmarks[j].x = face5point_x[j];
					landmarks[j].y = face5point_y[j];
				}

				// ImageData store data of an image without memory alignment.
				seeta::ImageData src_img_data(in_height, in_width, 3);
				src_img_data.data = (unsigned char*)img_ptr;

				// Create a image to store crop face.

				std::vector<unsigned char> dst_buffer(width*height * 3);
				seeta::ImageData dst_img_data(width, height, 3);
				dst_img_data.data = &dst_buffer[0];
				if (0 == net->CropFace(src_img_data, &landmarks[0], dst_img_data))
					return false;
				for (int h = 0; h < height; h++)
				{
					memcpy(crop_img + h*crop_widthStep, &dst_buffer[0] + h*width * 3, sizeof(unsigned char)*width * 3);
				}
				return true;
			}

			return false;
		}

		virtual bool ExtractFeature(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y, float* feat, bool normalize)
		{
			if (width == 0 || height == 0)
				return false;
			std::vector<unsigned char> buffer(width*height * 3);
			if (!CropImage(in_img, in_width, in_height, in_widthStep, pixFmt, face5point_x, face5point_y, &buffer[0], width * 3))
				return false;
			return ExtractFeature(&buffer[0], width * 3, pixFmt, feat, normalize);
		}

		virtual bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat pixFmt, float* feat, bool normalize)
		{
			if (pixFmt != ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR && pixFmt != ZQ_PixelFormat::ZQ_PIXEL_FMT_RGB)
				return false;

			std::vector<unsigned char> buffer(height*width * 3);
			unsigned char* ptr = &buffer[0];
			if (pixFmt == ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR)
			{
				for (int h = 0; h < height; h++)
				{
					memcpy(ptr + h*width * 3, img + h*widthStep, sizeof(unsigned char)*width * 3);
				}
			}
			else
			{
				for (int h = 0; h < height; h++)
				{
					const unsigned char* in_row = img + h*widthStep;
					unsigned char* cur_row = ptr + h*width * 3;
					for (int w = 0; w < width; w++)
					{
						cur_row[w * 3 + 0] = in_row[w * 3 + 2];
						cur_row[w * 3 + 1] = in_row[w * 3 + 1];
						cur_row[w * 3 + 2] = in_row[w * 3 + 0];
					}
				}
			}
			
			// ImageData store data of an image without memory alignment.
			seeta::ImageData src_img_data(width, height, 3);
			src_img_data.data = ptr;

			// Extract feature
			net->ExtractFeature(src_img_data, feat);
			if (normalize)
				ZQ_MathBase::Normalize(feat_dim, feat);
			return true;
		}

		virtual float CalSimilarity(const float* feat1, const float* feat2) const
		{
			return net->CalcSimilarity((float*)feat1, (float*)feat2);
		}

	private:
		seeta::FaceIdentification*  net;
		int width;
		int height;
		int feat_dim;
	};
}

#endif
