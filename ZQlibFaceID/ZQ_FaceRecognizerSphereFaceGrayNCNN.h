#ifndef _ZQ_FACE_RECOGNIZER_SPHERE_FACE_GRAY_NCNN_H_
#define _ZQ_FACE_RECOGNIZER_SPHERE_FACE_GRAY_NCNN_H_
#pragma once
#include "ZQ_FaceRecognizerSphereFace.h"
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ZQ_MathBase.h"
#include <string.h>
namespace ZQ
{
	class ZQ_FaceRecognizerSphereFaceGrayNCNN : public ZQ_FaceRecognizerSphereFace
	{
	public:
		ZQ_FaceRecognizerSphereFaceGrayNCNN()
		{
			feat_dim = 0;
			bgr_buffer.resize(GetCropHeight()*GetCropWidth());
		}
		~ZQ_FaceRecognizerSphereFaceGrayNCNN()
		{

		}

		virtual bool Init(const std::string model_name,
			const std::string param_file = "", const std::string model_file = "",
			const std::string out_blob_name = "")
		{
			int H = GetCropHeight();
			int W = GetCropWidth();
			this->param_file = param_file;
			this->bin_file = model_file;
			this->output_blob_name = out_blob_name;

			if (0 != net.load_param(this->param_file.c_str()))
			{
				feat_dim = 0;
				return false;
			}
			if (0 != net.load_model(this->bin_file.c_str()))
			{
				feat_dim = 0;
				return false;
			}
			this->feat_dim = feat_dim;
			
			ncnn::Mat out;
			ncnn::Mat face_image = ncnn::Mat(W, H, 1);
			ncnn::Extractor ex = net.create_extractor();
			ex.set_light_mode(true);
			ex.input("data", face_image);
			ex.extract(this->output_blob_name.c_str(), out);
			feat_dim = out.w;
			return true;
		}

		virtual int GetFeatDim() const
		{
			return feat_dim;
		}

		virtual bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat pixFmt, float* feat, bool normalize)
		{
			int crop_width = GetCropWidth();
			int crop_height = GetCropHeight();

			const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
			const float norm_vals[3] = { 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };
			
			
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						cur_pix_ptr[0] = ori_pix_ptr[0];
					}
				}
				input = ncnn::Mat::from_pixels_resize(img, ncnn::Mat::PIXEL_GRAY, crop_width, crop_height, crop_width, crop_height);
				input.substract_mean_normalize(mean_vals, norm_vals);
				break;
			case ZQ_PIXEL_FMT_BGR:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*3;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w*3;
						*cur_pix_ptr = ori_pix_ptr[0];
						cur_pix_ptr++;
						*cur_pix_ptr = ori_pix_ptr[1];
						cur_pix_ptr++;
						*cur_pix_ptr = ori_pix_ptr[2];
					}
				}
				input = ncnn::Mat::from_pixels_resize(img, ncnn::Mat::PIXEL_BGR2GRAY, crop_width, crop_height, crop_width, crop_height);
				input.substract_mean_normalize(mean_vals, norm_vals);
				break;
			default:
				return false;
				break;
			}

			ncnn::Mat out;
			ncnn::Extractor ex = net.create_extractor();

			ex.set_light_mode(true);
			ex.input("data", input);
			ex.extract(this->output_blob_name.c_str(), out);
			memcpy(feat, out.data, out.w * sizeof(float));
			if (normalize)
				ZQ_MathBase::Normalize(feat_dim, feat);
			return true;
		}

	private:

		ncnn::Mat input;
		std::vector<unsigned char> bgr_buffer;
		ncnn::Net net;
		int feat_dim;
		std::string param_file;
		std::string bin_file;
		std::string output_blob_name;
	};
}
#endif
