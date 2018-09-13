#ifndef _ZQ_FACE_RECOGNIZER_SPHERE_FACE_MINICAFFE_H_
#define _ZQ_FACE_RECOGNIZER_SPHERE_FACE_MINICAFFE_H_
#pragma once
#include "ZQ_FaceRecognizerSphereFace.h"
#include <caffe/caffe.hpp>
#include "ZQ_MathBase.h"
#include <string.h>
namespace ZQ
{
	class ZQ_FaceRecognizerSphereFaceMiniCaffe : public ZQ_FaceRecognizerSphereFace
	{
	public:
		ZQ_FaceRecognizerSphereFaceMiniCaffe()
		{
			feat_dim = 0;
			net = 0;
		}
		~ZQ_FaceRecognizerSphereFaceMiniCaffe()
		{
			if (net)
				delete net;
			net = 0;
		}

		virtual bool Init(const std::string model_name = "",
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_blob_name = "")
		{
			bgr_buffer.resize(GetCropHeight()*GetCropWidth() * 3);
			this->prototxt_file = prototxt_file;
			this->caffemodel_file = caffemodel_file;
			this->output_blob_name = out_blob_name;

			/* Load the network. */
			if (net) delete net;
			net = new caffe::Net(prototxt_file);

			if (net == 0)
			{
				printf("failed to load prototxt_file %s\n", prototxt_file.c_str());
				return false;
			}
			net->CopyTrainedLayersFrom(caffemodel_file);

			const std::shared_ptr<caffe::Blob> input_layer = net->blob_by_name("data");
			int num_channels_ = input_layer->channels();
			int width = input_layer->width();
			int height = input_layer->height();
			if (width != GetCropWidth() || height != GetCropHeight() || num_channels_ != 3)
			{
				printf("invalid caffemodel: input should be %d*%d*3\n", GetCropWidth(), GetCropHeight());
				return false;
				delete net;
				net = 0;
			}

			const std::shared_ptr<caffe::Blob> output_layer = net->blob_by_name(out_blob_name);
			feat_dim = output_layer->channels();
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
			int slice_size = crop_height*crop_width;
			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGR:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 3;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 3;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGRX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGBX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[0] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_XBGR:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[3] - mean_val) * std_val;
					}
				}
				break;
			case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 3;
						float* cur_pix_ptr = &bgr_buffer[0] + h*crop_width + w;
						*cur_pix_ptr = (ori_pix_ptr[3] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[2] - mean_val) * std_val;
						cur_pix_ptr += slice_size;
						*cur_pix_ptr = (ori_pix_ptr[1] - mean_val) * std_val;
					}
				}
				break;
			default:
				return false;
				break;
			}

			const std::shared_ptr<caffe::Blob> input_layer = net->blob_by_name("data");
			int width = input_layer->width();
			int height = input_layer->height();
			float* input_data = input_layer->mutable_cpu_data();
			memcpy(input_data, &bgr_buffer[0], sizeof(float)*width*height * 3);

			net->Forward();

			const std::shared_ptr<caffe::Blob> output_layer = net->blob_by_name(output_blob_name);
			const float* begin = output_layer->cpu_data();
			memcpy(feat, begin, sizeof(float)*feat_dim);
			if(normalize)
				ZQ_MathBase::Normalize(feat_dim, feat);
			return true;
		}

	private:

		const float mean_val = 127.5f;
		const float std_val = 0.0078125f;

		caffe::Net*  net;
		int feat_dim;
		std::string prototxt_file;
		std::string caffemodel_file;
		std::string output_blob_name;
		std::vector<float> bgr_buffer;
	};
}
#endif
