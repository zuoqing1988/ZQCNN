#ifndef _ZQ_FACE_RECOGNIZER_SPHERE_FACE_ZQCNN_H_
#define _ZQ_FACE_RECOGNIZER_SPHERE_FACE_ZQCNN_H_
#pragma once
#include "ZQ_FaceRecognizerSphereFace.h"
#include "ZQ_CNN_Net.h"
#include "ZQ_MathBase.h"
#include <string.h>
namespace ZQ
{
	class ZQ_FaceRecognizerSphereFaceZQCNN : public ZQ_FaceRecognizerSphereFace
	{
	public:
		ZQ_FaceRecognizerSphereFaceZQCNN()
		{
			feat_dim = 0;
			input.ChangeSize(1, GetCropHeight(), GetCropWidth(), 3, 1, 1);
			bgr_buffer.resize(GetCropHeight()*GetCropWidth() * 3);
		}
		~ZQ_FaceRecognizerSphereFaceZQCNN()
		{

		}

		virtual bool Init(const std::string model_name,
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_blob_name = "")
		{
			bool catch_predefined = false;

			if (_strcmpi(model_name.c_str(), "04bn256") == 0)
			{
				zqparam_file = "model\\sphereface04bn256.zqparams";
				nchwbin_file = "model\\sphereface04bn256_iter_26000.nchwbin";
				feat_dim = 256;
				output_blob_name = "fc5";
				catch_predefined = true;
			}
			else if (_strcmpi(model_name.c_str(), "06bn512") == 0)
			{
				zqparam_file = "model\\sphereface06bn512.zqparams";
				nchwbin_file = "model\\sphereface06bn512_iter_86000.nchwbin";
				feat_dim = 512;
				output_blob_name = "fc5";
				catch_predefined = true;
			}
			else if (_strcmpi(model_name.c_str(), "mobile-10bn512") == 0)
			{
				zqparam_file = "model\\mobilenet_sphereface10bn512.zqparams";
				nchwbin_file = "model\\mobilenet_sphereface10bn512_iter_50000.nchwbin";
				feat_dim = 512;
				output_blob_name = "fc5";
				catch_predefined = true;
			}
			
			if (catch_predefined)
			{
				if (!net.LoadFrom(zqparam_file, nchwbin_file))
				{
					feat_dim = 0;
					return false;
				}
				return true;
			}
			else
			{
				zqparam_file = prototxt_file;
				nchwbin_file = caffemodel_file;
				output_blob_name = out_blob_name;

				if (!net.LoadFrom(zqparam_file, nchwbin_file))
				{
					feat_dim = 0;
					return false;
				}
				
				int C, H, W;
				net.GetInputDim(C, H, W);
				ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
				input.ChangeSize(1, H, W, C, 0, 0);
				
				if (!net.Forward(input))
				{
					printf("failed to forward\n");
					return false;
				}
				const ZQ_CNN_Tensor4D* out = net.GetBlobByName(output_blob_name);
				if (out == NULL)
					return false;
				feat_dim = out->GetC();
				return true;
			}
		}

		virtual int GetFeatDim() const
		{
			return feat_dim;
		}

		virtual bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat pixFmt, float* feat, bool normalize)
		{
			int crop_width = GetCropWidth();
			int crop_height = GetCropHeight();

			switch (pixFmt)
			{
			case ZQ_PIXEL_FMT_GRAY:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[0];
						cur_pix_ptr[1] = ori_pix_ptr[0];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_BGR:
				input.ConvertFromBGR(img, crop_width, crop_height, widthStep, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 3;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[2];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_BGRX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[0];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[2];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_RGBX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[2];
						cur_pix_ptr[1] = ori_pix_ptr[1];
						cur_pix_ptr[2] = ori_pix_ptr[0];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_XBGR:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[1];
						cur_pix_ptr[1] = ori_pix_ptr[2];
						cur_pix_ptr[2] = ori_pix_ptr[3];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w * 4;
						unsigned char* cur_pix_ptr = &bgr_buffer[0] + (h*crop_width + w) * 3;
						cur_pix_ptr[0] = ori_pix_ptr[3];
						cur_pix_ptr[1] = ori_pix_ptr[2];
						cur_pix_ptr[2] = ori_pix_ptr[1];
					}
				}
				input.ConvertFromBGR(&bgr_buffer[0], crop_width, crop_height, crop_width * 3, mean_val, std_val);
				break;
			default:
				return false;
				break;
			}
			if (!net.Forward(input))
				return false;
			const ZQ_CNN_Tensor4D* blob = net.GetBlobByName(output_blob_name);
			if (blob == 0)
				return false;
			memcpy(feat, blob->GetFirstPixelPtr(), sizeof(float)*feat_dim);
			if(normalize)
				ZQ_MathBase::Normalize(feat_dim, feat);
			return true;
		}

	private:

		const float mean_val = 127.5f;
		const float std_val = 0.0078125f;

		ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
		std::vector<unsigned char> bgr_buffer;
		ZQ_CNN_Net net;
		int feat_dim;
		std::string zqparam_file;
		std::string nchwbin_file;
		std::string output_blob_name;
	};
}
#endif
