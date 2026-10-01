#ifndef _ZQ_FACE_RECOGNIZER_SPHERE_FACE_OPENCV_H_
#define _ZQ_FACE_RECOGNIZER_SPHERE_FACE_OPENCV_H_
#pragma once
#include "ZQ_FaceRecognizerSphereFace.h"
#include <opencv2/dnn.hpp>
#include "ZQ_MathBase.h"
#include <string.h>
namespace ZQ
{
	class ZQ_FaceRecognizerSphereFaceOpenCV : public ZQ_FaceRecognizerSphereFace
	{
	public:
		ZQ_FaceRecognizerSphereFaceOpenCV()
		{
			feat_dim = 0;
		}
		~ZQ_FaceRecognizerSphereFaceOpenCV()
		{
		}

		virtual bool Init(const std::string model_name = "",
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_layer_name = "")
		{
			int H = GetCropHeight();
			int W = GetCropWidth();
			
			this->prototxt_file = prototxt_file;
			this->caffemodel_file = caffemodel_file;
			this->output_layer_name = out_layer_name;

			/* OpenCV 的 dnn 模块用 CV_Error 报错, 默认错误处理是 terminate(),
			   不是抛 C++ 异常 —— 本文件若不接住, 一个"模型文件不存在"就会让整个
			   进程以 0xC0000409 崩掉, 调用方的 if (!Init(...)) 根本来不及执行。
			   实测: 删掉 model/mobilefacenet-res2-6-10-2-dim512-opencv.prototxt 后
			   SampleFaceRecognizerArcFaceOpenCV 直接 terminate。
			   下面统一 try/catch(cv::Exception) 转成 return false, 恢复 bool 契约。 */

			try
			{
				/* Load the network. */
				net = cv::dnn::readNetFromCaffe(prototxt_file, caffemodel_file);

				if (net.empty())
				{
					printf("failed to load prototxt_file %s, model_file %s\n", prototxt_file.c_str(), caffemodel_file.c_str());
					return false;
				}
				cv::Mat img(H, W, CV_MAKE_TYPE(8, 3));
				cv::Mat inputBlob = cv::dnn::blobFromImage(img, std_val, cv::Size(W, H), cv::Scalar(mean_val, mean_val, mean_val), true, false);
				net.setInput(inputBlob, "data");
				cv::Mat prob = net.forward(output_layer_name);
				feat_dim = prob.cols;
				bgr_buffer = img;
			}
			catch (const cv::Exception& e)
			{
				printf("Init failed for %s / %s: %s\n", prototxt_file.c_str(), caffemodel_file.c_str(), e.what());
				feat_dim = 0;
				return false;
			}
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
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*bgr_buffer.step[0] + w*3;
						*cur_pix_ptr = ori_pix_ptr[0];
						cur_pix_ptr++;
						*cur_pix_ptr = ori_pix_ptr[0];
						cur_pix_ptr++;
						*cur_pix_ptr = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGR:
				for (int h = 0; h < crop_height; h++)
				{
					const unsigned char* ori_pix_ptr = img + h*widthStep;
					unsigned char* cur_pix_ptr = bgr_buffer.data + h*bgr_buffer.step[0];
					memcpy(cur_pix_ptr, ori_pix_ptr, crop_width * 3);
				}
				break;
			case ZQ_PIXEL_FMT_RGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*3;
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*crop_width*3 + w*3;
						*cur_pix_ptr = ori_pix_ptr[2];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[1];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_BGRX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*4;
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*crop_width*3 + w*3;
						*cur_pix_ptr = ori_pix_ptr[0];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[1];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[2];
					}
				}
				break;
			case ZQ_PIXEL_FMT_RGBX:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*4;
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*crop_width*3 + w*3;
						*cur_pix_ptr = ori_pix_ptr[2];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[1];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[0];
					}
				}
				break;
			case ZQ_PIXEL_FMT_XBGR:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*4;
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*crop_width*3 + w*3;
						*cur_pix_ptr = ori_pix_ptr[1];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[2];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[3];
					}
				}
				break;
			case ZQ_PIXEL_FMT_XRGB:
				for (int h = 0; h < crop_height; h++)
				{
					for (int w = 0; w < crop_width; w++)
					{
						const unsigned char* ori_pix_ptr = img + h*widthStep + w*4;
						unsigned char* cur_pix_ptr = bgr_buffer.data + h*crop_width*3 + w*3;
						*cur_pix_ptr = ori_pix_ptr[3];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[2];
						cur_pix_ptr ++;
						*cur_pix_ptr = ori_pix_ptr[1];
					}
				}
				break;
			default:
				return false;
				break;
			}

			try
			{
				cv::Mat inputBlob = cv::dnn::blobFromImage(bgr_buffer, std_val, cv::Size(crop_width, crop_height), 
					cv::Scalar(mean_val, mean_val, mean_val), true, false);

				net.setInput(inputBlob, "data");
				cv::Mat prob = net.forward(output_layer_name);
				if (prob.empty() || prob.cols != feat_dim)
				{
					printf("unexpected forward output: cols=%d, expect %d\n", prob.cols, feat_dim);
					return false;
				}
				float* prob_data = prob.ptr<float>();
				memcpy(feat, prob_data, sizeof(float)*feat_dim);
			}
			catch (const cv::Exception& e)
			{
				printf("ExtractFeature failed: %s\n", e.what());
				return false;
			}
			if (normalize)
				ZQ_MathBase::Normalize(feat_dim, feat);
			return true;
		}

	private:

		const float mean_val = 127.5f;
		const float std_val = 0.0078125f;

		cv::dnn::Net net;
		int feat_dim;
		std::string prototxt_file;
		std::string caffemodel_file;
		std::string output_layer_name;
		cv::Mat bgr_buffer;
	};
}
#endif
