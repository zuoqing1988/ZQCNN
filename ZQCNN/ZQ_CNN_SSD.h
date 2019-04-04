#ifndef _ZQ_CNN_SSD_H_
#define _ZQ_CNN_SSD_H_
#pragma once

#include "ZQ_CNN_Net.h"
#include <vector>
#include <iostream>
namespace ZQ
{
	class ZQ_CNN_SSD
	{
	public:
		class BBox 
		{
		public:
			float col1, row1, col2, row2, score;
			int label;

			BBox()
			{
				label = -1;
				score = 0;
				col1 = row1 = col2 = row2 = 0;
			}
		};

	private:
		ZQ_CNN_Net net;
		bool mxnet_ssd;
		std::string proto_file;
		std::string model_file;
		std::string out_blob_name;
	public:

		bool Init(const std::string& proto_file, const std::string& model_file, const std::string& out_blob_name, bool mxnet_ssd = false)
		{
			if (!net.LoadFrom(proto_file, model_file))
			{
				printf("failed to load net (%s, %s)\n",proto_file.c_str(), model_file.c_str());
				return false;
			}
			printf("MulAdd = %.3f M\n", net.GetNumOfMulAdd() / (1024.0*1024.0));
			this->proto_file = proto_file;
			this->model_file = model_file;
			this->out_blob_name = out_blob_name;
			const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
			if (ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", out_blob_name.c_str());
				return false;
			}
			this->mxnet_ssd = mxnet_ssd;
			return true;
		}
		
		bool Detect(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float confidence_thresh,
			bool show_debug_info = false)
		{
			if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
				return false;
			int C, H, W;
			if (show_debug_info)
				net.TurnOnShowDebugInfo();
			net.GetInputDim(C, H, W);
			if (C != 3)
				return false;
			if (H == 0 || W == 0)
			{
				H = height;
				W = width;
			}
			ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
			if (mxnet_ssd)
			{
				if (!input0.ConvertFromBGR(bgr_img, width, height, widthStep))
					return false;
			}
			else
			{
				if (!input0.ConvertFromBGR(bgr_img, width, height, widthStep))
					return false;
			}
			if (width != W || height != H)
			{
				if (!input0.ResizeBilinear(input1, W, H, 0, 0))
				{
					return false;
				}
				if (!net.Forward(input1))
				{
					printf("failed to run net (%s, %s)!\n", proto_file.c_str(), model_file.c_str());
					return false;
				}
			}
			else
			{
				if (!net.Forward(input0))
				{
					printf("failed to run net (%s, %s)!\n", proto_file.c_str(), model_file.c_str());
					return false;
				}
			}

			const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
			// get output, shape is N x 7
			if (ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n",out_blob_name.c_str());
				return false;
			}

			const float* result_data = ptr->GetFirstPixelPtr();
			int sliceStep = ptr->GetSliceStep();
			int N = ptr->GetN();
			output.clear();
			float scale_X = width;
			float scale_Y = height;
			for (int k = 0; k < N; k++)
			{
				if (result_data[0] != -1 && result_data[2] >= confidence_thresh)
				{
					// [image_id, label, score, xmin, ymin, xmax, ymax]
					BBox bbox;
					if (mxnet_ssd)
					{
						bbox.col1 = result_data[3] * scale_X;
						bbox.row1 = result_data[4] * scale_Y;
						bbox.col2 = result_data[5] * scale_X;
						bbox.row2 = result_data[6] * scale_Y;
					}
					else
					{
						bbox.col1 = result_data[3] * scale_X;
						bbox.row1 = result_data[4] * scale_Y;
						bbox.col2 = result_data[5] * scale_X;
						bbox.row2 = result_data[6] * scale_Y;
					}
					bbox.score = result_data[2];
					bbox.label = static_cast<int>(result_data[1]);
					output.push_back(bbox);
				}
				result_data += sliceStep;
			}
			return true;
		}
	};
}

#endif
