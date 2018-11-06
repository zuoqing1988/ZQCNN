#ifndef _ZQ_CNN_TEXT_BOXES_H_
#define _ZQ_CNN_TEXT_BOXES_H_
#pragma once

#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_BBox.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <vector>
#include <iostream>
namespace ZQ
{
	class ZQ_CNN_TextBoxes
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
		std::string proto_file;
		std::string model_file;
		std::string out_blob_name;
	public:

		bool Init(const std::string& proto_file, const std::string& model_file, const std::string& out_blob_name)
		{
			if (!net.LoadFrom(proto_file, model_file))
			{
				printf("failed to load net (%s, %s)\n", proto_file.c_str(), model_file.c_str());
				return false;
			}
			this->proto_file = proto_file;
			this->model_file = model_file;
			this->out_blob_name = out_blob_name;
			const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
			if (ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", out_blob_name.c_str());
				return false;
			}
			return true;
		}

		bool Detect(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float confidence_thresh, 
			const std::vector<int>& target_W, const std::vector<int>& target_H,	bool show_debug_info = false)
		{
			if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
				return false;
			if (target_W.size() < 1 || target_H.size() < 1 || target_W.size() != target_H.size())
				return false;

			int C, H, W;
			if (show_debug_info)
				net.TurnOnShowDebugInfo();
			net.GetInputDim(C, H, W);
			if (C != 3)
				return false;
			ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
			std::vector<float> buffer(width*height * 3);
			int HW = height*width;
			int HW2 = HW * 2;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					buffer[offset] = (float)bgr_img[h*widthStep + w * 3 + 0] - 102.9801f;
					buffer[offset+HW] = (float)bgr_img[h*widthStep + w * 3 + 1] - 117.0f;
					buffer[offset+HW2] = (float)bgr_img[h*widthStep + w * 3 + 2] - 122.7717f;
				}
			}
			if (!input0.ConvertFromCompactNCHW(&buffer[0], 1, 3, height, width,1,1))
				return false;
			
			std::vector<ZQ_CNN_NormalizedBBox> bboxes;
			std::vector<float> scores;

			for (int res = 0; res < target_H.size(); res++)
			{
				if (width != target_W[res] || height != target_H[res])
				{
					if (!input0.ResizeBilinear(input1, target_W[res], target_H[res], 0, 0))
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
					printf("maybe the output blob name (%s) is incorrect\n", out_blob_name.c_str());
					return false;
				}

				const float* result_data = ptr->GetFirstPixelPtr();
				int sliceStep = ptr->GetSliceStep();
				int N = ptr->GetN();
				output.clear();
				for (int k = 0; k < N; k++)
				{
					if (result_data[0] != -1 && result_data[2] >= confidence_thresh)
					{
						ZQ_CNN_NormalizedBBox bbox;
						// [image_id, label, score, xmin, ymin, xmax, ymax]
						bbox.col1 = result_data[3];
						bbox.row1 = result_data[4];
						bbox.col2 = result_data[5];
						bbox.row2 = result_data[6];
						bbox.score = result_data[2];
						bbox.label = static_cast<int>(result_data[1]);
						bboxes.push_back(bbox);
						scores.push_back(bbox.score);
					}
					result_data += sliceStep;
				}
			}

			std::vector<int> indices;
			ZQ_CNN_BBoxUtils::ApplyNMSFast(bboxes, scores, confidence_thresh, 0.3, 1.0f, 400, &indices);
			output.resize(indices.size());
			for (int i = 0; i < indices.size(); i++)
			{
				int id = indices[i];
				output[i].col1 = bboxes[id].col1*width;
				output[i].col2 = bboxes[id].col2*width;
				output[i].row1 = bboxes[id].row1*height;
				output[i].row2 = bboxes[id].row2*height;
				output[i].label = bboxes[id].label;
				output[i].score = bboxes[id].score;
			}
			return true;
		}
	};
}

#endif
