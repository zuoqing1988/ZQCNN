#ifndef _ZQ_CNN_SSD_H_
#define _ZQ_CNN_SSD_H_
#pragma once

#include "ZQ_CNN_Net.h"
#include <vector>
#include <iostream>


namespace ZQ
{
	class ZQ_CNN_NSFW
	{
	private:
		ZQ_CNN_Net net;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit data;
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

		bool Detect(std::vector<float>& output, const unsigned char* bgr_img, int width, int height, int widthStep, bool show_debug_info = false)
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
			ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
			std::vector<float> buffer(width*height * 3);
			int HW = width*height;
			int HW2 = HW * 2;
			for (int h = 0; h < height; h++)
			{
				const unsigned char* ptr = bgr_img + h*widthStep;
				for (int w = 0; w < width; w++)
				{
					int off = h*width + w;
					buffer[off] = ptr[w * 3 + 0] - 102.9801f;
					buffer[off + HW] = ptr[w * 3 + 1] - 115.9465f;
					buffer[off + HW2] = ptr[w * 3 + 2] - 122.7717f;
				}
			}
			input.ConvertFromCompactNCHW(&buffer[0],1,3,height,width,1,1);
			
			if (width != W || height != H)
			{
				if (!input.ResizeBilinear(data, W, H, 1, 1))
				{
					return false;
				}
				if (!net.Forward(data))
				{
					printf("failed to run net (%s, %s)!\n", proto_file.c_str(), model_file.c_str());
					return false;
				}
			}
			else
			{
				data.CopyData(input);
			}

			if (!net.Forward(data))
			{
				printf("failed to run net (%s, %s)!\n", proto_file.c_str(), model_file.c_str());
				return false;
			}

			const ZQ_CNN_Tensor4D* ptr = net.GetBlobByName(out_blob_name);
			if (ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", out_blob_name.c_str());
				return false;
			}

			const float* result_data = ptr->GetFirstPixelPtr();
			int out_C = ptr->GetC();
			output.clear();
			for (int c = 0; c < out_C; c++)
			{
				output.push_back(result_data[c]);
			}
			return true;
		}
	};
}

#endif

