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
		// 审计修复 2026-10-06（附录 IQ.2）：原来无初值。
	// `Init` 里有**两条** `return false` 排在 `this->mxnet_ssd = mxnet_ssd;` **之前**
	// （网加载失败 / blob 查不到），而 `Detect` 在 :73 读它。调用方忽略 Init 的
	// 返回值继续 Detect 时读的是**不确定值**（UB），`std::vector<BBox>` 走哪一支随机。
	// 对照本仓既有正确写法：`ZQ_CNN_VideoFaceDetection_Interface.h:26-34` 把成员全部初始化。
	bool mxnet_ssd = false;
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
			// 审计修复 2026-10-06（附录 IQ.5）：`output.clear()` 原来排在**七条** return false
			// **之后**（:120）—— 任何一次 Detect 失败，调用方仍读 output 就会拿到
			// **上一次成功调用的框**。提到最前面：失败时「没有结果」和「结果是空的」必须一致。
			// 可达性：`ZQ_CNN_MouthDetector.h:289` 丢弃了返回值，但它的
			// `DetectedFace one_face` 是循环内新构造、`result_vec_mouth` 为空，所以**目前不发作**；
			// `SampleSSD.cpp:74-78` 失败即 EXIT_FAILURE，也不发作。
			// 但它正是「复用同一个 output 跑很多轮」那种写法下最自然的一颗雷。
			output.clear();
			if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
				return false;
			int C, H, W;
			// 审计修复 2026-10-06（附录 IQ.6）：原来**只开不开**。`show_debug_info` 是
			// 每次调用的形参，但一旦某次传 true，`ZQ_CNN_Net::show_debug_info` 就永远为真，
			// 之后所有 Detect 都刷屏；类里也没有 TurnOffShowDebugInfo 出口。
			// 改成对称：这一轮不打印就把 net 的开关**显式关掉**。
			if (show_debug_info)
				net.TurnOnShowDebugInfo();
			else
				net.TurnOffShowDebugInfo();

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
			if (sliceStep < 7)
			{
				printf("the output blob (%s) has slice step (%d) less than 7\n", out_blob_name.c_str(), sliceStep);
				return false;
			}
			// 这里原来还有一个 output.clear()：IQ.5 把它提到函数开头之后，
			// 这个已经走不到（前面所有失败路径都在开头就 return 了），留着是死代码。
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
