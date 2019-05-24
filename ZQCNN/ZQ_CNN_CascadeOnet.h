#ifndef _ZQ_CNN_CASCADE_ONET_H_
#define _ZQ_CNN_CASCADE_ONET_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <omp.h>
namespace ZQ
{
	class ZQ_CNN_CascadeOnet
	{
	public:
		using string = std::string;
		ZQ_CNN_CascadeOnet()
		{
			show_debug_info = false;
		}
		~ZQ_CNN_CascadeOnet()
		{

		}
	private:
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input, onet1_image, onet2_image, onet3_image;
		ZQ_CNN_Net onet1,onet2,onet3;
		int onet1_size, onet2_size, onet3_size;
		bool show_debug_info;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }

		bool Init(const string& onet1_param, const string& onet1_model,
			const string& onet2_param, const string& onet2_model,
			const string& onet3_param, const string& onet3_model)
		{
			bool ret;
			ret = onet1.LoadFrom(onet1_param, onet1_model, true, 1e-9, true)
				&& onet2.LoadFrom(onet2_param, onet2_model, true, 1e-9, true)
				&& onet3.LoadFrom(onet3_param, onet3_model, true, 1e-9, true);
				
			if (!ret)
			{
				return false;
			}
			if (show_debug_info)
			{
				printf("onet1 = %.2f M\n", onet1.GetNumOfMulAdd() / (1024.0*1024.0));
				printf("onet2 = %.2f M\n", onet2.GetNumOfMulAdd() / (1024.0*1024.0));
				printf("onet3 = %.2f M\n", onet3.GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			onet1.GetInputDim(C, H, W);
			onet1_size = H;
			onet2.GetInputDim(C, H, W);
			onet2_size = H;
			onet3.GetInputDim(C, H, W);
			onet3_size = H;
			return ret;
		}

		bool InitFromBuffer(
			const char* onet1_param, __int64 onet1_param_len, const char* onet1_model, __int64 onet1_model_len,
			const char* onet2_param, __int64 onet2_param_len, const char* onet2_model, __int64 onet2_model_len,
			const char* onet3_param, __int64 onet3_param_len, const char* onet3_model, __int64 onet3_model_len
		)
		{
			bool ret;
			ret = onet1.LoadFromBuffer(onet1_param, onet1_param_len, onet1_model, onet1_model_len, true, 1e-9, true)
				&& onet2.LoadFromBuffer(onet2_param, onet2_param_len, onet2_model, onet2_model_len, true, 1e-9, true)
				&& onet3.LoadFromBuffer(onet3_param, onet3_param_len, onet3_model, onet3_model_len, true, 1e-9, true);

			if (!ret)
			{
				return false;
			}
			if (show_debug_info)
			{
				printf("onet1 = %.2f M\n", onet1.GetNumOfMulAdd() / (1024.0*1024.0));
				printf("onet2 = %.2f M\n", onet2.GetNumOfMulAdd() / (1024.0*1024.0));
				printf("onet3 = %.2f M\n", onet3.GetNumOfMulAdd() / (1024.0*1024.0));
			}
			int C, H, W;
			onet1.GetInputDim(C, H, W);
			onet1_size = H;
			onet2.GetInputDim(C, H, W);
			onet2_size = H;
			onet3.GetInputDim(C, H, W);
			onet3_size = H;
			return ret;
		}

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, 
			int xmin, int ymin, int xmax, int ymax,
			std::vector<ZQ_CNN_BBox>& results, int nIters = 3)
		{
			results.clear();
			if (!input.ConvertFromBGR(bgr_img, _width, _height, _widthStep))
				return false;

			std::vector<int> onet_sizes;
			onet_sizes.push_back(onet1_size);
			onet_sizes.push_back(onet2_size);
			onet_sizes.push_back(onet3_size);
			for (int i = 3; i < nIters; i++)
				onet_sizes.push_back(onet3_size);
			std::vector<ZQ_CNN_Net*> nets;
			nets.push_back(&onet1);
			nets.push_back(&onet2);
			nets.push_back(&onet3);
			for (int i = 3; i < nIters; i++)
				nets.push_back(&onet3);
			std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit*> onet_images;
			onet_images.push_back(&onet1_image);
			onet_images.push_back(&onet2_image);
			onet_images.push_back(&onet3_image);
			for (int i = 3; i < nIters; i++)
				onet_images.push_back(&onet3_image);
			std::vector<ZQ_CNN_BBox> box(1);
			box[0].col1 = xmin;
			box[0].col2 = xmax;
			box[0].row1 = ymin;
			box[0].row2 = ymax;
			box[0].exist = true;
			

			for (int i = 0; i < nIters; i++)
			{
				ZQ_CNN_BBoxUtils::_square_bbox(box, _width, _height);
				int rect_x = box[0].col1, rect_y = box[0].row1, rect_w = box[0].col2 - box[0].col1, rect_h = box[0].row2 - box[0].row1;
				if (!input.ResizeBilinearRect(*(onet_images[i]), onet_sizes[i], onet_sizes[i], 0, 0,
					rect_x, rect_y, rect_w, rect_h))
					return false;
				nets[i]->Forward(*(onet_images[i]));
				const ZQ_CNN_Tensor4D* prob = nets[i]->GetBlobByName("prob1");
				const ZQ_CNN_Tensor4D* location = nets[i]->GetBlobByName("conv6-2");
				if (prob == 0)
				{
					std::cout << "failed to get blob prob1\n";
					return false;
				}
				if (location == 0)
				{
					std::cout << "failed to get blob conv6-2\n";
					return false;
				}
				const float* prob_ptr = prob->GetFirstPixelPtr();
				const float* location_ptr = location->GetFirstPixelPtr();
				
				box[0].score = prob_ptr[1];
				box[0].col1 = rect_x;
				box[0].col2 = rect_x + rect_w;
				box[0].row1 = rect_y;
				box[0].row2 = rect_y + rect_h;
				box[0].exist = true;
				for (int j = 0; j < 4; j++)
					box[0].regreCoord[j] = location_ptr[j];
				ZQ_CNN_BBoxUtils::_refine_and_square_bbox(box, _width, _height, false);
				results.push_back(box[0]);
			}
			return true;
		}
	};
}

#endif
