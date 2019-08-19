#ifndef _ZQ_CNN_LANDMARK_240_H_
#define _ZQ_CNN_LANDMARK_240_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_Tensor4D.h"
namespace ZQ
{
	class ZQ_CNN_Landmark240
	{
	public:
		using string = std::string;
		ZQ_CNN_Landmark240() { show_debug_info = false; }
		~ZQ_CNN_Landmark240() {}

	private:
		bool show_debug_info;
		ZQ_CNN_Net left_brow_eye_net;
		ZQ_CNN_Net right_brow_eye_net;
		ZQ_CNN_Net mouth_net;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit left_brow_eye_image;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit right_brow_eye_image;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit mouth_image;
		int left_brow_eye_net_size;
		int right_brow_eye_net_size;
		int mouth_net_size;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }

		bool Init(const string& left_brow_eye_param, const string& left_brow_eye_model,
			const string& right_brow_eye_param, const string& right_brow_eye_model,
			const string& mouth_param, const string& mouth_model)
		{
			bool ret = left_brow_eye_net.LoadFrom(left_brow_eye_param, left_brow_eye_model, true, 1e-9, true)
				&& right_brow_eye_net.LoadFrom(right_brow_eye_param, right_brow_eye_model, true, 1e-9, true)
				&& mouth_net.LoadFrom(mouth_param, mouth_model, true, 1e-9, true);
			if (ret)
			{
				int C, H, W;
				left_brow_eye_net.GetInputDim(C, H, W);
				left_brow_eye_net_size = H;
				right_brow_eye_net.GetInputDim(C, H, W);
				right_brow_eye_net_size = H;
				mouth_net.GetInputDim(C, H, W);
				mouth_net_size = H;

				if (show_debug_info)
				{
					printf("left_brow_eye num_MulAdd = %.1f M\n", left_brow_eye_net.GetNumOfMulAdd() / (1024.0*1024.0));
					printf("right_brow_eye num_MulAdd = %.1f M\n", right_brow_eye_net.GetNumOfMulAdd() / (1024.0*1024.0));
					printf("mouth num_MulAdd = %.1f M\n", mouth_net.GetNumOfMulAdd() / (1024.0*1024.0));
				}
			}
			return ret;
		}

		bool InitFromBuffer(
			const char* left_brow_eye_param, __int64 left_brow_eye_param_len, const char* left_brow_eye_model, __int64 left_brow_eye_model_len,
			const char* right_brow_eye_param, __int64 right_brow_eye_param_len, const char* right_brow_eye_model, __int64 right_brow_eye_model_len,
			const char* mouth_param, __int64 mouth_param_len, const char* mouth_model, __int64 mouth_model_len)
		{
			bool ret = left_brow_eye_net.LoadFromBuffer(left_brow_eye_param, left_brow_eye_param_len, left_brow_eye_model, left_brow_eye_model_len, true, 1e-9, true)
				&& right_brow_eye_net.LoadFromBuffer(right_brow_eye_param, right_brow_eye_param_len, right_brow_eye_model, right_brow_eye_model_len, true, 1e-9, true)
				&& mouth_net.LoadFromBuffer(mouth_param, mouth_param_len, mouth_model, mouth_model_len, true, 1e-9, true);
			
			if (ret)
			{
				int C, H, W;
				left_brow_eye_net.GetInputDim(C, H, W);
				left_brow_eye_net_size = H;
				right_brow_eye_net.GetInputDim(C, H, W);
				right_brow_eye_net_size = H;
				mouth_net.GetInputDim(C, H, W);
				mouth_net_size = H;

				if (show_debug_info)
				{
					printf("left_brow_eye num_MulAdd = %.1f M\n", left_brow_eye_net.GetNumOfMulAdd() / (1024.0*1024.0));
					printf("right_brow_eye num_MulAdd = %.1f M\n", right_brow_eye_net.GetNumOfMulAdd() / (1024.0*1024.0));
					printf("mouth num_MulAdd = %.1f M\n", mouth_net.GetNumOfMulAdd() / (1024.0*1024.0));
				}
			}
			return ret;
		}

		bool Find(const ZQ_CNN_Tensor4D_NHW_C_Align128bit& input, const ZQ_CNN_BBox106& in_box, ZQ_CNN_BBox240& out_box)
		{
			float left_brow_eye_xmin = FLT_MAX;
			float left_brow_eye_ymin = FLT_MAX;
			float left_brow_eye_xmax = -FLT_MAX;
			float left_brow_eye_ymax = -FLT_MAX;
			float right_brow_eye_xmin = FLT_MAX;
			float right_brow_eye_ymin = FLT_MAX;
			float right_brow_eye_xmax = -FLT_MAX;
			float right_brow_eye_ymax = -FLT_MAX;
			float mouth_xmin = FLT_MAX;
			float mouth_ymin = FLT_MAX;
			float mouth_xmax = -FLT_MAX;
			float mouth_ymax = -FLT_MAX;
			
			//mouth
			for (int i = 84; i <= 103; i++)
			{
				mouth_xmin = __min(mouth_xmin, in_box.ppoint[i * 2 + 0]);
				mouth_xmax = __max(mouth_xmax, in_box.ppoint[i * 2 + 0]);
				mouth_ymin = __min(mouth_ymin, in_box.ppoint[i * 2 + 1]);
				mouth_ymax = __max(mouth_ymax, in_box.ppoint[i * 2 + 1]);
			}

			//left brow & eye
			for (int i = 33; i <= 37; i++) // left brow up
			{
				left_brow_eye_xmin = __min(left_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_xmax = __max(left_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_ymin = __min(left_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				left_brow_eye_ymax = __max(left_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 64; i <= 67; i++) // left brow down
			{
				left_brow_eye_xmin = __min(left_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_xmax = __max(left_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_ymin = __min(left_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				left_brow_eye_ymax = __max(left_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 52; i <= 57; i++) // left eye 
			{
				left_brow_eye_xmin = __min(left_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_xmax = __max(left_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_ymin = __min(left_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				left_brow_eye_ymax = __max(left_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 72; i <= 73; i++) // left eye
			{
				left_brow_eye_xmin = __min(left_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_xmax = __max(left_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				left_brow_eye_ymin = __min(left_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				left_brow_eye_ymax = __max(left_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			

			// right brow & eye
			for (int i = 38; i <= 42; i++) // right brow up
			{
				right_brow_eye_xmin = __min(right_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_xmax = __max(right_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_ymin = __min(right_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				right_brow_eye_ymax = __max(right_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 68; i <= 71; i++) // right brow down
			{
				right_brow_eye_xmin = __min(right_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_xmax = __max(right_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_ymin = __min(right_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				right_brow_eye_ymax = __max(right_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 58; i <= 63; i++) // right eye
			{
				right_brow_eye_xmin = __min(right_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_xmax = __max(right_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_ymin = __min(right_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				right_brow_eye_ymax = __max(right_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}
			for (int i = 75; i <= 76; i++) // right eye
			{
				right_brow_eye_xmin = __min(right_brow_eye_xmin, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_xmax = __max(right_brow_eye_xmax, in_box.ppoint[i * 2 + 0]);
				right_brow_eye_ymin = __min(right_brow_eye_ymin, in_box.ppoint[i * 2 + 1]);
				right_brow_eye_ymax = __max(right_brow_eye_ymax, in_box.ppoint[i * 2 + 1]);
			}

			// expand

			float left_brow_eye_width = left_brow_eye_xmax - left_brow_eye_xmin;
			float left_brow_eye_height = left_brow_eye_ymax - left_brow_eye_ymin;
			float left_brow_eye_cx = 0.5*(left_brow_eye_xmin + left_brow_eye_xmax);
			float left_brow_eye_cy = 0.5*(left_brow_eye_ymin + left_brow_eye_ymax);
			int left_brow_eye_expanded_size = __max(left_brow_eye_width, left_brow_eye_height) * 1.5;
			
			float right_brow_eye_width = right_brow_eye_xmax - right_brow_eye_xmin;
			float right_brow_eye_height = right_brow_eye_ymax - right_brow_eye_ymin;
			float right_brow_eye_cx = 0.5*(right_brow_eye_xmin + right_brow_eye_xmax);
			float right_brow_eye_cy = 0.5*(right_brow_eye_ymin + right_brow_eye_ymax);
			int right_brow_eye_expanded_size = __max(right_brow_eye_width, right_brow_eye_height) * 1.5;

			float mouth_width = mouth_xmax - mouth_xmin;
			float mouth_height = mouth_ymax - mouth_ymin;
			float mouth_cx = 0.5*(mouth_xmin + mouth_xmax);
			float mouth_cy = 0.5*(mouth_ymin + mouth_ymax);
			int mouth_expanded_size = __max(mouth_width, mouth_height) * 1.5;

			// resize 
			int left_brow_eye_off_x = left_brow_eye_cx - 0.5*right_brow_eye_expanded_size;
			int left_brow_eye_off_y = left_brow_eye_cy - 0.5*right_brow_eye_expanded_size;
			input.ResizeBilinearRect(left_brow_eye_image, left_brow_eye_net_size, left_brow_eye_net_size, -1, -1, 
				left_brow_eye_off_x, left_brow_eye_off_y, left_brow_eye_expanded_size, left_brow_eye_expanded_size);

			float right_brow_eye_off_x = right_brow_eye_cx - 0.5*right_brow_eye_expanded_size;
			float right_brow_eye_off_y = right_brow_eye_cy - 0.5*right_brow_eye_expanded_size;
			input.ResizeBilinearRect(right_brow_eye_image, right_brow_eye_net_size, right_brow_eye_net_size, -1, -1,
				right_brow_eye_off_x, right_brow_eye_off_y, right_brow_eye_expanded_size, right_brow_eye_expanded_size);

			float mouth_off_x = mouth_cx - 0.5*mouth_expanded_size;
			float mouth_off_y = mouth_cy - 0.5*mouth_expanded_size;
			input.ResizeBilinearRect(mouth_image, mouth_net_size, mouth_net_size, -1, -1,
				mouth_off_x, mouth_off_y, mouth_expanded_size, mouth_expanded_size);


			//forward
			if (!left_brow_eye_net.Forward(left_brow_eye_image))
				return false;
			if (!right_brow_eye_net.Forward(right_brow_eye_image))
				return false;
			if (!mouth_net.Forward(mouth_image))
				return false;

			//get output
			const ZQ_CNN_Tensor4D* left_brow_eye_output = left_brow_eye_net.GetBlobByName("conv6-3");
			const ZQ_CNN_Tensor4D* right_brow_eye_output = right_brow_eye_net.GetBlobByName("conv6-3");
			const ZQ_CNN_Tensor4D* mouth_output = mouth_net.GetBlobByName("conv6-3");
			const float* left_data = left_brow_eye_output->GetFirstPixelPtr();
			const float* right_data = right_brow_eye_output->GetFirstPixelPtr();
			const float* mouth_data = mouth_output->GetFirstPixelPtr();
			for (int i = 0; i < 35; i++)
			{
				out_box.left_brow_eye_ppoint[i * 2 + 0] = left_brow_eye_off_x + left_data[i * 2 + 0] * left_brow_eye_expanded_size;
				out_box.left_brow_eye_ppoint[i * 2 + 1] = left_brow_eye_off_y + left_data[i * 2 + 1] * left_brow_eye_expanded_size;
			}
			for (int i = 0; i < 35; i++)
			{
				out_box.right_brow_eye_ppoint[i * 2 + 0] = right_brow_eye_off_x + right_data[i * 2 + 0] * right_brow_eye_expanded_size;
				out_box.right_brow_eye_ppoint[i * 2 + 1] = right_brow_eye_off_y + right_data[i * 2 + 1] * right_brow_eye_expanded_size;
			}
			for (int i = 0; i < 64; i++)
			{
				out_box.mouth_ppoint[i * 2 + 0] = mouth_off_x + mouth_data[i * 2 + 0] * mouth_expanded_size;
				out_box.mouth_ppoint[i * 2 + 1] = mouth_off_y + mouth_data[i * 2 + 1] * mouth_expanded_size;
			}
			out_box.box = in_box;
			return true;
		}
	};
}
#endif
