#ifndef _ZQ_CNN_PERSON_POSE_H_
#define _ZQ_CNN_PERSON_POSE_H_
#pragma once

#include "ZQ_CNN_Net.h"
#include <vector>
#include <iostream>
// 审计修复 2026-10-06（附录 IM.2）：下面用了 FLT_MAX 而没有 include <cfloat>。
// 随仓的 sample 恰好在别处间接 include 了 <float.h>/<cfloat>，所以一直编得过；
// 单独编这个头就报 'FLT_MAX' was not declared in this scope。
// （与 IM.1 是同一族：用到的宏/类型必须自己 include 它的定义。）
#include <cfloat>

namespace ZQ
{
	class ZQ_CNN_PersonPose
	{
	public:
		class BBox
		{
		public:
			float col1, row1, col2, row2, score;
			float points[54];
			int num_points;
			BBox()
			{
				score = 0;
				col1 = row1 = col2 = row2 = 0;
				num_points = 0;
				// 审计修复 2026-10-06（附录 IN.3）：原来写的是 `sizeof(float) * 51` —— 
				// `points` 是 `float[54]`，差 3 个 float = **第 17 个关键点**的 points[51..53]。
				// 拷贝来源 `ZQ_CNN_PersonPose2.h:30` 是 `sizeof(float) * 42` 对 `points[42]` —— 
				// **正确**。这就是两份拷贝的核心差异之一。
				// 后果：第 17 个关键点没超过阈值时，那三个格子保持构造时的**栈垃圾**，
				// 而 `num_points` 仍然是 18 —— 调用方被告知有 18 个点。
				// 随仓 sample 的 `Draw14` 最大读 13*3+2 = 41，落在原来 memset 覆盖的范围内，
				// 所以是**真缺陷 + 当前零覆盖**，不是活 bug。
				// 改用 sizeof 以后就和数组长度绑死，再加字段也不会再错。
				memset(points, 0, sizeof(points));
			}
		};

	private:
		ZQ_CNN_Net ssd_net;
		std::string ssd_out_blob_name;
		int person_class_id;

		ZQ_CNN_Net pose_net;
		std::string pose_out_blob_name;

		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
		// 审计修复 2026-10-06（附录 IN.4）：这一组成员原来**没有初值**。
		// `GetInputDim` 只在 `Init` 的**末尾**被调用，而 Init 里有 6 处 `return false`
		// （两个 blob 名为空、某个 net 加载失败、blob 查不到）都在它之前。
		// 调用方忽略 Init 的返回值（`detector.Init(...)` 不套 if，是最常见的写法）时，
		// Detect 上来就读它们：`pose_W` 若为 0 就是**除零 -> inf -> (int)inf 是 UB**。
		// 对照：`ZQ_CNN_Net.h` 的 input_C/H/W 上一轮已经补了 `= 0`（同一族、同一理由），
		// 这两个头没跟上。全部夹到 0。
		int ssd_C = 0, ssd_H = 0, ssd_W = 0;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit pose_input;
		int pose_C, pose_H, pose_W = 0;   // 见上面 ssd_C 那条注释
		int pose_npts = 0;   // 同上
	public:

		bool Init(const std::string& ssd_proto_file, const std::string& ssd_model_file, const std::string& ssd_out_blob_name, int person_class_id,
			const std::string& pose_proto_file, const std::string& pose_model_file, const std::string& pose_out_blob_name)
		{
			if (!ssd_net.LoadFrom(ssd_proto_file, ssd_model_file,true,1e-9,true))
			{
				printf("failed to load net (%s, %s)\n", ssd_proto_file.c_str(), ssd_model_file.c_str());
				return false;
			}
			if (!pose_net.LoadFrom(pose_proto_file, pose_model_file,true,1e-9,true))
			{
				printf("failed to load net (%s, %s)\n", pose_proto_file.c_str(), pose_model_file.c_str());
				return false;
			}

			printf("MulAdd = %.3f M, %.3f M\n", ssd_net.GetNumOfMulAdd() / (1024.0*1024.0), pose_net.GetNumOfMulAdd() / (1024.0*1024.0));
			this->ssd_out_blob_name = ssd_out_blob_name;
			this->pose_out_blob_name = pose_out_blob_name;
			this->person_class_id = person_class_id;
			const ZQ_CNN_Tensor4D* ssd_ptr = ssd_net.GetBlobByName(ssd_out_blob_name);
			if (ssd_ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", ssd_out_blob_name.c_str());
				return false;
			}
			const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
			if (pose_ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", pose_out_blob_name.c_str());
				return false;
			}
			int N, H, W;
			pose_ptr->GetShape(N,pose_npts,H,W);
			printf("npts = %d\n", pose_npts);
			ssd_net.GetInputDim(ssd_C, ssd_H, ssd_W);
			pose_net.GetInputDim(pose_C, pose_H, pose_W);
			if (ssd_C != 3 || pose_C != 3)
				return false;
			return true;
		}

		bool Detect(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float confidence_thresh,
			bool show_debug_info = false)
		{
			if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
				return false;
			
			if (show_debug_info)
			{
				ssd_net.TurnOnShowDebugInfo();
				pose_net.TurnOnShowDebugInfo();
			}
			
			if (ssd_H == 0 || ssd_W == 0)
			{
				ssd_H = height;
				ssd_W = width;
			}
			
			if (!input0.ConvertFromBGR(bgr_img, width, height, widthStep))
				return false;
			
			if (width != ssd_W || height != ssd_H)
			{
				if (!input0.ResizeBilinear(input1, ssd_W, ssd_H, 1, 1, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER))
				{
					return false;
				}
				if (!ssd_net.Forward(input1))
				{
					printf("failed to run ssd!\n");
					return false;
				}
			}
			else
			{
				if (!ssd_net.Forward(input0))
				{
					printf("failed to run ssd!\n");
					return false;
				}
			}

			const ZQ_CNN_Tensor4D* ssd_ptr = ssd_net.GetBlobByName(ssd_out_blob_name);
			// get output, shape is N x 7
			if (ssd_ptr == 0)
			{
				printf("maybe the output blob name (%s) is incorrect\n", ssd_out_blob_name.c_str());
				return false;
			}

			const float* result_data = ssd_ptr->GetFirstPixelPtr();
			int sliceStep = ssd_ptr->GetSliceStep();
			int N = ssd_ptr->GetN();
			if (sliceStep < 7)
			{
				printf("the output blob (%s) has slice step (%d) less than 7\n", ssd_out_blob_name.c_str(), sliceStep);
				return false;
			}
			output.clear();
			float scale_X = width;
			float scale_Y = height;
			for (int k = 0; k < N; k++)
			{
				if (result_data[0] != -1 && result_data[2] >= confidence_thresh && (int)result_data[1] == person_class_id)
				{
					// [image_id, label, score, xmin, ymin, xmax, ymax]
					BBox bbox;
					bbox.col1 = result_data[3] * scale_X;
					bbox.row1 = result_data[4] * scale_Y;
					bbox.col2 = result_data[5] * scale_X;
					bbox.row2 = result_data[6] * scale_Y;
					bbox.score = result_data[2];
					if (bbox.col2 - bbox.col1 >= 32 && bbox.row2 - bbox.row1 >= 32)
					{
						output.push_back(bbox);
					}
				}
				result_data += sliceStep;
			}
			//printf("!!\n");
			/***********        HeatMap     *****************/
			for (int nn = 0; nn < output.size(); nn++)
			{
				int col1 = output[nn].col1;
				int col2 = output[nn].col2;
				int row1 = output[nn].row1;
				int row2 = output[nn].row2;
				int rect_w = col2 - col1;
				int rect_h = row2 - row1;
				float max_side = __max((float)rect_w/pose_W, (float)rect_h/pose_H);
				float max_side_W = max_side*pose_W;
				float max_side_H = max_side*pose_H;
				col1 = col1 - max_side_W*0.15;
				col2 = col2 + max_side_W*0.15;
				row1 = row1 - max_side_H*0.15;
				row2 = row2 + max_side_H*0.15;
				rect_w = col2 - col1;
				rect_h = row2 - row1;
				int cx = (col1 + col2) / 2;
				int cy = (row1 + row2) / 2;
				float size = __max((float)rect_w/pose_W, (float)rect_h/pose_H);
				int size_W = size*pose_W;
				int size_H = size*pose_H;
				row2 = __min(row2, row1 + size_H);
				col2 = __min(col2, col1 + size_W);
				int box_col1 = cx - size_W / 2;
				int box_col2 = box_col1 + size_W;
				int box_row1 = cy - size_H / 2;
				int box_row2 = box_row1 + size_H;
				int start_w = __max(0,col1);
				int end_w = __min(width, col2);
				int start_h = __max(0, row1);
				int end_h = __min(height, row2);
				int pad_w_left = __max(0,start_w - box_col1);
                int pad_h_up = __max(0,start_h - box_row1);
				const __int64 buffer_size = (__int64)size_H * size_W * 3;
				if (buffer_size <= 0 || buffer_size > 0x7FFFFFFF)
					return false;
				std::vector<unsigned char> buffer((size_t)buffer_size, 0);
				
				for (int hh = start_h; hh < end_h; hh++)
				{
					int in_h = hh;
					int in_w = start_w;
					int out_h = hh - start_h + pad_h_up;
					int out_w = pad_w_left;
					memcpy(&buffer[(out_h*size_W+out_w)*3], bgr_img + in_h*widthStep + in_w * 3, sizeof(unsigned char) * 3 * (end_w - start_w));
				}
				ZQ_CNN_Tensor4D_NHW_C_Align128bit temp_img;
				// 审计修复 2026-10-06（附录 IN.8）：返回值原来被丢弃。
				// `ConvertFromBGR`（ZQ_CNN_Tensor4D.h:395-397）第一句就是
				// `if (!ChangeSize(...)) return false;` —— 失败时 `temp_img` 停在**上一次**的尺寸，
				// 紧接着的 ResizeBilinear 就按陈旧尺寸跑。
				// 同文件 :107 / :112 / :116 对同样的调用**全都检查了** —— 不一致。
				if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1))
				{
					printf("ConvertFromBGR failed (pose %dx%d)\n", size_W, size_H);
					return false;
				}
				/*cv::Mat img = cv::Mat(size_H, size_W, CV_8UC3);
				for (int hh = 0; hh < size_H; hh++)
				{
					memcpy(img.data + hh*img.step[0], &buffer[hh*size_W * 3], sizeof(unsigned char)*size_W * 3);
				}
				cv::namedWindow("roi");
				cv::imshow("roi", img);
				cv::waitKey(0);*/
				// 审计修复 2026-10-06（附录 IN.8）：同上一条，ResizeBilinear 也要查。
				temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
				if (!temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER))
				{
					printf("ResizeBilinear failed (pose %dx%d)\n", pose_W, pose_H);
					return false;
				}
				if (!pose_net.Forward(pose_input))
				{
					printf("failed to run landmark!\n");
					return false;
				}
				const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
					// 审计修复 2026-10-06（附录 IN.5）：GetBlobByName 找不到返回 **0**
					// （ZQ_CNN_Net.h:295-300），原来下一行直接 `pose_ptr->GetFirstPixelPtr()`。
					// 紧邻 100 行内的 **SSD 侧有**守卫（带 `maybe the output blob name (%s)
					// is incorrect` 的 printf），姿态侧一个字都没有 —— 同文件内的不对称，
					// 而 Init 里只查过一次就当成了不变量。
					// Init 成功即意味着 blob 当时存在，所以当前不可达；
					// 一旦有人改 blob 名、或加一层 UnloadBlob，就是空指针解引用。
				if (pose_ptr == 0)
				{
					printf("maybe the output blob name (%s) is incorrect\n", pose_out_blob_name.c_str());
					return false;
				}
				const float* heatmap_data = pose_ptr->GetFirstPixelPtr();
				int hm_H = pose_ptr->GetH();
				int hm_W = pose_ptr->GetW();
				int hm_C = pose_ptr->GetC();
				int hm_widthStep = pose_ptr->GetWidthStep();
				int hm_pixStep = pose_ptr->GetPixelStep();
				float thresh = 0.3f;
				output[nn].num_points = __min(18, hm_C);
				for (int c = 0; c < output[nn].num_points; c++)
				{
					float sum_weight = 0;
					float sum_x = 0;
					float sum_y = 0;
					float max_weight = -FLT_MAX;
					int max_h = -1;
					int max_w = -1;
					for (int h = 0; h < hm_H; h++)
					{
						for (int w = 0; w < hm_W; w++)
						{
							float tmp_val = heatmap_data[c + h*hm_widthStep + w*hm_pixStep];
							if (tmp_val > thresh)
							{
								sum_weight += tmp_val;
								sum_x += tmp_val*w;
								sum_y += tmp_val*h;
							}
							if (tmp_val > max_weight)
							{
								max_h = h;
								max_w = w;
								max_weight = tmp_val;
							}
						}
					}
					/*if (sum_weight > 0)
					{
						sum_x /= sum_weight;
						sum_y /= sum_weight;
						output[nn].points[c * 3 + 0] = (sum_x + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
						output[nn].points[c * 3 + 1] = (sum_y + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
						output[nn].points[c * 3 + 2] = 1;
					}*/
					if (max_weight > thresh)
					{
						output[nn].points[c * 3 + 0] = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
						output[nn].points[c * 3 + 1] = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
						output[nn].points[c * 3 + 2] = max_weight;
					}

					/*cv::Mat hm_img = cv::Mat(hm_H, hm_W, CV_8UC3);
					for (int h = 0; h < hm_H; h++)
					{
						for (int w = 0; w < hm_W; w++)
						{
							hm_img.data[h*hm_img.step[0] + w * 3 + 0] = 0;
							hm_img.data[h*hm_img.step[0] + w * 3 + 1] = 0;
							hm_img.data[h*hm_img.step[0] + w * 3 + 2] = __min(255, __max(0, heatmap_data[c + h*hm_widthStep + w*hm_pixStep] * 255));
						}
					}
					cv::resize(hm_img, hm_img, cv::Size(), 5, 5, CV_INTER_NN);
					char buf_name[100];
					sprintf(buf_name, "heatmap_%d", c);
					cv::namedWindow(buf_name);
					cv::imshow(buf_name, hm_img);*/
					
				}
				//cv::waitKey(0);
			}
			
			return true;
		}

		/*
		filter_type: 0 none, 1 linear, 2 complex
		*/
		bool DetectVideoSinglePerson(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float confidence_thresh,
			bool show_debug_info = false, int filter_type = 2)
		{
			bool need_ssd = false;
			if (output.size() == 0)
				need_ssd = true;
			if (bgr_img == 0 || width <= 0 || height <= 0 || widthStep < width * 3)
				return false;

			if (show_debug_info)
			{
				ssd_net.TurnOnShowDebugInfo();
				pose_net.TurnOnShowDebugInfo();
			}

			if (ssd_H == 0 || ssd_W == 0)
			{
				ssd_H = height;
				ssd_W = width;
			}
			if (!need_ssd)
			{
				for (int nn = output.size() - 1; nn >= 0; nn--)
				{
					int npts = output[nn].num_points;
				// 审计修复 2026-10-06（附录 IN.6）：`num_points` 是 **public 字段**，调用方可以预填
				// `output`。npts==0 时下面的两个守卫（`valid_num < npts*0.3`、`total_weight < valid_num*0.5`）
				// 与 0 比**恒假** -> 不 erase -> col1/col2 从未被更新，直接算出 `col1=1e9 > col2=-1e9` 的**反向框**。
				// 下一帧 `rect_w = -2e9` -> size_W/size_H 为负 -> `buffer_size = size_H*size_W*3` 为**正**，
				// `buffer_size <= 0` 那个守卫**挡不住** -> ConvertFromBGR(负宽, 负高)。
				// 拷贝 PersonPose2.h 的 npts 由 half_mode 推导（14 或 10），永远 >= 1，
				// 这是「这份有、那份没有」的第二条。
				if (npts <= 0) continue;
					int col1 = 1e9;
					int col2 = -1e9;
					int row1 = 1e9;
					int row2 = -1e9;
					float total_weight = 0;
					float valid_num = 0;
					for (int i = 0; i < npts; i++)
					{
						if (output[nn].points[i * 3 + 2] > 0)
						{
							col1 = __min(col1, output[nn].points[i * 3 + 0]);
							col2 = __max(col2, output[nn].points[i * 3 + 0]);
							row1 = __min(row1, output[nn].points[i * 3 + 1]);
							row2 = __max(row2, output[nn].points[i * 3 + 1]);
							total_weight += output[nn].points[i * 3 + 2];
							valid_num += 1;
						}
					}
					if (valid_num < npts*0.3 || total_weight < valid_num*0.5)
					{
						output.erase(output.begin() + nn);
						continue;
					}
					int cx = (col1 + col2) / 2;
					int cy = (row1 + row2) / 2;
					int size_x = col2 - col1;
					int size_y = row2 - row1;
					output[nn].col1 = cx - size_x*0.5;
					output[nn].col2 = cx + size_x*0.5;
					output[nn].row1 = cy - size_y*0.5;
					if (npts == 10)
					{
						if (output[nn].points[0 * 3 + 2] > 0 && output[nn].points[1 * 3 + 2] > 0
							&& output[nn].points[2 * 3 + 2] > 0 && output[nn].points[5 * 3 + 2] > 0)
						{
							float dir_x0 = output[nn].points[0 * 3 + 0] - output[nn].points[1 * 3 + 0];
							float dir_y0 = output[nn].points[0 * 3 + 1] - output[nn].points[1 * 3 + 1];
							float dir_x1 = output[nn].points[2 * 3 + 0] - output[nn].points[5 * 3 + 0];
							float dir_y1 = output[nn].points[2 * 3 + 1] - output[nn].points[5 * 3 + 1];
							float head_len = sqrt(dir_x0*dir_x0 + dir_y0*dir_y0);
							float shoulder_len = sqrt(dir_x1*dir_x1 + dir_y1*dir_y1);
							float standard_len = __max(head_len, shoulder_len*0.6);
							if (output[nn].points[8 * 3 + 2] == 0 && output[nn].points[9 * 3 + 2] == 0)
							{
								
								output[nn].row2 = output[nn].points[1 * 3 + 1] + standard_len * 2;
							}
							else
								output[nn].row2 = cy + size_y*0.5;
							output[nn].row1 = __min(output[nn].row1, output[nn].points[0 * 3 + 1] - standard_len*0.2);
							output[nn].col1 = __min(output[nn].col1, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) - standard_len*1.5);
							output[nn].col2 = __max(output[nn].col2, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) + standard_len*1.5);
						}
						else if(output[nn].points[2 * 3 + 2] > 0 && output[nn].points[5 * 3 + 2] > 0)
						{
							float dir_x1 = output[nn].points[2 * 3 + 0] - output[nn].points[5 * 3 + 0];
							float dir_y1 = output[nn].points[2 * 3 + 1] - output[nn].points[5 * 3 + 1];
							float shoulder_len = sqrt(dir_x1*dir_x1 + dir_y1*dir_y1);
							float standard_len = shoulder_len*0.6;
							if (output[nn].points[8 * 3 + 2] == 0 && output[nn].points[9 * 3 + 2] == 0)
							{

								output[nn].row2 = 0.5*(output[nn].points[2 * 3 + 1] + output[nn].points[5 * 3 + 1]) + standard_len * 2.0;
							}
							else
								output[nn].row2 = cy + size_y*0.5;
							output[nn].row1 = __min(output[nn].row1, 0.5*(output[nn].points[2 * 3 + 1] + output[nn].points[5 * 3 + 1]) - standard_len*1.0);
							output[nn].col1 = __min(output[nn].col1, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) - standard_len*1.5);
							output[nn].col2 = __max(output[nn].col2, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) + standard_len*1.5);
						}
					}
					else if (npts == 14)
					{
						if (output[nn].points[0 * 3 + 2] > 0 && output[nn].points[1 * 3 + 2] > 0
							&& output[nn].points[2 * 3 + 2] > 0 && output[nn].points[5 * 3 + 2] > 0)
						{
							float dir_x0 = output[nn].points[0 * 3 + 0] - output[nn].points[1 * 3 + 0];
							float dir_y0 = output[nn].points[0 * 3 + 1] - output[nn].points[1 * 3 + 1];
							float dir_x1 = output[nn].points[2 * 3 + 0] - output[nn].points[5 * 3 + 0];
							float dir_y1 = output[nn].points[2 * 3 + 1] - output[nn].points[5 * 3 + 1];
							float head_len = sqrt(dir_x0*dir_x0 + dir_y0*dir_y0);
							float shoulder_len = sqrt(dir_x1*dir_x1 + dir_y1*dir_y1);
							float standard_len = __max(head_len, shoulder_len*0.6);
							if (output[nn].points[9 * 3 + 2] == 0 && output[nn].points[12 * 3 + 2] == 0)
							{

								output[nn].row2 = output[nn].points[1 * 3 + 1] + standard_len * 3.5;
							}
							else
								output[nn].row2 = cy + size_y*0.5;
							output[nn].row1 = __min(output[nn].row1, output[nn].points[0 * 3 + 1] - standard_len*0.2);
							output[nn].col1 = __min(output[nn].col1, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) - standard_len*2.0);
							output[nn].col2 = __max(output[nn].col2, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) + standard_len*2.0);
						}
						else if (output[nn].points[2 * 3 + 2] > 0 && output[nn].points[5 * 3 + 2] > 0)
						{
							float dir_x1 = output[nn].points[2 * 3 + 0] - output[nn].points[5 * 3 + 0];
							float dir_y1 = output[nn].points[2 * 3 + 1] - output[nn].points[5 * 3 + 1];
							float shoulder_len = sqrt(dir_x1*dir_x1 + dir_y1*dir_y1);
							float standard_len = shoulder_len*0.6;
							if (output[nn].points[9 * 3 + 2] == 0 && output[nn].points[12 * 3 + 2] == 0)
							{

								output[nn].row2 = 0.5*(output[nn].points[2 * 3 + 1] + output[nn].points[5 * 3 + 1]) + standard_len * 3.5;
							}
							else
								output[nn].row2 = cy + size_y*0.5;
							output[nn].row1 = __min(output[nn].row1, 0.5*(output[nn].points[2 * 3 + 1] + output[nn].points[5 * 3 + 1]) - standard_len*1.2);
							output[nn].col1 = __min(output[nn].col1, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) - standard_len*2.0);
							output[nn].col2 = __max(output[nn].col2, 0.5*(output[nn].points[2 * 3 + 0] + output[nn].points[5 * 3 + 0]) + standard_len*2.0);
						}
					}
					else
					{
						output[nn].row2 = cy + size_y*0.5;
					}
					
				}
				if (output.size() == 0)
					need_ssd = true;
			}

			if (need_ssd)
			{
				//printf("need ssd\n");
				if (!input0.ConvertFromBGR(bgr_img, width, height, widthStep))
					return false;

				if (width != ssd_W || height != ssd_H)
				{
					if (!input0.ResizeBilinear(input1, ssd_W, ssd_H, 1, 1, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER))
					{
						return false;
					}
					if (!ssd_net.Forward(input1))
					{
						printf("failed to run ssd!\n");
						return false;
					}
				}
				else
				{
					if (!ssd_net.Forward(input0))
					{
						printf("failed to run ssd!\n");
						return false;
					}
				}

				const ZQ_CNN_Tensor4D* ssd_ptr = ssd_net.GetBlobByName(ssd_out_blob_name);
				// get output, shape is N x 7
				if (ssd_ptr == 0)
				{
					printf("maybe the output blob name (%s) is incorrect\n", ssd_out_blob_name.c_str());
					return false;
				}

				const float* result_data = ssd_ptr->GetFirstPixelPtr();
				int sliceStep = ssd_ptr->GetSliceStep();
				int N = ssd_ptr->GetN();
				if (sliceStep < 7)
				{
					printf("the output blob (%s) has slice step (%d) less than 7\n", ssd_out_blob_name.c_str(), sliceStep);
					return false;
				}
				output.clear();
				float scale_X = width;
				float scale_Y = height;
				for (int k = 0; k < N; k++)
				{
					if (result_data[0] != -1 && result_data[2] >= confidence_thresh && (int)result_data[1] == person_class_id)
					{
						// [image_id, label, score, xmin, ymin, xmax, ymax]
						BBox bbox;
						bbox.col1 = result_data[3] * scale_X;
						bbox.row1 = result_data[4] * scale_Y;
						bbox.col2 = result_data[5] * scale_X;
						bbox.row2 = result_data[6] * scale_Y;
						bbox.score = result_data[2];
						if (pose_npts == 10)
						{
							int rect_w = bbox.col2 - bbox.col1;
							int rect_h = bbox.row2 - bbox.row1;
							rect_h = __min(rect_h, __max(rect_h*0.5, rect_w));
							bbox.row2 = bbox.row1 + rect_h;
						}
						if (bbox.col2 - bbox.col1 >= 32 && bbox.row2 - bbox.row1 >= 32)
						{
							output.push_back(bbox);
						}
					}
					result_data += sliceStep;
				}
			}
			

			//printf("!!\n");
			/***********        HeatMap     *****************/
			for (int nn = 0; nn < output.size(); nn++)
			{
				
				int col1 = output[nn].col1;
				int col2 = output[nn].col2;
				int row1 = output[nn].row1;
				int row2 = output[nn].row2;
				int rect_w = col2 - col1;
				int rect_h = row2 - row1;
				float max_side = __max((float)rect_w / pose_W, (float)rect_h / pose_H);
				float max_side_W = max_side*pose_W;
				float max_side_H = max_side*pose_H;
				col1 = col1 - max_side_W*0.1;
				col2 = col2 + max_side_W*0.1;
				row1 = row1 - max_side_H*0.1;
				std::vector<float> filter_weights(pose_npts, 0.5f);
				if (pose_npts == 10)
				{
					filter_weights[4] = 0.7f;
					filter_weights[6] = 0.7f;
					filter_weights[8] = 0.8f;
					filter_weights[9] = 0.8f;
				}
				else if (pose_npts == 14)
				{
					filter_weights[8] = 0.7f;
					filter_weights[11] = 0.7f;
				}

				if (pose_npts == 10)
				{
					row2 = __max(row2,__min(row2 + max_side_H*0.1, height+max_side_H*0.05));	
				}
				else
					row2 = row2 + max_side_H*0.1;
				rect_w = col2 - col1;
				rect_h = row2 - row1;
				int cx = (col1 + col2) / 2;
				int cy = (row1 + row2) / 2;
				float size = __max((float)rect_w / pose_W, (float)rect_h / pose_H);
				int size_W = size*pose_W;
				int size_H = size*pose_H;
				row2 = __min(row2, row1 + size_H);
				col2 = __min(col2, col1 + size_W);
				int box_col1 = cx - size_W / 2;
				int box_col2 = box_col1 + size_W;
				int box_row1, box_row2;
				if (pose_npts == 10)
				{
					box_row2 = row2;
					box_row1 = box_row2 - size_H;
				}
				else
				{
					box_row1 = cy - size_H / 2;
					box_row2 = box_row1 + size_H;
				}
				int start_w = __max(0, col1);
				int end_w = __min(width, col2);
				int start_h = __max(0, row1);
				int end_h = __min(height, row2);
				int pad_w_left = __max(0, start_w - box_col1);
				int pad_h_up = __max(0, start_h - box_row1);
				const __int64 buffer_size = (__int64)size_H * size_W * 3;
				if (buffer_size <= 0 || buffer_size > 0x7FFFFFFF)
					return false;
				std::vector<unsigned char> buffer((size_t)buffer_size, 0);

				for (int hh = start_h; hh < end_h; hh++)
				{
					int in_h = hh;
					int in_w = start_w;
					int out_h = hh - start_h + pad_h_up;
					int out_w = pad_w_left;
					memcpy(&buffer[(out_h*size_W + out_w) * 3], bgr_img + in_h*widthStep + in_w * 3, sizeof(unsigned char) * 3 * (end_w - start_w));
				}
				ZQ_CNN_Tensor4D_NHW_C_Align128bit temp_img;
				// 审计修复 2026-10-06（附录 IN.8）：返回值原来被丢弃。
				// `ConvertFromBGR`（ZQ_CNN_Tensor4D.h:395-397）第一句就是
				// `if (!ChangeSize(...)) return false;` —— 失败时 `temp_img` 停在**上一次**的尺寸，
				// 紧接着的 ResizeBilinear 就按陈旧尺寸跑。
				// 同文件 :107 / :112 / :116 对同样的调用**全都检查了** —— 不一致。
				if (!temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1))
				{
					printf("ConvertFromBGR failed (pose %dx%d)\n", size_W, size_H);
					return false;
				}
				/*cv::Mat img = cv::Mat(size_H, size_W, CV_8UC3);
				for (int hh = 0; hh < size_H; hh++)
				{
				memcpy(img.data + hh*img.step[0], &buffer[hh*size_W * 3], sizeof(unsigned char)*size_W * 3);
				}
				cv::namedWindow("roi");
				cv::imshow("roi", img);
				cv::waitKey(0);*/
				// 审计修复 2026-10-06（附录 IN.8）：同上一条，ResizeBilinear 也要查。
				temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
				if (!temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER))
				{
					printf("ResizeBilinear failed (pose %dx%d)\n", pose_W, pose_H);
					return false;
				}
				if (!pose_net.Forward(pose_input))
				{
					printf("failed to run landmark!\n");
					return false;
				}
				const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
					// 审计修复 2026-10-06（附录 IN.5）：GetBlobByName 找不到返回 **0**
					// （ZQ_CNN_Net.h:295-300），原来下一行直接 `pose_ptr->GetFirstPixelPtr()`。
					// 紧邻 100 行内的 **SSD 侧有**守卫（带 `maybe the output blob name (%s)
					// is incorrect` 的 printf），姿态侧一个字都没有 —— 同文件内的不对称，
					// 而 Init 里只查过一次就当成了不变量。
					// Init 成功即意味着 blob 当时存在，所以当前不可达；
					// 一旦有人改 blob 名、或加一层 UnloadBlob，就是空指针解引用。
				if (pose_ptr == 0)
				{
					printf("maybe the output blob name (%s) is incorrect\n", pose_out_blob_name.c_str());
					return false;
				}
				const float* heatmap_data = pose_ptr->GetFirstPixelPtr();
				int hm_H = pose_ptr->GetH();
				int hm_W = pose_ptr->GetW();
				int hm_C = pose_ptr->GetC();
				int hm_widthStep = pose_ptr->GetWidthStep();
				int hm_pixStep = pose_ptr->GetPixelStep();
				float thresh = 0.3f;
				output[nn].num_points = __min(18, __min(hm_C, pose_npts));
				for (int c = 0; c < output[nn].num_points; c++)
				{
					float sum_weight = 0;
					float sum_x = 0;
					float sum_y = 0;
					float max_weight = -FLT_MAX;
					int max_h = -1;
					int max_w = -1;
					for (int h = 0; h < hm_H; h++)
					{
						for (int w = 0; w < hm_W; w++)
						{
							float tmp_val = heatmap_data[c + h*hm_widthStep + w*hm_pixStep];
							if (tmp_val > thresh)
							{
								sum_weight += tmp_val;
								sum_x += tmp_val*w;
								sum_y += tmp_val*h;
							}
							if (tmp_val > max_weight)
							{
								max_h = h;
								max_w = w;
								max_weight = tmp_val;
							}
						}
					}
					/*if (sum_weight > 0)
					{
					sum_x /= sum_weight;
					sum_y /= sum_weight;
					output[nn].points[c * 3 + 0] = (sum_x + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
					output[nn].points[c * 3 + 1] = (sum_y + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
					output[nn].points[c * 3 + 2] = 1;
					}*/
					if (max_weight > thresh)
					{
						if (filter_type == 2)
						{
							if (need_ssd || output[nn].points[c * 3 + 2] < 0.5)
							{
								output[nn].points[c * 3 + 0] = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
								output[nn].points[c * 3 + 1] = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
							}
							else
							{
								float cur_x = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
								float cur_y = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
								float last_x = output[nn].points[c * 3 + 0];
								float last_y = output[nn].points[c * 3 + 1];
								float thresh = __min(max_side_H, max_side_W)*0.03;
								float thresh2 = thresh*thresh;
								if ((cur_x - last_x)*(cur_x - last_x) + (cur_y - last_y)*(cur_y - last_y) <= thresh2)
								{
									output[nn].points[c * 3 + 0] = last_x;
									output[nn].points[c * 3 + 1] = last_y;
								}
								else
								{
									output[nn].points[c * 3 + 0] = last_x*filter_weights[c] + cur_x*(1.0f - filter_weights[c]);
									output[nn].points[c * 3 + 1] = last_y*filter_weights[c] + cur_y*(1.0f - filter_weights[c]);
								}
							}

							output[nn].points[c * 3 + 2] = max_weight;
						}
						else if (filter_type == 1)
						{
							if (need_ssd || output[nn].points[c * 3 + 2] < 0.5)
							{
								output[nn].points[c * 3 + 0] = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
								output[nn].points[c * 3 + 1] = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
							}
							else
							{
								float cur_x = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
								float cur_y = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
								float last_x = output[nn].points[c * 3 + 0];
								float last_y = output[nn].points[c * 3 + 1];
								output[nn].points[c * 3 + 0] = last_x*filter_weights[c] + cur_x*(1.0f - filter_weights[c]);
								output[nn].points[c * 3 + 1] = last_y*filter_weights[c] + cur_y*(1.0f - filter_weights[c]);
							}

							output[nn].points[c * 3 + 2] = max_weight;
						}
						else
						{
							output[nn].points[c * 3 + 0] = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
							output[nn].points[c * 3 + 1] = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
							output[nn].points[c * 3 + 2] = max_weight;
						}
					}
					else
					{
						output[nn].points[c * 3 + 0] = 0;
						output[nn].points[c * 3 + 1] = 0;
						output[nn].points[c * 3 + 2] = 0;
					}

					/*cv::Mat hm_img = cv::Mat(hm_H, hm_W, CV_8UC3);
					for (int h = 0; h < hm_H; h++)
					{
					for (int w = 0; w < hm_W; w++)
					{
					hm_img.data[h*hm_img.step[0] + w * 3 + 0] = 0;
					hm_img.data[h*hm_img.step[0] + w * 3 + 1] = 0;
					hm_img.data[h*hm_img.step[0] + w * 3 + 2] = __min(255, __max(0, heatmap_data[c + h*hm_widthStep + w*hm_pixStep] * 255));
					}
					}
					cv::resize(hm_img, hm_img, cv::Size(), 5, 5, CV_INTER_NN);
					char buf_name[100];
					sprintf(buf_name, "heatmap_%d", c);
					cv::namedWindow(buf_name);
					cv::imshow(buf_name, hm_img);*/

				}
				//cv::waitKey(0);
			}

			return true;
		}
	};
}

#endif
