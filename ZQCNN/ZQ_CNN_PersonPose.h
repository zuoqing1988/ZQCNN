#ifndef _ZQ_CNN_PERSON_POSE_H_
#define _ZQ_CNN_PERSON_POSE_H_
#pragma once

#include "ZQ_CNN_Net.h"
#include <vector>
#include <iostream>

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
				memset(points, 0, sizeof(float) * 51);
			}
		};

	private:
		ZQ_CNN_Net ssd_net;
		std::string ssd_out_blob_name;
		int person_class_id;

		ZQ_CNN_Net pose_net;
		std::string pose_out_blob_name;

		ZQ_CNN_Tensor4D_NHW_C_Align128bit input0, input1;
		int ssd_C, ssd_H, ssd_W;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit pose_input;
		int pose_C, pose_H, pose_W;
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
				std::vector<unsigned char> buffer(size_H*size_W*3,0);
				
				for (int hh = start_h; hh < end_h; hh++)
				{
					int in_h = hh;
					int in_w = start_w;
					int out_h = hh - start_h + pad_h_up;
					int out_w = pad_w_left;
					memcpy(&buffer[(out_h*size_W+out_w)*3], bgr_img + in_h*widthStep + in_w * 3, sizeof(unsigned char) * 3 * (end_w - start_w));
				}
				ZQ_CNN_Tensor4D_NHW_C_Align128bit temp_img;
				temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1);
				/*cv::Mat img = cv::Mat(size_H, size_W, CV_8UC3);
				for (int hh = 0; hh < size_H; hh++)
				{
					memcpy(img.data + hh*img.step[0], &buffer[hh*size_W * 3], sizeof(unsigned char)*size_W * 3);
				}
				cv::namedWindow("roi");
				cv::imshow("roi", img);
				cv::waitKey(0);*/
				temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
				if (!pose_net.Forward(pose_input))
				{
					printf("failed to run landmark!\n");
					return false;
				}
				const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
				const float* heatmap_data = pose_ptr->GetFirstPixelPtr();
				int hm_H = pose_ptr->GetH();
				int hm_W = pose_ptr->GetW();
				int hm_C = pose_ptr->GetC();
				int hm_widthStep = pose_ptr->GetWidthStep();
				int hm_pixStep = pose_ptr->GetPixelStep();
				float thresh = 0.3f;
				output[nn].num_points = hm_C;
				for (int c = 0; c < hm_C; c++)
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

		bool DetectVideoSinglePerson(std::vector<BBox>& output, const unsigned char* bgr_img, int width, int height, int widthStep, float confidence_thresh,
			bool show_debug_info = false)
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
					if (valid_num < npts*0.6 || total_weight < valid_num*0.5)
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
							float top_x = output[nn].points[0 * 3 + 0];
							float top_y = output[nn].points[0 * 3 + 1];
							float neck_x = output[nn].points[1 * 3 + 0];
							float neck_y = output[nn].points[1 * 3 + 1];
							float dir_x = top_x - neck_x;
							float dir_y = top_y - neck_y;
							float head_len = sqrt(dir_x*dir_x + dir_y*dir_y);
							if (output[nn].points[8 * 3 + 2] == 0 && output[nn].points[9 * 3 + 2] == 0)
							{
								
								output[nn].row2 = neck_y + head_len * 2;
							}
							else
								output[nn].row2 = cy + size_y*0.5;
							output[nn].col1 = __min(output[nn].col1, __min(output[nn].points[2 * 3 + 0], output[nn].points[5 * 3 + 0]) - head_len*0.6);
							output[nn].col2 = __max(output[nn].col2, __max(output[nn].points[2 * 3 + 0], output[nn].points[5 * 3 + 0]) + head_len*0.6);
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
				int box_row1 = cy - size_H / 2;
				int box_row2 = box_row1 + size_H;
				int start_w = __max(0, col1);
				int end_w = __min(width, col2);
				int start_h = __max(0, row1);
				int end_h = __min(height, row2);
				int pad_w_left = __max(0, start_w - box_col1);
				int pad_h_up = __max(0, start_h - box_row1);
				std::vector<unsigned char> buffer(size_H*size_W * 3, 0);

				for (int hh = start_h; hh < end_h; hh++)
				{
					int in_h = hh;
					int in_w = start_w;
					int out_h = hh - start_h + pad_h_up;
					int out_w = pad_w_left;
					memcpy(&buffer[(out_h*size_W + out_w) * 3], bgr_img + in_h*widthStep + in_w * 3, sizeof(unsigned char) * 3 * (end_w - start_w));
				}
				ZQ_CNN_Tensor4D_NHW_C_Align128bit temp_img;
				temp_img.ConvertFromBGR(&buffer[0], size_W, size_H, size_W * 3, 0, 1);
				/*cv::Mat img = cv::Mat(size_H, size_W, CV_8UC3);
				for (int hh = 0; hh < size_H; hh++)
				{
				memcpy(img.data + hh*img.step[0], &buffer[hh*size_W * 3], sizeof(unsigned char)*size_W * 3);
				}
				cv::namedWindow("roi");
				cv::imshow("roi", img);
				cv::waitKey(0);*/
				temp_img.ResizeBilinear(pose_input, pose_W, pose_H, 0, 0, ZQ_CNN_Tensor4D::SAMPLE_ALIGN_CENTER);
				if (!pose_net.Forward(pose_input))
				{
					printf("failed to run landmark!\n");
					return false;
				}
				const ZQ_CNN_Tensor4D* pose_ptr = pose_net.GetBlobByName(pose_out_blob_name);
				const float* heatmap_data = pose_ptr->GetFirstPixelPtr();
				int hm_H = pose_ptr->GetH();
				int hm_W = pose_ptr->GetW();
				int hm_C = pose_ptr->GetC();
				int hm_widthStep = pose_ptr->GetWidthStep();
				int hm_pixStep = pose_ptr->GetPixelStep();
				float thresh = 0.3f;
				output[nn].num_points = hm_C;
				for (int c = 0; c < hm_C; c++)
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
						if (need_ssd || output[nn].points[c * 3 + 2] < 0.5)
						{
							output[nn].points[c * 3 + 0] = (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
							output[nn].points[c * 3 + 1] = (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
						}
						else
						{
							output[nn].points[c * 3 + 0] += (max_w + 0.5) / hm_W*size_W - 0.5 - pad_w_left + start_w;
							output[nn].points[c * 3 + 1] += (max_h + 0.5) / hm_H*size_H - 0.5 - pad_h_up + start_h;
							output[nn].points[c * 3 + 0] /= 2;
							output[nn].points[c * 3 + 1] /= 2;
						}
						
						output[nn].points[c * 3 + 2] = max_weight;
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
