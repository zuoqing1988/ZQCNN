#ifndef _TRAIN_MTCNN_PROCESSOR_H_
#define _TRAIN_MTCNN_PROCESSOR_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include <vector>
#include <string>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"

namespace ZQ
{
	class TrainMTCNNprocessor
	{
	public:
		static bool generateWiderProb(const char* anno_file, const char* prob_file,
			const char* param_file, const char* model_file, const char* out_blob_name)
		{
			ZQ_CNN_Net Onet;
			ZQ_CNN_Tensor4D_NHW_C_Align128bit input;
			if (!Onet.LoadFrom(param_file, model_file))
			{
				printf("failed to load Onet: %s %s\n", param_file, model_file);
				return false;
			}

			FILE* in = 0, *out = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&in, anno_file, "r"))
#else
			if (0 == (in = fopen(anno_file, "r")))
#endif
			{
				printf("failed to open %s\n", anno_file);
				return false;
			}
#if defined(_WIN32)
			if (0 != fopen_s(&out, prob_file, "w"))
#else
			if (0 == (out = fopen(prob_file, "w")))
#endif
			{
				printf("failed to create %s\n", prob_file);
				fclose(in);
				return false;
			}

			const int BUF_LEN = 1024 * 1024;
			char* buf = (char*)malloc(BUF_LEN);
			memset(buf, 0, BUF_LEN);

			int handled = 0;
			while (true)
			{
				buf[0] = '\0';
				fgets(buf, BUF_LEN - 1, in);
				if (buf[0] == '\0')
					break;
				int len = strlen(buf);
				if (buf[len - 1] == '\n')
					buf[--len] = '\0';
				std::vector<std::string> splits = _split_blank(buf);
				int split_num = splits.size();
				if (split_num % 4 != 1)
				{
					printf("something is wrong: %s\n", buf);
					fclose(in);
					fclose(out);
					free(buf);
					return false;
				}

				std::string& img_file = splits[0];
				int bbox_num = split_num / 4;
				cv::Mat img = cv::imread(img_file, 1);
				if (img.empty())
				{
					printf("failed to load image %s\n", img_file.c_str());
					fclose(in);
					fclose(out);
					free(buf);
					return false;
				}
				int img_width = img.cols;
				int img_height = img.rows;
				fprintf(out, "%s", img_file.c_str());
				for (int i = 0; i < bbox_num; i++)
				{
					int x1 = atoi(splits[i * 4 + 1].c_str());
					int y1 = atoi(splits[i * 4 + 2].c_str());
					int x2 = atoi(splits[i * 4 + 3].c_str());
					int y2 = atoi(splits[i * 4 + 4].c_str());
					int w = x2 - x1;
					int h = y2 - y1;
					int max_side = __max(w, h);
					int crop_x1 = __min(img_width - 1, __max(0, x1 + w / 2 - max_side / 2));
					int crop_y1 = __min(img_height - 1, __max(0, y1 + h / 2 - max_side / 2));
					int crop_x2 = __min(img_width - 1, __max(0, x1 + w / 2 + max_side / 2));
					int crop_y2 = __min(img_height - 1, __max(0, y1 + h / 2 + max_side / 2));
					cv::Mat crop_im = img(cv::Rect(cv::Point(crop_x1, crop_y1), cv::Point(crop_x2, crop_y2)));
					if (crop_im.empty())
					{
						fprintf(out, " 0.00");
						continue;
					}
					cv::Mat resize_im;
					cv::resize(crop_im, resize_im, cv::Size(48, 48));
					input.ConvertFromBGR(resize_im.data, resize_im.cols, resize_im.rows, resize_im.step[0]);
					if (!Onet.Forward(input))
					{
						printf("failed to forward\n");
						fclose(in);
						fclose(out);
						free(buf);
						return false;
					}
					const ZQ_CNN_Tensor4D* blob = Onet.GetBlobByName(std::string(out_blob_name));
					if (blob == 0)
					{
						printf("failed to get blob %s\n", out_blob_name);
						fclose(in);
						fclose(out);
						free(buf);
						return false;
					}
					const float* ptr = blob->GetFirstPixelPtr();

					fprintf(out, " %.2f", ptr[1]);
				}
				fprintf(out, "\n");

				handled++;
				if (handled % 100 == 0)
				{
					printf("%d handled\n", handled);
				}
			}
			free(buf);
			fclose(in);
			fclose(out);
			return true;
		}

		static bool generate_data(int size, const char* root, const char* anno_file, const char* prob_file,
			int base_num = 1, int thread_num = 4, float prob_thresh = 0.3)
		{
			const int BUF_LEN = 1000;
			char save_dir[BUF_LEN] = { 0 };
			char pos_save_dir[BUF_LEN] = { 0 };
			char part_save_dir[BUF_LEN] = { 0 };
			char neg_save_dir[BUF_LEN] = { 0 };
			int len = strlen(root);
			
			if (root[len - 1] == '/' || root[len - 1] == '\\')
			{
#if defined(_WIN32)
				sprintf_s(save_dir, BUF_LEN - 1, "%sprepare_data/%d", root, size);
#else
				sprintf(save_dir, "%sprepare_data/%d", root, size);
#endif
			}
			else
			{
#if defined(_WIN32)
				sprintf_s(save_dir, BUF_LEN - 1, "%s/prepare_data/%d", root, size);
#else
				sprintf(save_dir, "%s/prepare_data/%d", root, size);
#endif
			}
#if defined(_WIN32)
			sprintf_s(pos_save_dir, BUF_LEN - 1, "%s/positive", save_dir);
			sprintf_s(part_save_dir, BUF_LEN - 1, "%s/part", save_dir);
			sprintf_s(neg_save_dir, BUF_LEN - 1, "%s/negative", save_dir);
#else
			sprintf(pos_save_dir, "%s/positive", save_dir);
			sprintf(part_save_dir, "%s/part", save_dir);
			sprintf(neg_save_dir, "%s/negative", save_dir);
#endif

			std::string pos_file = std::string(save_dir) + "/pos.txt";
			std::string part_file = std::string(save_dir) + "/part.txt";
			std::string neg_file = std::string(save_dir) + "/neg.txt";
			
			std::vector<std::string> image_files;
			std::vector<std::vector<float> > all_boxes;
			std::vector<std::vector<float> > all_probs;

			if (!_load_anno_and_prob(anno_file, prob_file, image_files, all_boxes, all_probs))
			{
				printf("failed to load anno and prob file\n");
				return false;
			}

			FILE* out_pos = 0, *out_part = 0, *out_neg = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&out_pos, pos_file.c_str(), "w"))
#else
			if (0 == (out_pos = fopen(pos_file.c_str(), "w")))
#endif
			{
				printf("failed to create file %s\n", pos_file.c_str());
				return false;
			}
#if defined(_WIN32)
			if (0 != fopen_s(&out_part, part_file.c_str(), "w"))
#else
			if (0 == (out_part = fopen(part_file.c_str(), "w")))
#endif
			{
				printf("failed to create file %s\n", part_file.c_str());
				fclose(out_pos);
				return false;
			}
#if defined(_WIN32)
			if (0 != fopen_s(&out_neg, neg_file.c_str(), "w"))
#else 
			if (0 == (out_neg = fopen(neg_file.c_str(), "w")))
#endif
			{
				printf("failed to create file %s\n", neg_file.c_str());
				fclose(out_pos);
				fclose(out_part);
				return false;
			}

			int image_num = image_files.size();
			int handled[1] = { 0 };
			bool ret[1] = { true };
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,10)
			for (int i = 0; i < image_num; i++)
			{
				std::vector<std::string> pos_names, part_names, neg_names;
				bool flag = true;
				if (ret[0])
				{
					flag = _generate_data_for_one_image(i, size, image_files[i], all_boxes[i],
						all_probs[i], prob_thresh,
						pos_save_dir, part_save_dir, neg_save_dir,
						base_num, pos_names, part_names, neg_names);
				}
#pragma omp critical
				{
					if (!flag)
						ret[0] = false;
					
					for (int j = 0; j < pos_names.size(); j++)
					{
						fprintf(out_pos, "%s\n", pos_names[j].c_str());
					}
					for (int j = 0; j < part_names.size(); j++)
					{
						fprintf(out_part, "%s\n", part_names[j].c_str());
					}
					for (int j = 0; j < neg_names.size(); j++)
					{
						fprintf(out_neg, "%s\n", neg_names[j].c_str());
					}
					handled[0]++;
					if (handled[0] % 100 == 0)
					{
						printf("%d handled\n", handled[0]);
					}
				}
			}
			fclose(out_pos);
			fclose(out_part);
			fclose(out_neg);
			return ret[0];
		}

		static bool generate_landmark(int size, const char* root, const char* celeba_img_fold, 
			const char* celeba_bbox_file, const char* celeba_landmark_file, 
			int base_num = 1, int thread_num = 4)
		{
			const int BUF_LEN = 1000;
			char save_dir[BUF_LEN] = { 0 };
			char landmark_save_dir[BUF_LEN] = { 0 };
			std::string celeba_img_root;
#if defined(_WIN32)
			strcpy_s(save_dir, BUF_LEN - 1, celeba_img_fold);
#else
			strcpy(save_dir, celeba_img_fold);
#endif
			int len = strlen(save_dir);

			if (save_dir[len - 1] == '/' || save_dir[len - 1] == '\\')
				save_dir[--len] = '\0';
			celeba_img_root = save_dir;

			len = strlen(root);
			if (root[len - 1] == '/' || root[len - 1] == '\\')
			{
#if defined(_WIN32)
				sprintf_s(save_dir, BUF_LEN - 1, "%sprepare_data/%d", root, size);
#else
				sprintf(save_dir, "%sprepare_data/%d", root, size);
#endif
			}
			else
			{
#if defined(_WIN32)
				sprintf_s(save_dir, BUF_LEN - 1, "%s/prepare_data/%d", root, size);
#else
				sprintf(save_dir, "%s/prepare_data/%d", root, size);
#endif
			}
#if defined(_WIN32)
			sprintf_s(landmark_save_dir, BUF_LEN - 1, "%s/landmark", save_dir);
#else
			sprintf(landmark_save_dir, "%s/landmark", save_dir);
#endif
			
			std::string landmark_file = std::string(save_dir) + "/landmark.txt";
			
			std::vector<std::string> image_files;
			std::vector<std::vector<float> > all_boxes;
			std::vector<std::vector<float> > all_landmarks;

			if (!_load_celeba_bbox_and_landmarks(celeba_bbox_file, celeba_landmark_file, image_files, all_boxes, all_landmarks))
			{
				printf("failed to load bbox and landmark file\n");
				return false;
			}

			FILE* out = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&out, landmark_file.c_str(), "w"))
#else 
			if (0 == (out = fopen(landmark_file.c_str(), "w")))
#endif
			{
				printf("failed to create file %s\n", landmark_file.c_str());
				return false;
			}
			

			int image_num = image_files.size();
			int handled[1] = { 0 };
			bool ret[1] = { true };
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,10)
			for (int i = 0; i < image_num; i++)
			{
				std::vector<std::string> landmark_names;
				bool flag = true;
				if (ret[0])
				{
					std::string img_file = celeba_img_root + "/" + image_files[i];
					flag = _generate_landmark_for_one_image(i, size, img_file, all_boxes[i],
						all_landmarks[i], landmark_save_dir, base_num, landmark_names);
				}
#pragma omp critical
				{
					if (!flag)
						ret[0] = false;

					for (int j = 0; j < landmark_names.size(); j++)
					{
						fprintf(out, "%s\n", landmark_names[j].c_str());
					}
				
					handled[0]++;
					if (handled[0] % 100 == 0)
					{
						printf("%d handled\n", handled[0]);
					}
				}
			}
			fclose(out);
			return ret[0];
		}
	public:
		static bool _is_blank_c(char c)
		{
			return c == ' ' || c == '\t' || c == '\n';
		}

		static std::vector<std::string>  _split_blank(const char* str)
		{
			std::vector<std::string> out;
			int len = strlen(str);
			std::vector<char> buf(len + 1);
			int i = 0, j = 0;
			while (1)
			{
				//skip blank
				for (; i < len && _is_blank_c(str[i]); i++);
				if (i >= len)
					break;

				for (j = i; j < len && !_is_blank_c(str[j]); j++);
				int tmp_len = j - i;
				if (tmp_len == 0)
					break;
				memcpy(&buf[0], str + i, tmp_len * sizeof(char));
				buf[tmp_len] = '\0';

				out.push_back(std::string(&buf[0]));
				i = j;
			}
			return out;
		}

		static float _IOU(const float cur_box[4], const std::vector<float>& all_boxes, const std::string mode = "Union")
		{
			int box_num = all_boxes.size() / 4;
			//the iou
			float max_iou = 0;
			float area1 = (cur_box[2] - cur_box[0])*(cur_box[3] - cur_box[1]);
			for (int i = 0; i < box_num; i++)
			{
				float maxY = __max(cur_box[1], all_boxes[i * 4 + 1]);
				float maxX = __max(cur_box[0], all_boxes[i * 4 + 0]);
				float minY = __min(cur_box[3], all_boxes[i * 4 + 3]);
				float minX = __min(cur_box[2], all_boxes[i * 4 + 2]);
				//maxX1 and maxY1 reuse 
				maxX = __max(minX - maxX + 1, 0);
				maxY = __max(minY - maxY + 1, 0);
				//IOU reuse for the area of two bbox
				float IOU = maxX * maxY;
				float area2 = (all_boxes[i * 4 + 2] - all_boxes[i * 4])
					*(all_boxes[i * 4 + 3] - all_boxes[i * 4 + 1]);
				if (!mode.compare("Union"))
					IOU = IOU / (area1 + area2 - IOU);
				else if (!mode.compare("Min"))
				{
					IOU = IOU / __min(area1, area2);
				}
				max_iou = __max(max_iou, IOU);
			}
			return max_iou;
		}

		static int _randint(int low, int high)
		{
			if (high > low)
				return rand() % (high - low) + low;
			else
				return low;
		}

		static bool _load_anno_and_prob(const char* anno_file, const char* prob_file,
			std::vector<std::string>& image_files, std::vector<std::vector<float> >& all_boxes, 
			std::vector<std::vector<float> >& all_probs)
		{
			image_files.clear();
			all_boxes.clear();
			all_probs.clear();

			FILE* in = 0, *in2 = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&in, anno_file, "r"))
#else
			if (0 == (in = fopen(anno_file, "r")))
#endif
			{
				printf("failed to open %s\n", anno_file);
				return false;
			}
#if defined(_WIN32)
			if (0 != fopen_s(&in2, prob_file, "r"))
#else
			if (0 == (in2 = fopen(prob_file, "r")))
#endif
			{
				printf("failed to open %s\n", prob_file);
				fclose(in);
				return false;
			}


			const int BUF_LEN = 1024 * 1024;
			char* buf = (char*)malloc(BUF_LEN);
			char* buf2 = (char*)malloc(BUF_LEN);
			memset(buf, 0, BUF_LEN);
			memset(buf2, 0, BUF_LEN);
			int handled[1] = { 0 };
			while (true)
			{
				buf[0] = '\0';
				buf2[0] = '\0';
				fgets(buf, BUF_LEN - 1, in);
				fgets(buf2, BUF_LEN - 1, in2);
				if (buf[0] == '\0')
					break;
				int len = strlen(buf);
				if (buf[len - 1] == '\n')
					buf[--len] = '\0';
				int len2 = strlen(buf2);
				if (buf2[len2 - 1] == '\n')
					buf2[--len2] = '\0';
				std::vector<std::string> splits = _split_blank(buf);
				std::vector<std::string> splits2 = _split_blank(buf2);
				int split_num = splits.size();
				int split_num2 = splits2.size();
				if (split_num % 4 != 1)
				{
					printf("something is wrong: %s\n", buf);
					fclose(in);
					fclose(in2);
					free(buf);
					free(buf2);
					return false;
				}

				std::string& img_file = splits[0];
				int bbox_num = split_num / 4;
				if (split_num2 != bbox_num + 1)
				{
					printf("something is wrong: %s, %s\n", buf, buf2);
					fclose(in);
					fclose(in2);
					free(buf);
					free(buf2);
					return false;
				}
				image_files.push_back(img_file);
				std::vector<float> boxes(bbox_num * 4);
				std::vector<float> probs(bbox_num);
				for (int i = 0; i < bbox_num; i++)
				{
					boxes[i * 4] = atoi(splits[i * 4 + 1].c_str());
					boxes[i * 4 + 1] = atoi(splits[i * 4 + 2].c_str());
					boxes[i * 4 + 2] = atoi(splits[i * 4 + 3].c_str());
					boxes[i * 4 + 3] = atoi(splits[i * 4 + 4].c_str());
					probs[i] = atof(splits2[i + 1].c_str());
				}
				all_boxes.push_back(boxes);
				all_probs.push_back(probs);
			}

			fclose(in);
			fclose(in2);
			free(buf);
			free(buf2);
			return true;
		}

		static bool _load_celeba_bbox_and_landmarks(const char* celeba_bbox_file, 
			const char* celeba_landmark_file, std::vector<std::string>& image_files, 
			std::vector<std::vector<float> >& all_boxes, 
			std::vector<std::vector<float> >& all_landmarks)
		{
			image_files.clear();
			all_boxes.clear();
			all_landmarks.clear();

			FILE* in = 0, *in2 = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&in, celeba_bbox_file, "r"))
#else
			if (0 == (in = fopen(celeba_bbox_file, "r")))
#endif
			{
				printf("failed to open %s\n", celeba_bbox_file);
				return false;
			}
#if defined(_WIN32)
			if (0 != fopen_s(&in2, celeba_landmark_file, "r"))
#else
			if (0 == (in2 = fopen(celeba_landmark_file, "r")))
#endif
			{
				printf("failed to open %s\n", celeba_landmark_file);
				fclose(in);
				return false;
			}

			int line_id = 0;
			int image_num = 0;
			const int BUF_LEN = 1024 * 1024;
			char* buf = (char*)malloc(BUF_LEN);
			char* buf2 = (char*)malloc(BUF_LEN);
			memset(buf, 0, BUF_LEN);
			memset(buf2, 0, BUF_LEN);
			int handled[1] = { 0 };
			while (true)
			{
				buf[0] = '\0';
				buf2[0] = '\0';
				fgets(buf, BUF_LEN - 1, in);
				fgets(buf2, BUF_LEN - 1, in2);
				if (buf[0] == '\0')
					break;
				
				if (line_id == 0)
				{
#if defined(_WIN32)
					sscanf_s(buf, "%d", &image_num);
#else
					sscanf(buf, "%d", &image_num);
#endif
				}
				
				if (line_id <= 1)
				{
					line_id++;
					continue;
				}

				if (line_id + 2 >= image_num)
					break;
				line_id++;
				
				int len = strlen(buf);
				if (buf[len - 1] == '\n')
					buf[--len] = '\0';
				int len2 = strlen(buf2);
				if (buf2[len2 - 1] == '\n')
					buf2[--len2] = '\0';
				std::vector<std::string> splits = _split_blank(buf);
				std::vector<std::string> splits2 = _split_blank(buf2);
				int split_num = splits.size();
				int split_num2 = splits2.size();
				if (split_num % 4 != 1)
				{
					printf("something is wrong: %s\n", buf);
					fclose(in);
					fclose(in2);
					free(buf);
					free(buf2);
					return false;
				}

				std::string& img_file = splits[0];
				int bbox_num = split_num / 4;
				if (split_num2 != bbox_num*10 + 1)
				{
					printf("something is wrong: %s, %s\n", buf, buf2);
					fclose(in);
					fclose(in2);
					free(buf);
					free(buf2);
					return false;
				}
				image_files.push_back(img_file);
				std::vector<float> boxes(bbox_num * 4);
				std::vector<float> landmarks(bbox_num * 10);
				for (int i = 0; i < bbox_num; i++)
				{
					for (int j = 0; j < 4; j++)
						boxes[i * 4 + j] = atoi(splits[i * 4 + j + 1].c_str());
					for (int j = 0; j < 10; j++)
						landmarks[i * 10 + j] = atof(splits2[i * 10 + j + 1].c_str());
				}
				all_boxes.push_back(boxes);
				all_landmarks.push_back(landmarks);
			}

			fclose(in);
			fclose(in2);
			free(buf);
			free(buf2);
			return true;
		}

		static bool _generate_data_for_one_image(int idx, int size, const std::string& image_file,
			const std::vector<float>& boxes, const std::vector<float>& probs, float prob_thresh,
			const std::string& pos_save_dir, const std::string& part_save_dir,
			const std::string& neg_save_dir, int base_num, std::vector<std::string>& pos_names,
			std::vector<std::string>& part_names, std::vector<std::string>& neg_names)
		{
			const int BUF_LEN = 500;
			char tmp_buf[BUF_LEN];
			std::string file_name, line;
			pos_names.clear();
			part_names.clear();
			neg_names.clear();

			int box_num = boxes.size() / 4;

			cv::Mat img = cv::imread(image_file, 1);
			if (img.empty())
			{
				printf("failed to load image %s\n", image_file.c_str());
				return false;
			}
			int width = img.cols;
			int height = img.rows;
			int min_size = __min(width, height) / 2;
			if (min_size <= size)
			{
				return true;
			}

			cv::Mat resized_im, brighter_im, darker_im;
			// neg
			int neg_num = 0, pos_num = 0, part_num = 0;
			while (neg_num < base_num * 50)
			{
				int cur_size = _randint(size, min_size);
				int nx = rand() % (width - cur_size);
				int ny = rand() % (height - cur_size);

				float crop_box[4] = { nx, ny, nx + cur_size, ny + cur_size };

				float iou = _IOU(crop_box, boxes);

				cv::Mat cropped_im = img(cv::Rect(cv::Point(crop_box[0], crop_box[1]), cv::Point(crop_box[2], crop_box[3])));

				if (cropped_im.empty())
					continue;

				if (iou < 0.3)
				{
					cv::resize(cropped_im, resized_im, cv::Size(size, size));
					resized_im.convertTo(brighter_im, resized_im.type(), 1.25);
					resized_im.convertTo(darker_im, resized_im.type(), 0.8);
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
					file_name = neg_save_dir + "/" + std::string(tmp_buf);
					if (!cv::imwrite(file_name, resized_im))
					{
						printf("failed to write image %s\n", file_name.c_str());
						return false;
					}
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
					line = neg_save_dir + "/" + std::string(tmp_buf);
					neg_names.push_back(line);
					neg_num++;
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
					file_name = neg_save_dir + "/" + std::string(tmp_buf);
					if (!cv::imwrite(file_name, brighter_im))
					{
						printf("failed to write image %s\n", file_name.c_str());
						return false;
					}
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
					line = neg_save_dir + "/" + std::string(tmp_buf);
					neg_names.push_back(line);
					neg_num++;
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
					file_name = neg_save_dir + "/" + std::string(tmp_buf);
					if (!cv::imwrite(file_name, darker_im))
					{
						printf("failed to write image %s\n", file_name.c_str());
						return false;
					}
#if defined(_WIN32)
					sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
					sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
					line = neg_save_dir + "/" + std::string(tmp_buf);
					neg_names.push_back(line);
					neg_num++;
				}
			}

			for (int bb = 0; bb < box_num; bb++)
			{
				int x1 = boxes[bb * 4];
				int y1 = boxes[bb * 4 + 1];
				int x2 = boxes[bb * 4 + 2];
				int y2 = boxes[bb * 4 + 3];
				int w = x2 - x1 + 1;
				int h = y2 - y1 + 1;
				if (__max(w, h) < 40 || x1 < 0 || y1 < 0 || probs[bb] < prob_thresh)
					continue;


				//neg
				for (int i = 0; i < base_num * 2; i++)
				{
					int cur_size = _randint(size, min_size);
					int delta_x = _randint(__max(-cur_size, -x1), w);
					int delta_y = _randint(__max(-cur_size, -y1), h);
					int nx1 = __max(0, x1 + delta_x);
					int ny1 = __max(0, y1 + delta_y);

					if (nx1 + cur_size > width || ny1 + cur_size > height)
						continue;

					float crop_box[4] = { nx1, ny1, nx1 + cur_size, ny1 + cur_size };

					float iou = _IOU(crop_box, boxes);

					cv::Mat cropped_im = img(cv::Rect(cv::Point(crop_box[0], crop_box[1]), cv::Point(crop_box[2], crop_box[3])));

					if (cropped_im.empty())
						continue;

					if (iou < 0.3)
					{
						cv::resize(cropped_im, resized_im, cv::Size(size, size));
						resized_im.convertTo(brighter_im, resized_im.type(), 1.25);
						resized_im.convertTo(darker_im, resized_im.type(), 0.8);
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
						file_name = neg_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, resized_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
						line = neg_save_dir + "/" + std::string(tmp_buf);
						neg_names.push_back(line);
						neg_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
						file_name = neg_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, brighter_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
						line = neg_save_dir + "/" + std::string(tmp_buf);
						neg_names.push_back(line);
						neg_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, neg_num);
#endif
						file_name = neg_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, darker_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 0", idx, neg_num);
#else
						sprintf(tmp_buf, "%d_%d 0", idx, neg_num);
#endif
						line = neg_save_dir + "/" + std::string(tmp_buf);
						neg_names.push_back(line);
						neg_num++;
					}
				}

				//pos & part
				for (int i = 0; i < base_num * 8; i++)
				{
					int cur_size = _randint(__min(w, h) * 0.8, ceil(1.25 * __max(w, h)));
					int delta_x = _randint(-w * 0.2, w * 0.2);
					int delta_y = _randint(-h * 0.2, h * 0.2);
					int nx1 = int(__max(x1 + w / 2 + delta_x - cur_size / 2, 0));
					int ny1 = int(__max(y1 + h / 2 + delta_y - cur_size / 2, 0));
					int nx2 = nx1 + cur_size;
					int ny2 = ny1 + cur_size;

					if (nx2 > width || ny2 > height)
						continue;

					float crop_box[4] = { nx1, ny1, nx1 + cur_size, ny1 + cur_size };

					float iou = _IOU(crop_box, boxes);

					cv::Mat cropped_im = img(cv::Rect(cv::Point(crop_box[0], crop_box[1]), cv::Point(crop_box[2], crop_box[3])));

					if (cropped_im.empty())
						continue;

					float offset_x1 = (x1 - nx1) / float(cur_size);
					float offset_y1 = (y1 - ny1) / float(cur_size);
					float offset_x2 = (x2 - nx2) / float(cur_size);
					float offset_y2 = (y2 - ny2) / float(cur_size);
					if (iou >= 0.65)
					{
						cv::resize(cropped_im, resized_im, cv::Size(size, size));
						resized_im.convertTo(brighter_im, resized_im.type(), 1.25);
						resized_im.convertTo(darker_im, resized_im.type(), 0.8);
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, pos_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, pos_num);
#endif
						file_name = pos_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, resized_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#else
						sprintf(tmp_buf, "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#endif
							offset_x1, offset_y1, offset_x2, offset_y2);
						line = pos_save_dir + "/" + std::string(tmp_buf);
						pos_names.push_back(line);
						pos_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, pos_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, pos_num);
#endif
						file_name = pos_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, brighter_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#else
						sprintf(tmp_buf, "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#endif
							offset_x1, offset_y1, offset_x2, offset_y2);
						line = pos_save_dir + "/" + std::string(tmp_buf);
						pos_names.push_back(line);
						pos_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, pos_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, pos_num);
#endif
						file_name = pos_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, darker_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#else
						sprintf(tmp_buf, "%d_%d 1 %.2f %.2f %.2f %.2f", idx, pos_num,
#endif
							offset_x1, offset_y1, offset_x2, offset_y2);
						line = pos_save_dir + "/" + std::string(tmp_buf);
						pos_names.push_back(line);
						pos_num++;
					}
					else if (iou >= 0.4)
					{
						if (rand() % 100 <= 40)
						{
							cv::resize(cropped_im, resized_im, cv::Size(size, size));
							resized_im.convertTo(brighter_im, resized_im.type(), 1.25);
							resized_im.convertTo(darker_im, resized_im.type(), 0.8);
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, part_num);
#else
							sprintf(tmp_buf, "%d_%d.jpg", idx, part_num);
#endif
							file_name = part_save_dir + "/" + std::string(tmp_buf);
							if (!cv::imwrite(file_name, resized_im))
							{
								printf("failed to write image %s\n", file_name.c_str());
								return false;
							}
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#else
							sprintf(tmp_buf, "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#endif
								offset_x1, offset_y1, offset_x2, offset_y2);
							line = part_save_dir + "/" + std::string(tmp_buf);
							part_names.push_back(line);
							part_num++;
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, part_num);
#else
							sprintf(tmp_buf, "%d_%d.jpg", idx, part_num);
#endif
							file_name = part_save_dir + "/" + std::string(tmp_buf);
							if (!cv::imwrite(file_name, brighter_im))
							{
								printf("failed to write image %s\n", file_name.c_str());
								return false;
							}
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#else
							sprintf(tmp_buf, "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#endif
								offset_x1, offset_y1, offset_x2, offset_y2);
							line = part_save_dir + "/" + std::string(tmp_buf);
							part_names.push_back(line);
							part_num++;
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d.jpg", idx, part_num);
#else
							sprintf(tmp_buf, "%d_%d.jpg", idx, part_num);
#endif
							file_name = part_save_dir + "/" + std::string(tmp_buf);
							if (!cv::imwrite(file_name, darker_im))
							{
								printf("failed to write image %s\n", file_name.c_str());
								return false;
							}
#if defined(_WIN32)
							sprintf_s(tmp_buf, BUF_LEN-1,  "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#else
							sprintf(tmp_buf, "%d_%d -1 %.2f %.2f %.2f %.2f", idx, part_num,
#endif
								offset_x1, offset_y1, offset_x2, offset_y2);
							line = part_save_dir + "/" + std::string(tmp_buf);
							part_names.push_back(line);
							part_num++;
						}
					}
				}
			}
			return true;
		}


		static bool _generate_landmark_for_one_image(int idx, int size, const std::string& image_file,
			const std::vector<float>& boxes, const std::vector<float>& landmarks, 
			const std::string& landmark_save_dir, int base_num, std::vector<std::string>& landmark_names)
		{
			const int BUF_LEN = 500;
			char tmp_buf[BUF_LEN];
			std::string file_name, line;
			landmark_names.clear();
			
			int box_num = boxes.size() / 4;

			cv::Mat img = cv::imread(image_file, 1);
			if (img.empty())
			{
				printf("failed to load image %s\n", image_file.c_str());
				return false;
			}
			int width = img.cols;
			int height = img.rows;
			int min_size = __min(width, height) / 2;
			if (min_size <= size)
			{
				return true;
			}
			
			cv::Mat rot_im;
			cv::Mat resized_im, brighter_im, darker_im;
			double angles[13] = { 0,-15,-30,-45,-60,-75,-90,15,30,45,60,75,90 };
			int angle_num = 13;
			float rot_landmark[10];
			int landmark_num = 0;
			for (int bb = 0; bb < box_num; bb++)
			{
				int x1 = boxes[bb * 4];
				int y1 = boxes[bb * 4 + 1];
				int w = boxes[bb * 4 + 2];
				int h = boxes[bb * 4 + 3];
				if (__max(w, h) < 40 || x1 < 0 || y1 < 0)
					continue;
				float cx = landmarks[bb * 10 + 4];
				float cy = landmarks[bb * 10 + 5];
				
				for (int aa = 0; aa < angle_num; aa++)
				{
					cv::Mat rotM = cv::getRotationMatrix2D(cv::Point2f(cx, cy), angles[aa], 1);
					for (int i = 0; i < 5; i++)
					{
						rot_landmark[i * 2 + 0] = rotM.at<double>(0, 0)*landmarks[bb * 10 + i * 2]
							+ rotM.at<double>(0, 1)*landmarks[bb * 10 + i * 2 + 1]
							+ rotM.at<double>(0, 2);
						rot_landmark[i * 2 + 1] = rotM.at<double>(1, 0)*landmarks[bb * 10 + i * 2]
							+ rotM.at<double>(1, 1)*landmarks[bb * 10 + i * 2 + 1]
							+ rotM.at<double>(1, 2);
					}

					cv::warpAffine(img, rot_im, rotM, cv::Size(width, height));

					for (int i = 0; i < base_num; i++)
					{
						int cur_size = _randint(__min(w, h) * 0.8, ceil(1.25 * __max(w, h)));
						int delta_x = _randint(-w * 0.15, w * 0.15);
						int delta_y = _randint(-h * 0.15, h * 0.15);
						int nx1 = int(__max(x1 + w / 2 + delta_x - cur_size / 2, 0));
						int ny1 = int(__max(y1 + h / 2 + delta_y - cur_size / 2, 0));
						int nx2 = nx1 + cur_size;
						int ny2 = ny1 + cur_size;

						if (nx2 > width || ny2 > height)
							continue;

						float crop_box[4] = { nx1, ny1, nx1 + cur_size, ny1 + cur_size };

						cv::Mat cropped_im = img(cv::Rect(cv::Point(crop_box[0], crop_box[1]), cv::Point(crop_box[2], crop_box[3])));

						if (cropped_im.empty())
							continue;

						float offset_x1 = (rot_landmark[0] - nx1 + 0.5) / float(cur_size);
						float offset_y1 = (rot_landmark[1] - ny1 + 0.5) / float(cur_size);
						float offset_x2 = (rot_landmark[2] - nx1 + 0.5) / float(cur_size);
						float offset_y2 = (rot_landmark[3] - ny1 + 0.5) / float(cur_size);
						float offset_x3 = (rot_landmark[4] - nx1 + 0.5) / float(cur_size);
						float offset_y3 = (rot_landmark[5] - ny1 + 0.5) / float(cur_size);
						float offset_x4 = (rot_landmark[6] - nx1 + 0.5) / float(cur_size);
						float offset_y4 = (rot_landmark[7] - ny1 + 0.5) / float(cur_size);
						float offset_x5 = (rot_landmark[8] - nx1 + 0.5) / float(cur_size);
						float offset_y5 = (rot_landmark[9] - ny1 + 0.5) / float(cur_size);

						cv::resize(cropped_im, resized_im, cv::Size(size, size));
						resized_im.convertTo(brighter_im, resized_im.type(), 1.25);
						resized_im.convertTo(darker_im, resized_im.type(), 0.8);
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d.jpg", idx, landmark_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, landmark_num);
#endif
						file_name = landmark_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, resized_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#else
						sprintf(tmp_buf, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#endif
							idx, landmark_num,
							offset_x1, offset_x2, offset_x3, offset_x4, offset_x5,
							offset_y1, offset_y2, offset_y3, offset_y4, offset_y5);
						line = landmark_save_dir + "/" + std::string(tmp_buf);
						landmark_names.push_back(line);
						landmark_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d.jpg", idx, landmark_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, landmark_num);
#endif
						file_name = landmark_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, brighter_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#else
						sprintf(tmp_buf, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#endif
							idx, landmark_num,
							offset_x1, offset_x2, offset_x3, offset_x4, offset_x5,
							offset_y1, offset_y2, offset_y3, offset_y4, offset_y5);
						line = landmark_save_dir + "/" + std::string(tmp_buf);
						landmark_names.push_back(line);
						landmark_num++;
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d.jpg", idx, landmark_num);
#else
						sprintf(tmp_buf, "%d_%d.jpg", idx, landmark_num);
#endif
						file_name = landmark_save_dir + "/" + std::string(tmp_buf);
						if (!cv::imwrite(file_name, darker_im))
						{
							printf("failed to write image %s\n", file_name.c_str());
							return false;
						}
#if defined(_WIN32)
						sprintf_s(tmp_buf, BUF_LEN - 1, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#else
						sprintf(tmp_buf, "%d_%d -2 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f",
#endif
							idx, landmark_num,
							offset_x1, offset_x2, offset_x3, offset_x4, offset_x5,
							offset_y1, offset_y2, offset_y3, offset_y4, offset_y5);
						line = landmark_save_dir + "/" + std::string(tmp_buf);
						landmark_names.push_back(line);
						landmark_num++;
					}
				}
				
			}
			return true;
		}
	};
}

#endif
