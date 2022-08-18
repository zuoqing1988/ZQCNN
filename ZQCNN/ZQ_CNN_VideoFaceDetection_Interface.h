#ifndef _ZQ_CNN_VIDEO_FACE_DETECTION_INTERFACE_H_
#define _ZQ_CNN_VIDEO_FACE_DETECTION_INTERFACE_H_
#pragma once

#include "ZQ_CNN_Net_Interface.h"
#include "ZQ_CNN_BBoxUtils.h"
#include "ZQ_CNN_MTCNN_Interface.h"
#include "ZQ_CNN_CascadeOnet_Interface.h"
#include "ZQ_CNN_FaceCropUtils.h"
#include "ZQlib/ZQ_SVD.h"
#include <float.h>
#include <vector>
namespace ZQ
{
	template<class ZQ_CNN_Net_Interface, class ZQ_CNN_Tensor4D_Interface, class ZQ_CNN_Tensor4D_Interface_Base>
	class ZQ_CNN_VideoFaceDetection_Interface
	{
		using string = std::string;
	public:
		enum VFD_MSG {
			VFD_MSG_MAX_TRACE_NUM,
			VFD_MSG_WEIGHT_DECAY,
			VFD_MSG_FORCE_FIRST_FRAME
		};

		ZQ_CNN_VideoFaceDetection_Interface()
		{
			max_trace_num = 4;
			weight_decay = 0;
			show_debug_info = false;
			enable_iou_filter = false;
			is_first_frame = true;
			thread_num = 1;
			has_lnet106 = false;
			key_cooldown = 50;
		}
		~ZQ_CNN_VideoFaceDetection_Interface() {}

	private:
		int max_trace_num;
		float weight_decay;
		bool show_debug_info;
		bool enable_iou_filter;
		float othresh;
		bool is_first_frame;
		int thread_num;
		ZQ_CNN_MTCNN_Interface<ZQ_CNN_Net_Interface, ZQ_CNN_Tensor4D_Interface, ZQ_CNN_Tensor4D_Interface_Base> mtcnn;
		std::vector<ZQ_CNN_CascadeOnet_Interface<ZQ_CNN_Net_Interface, ZQ_CNN_Tensor4D_Interface, ZQ_CNN_Tensor4D_Interface_Base>> cascade_Onets;
		std::vector<ZQ_CNN_Net_Interface> onets;
		bool has_lnet106;
		std::vector<ZQ_CNN_Net_Interface> lnets106;
		bool has_headposegaze;
		std::vector<ZQ_CNN_Net_Interface> headposegaze_nets;
		int key_cooldown;
		int cur_key_cooldown;
		std::vector<std::vector<ZQ_CNN_BBox106> > trace;
		std::vector<ZQ_CNN_BBox106> backup_results;
		ZQ_CNN_Tensor4D_Interface input, lnet106_image;
		int lnet106_size;
		int onet_size;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		void TurnOnFilterIOU() { enable_iou_filter = true; }
		void TurnOffFilterIOU() { enable_iou_filter = false; }

		bool Init(const string& pnet_param, const string& pnet_model, const string& rnet_param, const string& rnet_model,
			const string& onet_param, const string& onet_model, int thread_num = 1,
			bool has_lnet106 = false, const string& lnet106_param = "", const std::string& lnet106_model = "",
			bool has_headposegaze = false, const string& headposegaze_param = "", const std::string& headposegaze_model = "")
		{
			if (!mtcnn.Init(pnet_param, pnet_model, rnet_param, rnet_model, onet_param, onet_model, thread_num, has_lnet106, lnet106_param, lnet106_model))
				return false;
			this->thread_num = __max(1, thread_num);
			cascade_Onets.resize(this->thread_num);
			for (int i = 0; i < cascade_Onets.size(); i++)
			{
				if (!cascade_Onets[i].Init(onet_param, onet_model, onet_param, onet_model, onet_param, onet_model))
					return false;
			}
			onets.resize(this->thread_num);
			for (int i = 0; i < onets.size(); i++)
			{
				if (!onets[i].LoadFrom(onet_param, onet_model, true, 1e-9, true))
					return false;
			}
			int C, H, W;
			onets[0].GetInputDim(C, H, W);
			onet_size = H;
			
			this->has_lnet106 = has_lnet106;
			if (has_lnet106)
			{
				lnets106.resize(this->thread_num);
				for (int i = 0; i < lnets106.size(); i++)
				{
					if (!lnets106[i].LoadFrom(lnet106_param, lnet106_model, true, 1e-9, true))
						return false;
				}
				int C, H, W;
				lnets106[0].GetInputDim(C, H, W);
				lnet106_size = H;
			}

			this->has_headposegaze = has_headposegaze;
			if (has_headposegaze)
			{
				headposegaze_nets.resize(this->thread_num);
				for (int i = 0; i < headposegaze_nets.size(); i++)
				{
					if (!headposegaze_nets[i].LoadFrom(headposegaze_param, headposegaze_model, true, 1e-9, true))
						return false;
				}
			}
			return true;
		}

		void SetPara(int w, int h, int min_face_size = 60, float pthresh = 0.6, float rthresh = 0.7, float othresh = 0.7,
			float nms_pthresh = 0.6, float nms_rthresh = 0.7, float nms_othresh = 0.7, float scale_factor = 0.709,
			int pnet_overlap_thresh_count = 3, int pnet_size = 20, int pnet_stride = 4, int key_cooldown = 50)
		{
			this->key_cooldown = key_cooldown;
			mtcnn.SetPara(w, h, min_face_size, pthresh, rthresh, othresh, nms_pthresh, nms_rthresh, nms_othresh,
				scale_factor, pnet_overlap_thresh_count, pnet_size, pnet_stride, true, 1.0);
			this->othresh = othresh;
		}

		void Message(const VFD_MSG msg, double val)
		{
			switch (msg)
			{
			case VFD_MSG_FORCE_FIRST_FRAME:
				is_first_frame = true;
				break;
			case VFD_MSG_MAX_TRACE_NUM:
				max_trace_num = __max(0, val);
				break;
			case VFD_MSG_WEIGHT_DECAY:
				weight_decay = val;
				break;
			}
		}

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox106>& results)
		{
			if (!input.ConvertFromBGR(bgr_img, _width, _height, _widthStep))
				return false;

			results.clear();

			if (is_first_frame)
			{
				mtcnn.EnableLnet(true);
				if (!mtcnn.Find106(input, results))
					return false;
				_refine_landmark106(results, true);
				_recompute_bbox(results);
				int cur_box_num = results.size();
				trace.clear();
				trace.resize(cur_box_num);
				for (int i = 0; i < cur_box_num; i++)
					trace[i].push_back(results[i]);
				is_first_frame = results.size() == 0;
				cur_key_cooldown = key_cooldown;
				backup_results = results;
			}
			else
			{
				std::vector<ZQ_CNN_BBox106> results106_part1, results106_part2;
				std::vector<int> good_idx;
				std::vector<ZQ_CNN_OrderScore> orders;
				std::vector<ZQ_CNN_BBox106> tmp_boxes;
				ZQ_CNN_OrderScore tmp_order;
				int ori_count = 0;
				//static int fr_id = 0;
				//fr_id++;
				/**********   Stage 1: detect around the old positions         ************/
				const double m_pi = 4 * atan(1.0);
				std::vector<ZQ_CNN_BBox106> boxes;
				std::vector<ZQ_CNN_BBox106> last_boxes(trace.size());
				std::vector<ZQ_CNN_Tensor4D_Interface> task_images(trace.size());
				std::vector<ZQ_CNN_Tensor4D_Interface> task_images_gray(trace.size());
				for (int i = 0; i < trace.size(); i++)
				{
					/**************** First: L106 ******************/

					float last_rot = _get_rot_of_landmark106(trace[i][0].ppoint, m_pi);
					float cx, cy, min_x, max_x, min_y, max_y;
					_get_landmark106_info(trace[i][0].ppoint, cx, cy, min_x, max_x, min_y, max_y);
					float cur_w = max_x - min_x;
					float cur_h = max_y - min_y;
					float cur_size = 1.15*__max(cur_w, cur_h);

					std::vector<float> map_x, map_y;
					_compute_map(cx, cy, last_rot, cur_size, cur_size, lnet106_size, lnet106_size, map_x, map_y);

					input.Remap(task_images[0], lnet106_size, lnet106_size, 0, 0, map_x, map_y, true, 0);

					task_images[0].ConvertColor_BGR2GRAY(task_images_gray[0], 1, 1);
					lnets106[0].Forward(task_images_gray[0]);

					ZQ_CNN_BBox106 tmp_box106;
					//const ZQ_CNN_Tensor4D_Interface_Base* keyPoint = lnets106[0].GetBlobByName("conv6-3");
					const ZQ_CNN_Tensor4D_Interface_Base* keyPoint = lnets106[0].GetBlobByName("landmark_fc2/BiasAdd");
					const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
					int keypoint_num = keyPoint->GetC() / 2;
					int keyPoint_sliceStep = keyPoint->GetSliceStep();
					float cos_rot = cos(last_rot);
					float sin_rot = sin(last_rot);
					for (int num = 0; num < keypoint_num; num++)
					{
						float tmp_w = cur_size * (keyPoint_ptr[num * 2] - 0.5);
						float tmp_h = cur_size * (keyPoint_ptr[num * 2 + 1] - 0.5);
						tmp_box106.ppoint[num * 2] = cx + tmp_w*cos_rot + tmp_h*sin_rot;
						tmp_box106.ppoint[num * 2 + 1] = cy - tmp_w*sin_rot + tmp_h*cos_rot;
					}

					/**************** Second: Onet ******************/
					float rot1 = _get_rot_of_landmark106(tmp_box106.ppoint, m_pi);
					_get_landmark106_info(tmp_box106.ppoint, cx, cy, min_x, max_x, min_y, max_y);
					cur_w = max_x - min_x;
					cur_h = max_y - min_y;
					cur_size = 1.1*__max(cur_w, cur_h);

					_compute_map(cx, cy, rot1, cur_size, cur_size, onet_size, onet_size, map_x, map_y);

					input.Remap(task_images[0], onet_size, onet_size, 0, 0, map_x, map_y, true, 0);

					onets[0].Forward(task_images[0]);

					const ZQ_CNN_Tensor4D_Interface_Base* prob = onets[0].GetBlobByName("prob1");
					const float* prob_ptr = prob->GetFirstPixelPtr();


					if (prob_ptr[1] < __max(0.2, othresh - 0.3))
					{
						printf("here:lost %f\n", prob_ptr[1]);
						continue;
					}

					ZQ_CNN_BBox106 tmp_box;
					_get_landmark106_info(tmp_box106.ppoint, cx, cy, min_x, max_x, min_y, max_y);
					cur_w = 1.1*(max_x - min_x);
					cur_h = 1.1*(max_y - min_y);
					tmp_box.col1 = cx - 0.5*cur_w;
					tmp_box.col2 = cx + 0.5*cur_w;
					tmp_box.row1 = cy - 0.5*cur_h;
					tmp_box.row2 = cy + 0.5*cur_h;
					tmp_box.score = 2.0;
					tmp_box.exist = true;
					tmp_box106.col1 = tmp_box.col1;
					tmp_box106.col2 = tmp_box.col2;
					tmp_box106.row1 = tmp_box.row1;
					tmp_box106.row2 = tmp_box.row2;
					tmp_box106.score = tmp_box.score;
					tmp_box106.exist = tmp_box.exist;
					good_idx.push_back(i);
					tmp_order.score = 2.0;
					tmp_order.oriOrder = ori_count++;
					boxes.push_back(tmp_box);
					orders.push_back(tmp_order);
					results106_part1.push_back(tmp_box106);
				}


				/**********   Stage 2: detect globally         ************/
				if (cur_key_cooldown <= 0 || results106_part1.size() == 0)
				{
					std::vector<ZQ_CNN_BBox106> tmp_boxes;
					cur_key_cooldown = key_cooldown;
					mtcnn.Find106(input, tmp_boxes);
					for (int j = 0; j < tmp_boxes.size(); j++)
					{
						ZQ_CNN_BBox106& cur_box = tmp_boxes[j];
						float ori_area = (cur_box.col2 - cur_box.col1) * (cur_box.row2 - cur_box.row1);
						float valid_col1 = __max(0, cur_box.col1);
						float valid_col2 = __min(_width - 1, cur_box.col2);
						float valid_row1 = __max(0, cur_box.row1);
						float valid_row2 = __min(_height - 1, cur_box.row2);
						float valid_area = (valid_col2 - valid_col1) * (valid_row2 - valid_row1);
						if (valid_area < ori_area*0.95)
						{
							//printf("here\n");
							continue;
						}
						//printf("here:global\n");
						tmp_order.oriOrder = ori_count++;
						tmp_order.score = tmp_boxes[j].score;
						boxes.push_back(tmp_boxes[j]);
						orders.push_back(tmp_order);
						good_idx.push_back(-1);
					}
				}

				/**********   Stage 3: nms         ************/
				std::vector<int> keep_orders;
				_nms(boxes, orders, keep_orders, 0.3, "Min");
				std::vector<int> old_good_idx = good_idx;
				std::vector<ZQ_CNN_BBox106> old_boxes = boxes;
				good_idx.clear();
				boxes.clear();
				for (int i = 0; i < keep_orders.size(); i++)
				{
					good_idx.push_back(old_good_idx[keep_orders[i]]);
					if (keep_orders[i] >= results106_part1.size())
						boxes.push_back(old_boxes[keep_orders[i]]);
				}

				/**********   Stage 4: get 106 & 240 landmark         ************/
				_Lnet106_stage(boxes, results106_part2);
				if (true)
				{
					results106_part1.insert(results106_part1.end(), results106_part2.begin(), results106_part2.end());

					_refine_landmark106(results106_part1, true);
				}
				else
				{
					_refine_landmark106(results106_part2, true);
					results106_part1.insert(results106_part1.end(), results106_part2.begin(), results106_part2.end());
				}

				results.swap(results106_part1);


				/**********   Stage 5: filtering        ************/
				int cur_box_num = results.size();
				std::vector<std::vector<ZQ_CNN_BBox106> > old_trace(trace);
				trace.clear();
				trace.resize(cur_box_num);
				for (int i = 0; i < cur_box_num; i++)
				{
					trace[i].push_back(results[i]);
					if (good_idx[i] >= 0)
					{
						std::vector<ZQ_CNN_BBox106>& tmp_old_trace = old_trace[good_idx[i]];
						for (int j = 0; j < tmp_old_trace.size() && j < max_trace_num; j++)
							trace[i].push_back(tmp_old_trace[j]);
					}
				}
				_filtering(trace, results);

				if (enable_iou_filter)
				{
					//_filtering_iou(results, good_idx, backup_results);
				}

				/**********   Stage 6: update bbox        ************/
				_recompute_bbox(results);

				/**********   Stage 7: compute headposegaze     ************/
				_headposegaze_stage(results, has_headposegaze);

				if (results.size() == 0)
					is_first_frame = true;
				backup_results = results;
			}
			cur_key_cooldown--;
			return true;
		}

	private:

		static bool _cmp_score(const ZQ_CNN_OrderScore& lsh, const ZQ_CNN_OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}

		static void _nms(const std::vector<ZQ_CNN_BBox106> &ori_boundingBox, const std::vector<ZQ_CNN_OrderScore> &orderScore, std::vector<int>& keep_orders,
			const float overlap_threshold, const std::string& modelname = "Union")
		{
			std::vector<ZQ_CNN_BBox106> boundingBox = ori_boundingBox;
			std::vector<ZQ_CNN_OrderScore> bboxScore = orderScore;
			if (boundingBox.empty() || overlap_threshold >= 1.0)
			{
				return;
			}
			std::vector<int> heros;
			//sort the score

			sort(bboxScore.begin(), bboxScore.end(), _cmp_score);

			int order = 0;
			float IOU = 0;
			float maxX = 0;
			float maxY = 0;
			float minX = 0;
			float minY = 0;
			while (bboxScore.size() > 0)
			{
				order = bboxScore.back().oriOrder;
				bboxScore.pop_back();
				if (order < 0)continue;
				heros.push_back(order);
				boundingBox[order].exist = false;//delete it
				int box_num = boundingBox.size();
				for (int num = 0; num < box_num; num++)
				{
					if (boundingBox[num].exist)
					{
						//the iou
						maxY = __max(boundingBox[num].row1, boundingBox[order].row1);
						maxX = __max(boundingBox[num].col1, boundingBox[order].col1);
						minY = __min(boundingBox[num].row2, boundingBox[order].row2);
						minX = __min(boundingBox[num].col2, boundingBox[order].col2);
						//maxX1 and maxY1 reuse 
						maxX = __max(minX - maxX + 1, 0);
						maxY = __max(minY - maxY + 1, 0);
						//IOU reuse for the area of two bbox
						IOU = maxX * maxY;
						float area1 = boundingBox[num].area;
						float area2 = boundingBox[order].area;
						if (!modelname.compare("Union"))
							IOU = IOU / (area1 + area2 - IOU);
						else if (!modelname.compare("Min"))
						{
							IOU = IOU / __min(area1, area2);
						}
						if (IOU > overlap_threshold)
						{
							boundingBox[num].exist = false;
							for (std::vector<ZQ_CNN_OrderScore>::iterator it = bboxScore.begin(); it != bboxScore.end(); it++)
							{
								if ((*it).oriOrder == num)
								{
									(*it).oriOrder = -1;
									break;
								}
							}
						}
					}
				}
			}


			for (int i = 0; i < heros.size(); i++)
			{
				boundingBox[heros[i]].exist = true;
			}
			//clear exist= false;
			for (int i = boundingBox.size() - 1; i >= 0; i--)
			{
				if (!boundingBox[i].exist)
				{
					boundingBox.erase(boundingBox.begin() + i);
				}
			}
			keep_orders = heros;
		}

		bool _Lnet106_stage(std::vector<ZQ_CNN_BBox106>& thirdBbox, std::vector<ZQ_CNN_BBox106>& resultBbox)
		{
			double t4 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox106> fourthBbox;
			std::vector<ZQ_CNN_BBox106>::iterator it = thirdBbox.begin();
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int l_count = 0;
			for (; it != thirdBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;

					l_count++;
					fourthBbox.push_back(*it);
				}
			}
			std::vector<ZQ_CNN_BBox106> copy_fourthBbox = fourthBbox;
			ZQ_CNN_BBoxUtils::_square_bbox(copy_fourthBbox, input.GetW(), input.GetH());
			for (it = copy_fourthBbox.begin(); it != copy_fourthBbox.end(); ++it)
			{
				int off_x = it->col1;
				int off_y = it->row1;
				int rect_w = it->col2 - off_x;
				int rect_h = it->row2 - off_y;
				src_off_x.push_back(off_x);
				src_off_y.push_back(off_y);
				src_rect_w.push_back(rect_w);
				src_rect_h.push_back(rect_h);
			}

			int batch_size = 1;
			int per_num = ceil((float)l_count / thread_num);
			int need_thread_num = thread_num;
			if (per_num > batch_size)
			{
				need_thread_num = ceil((float)l_count / batch_size);
				per_num = batch_size;
			}

			std::vector<ZQ_CNN_Tensor4D_Interface> task_lnet_images(need_thread_num);
			std::vector<ZQ_CNN_Tensor4D_Interface> task_lnet_images_gray(need_thread_num);
			std::vector<std::vector<int> > task_src_off_x(need_thread_num);
			std::vector<std::vector<int> > task_src_off_y(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_w(need_thread_num);
			std::vector<std::vector<int> > task_src_rect_h(need_thread_num);
			std::vector<std::vector<ZQ_CNN_BBox106> > task_fourthBbox(need_thread_num);

			for (int i = 0; i < need_thread_num; i++)
			{
				int st_id = per_num*i;
				int end_id = __min(l_count, per_num*(i + 1));
				int cur_num = end_id - st_id;
				if (cur_num > 0)
				{
					task_src_off_x[i].resize(cur_num);
					task_src_off_y[i].resize(cur_num);
					task_src_rect_w[i].resize(cur_num);
					task_src_rect_h[i].resize(cur_num);
					task_fourthBbox[i].resize(cur_num);
					for (int j = 0; j < cur_num; j++)
					{
						task_src_off_x[i][j] = src_off_x[st_id + j];
						task_src_off_y[i][j] = src_off_y[st_id + j];
						task_src_rect_w[i][j] = src_rect_w[st_id + j];
						task_src_rect_h[i][j] = src_rect_h[st_id + j];
						task_fourthBbox[i][j].col1 = copy_fourthBbox[st_id + j].col1;
						task_fourthBbox[i][j].col2 = copy_fourthBbox[st_id + j].col2;
						task_fourthBbox[i][j].row1 = copy_fourthBbox[st_id + j].row1;
						task_fourthBbox[i][j].row2 = copy_fourthBbox[st_id + j].row2;
						task_fourthBbox[i][j].area = copy_fourthBbox[st_id + j].area;
						task_fourthBbox[i][j].score = copy_fourthBbox[st_id + j].score;
						task_fourthBbox[i][j].exist = copy_fourthBbox[st_id + j].exist;
					}
				}
			}

			resultBbox.resize(l_count);
			for (int i = 0; i < l_count; i++)
			{
				resultBbox[i].col1 = fourthBbox[i].col1;
				resultBbox[i].col2 = fourthBbox[i].col2;
				resultBbox[i].row1 = fourthBbox[i].row1;
				resultBbox[i].row2 = fourthBbox[i].row2;
				resultBbox[i].score = fourthBbox[i].score;
				resultBbox[i].exist = fourthBbox[i].exist;
				resultBbox[i].area = fourthBbox[i].area;
			}

			for (int pp = 0; pp < need_thread_num; pp++)
			{
				if (task_src_off_x[pp].size() == 0)
					continue;
				if (!input.ResizeBilinearRect(task_lnet_images[pp], lnet106_size, lnet106_size, 0, 0,
					task_src_off_x[pp], task_src_off_y[pp], task_src_rect_w[pp], task_src_rect_h[pp]))
				{
					continue;
				}
				if (!task_lnet_images[pp].ConvertColor_BGR2GRAY(task_lnet_images_gray[pp], 1, 1))
				{
					continue;
				}
				task_lnet_images_gray[pp].MulScalar(128.0f);
				task_lnet_images_gray[pp].AddScalar(127.5f);
				double t31 = omp_get_wtime();
				lnets106[0].Forward(task_lnet_images_gray[pp]);
				double t32 = omp_get_wtime();
				//const ZQ_CNN_Tensor4D_Interface_Base* keyPoint = lnets106[0].GetBlobByName("conv6-3");
				const ZQ_CNN_Tensor4D_Interface_Base* keyPoint = lnets106[0].GetBlobByName("landmark_fc2/BiasAdd");
				const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
				int keypoint_num = keyPoint->GetC() / 2;
				int keyPoint_sliceStep = keyPoint->GetSliceStep();
				for (int i = 0; i < task_fourthBbox[pp].size(); i++)
				{
					for (int num = 0; num < keypoint_num; num++)
					{
						if ((num >= 33 && num < 43) || (num >= 64 && num < 72) || (num >= 84 && num < 104))
						{
							task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
								(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2]/**0.25*/;
							task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
								(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1]/**0.25*/;
						}
						else
						{
							task_fourthBbox[pp][i].ppoint[num * 2] = task_fourthBbox[pp][i].col1 +
								(task_fourthBbox[pp][i].col2 - task_fourthBbox[pp][i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2];
							task_fourthBbox[pp][i].ppoint[num * 2 + 1] = task_fourthBbox[pp][i].row1 +
								(task_fourthBbox[pp][i].row2 - task_fourthBbox[pp][i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num * 2 + 1];
						}
					}
				}
			}

			int count = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				count += task_fourthBbox[i].size();
			}
			resultBbox.resize(count);
			int id = 0;
			for (int i = 0; i < need_thread_num; i++)
			{
				for (int j = 0; j < task_fourthBbox[i].size(); j++)
				{
					memcpy(resultBbox[id].ppoint, task_fourthBbox[i][j].ppoint, sizeof(float) * 212);
					id++;
				}
			}
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Lnet [%d] times \n", l_count);
			if (show_debug_info)
				printf("stage 4: cost %.3f ms\n", 1000 * (t5 - t4));

			return true;
		}

		bool _refine_landmark106(std::vector<ZQ_CNN_BBox106>& resultBbox, bool refine_lnet106)
		{
			if (!refine_lnet106)
				return true;
			const double m_pi = atan(1.0) * 4;
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_Tensor4D_Interface> task_lnet_images(thread_num);
			std::vector<ZQ_CNN_Tensor4D_Interface> task_lnet_images_gray(thread_num);
			for (int pp = 0; pp < resultBbox.size(); pp++)
			{
				float min_x, max_x, min_y, max_y, cx, cy;
				_get_landmark106_info(resultBbox[pp].ppoint, cx, cy, min_x, max_x, min_y, max_y);
				float cur_w = max_x - min_x;
				float cur_h = max_y - min_y;
				float cur_size = 1.2*__max(cur_w, cur_h);
				float half_size = ceil(0.5*cur_size);

				//get rot of 106 landmark
				float cur_rot = _get_rot_of_landmark106(resultBbox[pp].ppoint, m_pi);

				// compute map
				std::vector<float> map_x, map_y;
				_compute_map(cx, cy, cur_rot, cur_size, cur_size, lnet106_size, lnet106_size, map_x, map_y);

				if (!input.Remap(task_lnet_images[0], lnet106_size, lnet106_size, 0, 0, map_x, map_y, true, 0))
				{
					continue;
				}
				task_lnet_images[0].ConvertColor_BGR2GRAY(task_lnet_images_gray[0], 1, 1);
				task_lnet_images_gray[0].MulScalar(128.0f);
				task_lnet_images_gray[0].AddScalar(127.5f);
				lnets106[0].Forward(task_lnet_images_gray[0]);
				//const ZQ_CNN_Tensor4D* keyPoint = lnets106[0].GetBlobByName("conv6-3");
				const ZQ_CNN_Tensor4D_Interface_Base* keyPoint = lnets106[0].GetBlobByName("landmark_fc2/BiasAdd");
				const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
				int keypoint_num = keyPoint->GetC() / 2;
				int keyPoint_sliceStep = keyPoint->GetSliceStep();

				float cos_rot = cos(cur_rot);
				float sin_rot = sin(cur_rot);
				for (int num = 0; num < keypoint_num; num++)
				{
					float tmp_w = cur_size * (keyPoint_ptr[num * 2] - 0.5);
					float tmp_h = cur_size * (keyPoint_ptr[num * 2 + 1] - 0.5);
					resultBbox[pp].ppoint[num * 2] = cx + tmp_w*cos_rot + tmp_h*sin_rot;
					resultBbox[pp].ppoint[num * 2 + 1] = cy - tmp_w*sin_rot + tmp_h*cos_rot;
				}
			}

			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("run refine_Lnet [%d] times, cost %.3f ms\n", resultBbox.size(), 1000 * (t2 - t1));

			return true;
		}

		bool _headposegaze_stage(std::vector<ZQ_CNN_BBox106>& resultBbox, bool enable_headposegaze)
		{
			if (!enable_headposegaze)
			{
				for (int i = 0; i < resultBbox.size(); i++)
					resultBbox[i].has_headposegaze = false;
				return true;
			}
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_Tensor4D_Interface> task_hpg_images(thread_num);
			std::vector<ZQ_CNN_Tensor4D_Interface> task_hpg_images_gray(thread_num);
			for (int pp = 0; pp < resultBbox.size(); pp++)
			{
				const float* cur_pts = resultBbox[pp].ppoint;
				//5 points
				float face5pts[10] =
				{
					0.5f*(cur_pts[72 * 2 + 0] + cur_pts[73 * 2 + 0]), //left eye cx (left of image)
					0.5f*(cur_pts[72 * 2 + 1] + cur_pts[73 * 2 + 1]), //left eye cy (left of image)
					0.5f*(cur_pts[75 * 2 + 0] + cur_pts[76 * 2 + 0]), //right eye cx (right of image)
					0.5f*(cur_pts[75 * 2 + 1] + cur_pts[76 * 2 + 1]), //right eye cy (right of image)
					cur_pts[46 * 2 + 0],		//nose x
					cur_pts[46 * 2 + 1],		//nose y
					cur_pts[84 * 2 + 0],	//left mouth corner x (left of image)
					cur_pts[84 * 2 + 1],	//left mouth corner y (left of image)
					cur_pts[90 * 2 + 0],	//right mouth corner x (right of image)
					cur_pts[90 * 2 + 1]		//right mouth corner y (right of image)
				};
				float transform[6];
				
				ZQ_CNN_FaceCropUtils::CropImage_112x112_translate_scale_roll(input, face5pts, task_hpg_images[pp], transform, -1);

				float sc = transform[0];
				float ss = transform[1];
				float tx = transform[2];
				float ty = transform[5];
				float rot = atan2(ss, sc);
				resultBbox[pp].center_and_rot[0] = tx;
				resultBbox[pp].center_and_rot[1] = ty;
				resultBbox[pp].center_and_rot[2] = rot;

				task_hpg_images[pp].ConvertColor_BGR2GRAY(task_hpg_images_gray[pp],1,1);
				task_hpg_images_gray[pp].MulScalar(128.0f);
				task_hpg_images_gray[pp].AddScalar(127.5f);
				
				headposegaze_nets[0].Forward(task_hpg_images_gray[0]);
				const ZQ_CNN_Tensor4D_Interface_Base* hpg = headposegaze_nets[0].GetBlobByName("headposegaze_fc3/BiasAdd");
				const float* hpg_ptr = hpg->GetFirstPixelPtr();
				memcpy(resultBbox[pp].headposegaze, hpg_ptr, sizeof(float) * 9);
				resultBbox[pp].has_headposegaze = true;

			}

			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("run headposegaze_stage [%d] times, cost %.3f ms\n", resultBbox.size(), 1000 * (t2 - t1));

			return true;
		}

		void _filtering(const std::vector<std::vector<ZQ_CNN_BBox106> >& trace, std::vector<ZQ_CNN_BBox106>& results)
		{
			float reproj_coords[212];
			for (int i = 0; i < results.size(); i++)
			{
				const std::vector<ZQ_CNN_BBox106>& cur_trace = trace[i];
				results[i] = cur_trace[0];
				const ZQ_CNN_BBox106& cur_box = trace[i][0];
				const float ori_thresh_L1 = 0.01f;
				const float ori_thresh_L2 = 0.01f;
				const float ori_thresh_Linf = 0.015f;
				const float reproj_thresh_L1 = 0.005f;
				const float reproj_thresh_L2 = 0.005f;
				const float reproj_thresh_Linf = 0.008f;
				float box_len_sum = cur_box.col2 - cur_box.col1 + cur_box.row2 - cur_box.row1;
				float real_ori_thresh_L1 = ori_thresh_L1*box_len_sum;
				float real_ori_thresh_L2 = ori_thresh_L2*box_len_sum;
				float real_ori_thresh_Linf = ori_thresh_Linf*box_len_sum;
				float real_thresh_L1 = reproj_thresh_L1*box_len_sum;
				float real_thresh_L2 = reproj_thresh_L2*box_len_sum;
				float real_thresh_Linf = reproj_thresh_Linf*box_len_sum;
				float sum_weight = 1.0f;
				int cur_trace_len = cur_trace.size();
				if (cur_trace_len <= 1)
					continue;

				for (int j = 1; j < cur_trace_len; j++)
				{
					double ori_dis_L2 = 0;
					double ori_dis_L1 = 0;
					double ori_dis_Linf = 0;
					double reproj_err_L2 = 0;
					double reproj_err_L1 = 0;
					double reproj_err_Linf = 0;
					float last_weight = exp(-weight_decay*j);
					const ZQ_CNN_BBox106& last_box = trace[i][j];
					_compute_transform(last_box.ppoint, cur_box.ppoint,
						ori_dis_L2, ori_dis_L1, ori_dis_Linf,
						reproj_err_L2, reproj_err_L1, reproj_err_Linf, reproj_coords);

					if (ori_dis_L2 < real_ori_thresh_L2 && ori_dis_L1 < real_ori_thresh_L1 && ori_dis_Linf < real_ori_thresh_Linf
						&& reproj_err_L2 < real_thresh_L2 && reproj_err_L1 < real_thresh_L1 && reproj_err_Linf < real_thresh_Linf)
					{
						//printf("[%d,%d]reproj_err_ratio = %5.2f,%5.2f,%5.2f\n", i, j, reproj_err_L2 / real_thresh_L2,
						//	reproj_err_L1 / real_thresh_L1, reproj_err_Linf / real_thresh_Linf);
						for (int j = 0; j < 212; j++)
						{
							results[i].ppoint[j] += last_box.ppoint[j] * last_weight;
						}
						sum_weight += last_weight;
					}
					else
					{
						//printf("reproj_err = %f\n", reproj_err);
					}
				}

				for (int j = 0; j < 212; j++)
				{
					results[i].ppoint[j] /= sum_weight;
				}
			}
		}

		void _compute_transform(const float last_pts[], const float cur_pts[],
			double& ori_dis_L2, double& ori_dis_L1, double& ori_dis_Linf,
			double& reproj_err_L2, double& reproj_err_L1, double& reproj_err_Linf, float reproj_coords[])
		{
			ZQ_Matrix<double> A(212, 6), b(212, 1), x(6, 1);
			for (int i = 0; i < 106; i++)
			{
				A.SetData(i * 2, 0, last_pts[i * 2]);
				A.SetData(i * 2, 1, last_pts[i * 2 + 1]);
				A.SetData(i * 2, 4, 1);
				b.SetData(i * 2, 0, cur_pts[i * 2]);
				A.SetData(i * 2 + 1, 2, last_pts[i * 2]);
				A.SetData(i * 2 + 1, 3, last_pts[i * 2 + 1]);
				A.SetData(i * 2 + 1, 5, 1);
				b.SetData(i * 2 + 1, 0, cur_pts[i * 2 + 1]);
			}
			if (!ZQ_SVD::Solve(A, x, b))
			{
				ori_dis_L2 = FLT_MAX;
				ori_dis_L1 = FLT_MAX;
				ori_dis_Linf = FLT_MAX;
				reproj_err_L2 = FLT_MAX;
				reproj_err_L1 = FLT_MAX;
				reproj_err_Linf = FLT_MAX;
			}
			else
			{
				bool flag;
				float R[4], T[2];
				R[0] = x.GetData(0, 0, flag);
				R[1] = x.GetData(1, 0, flag);
				R[2] = x.GetData(2, 0, flag);
				R[3] = x.GetData(3, 0, flag);
				T[0] = x.GetData(4, 0, flag);
				T[1] = x.GetData(5, 0, flag);

				ori_dis_L2 = 0;
				ori_dis_L1 = 0;
				ori_dis_Linf = 0;
				reproj_err_L2 = 0;
				reproj_err_L1 = 0;
				reproj_err_Linf = 0;
				for (int i = 0; i < 106; i++)
				{
					reproj_coords[i * 2 + 0] = last_pts[i * 2 + 0] * R[0] + last_pts[i * 2 + 1] * R[1] + T[0];
					reproj_coords[i * 2 + 1] = last_pts[i * 2 + 0] * R[2] + last_pts[i * 2 + 1] * R[3] + T[1];
					double dis_x = fabs(reproj_coords[i * 2 + 0] - cur_pts[i * 2 + 0]);
					double dis_y = fabs(reproj_coords[i * 2 + 1] - cur_pts[i * 2 + 1]);
					reproj_err_L1 += dis_x + dis_y;
					reproj_err_L2 += dis_x*dis_x + dis_y*dis_y;
					reproj_err_Linf = __max(reproj_err_Linf, __max(dis_x, dis_y));
					double ori_dis_x = fabs(last_pts[i * 2 + 0] - cur_pts[i * 2 + 0]);
					double ori_dis_y = fabs(last_pts[i * 2 + 1] - cur_pts[i * 2 + 1]);
					ori_dis_L1 += ori_dis_x + ori_dis_y;
					ori_dis_L2 += ori_dis_x*ori_dis_x + ori_dis_y*ori_dis_y;
					ori_dis_Linf = __max(ori_dis_Linf, __max(ori_dis_x, ori_dis_y));
				}
				reproj_err_L1 /= 212.0;
				reproj_err_L2 = sqrt(reproj_err_L2 / 212.0);
				ori_dis_L1 /= 212.0;
				ori_dis_L2 = sqrt(ori_dis_L2 / 212.0);
			}
		}

		void _filtering_iou(std::vector<ZQ_CNN_BBox106>& results, const std::vector<int>& good_idx, const std::vector<ZQ_CNN_BBox106>& backup_results)
		{
			const float thresh = 1.0f;
			int last_num = backup_results.size();
			for (int bb = 0; bb < results.size(); bb++)
			{
				int last_id = good_idx[bb];
				if (last_id >= 0)
				{
					bool flag = true;
					for (int k = 0; k < 212; k++)
					{
						if (fabs(results[bb].ppoint[k] - backup_results[last_id].ppoint[k]) > thresh)
						{
							flag = false;
							break;
						}
					}
					if (flag)
						results[bb] = backup_results[last_id];
				}
			}
		}

		void _recompute_bbox(std::vector<ZQ_CNN_BBox106>& boxes)
		{
			for (int i = 0; i < boxes.size(); i++)
			{
				float xmin = FLT_MAX;
				float ymin = FLT_MAX;
				float xmax = -FLT_MAX;
				float ymax = -FLT_MAX;
				float cx, cy, max_side;
				for (int j = 0; j < 106; j++)
				{
					float* coords = boxes[i].ppoint;
					xmin = __min(xmin, coords[j * 2]);
					xmax = __max(xmax, coords[j * 2]);
					ymin = __min(ymin, coords[j * 2 + 1]);
					ymax = __max(ymax, coords[j * 2 + 1]);
				}
				cx = 0.5*(xmin + xmax);
				cy = 0.5*(ymin + ymax);
				max_side = __max((xmax - xmin), (ymax - ymin));
				xmin = round(cx - 0.5*max_side);
				xmax = round(cx + 0.5*max_side);
				ymin = round(cy - 0.5*max_side);
				ymax = round(cy + 0.5*max_side);
				boxes[i].col1 = xmin;
				boxes[i].col2 = xmax;
				boxes[i].row1 = ymin;
				boxes[i].row2 = ymax;
				boxes[i].area = (xmax - xmin)*(ymax - ymin);
			}
		}

		static float _get_rot_of_landmark106(const float* pp, double m_pi)
		{
			float eye_cx = 0.25*(pp[52 * 2 + 0] + pp[55 * 2 + 0] + pp[58 * 2 + 0] + pp[61 * 2 + 0]);
			float eye_cy = 0.25*(pp[52 * 2 + 1] + pp[55 * 2 + 1] + pp[58 * 2 + 1] + pp[61 * 2 + 1]);
			float mouth_cx = 0.25*(pp[84 * 2 + 0] + pp[96 * 2 + 0] + pp[100 * 2 + 0] + pp[90 * 2 + 0]);
			float mouth_cy = 0.25*(pp[84 * 2 + 1] + pp[96 * 2 + 1] + pp[100 * 2 + 1] + pp[90 * 2 + 1]);
			float dir_x = mouth_cx - eye_cx;
			float dir_y = mouth_cy - eye_cy;
			float init_rot = 0.5*m_pi - atan2(dir_y, dir_x);
			return init_rot;
		}

		static void _get_landmark106_info(const float* pp, float& cx, float& cy, float& min_x, float& max_x, float& min_y, float& max_y)
		{
			min_x = FLT_MAX;
			min_y = FLT_MAX;
			max_x = -FLT_MAX;
			max_y = -FLT_MAX;
			for (int i = 0; i < 106; i++)
			{
				min_x = __min(min_x, pp[i * 2]);
				max_x = __max(max_x, pp[i * 2]);
				min_y = __min(min_y, pp[i * 2 + 1]);
				max_y = __max(max_y, pp[i * 2 + 1]);
			}
			cx = 0.5*(min_x + max_x);
			cy = 0.5*(min_y + max_y);
		}

		static void _compute_map(float cx, float cy, float rot, float cur_size_w, float cur_size_h,
			int dst_W, int dst_H, std::vector<float>& map_x, std::vector<float>& map_y)
		{
			map_x.resize(dst_H*dst_W);
			map_y.resize(dst_H*dst_W);
			float half_net_size_W = (dst_W - 1) / 2.0;
			float half_net_size_H = (dst_H - 1) / 2.0;
			float sin_rot = sin(rot);
			float cos_rot = cos(rot);
			float step_w = cur_size_w / dst_W;
			float step_h = cur_size_h / dst_H;
			for (int h = 0; h < dst_H; h++)
			{
				for (int w = 0; w < dst_W; w++)
				{
					map_x[h*dst_W + w] = cx + (w - half_net_size_W)*step_w*cos_rot + (h - half_net_size_H)*step_h*sin_rot;
					map_y[h*dst_W + w] = cy - (w - half_net_size_W)*step_w*sin_rot + (h - half_net_size_H)*step_h*cos_rot;
				}
			}
		}
	};
}
#endif
