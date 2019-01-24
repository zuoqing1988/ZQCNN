#ifndef _ZQ_CNN_MTCNN_OLD_H_
#define _ZQ_CNN_MTCNN_OLD_H_
#pragma once
#include "ZQ_CNN_Net.h"
#include "ZQ_CNN_BBoxUtils.h"
#include <omp.h>
namespace ZQ
{
	class ZQ_CNN_MTCNN_old
	{
	public:
		using string = std::string;
		ZQ_CNN_MTCNN_old()
		{
			min_size = 60;
			thresh[0] = 0.6;
			thresh[1] = 0.7;
			thresh[2] = 0.7;
			nms_thresh[0] = 0.6;
			nms_thresh[1] = 0.7;
			nms_thresh[2] = 0.7;
			width = 0;
			height = 0;
			factor = 0.709;
			pnet_overlap_thresh_count = 4;
			pnet_size = 12;
			pnet_stride = 2;
			special_handle_very_big_face = false;
			show_debug_info = false;
		}
		~ZQ_CNN_MTCNN_old()
		{

		}

	private:
		ZQ_CNN_Net pnet, rnet, onet;
		float thresh[3], nms_thresh[3];
		int min_size;
		int width, height;
		float factor;
		int pnet_overlap_thresh_count;
		int pnet_size;
		int pnet_stride;
		bool special_handle_very_big_face;
		std::vector<float> scales;
		std::vector<ZQ_CNN_Tensor4D_NHW_C_Align128bit> pnet_images;
		ZQ_CNN_Tensor4D_NHW_C_Align128bit input, rnet_image, onet_image;
		bool show_debug_info;
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		bool Init(const string& pnet_param, const string& pnet_model, const string& rnet_param, const string& rnet_model,
			const string& onet_param, const string& onet_model)
		{
			return pnet.LoadFrom(pnet_param, pnet_model) && rnet.LoadFrom(rnet_param, rnet_model) && onet.LoadFrom(onet_param, onet_model);
		}

		void SetPara(int w, int h, int min_face_size = 60, float pthresh = 0.6, float rthresh = 0.7, float othresh = 0.7,
			float nms_pthresh = 0.6, float nms_rthresh = 0.7, float nms_othresh = 0.7, float scale_factor = 0.709, 
			int pnet_overlap_thresh_count = 4, int pnet_size = 12, int pnet_stride = 2, bool special_handle_very_big_face = false)
		{
			min_size = __max(pnet_size, min_face_size);
			thresh[0] = __max(0.1, pthresh); thresh[1] = __max(0.1, rthresh); thresh[2] = __max(0.1, othresh);
			nms_thresh[0] = __max(0.1, nms_pthresh); nms_thresh[1] = __max(0.1, nms_rthresh); nms_thresh[2] = __max(0.1, nms_othresh);
			scale_factor = __max(0.5, __min(0.9, scale_factor));
			this->pnet_overlap_thresh_count = __max(0, pnet_overlap_thresh_count);
			this->pnet_size = pnet_size;
			this->pnet_stride = pnet_stride;
			this->special_handle_very_big_face = special_handle_very_big_face;
			if (width != w || height != h || factor != scale_factor)
			{
				scales.clear();
				pnet_images.clear();

				width = w; height = h;
				float minside = __min(width, height);
				int MIN_DET_SIZE = pnet_size;
				float m = (float)MIN_DET_SIZE / min_size;
				minside *= m;
				while (minside > MIN_DET_SIZE)
				{
					scales.push_back(m);
					minside *= factor;
					m *= factor;
				}
				minside = __min(width, height);
				int count = scales.size();
				for (int i = scales.size() - 1; i >= 0; i--)
				{
					if (ceil(scales[i] * minside) <= pnet_size)
					{
						count--;
					}
				}
				if (special_handle_very_big_face)
				{
					if (count > 2)
						count--;

					scales.resize(count);
					if (count > 0)
					{
						float last_size = ceil(scales[count - 1] * minside);
						for (int tmp_size = last_size - 1; tmp_size >= pnet_size + 1; tmp_size -= 2)
						{
							scales.push_back((float)tmp_size / minside);
							count++;
						}
					}
					
					scales.push_back((float)pnet_size / minside);
					count++;
				}
				else
				{
					scales.push_back((float)pnet_size / minside);
					count++;
				}

				pnet_images.resize(count);
				
				
			}
		}

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& results, int num_threads = 1)
		{
			double t1 = omp_get_wtime();
			if (width != _width || height != _height)
				return false;
			std::vector<unsigned char> buffer_bgr(width* height * 3);
			for (int h = 0; h < height; h++)
			{
				const unsigned char* bgr_row = bgr_img + h*_widthStep;
				unsigned char* cur_bgr_row = &buffer_bgr[0] + h*width * 3;
				for (int w = 0; w < width; w++)
				{
					cur_bgr_row[w * 3 + 2] = bgr_row[w * 3 + 2];
					cur_bgr_row[w * 3 + 1] = bgr_row[w * 3 + 1];
					cur_bgr_row[w * 3 + 0] = bgr_row[w * 3 + 0];
				}
			}
			
			if (!input.ConvertFromBGR(&buffer_bgr[0], width, height, width*3))
				return false;
			/*ZQ_CNN_Tensor4D_NHW_C_Align128bit tmp;
			tmp.CopyData(input);
			std::vector<ZQ_CNN_Tensor4D*> in_blobs(4);
			in_blobs[0] = &tmp;
			in_blobs[1] = &tmp;
			in_blobs[2] = &tmp;
			in_blobs[3] = &tmp;
			ZQ_CNN_Forward_SSEUtils::Concat_NCHW(in_blobs, 0, input);*/
			double t2 = omp_get_wtime();
			if(show_debug_info)
				printf("convert cost: %.3f ms\n", 1000 * (t2 - t1));

			ZQ_CNN_OrderScore order;
			pnet.TurnOffShowDebugInfo();
			//pnet.TurnOnShowDebugInfo();
			std::vector<std::vector<ZQ_CNN_BBox> > bounding_boxes(scales.size());
			std::vector<std::vector<ZQ_CNN_OrderScore> > bounding_scores(scales.size());
			const int block_size = 64;
			int stride = pnet_stride;
			int cellsize = pnet_size;
			int border_size = cellsize / stride;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				double t10 = omp_get_wtime();
				if (scales[i] != 1)
				{
					input.ResizeBilinear(pnet_images[i], changedW, changedH, 0, 0);
				}
				
				double t11 = omp_get_wtime();
				if (scales[i] != 1)
					pnet.Forward(pnet_images[i]);
				else
					pnet.Forward(input);
				double t12 = omp_get_wtime();
				if (show_debug_info)
					printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n", 
						i, changedW, changedH, 1000*(t11-t10), 1000 * (t12 - t11));
				double t13 = omp_get_wtime();
				//
				const ZQ_CNN_Tensor4D* score = pnet.GetBlobByName("prob1");
				const ZQ_CNN_Tensor4D* location = pnet.GetBlobByName("conv4-2");
				//for pooling 
				
				int count = 0;
				//score p
				int scoreH = score->GetH();
				int scoreW = score->GetW();
				int scorePixStep = score->GetPixelStep();
				const float *p = score->GetFirstPixelPtr() + 1;
				/*const float* location_ptr = location->GetFirstPixelPtr();
				int locationPixStep = location->GetPixelStep();
				int locationWidthStep = location->GetWidthStep();*/
				//printf("p[0]=%f p[1]=%f\n", p[-1], p[0]);
				if (scoreW <= block_size && scoreH < block_size)
				{
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							if (*p > thresh[0])
							{
								bbox.score = *p;
								order.score = *p;
								order.oriOrder = count;
								bbox.row1 = round((stride*row + 1) / scales[i]);
								bbox.col1 = round((stride*col + 1) / scales[i]);
								bbox.row2 = round((stride*row + 1 + cellsize) / scales[i]);
								bbox.col2 = round((stride*col + 1 + cellsize) / scales[i]);
								/*const float* cur_location_ptr = location_ptr + row*locationWidthStep + col*locationPixStep;
								for (int j = 0; j < 4; j++)
									bbox.regreCoord[j] = cur_location_ptr[j];*/
								bbox.exist = true;
								bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
								bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
									&& (col >= border_size && col < scoreW - border_size);
								bounding_boxes[i].push_back(bbox);
								bounding_scores[i].push_back(order);
								count++;
							}
							p += scorePixStep;
						}
					}
					int before_count = bounding_boxes[i].size();
					ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], 0.5f/*nms_threshold[0]*/, "Union", pnet_overlap_thresh_count);
					int after_count = bounding_boxes[i].size();
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}
				else
				{
					int before_count = 0, after_count = 0;
					int block_H_num = __max(1,scoreH / block_size);
					int block_W_num = __max(1,scoreW / block_size);
					int block_num = block_H_num*block_W_num;
					int width_per_block = scoreW / block_W_num;
					int height_per_block = scoreH / block_H_num;
					std::vector<std::vector<ZQ_CNN_BBox> > tmp_bounding_boxes(block_num);
					std::vector<std::vector<ZQ_CNN_OrderScore> > tmp_bounding_scores(block_num);
					std::vector<int> block_start_w(block_num), block_end_w(block_num);
					std::vector<int> block_start_h(block_num), block_end_h(block_num);
					for (int bh = 0; bh < block_H_num; bh++)
					{
						for (int bw = 0; bw < block_W_num; bw++)
						{
							int bb = bh * block_W_num + bw;
							block_start_w[bb] = (bw == 0) ? 0 : (bw*width_per_block - border_size);
							block_end_w[bb] = (bw == block_num - 1) ? scoreW : ((bw + 1)*width_per_block);
							block_start_h[bb] = (bh == 0) ? 0 : (bh*height_per_block - border_size);
							block_end_h[bb] = (bh == block_num - 1) ? scoreH : ((bh + 1)*height_per_block);
						}
					}
					
					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for(int bb = 0;bb < block_num;bb++)
					{
						count = 0;
						for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
						{
							p = score->GetFirstPixelPtr() + 1 
								+ row*score->GetWidthStep() 
								+ block_start_w[bb]*scorePixStep;
							for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
							{
								if (*p > thresh[0])
								{
									bbox.score = *p;
									order.score = *p;
									order.oriOrder = count;
									bbox.row1 = round((stride*row + 1) / scales[i]);
									bbox.col1 = round((stride*col + 1) / scales[i]);
									bbox.row2 = round((stride*row + 1 + cellsize) / scales[i]) - 1;
									bbox.col2 = round((stride*col + 1 + cellsize) / scales[i]) - 1;
									/*const float* cur_location_ptr = location_ptr + row*locationWidthStep + col*locationPixStep;
									for (int j = 0; j < 4; j++)
										bbox.regreCoord[j] = cur_location_ptr[j];*/
									bbox.exist = true;
									bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
										&& (col >= border_size && col < scoreW - border_size);
									bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
									tmp_bounding_boxes[bb].push_back(bbox);
									tmp_bounding_scores[bb].push_back(order);
									count++;
								}
								p += scorePixStep;
							}
						}
						int tmp_before_count = tmp_bounding_boxes[bb].size();
						ZQ_CNN_BBoxUtils::_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], 0.5f/*nms_threshold[0]*/, "Union", pnet_overlap_thresh_count);	
						int tmp_after_count = tmp_bounding_boxes[bb].size();
						before_count += tmp_before_count;
						after_count += tmp_after_count;
					}
					
					for (int bb = 0; bb < block_num; bb++)
					{
						std::vector<ZQ_CNN_BBox>::iterator it = tmp_bounding_boxes[bb].begin();
						for (; it != tmp_bounding_boxes[bb].end(); it++)
						{
							if ((*it).exist)
							{
								bounding_boxes[i].push_back(*it);
								order.score = (*it).score;
								order.oriOrder = count;
								bounding_scores[i].push_back(order);
								count++;
							}
						}
					}

					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count,after_count);
				}
				
			}
			
			std::vector<ZQ_CNN_BBox> firstBbox;
			std::vector<ZQ_CNN_OrderScore> firstOrderScore;
			int count = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				std::vector<ZQ_CNN_BBox>::iterator it = bounding_boxes[i].begin();
				for (; it != bounding_boxes[i].end(); it++)
				{
					if ((*it).exist)
					{
						firstBbox.push_back(*it);
						order.score = (*it).score;
						order.oriOrder = count;
						firstOrderScore.push_back(order);
						count++;
					}
				}
			}


			//the first stage's nms
			if (count < 1) return false;
			double t15 = omp_get_wtime();
			ZQ_CNN_BBoxUtils::_nms(firstBbox, firstOrderScore, nms_thresh[0], "Union");
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(firstBbox, width, height);
			double t16 = omp_get_wtime();
			if (show_debug_info)
				printf("nms cost: %.3f ms\n", 1000 * (t16 - t15));
			if (show_debug_info)
				printf("first stage candidate count: %d\n", count);
			double t3 = omp_get_wtime();
			if (show_debug_info)
				printf("stage 1: cost %.3f ms\n", 1000 * (t3 - t2));


			/////////////////
			//second stage
			rnet.TurnOffShowDebugInfo();
			//rnet.TurnOnShowDebugInfo();
			count = 0;
			std::vector<ZQ_CNN_BBox>::iterator it = firstBbox.begin();
			std::vector<ZQ_CNN_BBox> secondBbox;
			std::vector<ZQ_CNN_OrderScore> secondScore;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int r_count = 0;
			for (; it != firstBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height || rect_w <= 2 || rect_h <= 2)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						src_off_x.push_back(off_x);
						src_off_y.push_back(off_y);
						src_rect_w.push_back(rect_w);
						src_rect_h.push_back(rect_h);
						r_count++;
						secondBbox.push_back(*it);
					}
				}
			}


			if (!input.ResizeBilinearRect(rnet_image, 24, 24, 0, 0, src_off_x, src_off_y, src_rect_w, src_rect_h))
			{
				return false;
			}
			double t21 = omp_get_wtime();
			rnet.Forward(rnet_image);
			double t22 = omp_get_wtime();
			const ZQ_CNN_Tensor4D* score = rnet.GetBlobByName("prob1");
			const ZQ_CNN_Tensor4D* location = rnet.GetBlobByName("conv5-2");
			const float* score_ptr = score->GetFirstPixelPtr();
			const float* location_ptr = location->GetFirstPixelPtr();
			int score_sliceStep = score->GetSliceStep();
			int location_sliceStep = location->GetSliceStep();
			for(int i = 0; i < r_count;i++)
			{
				if (score_ptr[i*score_sliceStep+1] > thresh[1])
				{
					for (int j = 0; j < 4; j++)
						secondBbox[i].regreCoord[j] = location_ptr[i*location_sliceStep+j];
					secondBbox[i].area = src_rect_w[i] * src_rect_h[i];
					secondBbox[i].score = score_ptr[i*score_sliceStep + 1];
					order.score = secondBbox[i].score;
					order.oriOrder = count++;
					secondScore.push_back(order);
				}
				else
				{
					secondBbox[i].exist = false;
				}
			}
			
			if (count < 1)
				return false;

			
			for (int i = secondBbox.size()-1; i >= 0; i--)
			{
				if (!secondBbox[i].exist)
					secondBbox.erase(secondBbox.begin() + i);
			}
			ZQ_CNN_BBoxUtils::_nms(secondBbox, secondScore, nms_thresh[1], "Union");
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(secondBbox, width, height);

			double t4 = omp_get_wtime();
			if (show_debug_info)
				printf("run Rnet [%d] times (%.3f ms), candidate after nms: %d \n", r_count, 1000*(t22-t21), count);
			if (show_debug_info)
				printf("stage 2: cost %.3f ms\n", 1000 * (t4 - t3));

			//////

			//third stage 
			onet.TurnOffShowDebugInfo();
			count = 0;
			it = secondBbox.begin();
			std::vector<ZQ_CNN_BBox> thirdBbox;
			std::vector<ZQ_CNN_OrderScore> thirdScore;
			double t9 = omp_get_wtime();
			double part1 = 0, part2 = 0;
			int o_count = 0;
			src_off_x.clear(); src_off_y.clear(); src_rect_w.clear(); src_rect_h.clear();
			for (; it != secondBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height || rect_w <= 2 || rect_h <= 2)
					{
						(*it).exist = false;
						continue;
					}
					else
					{
						src_off_x.push_back(off_x);
						src_off_y.push_back(off_y);
						src_rect_w.push_back(rect_w);
						src_rect_h.push_back(rect_h);
						o_count++;
						thirdBbox.push_back(*it);
					}
				}
			}


			if (!input.ResizeBilinearRect(onet_image, 48, 48, 0, 0, src_off_x, src_off_y, src_rect_w, src_rect_h))
			{
				return false;
			}

			double t31 = omp_get_wtime();
			onet.Forward(onet_image);
			double t32 = omp_get_wtime();
			score = onet.GetBlobByName("prob1");
			location = onet.GetBlobByName("conv6-2");
			const ZQ_CNN_Tensor4D* keyPoint = onet.GetBlobByName("conv6-3");
			score_ptr = score->GetFirstPixelPtr();
			location_ptr = location->GetFirstPixelPtr();
			const float* keyPoint_ptr = keyPoint->GetFirstPixelPtr();
			score_sliceStep = score->GetSliceStep();
			location_sliceStep = location->GetSliceStep();
			int keyPoint_sliceStep = keyPoint->GetSliceStep();
			for (int i = 0; i < o_count; i++)
			{
				if (score_ptr[i*score_sliceStep + 1] > thresh[2])
				{
					for (int j = 0; j < 4; j++)
						thirdBbox[i].regreCoord[j] = location_ptr[i*location_sliceStep + j];
					for (int num = 0; num < 5; num++)
					{
						thirdBbox[i].ppoint[num] = thirdBbox[i].col1 + (thirdBbox[i].col2 - thirdBbox[i].col1)*keyPoint_ptr[i*keyPoint_sliceStep + num];
						thirdBbox[i].ppoint[num+5] = thirdBbox[i].row1 + (thirdBbox[i].row2 - thirdBbox[i].row1)*keyPoint_ptr[i*keyPoint_sliceStep + num + 5];
					}
					thirdBbox[i].area = src_rect_w[i] * src_rect_h[i];
					thirdBbox[i].score = score_ptr[i*score_sliceStep + 1];
					order.score = thirdBbox[i].score;
					order.oriOrder = count++;
					thirdScore.push_back(order);
				}
				else
				{
					thirdBbox[i].exist = false;
				}
			}

			
			if (count < 1)
				return false;
			
			for (int i = thirdBbox.size() - 1; i >= 0; i--)
			{
				if (!thirdBbox[i].exist)
					thirdBbox.erase(thirdBbox.begin() + i);
			}
			ZQ_CNN_BBoxUtils::_refine_and_square_bbox(thirdBbox, width, height);
			ZQ_CNN_BBoxUtils::_nms(thirdBbox, thirdScore, nms_thresh[2], "Min");
			double t5 = omp_get_wtime();
			if (show_debug_info)
				printf("run Onet [%d] times (%.3f ms), candidate before nms: %d \n", o_count,1000*(t32-t31), count);
			if (show_debug_info)
				printf("stage 3: cost %.3f ms\n", 1000 * (t5 - t4));
			results = thirdBbox;
			if (show_debug_info)
				printf("final found num : %d, cost: %.3f ms\n", (int)(results.size()), 1000*(t5-t1));
			return true;
		}
	};
}
#endif
