#ifndef _ZQ_CNN_MTCNN_H_
#define _ZQ_CNN_MTCNN_H_
#pragma once
#include "net.h"
#include <algorithm>
#include <omp.h>
#ifndef __max
#define __max(x,y) ((x>y)?(x):(y))
#endif
#ifndef __min
#define __min(x,y) ((x<y)?(x):(y))
#endif
namespace ZQ
{
	class ZQ_CNN_MTCNN_ncnn
	{
	public:
		class ZQ_CNN_BBox
		{
		public:
			float score;
			int row1;
			int col1;
			int row2;
			int col2;
			float area;
			bool exist;
			bool need_check_overlap_count;
			float ppoint[10];
			float regreCoord[4];

			ZQ_CNN_BBox()
			{
				memset(this, 0, sizeof(ZQ_CNN_BBox));
			}

			~ZQ_CNN_BBox() {}

			bool ReadFromBinary(FILE* in)
			{
				if (fread(this, sizeof(ZQ_CNN_BBox), 1, in) != 1)
					return false;
				return true;
			}

			bool WriteBinary(FILE* out) const
			{
				if (fwrite(this, sizeof(ZQ_CNN_BBox), 1, out) != 1)
					return false;
				return true;
			}
		};

		class ZQ_CNN_BBox106
		{
		public:
			float score;
			int row1;
			int col1;
			int row2;
			int col2;
			float area;
			bool exist;
			bool need_check_overlap_count;
			float ppoint[212];
			float regreCoord[4];

			ZQ_CNN_BBox106()
			{
				memset(this, 0, sizeof(ZQ_CNN_BBox106));
			}

			~ZQ_CNN_BBox106() {}

			bool ReadFromBinary(FILE* in)
			{
				if (fread(this, sizeof(ZQ_CNN_BBox106), 1, in) != 1)
					return false;
				return true;
			}

			bool WriteBinary(FILE* out) const
			{
				if (fwrite(this, sizeof(ZQ_CNN_BBox106), 1, out) != 1)
					return false;
				return true;
			}
		};

		class ZQ_CNN_OrderScore
		{
		public:
			float score;
			int oriOrder;

			ZQ_CNN_OrderScore()
			{
				memset(this, 0, sizeof(ZQ_CNN_OrderScore));
			}
		};

		static bool _cmp_score(const ZQ_CNN_OrderScore& lsh, const ZQ_CNN_OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}

		static void _nms(std::vector<ZQ_CNN_BBox> &boundingBox, std::vector<ZQ_CNN_OrderScore> &bboxScore, const float overlap_threshold,
			const std::string& modelname = "Union", int overlap_count_thresh = 0)
		{
			if (boundingBox.empty() || overlap_threshold >= 1.0)
			{
				return;
			}
			std::vector<int> heros;
			std::vector<int> overlap_num;
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
				int cur_overlap = 0;
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
							cur_overlap++;
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

				
				overlap_num.push_back(cur_overlap);
			}
			for (int i = 0; i < heros.size(); i++)
			{
				if (!boundingBox[heros[i]].need_check_overlap_count
					|| overlap_num[i] >= overlap_count_thresh)
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
		}

		static void _refine_and_square_bbox(std::vector<ZQ_CNN_BBox> &vecBbox, const int width, const int height,
			bool square = true)
		{
			float bbw = 0, bbh = 0, bboxSize = 0;
			float h = 0, w = 0;
			float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
			for (std::vector<ZQ_CNN_BBox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
			{
				if ((*it).exist)
				{
					bbh = (*it).row2 - (*it).row1 + 1;
					bbw = (*it).col2 - (*it).col1 + 1;
					y1 = (*it).row1 + (*it).regreCoord[1] * bbh;
					x1 = (*it).col1 + (*it).regreCoord[0] * bbw;
					y2 = (*it).row2 + (*it).regreCoord[3] * bbh;
					x2 = (*it).col2 + (*it).regreCoord[2] * bbw;

					w = x2 - x1 + 1;
					h = y2 - y1 + 1;
					if (square)
					{
						bboxSize = (h > w) ? h : w;
						y1 = y1 + h*0.5 - bboxSize*0.5;
						x1 = x1 + w*0.5 - bboxSize*0.5;
						(*it).row2 = round(y1 + bboxSize - 1);
						(*it).col2 = round(x1 + bboxSize - 1);
						(*it).row1 = round(y1);
						(*it).col1 = round(x1);
					}
					else
					{
						(*it).row2 = round(y1 + h - 1);
						(*it).col2 = round(x1 + w - 1);
						(*it).row1 = round(y1);
						(*it).col1 = round(x1);
					}

					//boundary check
					/*if ((*it).row1 < 0)(*it).row1 = 0;
					if ((*it).col1 < 0)(*it).col1 = 0;
					if ((*it).row2 > height)(*it).row2 = height - 1;
					if ((*it).col2 > width)(*it).col2 = width - 1;*/

					it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
				}
			}
		}

		static void _square_bbox(std::vector<ZQ_CNN_BBox> &vecBbox, const int width, const int height)
		{
			float bbw = 0, bbh = 0, bboxSize = 0;
			float h = 0, w = 0;
			float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
			for (std::vector<ZQ_CNN_BBox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
			{
				if ((*it).exist)
				{
					y1 = (*it).row1;
					x1 = (*it).col1;
					h = (*it).row2 - (*it).row1 + 1;
					w = (*it).col2 - (*it).col1 + 1;
					bboxSize = (h > w) ? h : w;
					y1 = y1 + h*0.5 - bboxSize*0.5;
					x1 = x1 + w*0.5 - bboxSize*0.5;
					(*it).row2 = round(y1 + bboxSize - 1);
					(*it).col2 = round(x1 + bboxSize - 1);
					(*it).row1 = round(y1);
					(*it).col1 = round(x1);

					//boundary check
					/*if ((*it).row1 < 0)(*it).row1 = 0;
					if ((*it).col1 < 0)(*it).col1 = 0;
					if ((*it).row2 > height)(*it).row2 = height - 1;
					if ((*it).col2 > width)(*it).col2 = width - 1;*/

					it->area = (it->row2 - it->row1)*(it->col2 - it->col1);
				}
			}
		}

	public:
		ZQ_CNN_MTCNN_ncnn()
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
			force_run_pnet_multithread = false;
			show_debug_info = false;
			limit_r_num = 0;
			limit_o_num = 0;
			limit_l_num = 0;
		}
		~ZQ_CNN_MTCNN_ncnn()
		{

		}

	private:
		std::vector<ncnn::Net> pnet, rnet, onet, lnet;
		std::vector<ncnn::UnlockedPoolAllocator> g_blob_pool_allocator;
		std::vector<ncnn::UnlockedPoolAllocator> g_workspace_pool_allocator;
		bool has_lnet;
		int thread_num;
		float thresh[3], nms_thresh[3];
		int min_size;
		int width, height;
		float factor;
		int pnet_overlap_thresh_count;
		int pnet_size;
		int pnet_stride;
		int rnet_size;
		int onet_size;
		int lnet_size;
		bool special_handle_very_big_face;
		bool do_landmark;
		float early_accept_thresh;
		float nms_thresh_per_scale;
		bool force_run_pnet_multithread;
		std::vector<float> scales;
		std::vector<ncnn::Mat> pnet_images;
		ncnn::Mat input, rnet_image, onet_image;
		bool show_debug_info;
		int limit_r_num;
		int limit_o_num;
		int limit_l_num;
		
	public:
		void TurnOnShowDebugInfo() { show_debug_info = true; }
		void TurnOffShowDebugInfo() { show_debug_info = false; }
		void SetLimit(int limit_r = 0, int limit_o = 0, int limit_l = 0)
		{
			limit_r_num = limit_r;
			limit_o_num = limit_o;
			limit_l_num = limit_l;
		}

	private:
		static bool _load(ncnn::Net& net, const std::string& param, const std::string& model)
		{
			if (-1 == net.load_param(param.c_str()))
				return false;
			if (-1 == net.load_model(model.c_str()))
				return false;
			return true;
		}

		static bool _roi(const ncnn::Mat& input, ncnn::Mat& output, int off_x, int off_y, int width, int height)
		{
			if (off_x >= 0 && off_y >= 0 && width > 0 && height > 0 && off_x + width <= input.w && off_y + height <= input.h)
			{
				copy_cut_border(input, output, off_y, input.h - off_y - height, off_x, input.w - off_x - width);
				return true;
			}
			else
				return false;
		}

	public:
		bool Init(const std::string& pnet_param, const std::string& pnet_model, 
			const std::string& rnet_param, const std::string& rnet_model,
			const std::string& onet_param, const std::string& onet_model, int thread_num = 1,
			bool has_lnet = false, const std::string& lnet_param = "", const std::string& lnet_model = "")
		{
			if (thread_num < 1)
				force_run_pnet_multithread = true;
			else
				force_run_pnet_multithread = false;
			thread_num = __max(1, thread_num);
			pnet.resize(thread_num);
			rnet.resize(thread_num);
			onet.resize(thread_num);
			this->has_lnet = has_lnet;
			if (has_lnet)
			{
				lnet.resize(thread_num);
			}
			
			g_blob_pool_allocator.resize(thread_num);
			g_workspace_pool_allocator.resize(thread_num);
			
			bool ret = true;
			for (int i = 0; i < thread_num; i++)
			{
				ret = _load(pnet[i], pnet_param, pnet_model)
					&& _load(rnet[i], rnet_param, rnet_model)
					&& _load(onet[i], onet_param, onet_model);
				if (has_lnet && ret)
					ret = _load(lnet[i], lnet_param, lnet_model);
				if (!ret)
					break;
			}
			if (!ret)
			{
				pnet.clear();
				rnet.clear();
				onet.clear();
				if (has_lnet)
					lnet.clear();
				this->thread_num = 0;
			}
			else
				this->thread_num = thread_num;
			for (int i = 0; i < thread_num; i++)
			{
				g_blob_pool_allocator[i].clear();
				g_workspace_pool_allocator[i].clear();
			}
			return ret;
		}

		void SetPara(int w, int h, int min_face_size = 60, float pthresh = 0.6, float rthresh = 0.7, float othresh = 0.7,
			float nms_pthresh = 0.6, float nms_rthresh = 0.7, float nms_othresh = 0.7, float scale_factor = 0.709,
			int pnet_overlap_thresh_count = 4, int pnet_size = 12, int pnet_stride = 2, bool special_handle_very_big_face = false,
			bool do_landmark = true, float early_accept_thresh = 1.00)
		{
			min_size = __max(pnet_size, min_face_size);
			thresh[0] = __max(0.1, pthresh); thresh[1] = __max(0.1, rthresh); thresh[2] = __max(0.1, othresh);
			nms_thresh[0] = __max(0.1, nms_pthresh); nms_thresh[1] = __max(0.1, nms_rthresh); nms_thresh[2] = __max(0.1, nms_othresh);
			scale_factor = __max(0.5, __min(0.97, scale_factor));
			this->pnet_overlap_thresh_count = __max(0, pnet_overlap_thresh_count);
			this->pnet_size = pnet_size;
			this->pnet_stride = pnet_stride;
			this->special_handle_very_big_face = special_handle_very_big_face;
			this->do_landmark = do_landmark;
			this->early_accept_thresh = early_accept_thresh;
			if (pnet_size == 20 && pnet_stride == 4)
				nms_thresh_per_scale = 0.45;
			else
				nms_thresh_per_scale = 0.495;
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

		bool Find(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& results)
		{
			double t1 = omp_get_wtime();
			std::vector<ZQ_CNN_BBox> firstBbox, secondBbox, thirdBbox;
			if (!_Pnet_stage(bgr_img, _width, _height, _widthStep, firstBbox))
				return false;
			//results = firstBbox;
			//return true;
			if (limit_r_num > 0)
			{
				_select(firstBbox, limit_r_num, _width, _height);
			}

			double t2 = omp_get_wtime();
			if (!_Rnet_stage(firstBbox, secondBbox))
				return false;
			//results = secondBbox;
			//return true;

			if (limit_o_num > 0)
			{
				_select(secondBbox, limit_o_num, _width, _height);
			}


			double t3 = omp_get_wtime();
			if (!_Onet_stage(secondBbox, results))
				return false;

			double t4 = omp_get_wtime();
			if (show_debug_info)
			{
				printf("final found num: %d\n", (int)results.size());
				printf("total cost: %.3f ms (P: %.3f ms, R: %.3f ms, O: %.3f ms)\n",
					1000 * (t4 - t1), 1000 * (t2 - t1), 1000 * (t3 - t2), 1000 * (t4 - t3));
			}
			return true;
		}

	private:
		void _compute_Pnet_single_thread(std::vector<std::vector<float> >& maps,
			std::vector<int>& mapH, std::vector<int>& mapW)
		{
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			maps.resize(scale_num);
			for (int i = 0; i < scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}

			for (int i = 0; i < scale_num; i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				float cur_scale_x = (float)width / changedW;
				float cur_scale_y = (float)height / changedH;
				double t10 = omp_get_wtime();
				if (scales[i] != 1)
				{
					ncnn::resize_bilinear(input, pnet_images[i], changedW, changedH);
				}

				double t11 = omp_get_wtime();
				ncnn::Extractor ex = pnet[0].create_extractor();
				ex.set_light_mode(true);
				ex.set_blob_allocator(&g_blob_pool_allocator[0]);
				ex.set_workspace_allocator(&g_workspace_pool_allocator[0]);
				ex.set_num_threads(1);
				if (scales[i] == 1)
					ex.input("data", input);
				else
					ex.input("data", pnet_images[i]);
				ncnn::Mat score, location;
				ex.extract("prob1", score);
				ex.extract("conv4-2", location);
				double t12 = omp_get_wtime();
				if (show_debug_info)
					printf("Pnet [%d]: resolution [%dx%d], resize:%.3f ms, cost:%.3f ms\n",
						i, changedW, changedH, 1000 * (t11 - t10), 1000 * (t12 - t11));
				//score p
				float *p = score.channel(1);
				int scoreH = score.h;
				int scoreW = score.w;
				for (int row = 0; row < scoreH; row++)
				{
					for (int col = 0; col < scoreW; col++)
					{
						if (row < mapH[i] && col < mapW[i])
							maps[i][row*mapW[i] + col] = *p;
						p++;
					}
				}
			}
		}
		void _compute_Pnet_multi_thread(std::vector<std::vector<float> >& maps,
			std::vector<int>& mapH, std::vector<int>& mapW)
		{
			if (thread_num <= 1)
			{
				for (int i = 0; i < scales.size(); i++)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales[i] != 1)
					{
						ncnn::resize_bilinear(input, pnet_images[i], changedW, changedH);
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num)
				for (int i = 0; i < scales.size(); i++)
				{
					int changedH = (int)ceil(height*scales[i]);
					int changedW = (int)ceil(width*scales[i]);
					if (changedH < pnet_size || changedW < pnet_size)
						continue;
					if (scales[i] != 1)
					{
						ncnn::resize_bilinear(input, pnet_images[i], changedW, changedH);
					}
				}
			}
			int scale_num = 0;
			for (int i = 0; i < scales.size(); i++)
			{
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				scale_num++;
				mapH.push_back((changedH - pnet_size) / pnet_stride + 1);
				mapW.push_back((changedW - pnet_size) / pnet_stride + 1);
			}
			maps.resize(scale_num);
			for (int i = 0; i < scale_num; i++)
			{
				maps[i].resize(mapH[i] * mapW[i]);
			}

			std::vector<int> task_rect_off_x;
			std::vector<int> task_rect_off_y;
			std::vector<int> task_rect_width;
			std::vector<int> task_rect_height;
			std::vector<float> task_scale;
			std::vector<int> task_scale_id;

			int stride = pnet_stride;
			const int block_size = 64 * stride;
			int cellsize = pnet_size;
			int border_size = cellsize - stride;
			int overlap_border_size = cellsize / stride;
			int jump_size = block_size - border_size;
			for (int i = 0; i < scales.size(); i++)
			{
				int changeH = (int)ceil(height*scales[i]);
				int changeW = (int)ceil(width*scales[i]);
				if (changeH < pnet_size || changeW < pnet_size)
					continue;
				int block_H_num = 0;
				int block_W_num = 0;
				int start = 0;
				while (start < changeH)
				{
					block_H_num++;
					if (start + block_size >= changeH)
						break;
					start += jump_size;
				}
				start = 0;
				while (start < changeW)
				{
					block_W_num++;
					if (start + block_size >= changeW)
						break;
					start += jump_size;
				}
				for (int s = 0; s < block_H_num; s++)
				{
					for (int t = 0; t < block_W_num; t++)
					{
						int rect_off_x = t * jump_size;
						int rect_off_y = s * jump_size;
						int rect_width = __min(changeW, rect_off_x + block_size) - rect_off_x;
						int rect_height = __min(changeH, rect_off_y + block_size) - rect_off_y;
						if (rect_width >= cellsize && rect_height >= cellsize)
						{
							task_rect_off_x.push_back(rect_off_x);
							task_rect_off_y.push_back(rect_off_y);
							task_rect_width.push_back(rect_width);
							task_rect_height.push_back(rect_height);
							task_scale.push_back(scales[i]);
							task_scale_id.push_back(i);
						}
					}
				}
			}

			//
			int task_num = task_scale.size();
			std::vector<ncnn::Mat> task_pnet_images(thread_num);

			if (thread_num <= 1)
			{
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
					if (scale_id == 0 && scales[0] == 1)
					{
						if (!_roi(input, task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height))
							continue;
					}
					else
					{
						if (!_roi(pnet_images[scale_id], task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height))
							continue;
					}

					ncnn::Extractor ex = pnet[thread_id].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[0]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[0]);
					ex.set_num_threads(1);
					ex.input("data", task_pnet_images[thread_id]);
					ncnn::Mat score, location;
					ex.extract("prob1", score);
					ex.extract("conv4-2", location);

					//score p
					float *p = score.channel(1);
					int scoreH = score.h;
					int scoreW = score.w;

					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p++;
						}
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num)
				for (int i = 0; i < task_num; i++)
				{
					int thread_id = omp_get_thread_num();
					int scale_id = task_scale_id[i];
					float cur_scale = task_scale[i];
					int i_rect_off_x = task_rect_off_x[i];
					int i_rect_off_y = task_rect_off_y[i];
					int i_rect_width = task_rect_width[i];
					int i_rect_height = task_rect_height[i];
					if (scale_id == 0 && scales[0] == 1)
					{
						if (!_roi(input, task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height))
							continue;
					}
					else
					{
						if (!_roi(pnet_images[scale_id], task_pnet_images[thread_id],
							i_rect_off_x, i_rect_off_y, i_rect_width, i_rect_height))
							continue;
					}

					ncnn::Extractor ex = pnet[thread_id].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[thread_id]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[thread_id]);
					ex.input("data", task_pnet_images[thread_id]);
					ncnn::Mat score, location;
					ex.extract("prob1", score);
					ex.extract("conv4-2", location);

					//score p
					float *p = score.channel(1);
					int scoreH = score.h;
					int scoreW = score.w;

					ZQ_CNN_BBox bbox;
					ZQ_CNN_OrderScore order;
					for (int row = 0; row < scoreH; row++)
					{
						for (int col = 0; col < scoreW; col++)
						{
							int real_row = row + i_rect_off_y / stride;
							int real_col = col + i_rect_off_x / stride;
							if (real_row < mapH[scale_id] && real_col < mapW[scale_id])
								maps[scale_id][real_row*mapW[scale_id] + real_col] = *p;

							p++;
						}
					}
				}
			}
		}

		bool _Pnet_stage(const unsigned char* bgr_img, int _width, int _height, int _widthStep, std::vector<ZQ_CNN_BBox>& firstBbox)
		{
			if (thread_num <= 0)
				return false;

			double t1 = omp_get_wtime();
			firstBbox.clear();
			if (width != _width || height != _height)
				return false;
			input = ncnn::Mat::from_pixels(bgr_img, ncnn::Mat::PIXEL_BGR, _width, _height);
			float mean_vals[3] = { 127.5,127.5,127.5 };
			float norm_vals[3] = { 1.0 / 128,1.0 / 128,1.0 / 128 };
			input.substract_mean_normalize(mean_vals, norm_vals);
			double t2 = omp_get_wtime();
			if (show_debug_info)
				printf("convert cost: %.3f ms\n", 1000 * (t2 - t1));

			std::vector<std::vector<float> > maps;
			std::vector<int> mapH;
			std::vector<int> mapW;
			if (thread_num == 1 && !force_run_pnet_multithread)
			{
				_compute_Pnet_single_thread(maps, mapH, mapW);
			}
			else
			{
				_compute_Pnet_multi_thread(maps, mapH, mapW);
			}
			ZQ_CNN_OrderScore order;
			std::vector<std::vector<ZQ_CNN_BBox> > bounding_boxes(scales.size());
			std::vector<std::vector<ZQ_CNN_OrderScore> > bounding_scores(scales.size());
			const int block_size = 32;
			int stride = pnet_stride;
			int cellsize = pnet_size;
			int border_size = cellsize / stride;

			for (int i = 0; i < maps.size(); i++)
			{
				double t13 = omp_get_wtime();
				int changedH = (int)ceil(height*scales[i]);
				int changedW = (int)ceil(width*scales[i]);
				if (changedH < pnet_size || changedW < pnet_size)
					continue;
				float cur_scale_x = (float)width / changedW;
				float cur_scale_y = (float)height / changedH;

				int count = 0;
				//score p
				int scoreH = mapH[i];
				int scoreW = mapW[i];
				const float *p = &maps[i][0];
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
								bbox.row1 = stride*row;
								bbox.col1 = stride*col;
								bbox.row2 = stride*row + cellsize;
								bbox.col2 = stride*col + cellsize;
								bbox.exist = true;
								bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
								bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
									&& (col >= border_size && col < scoreW - border_size);
								bounding_boxes[i].push_back(bbox);
								bounding_scores[i].push_back(order);
								count++;
							}
							p++;
						}
					}
					int before_count = bounding_boxes[i].size();
					_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
					int after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}
				else
				{
					int before_count = 0, after_count = 0;
					int block_H_num = __max(1, scoreH / block_size);
					int block_W_num = __max(1, scoreW / block_size);
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
					int chunk_size = 1;
					if (thread_num <= 1)
					{
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}
					else
					{
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(thread_num)
						for (int bb = 0; bb < block_num; bb++)
						{
							ZQ_CNN_BBox bbox;
							ZQ_CNN_OrderScore order;
							int count = 0;
							for (int row = block_start_h[bb]; row < block_end_h[bb]; row++)
							{
								const float* p = &maps[i][0] + row*scoreW + block_start_w[bb];
								for (int col = block_start_w[bb]; col < block_end_w[bb]; col++)
								{
									if (*p > thresh[0])
									{
										bbox.score = *p;
										order.score = *p;
										order.oriOrder = count;
										bbox.row1 = stride*row;
										bbox.col1 = stride*col;
										bbox.row2 = stride*row + cellsize;
										bbox.col2 = stride*col + cellsize;
										bbox.exist = true;
										bbox.need_check_overlap_count = (row >= border_size && row < scoreH - border_size)
											&& (col >= border_size && col < scoreW - border_size);
										bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
										tmp_bounding_boxes[bb].push_back(bbox);
										tmp_bounding_scores[bb].push_back(order);
										count++;
									}
									p++;
								}
							}
							int tmp_before_count = tmp_bounding_boxes[bb].size();
							_nms(tmp_bounding_boxes[bb], tmp_bounding_scores[bb], nms_thresh_per_scale, "Union", pnet_overlap_thresh_count);
							int tmp_after_count = tmp_bounding_boxes[bb].size();
							before_count += tmp_before_count;
							after_count += tmp_after_count;
						}
					}

					count = 0;
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

					//ZQ_CNN_BBoxUtils::_nms(bounding_boxes[i], bounding_scores[i], nms_thresh_per_scale, "Union", 0);
					after_count = bounding_boxes[i].size();
					for (int j = 0; j < after_count; j++)
					{
						ZQ_CNN_BBox& bbox = bounding_boxes[i][j];
						bbox.row1 = round(bbox.row1 *cur_scale_y);
						bbox.col1 = round(bbox.col1 *cur_scale_x);
						bbox.row2 = round(bbox.row2 *cur_scale_y);
						bbox.col2 = round(bbox.col2 *cur_scale_x);
						bbox.area = (bbox.row2 - bbox.row1)*(bbox.col2 - bbox.col1);
					}
					double t14 = omp_get_wtime();
					if (show_debug_info)
						printf("nms cost: %.3f ms, (%d-->%d)\n", 1000 * (t14 - t13), before_count, after_count);
				}

			}

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
			_nms(firstBbox, firstOrderScore, nms_thresh[0], "Union", 0);
			_refine_and_square_bbox(firstBbox, width, height, true);
			double t16 = omp_get_wtime();
			if (show_debug_info)
				printf("nms cost: %.3f ms\n", 1000 * (t16 - t15));
			if (show_debug_info)
				printf("first stage candidate count: %d\n", count);
			double t3 = omp_get_wtime();
			if (show_debug_info)
				printf("stage 1: cost %.3f ms\n", 1000 * (t3 - t2));
			return true;
		}

		bool _Rnet_stage(std::vector<ZQ_CNN_BBox>& firstBbox, std::vector<ZQ_CNN_BBox>& secondBbox)
		{
			double t3 = omp_get_wtime();
			secondBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = firstBbox.begin();
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
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
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

			secondBbox.resize(r_count);

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < r_count; pp++)
				{
					ncnn::Mat task_rnet_images;
					ncnn::Mat tempIm;
					copy_cut_border(input, tempIm, src_off_y[pp], input.h - src_off_y[pp] - src_rect_h[pp], src_off_x[pp], input.w - src_off_x[pp] - src_rect_w[pp]);
					resize_bilinear(tempIm, task_rnet_images, 24, 24);
					ncnn::Extractor ex = rnet[0].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[0]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[0]);
					ex.set_num_threads(1);
					ex.input("data", task_rnet_images);
					ncnn::Mat score, bbox, keyPoint;
					ex.extract("prob1", score);
					ex.extract("conv5-2", bbox);
					if ((float)score[1] > thresh[1])
					{
						for (int j = 0; j < 4; j++)
							secondBbox[pp].regreCoord[j] = (float)bbox[j];
						secondBbox[pp].area = src_rect_w[pp] * src_rect_h[pp];
						secondBbox[pp].score = (float)score[1];
					}
					else
					{
						secondBbox[pp].exist = false;
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < r_count; pp++)
				{
					int thread_id = omp_get_thread_num();
					ncnn::Mat task_rnet_images;
					ncnn::Mat tempIm;
					copy_cut_border(input, tempIm, src_off_y[pp], input.h - src_off_y[pp] - src_rect_h[pp], src_off_x[pp], input.w - src_off_x[pp] - src_rect_w[pp]);
					resize_bilinear(tempIm, task_rnet_images, 24, 24);
					ncnn::Extractor ex = rnet[thread_id].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[thread_id]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[thread_id]);
					ex.set_num_threads(1);
					ex.input("data", task_rnet_images);
					ncnn::Mat score, bbox, keyPoint;
					ex.extract("prob1", score);
					ex.extract("conv5-2", bbox);
					if ((float)score[1] > thresh[1])
					{
						for (int j = 0; j < 4; j++)
							secondBbox[pp].regreCoord[j] = (float)bbox[j];
						secondBbox[pp].area = src_rect_w[pp] * src_rect_h[pp];
						secondBbox[pp].score = (float)score[1];
					}
					else
					{
						secondBbox[pp].exist = false;
					}
				}
			}

			for (int i = secondBbox.size() - 1; i >= 0; i--)
			{
				if (!secondBbox[i].exist)
					secondBbox.erase(secondBbox.begin() + i);
			}

			int count = secondBbox.size();
			secondScore.resize(count);
			for (int i = 0; i < count; i++)
			{
				secondScore[i].score = secondBbox[i].score;
				secondScore[i].oriOrder = i;
			}

			//_nms(secondBbox, secondScore, nms_thresh[1], "Union");
			_nms(secondBbox, secondScore, nms_thresh[1], "Min");
			_refine_and_square_bbox(secondBbox, width, height, true);
			count = secondBbox.size();

			double t4 = omp_get_wtime();
			if (show_debug_info)
				printf("run Rnet [%d] times, candidate after nms: %d \n", r_count, count);
			if (show_debug_info)
				printf("stage 2: cost %.3f ms\n", 1000 * (t4 - t3));

			return true;
		}

		bool _Onet_stage(std::vector<ZQ_CNN_BBox>& secondBbox, std::vector<ZQ_CNN_BBox>& thirdBbox)
		{
			double t3 = omp_get_wtime();
			thirdBbox.clear();
			std::vector<ZQ_CNN_BBox>::iterator it = secondBbox.begin();
			std::vector<ZQ_CNN_OrderScore> thirdScore;
			std::vector<int> src_off_x, src_off_y, src_rect_w, src_rect_h;
			int o_count = 0;
			for (; it != secondBbox.end(); it++)
			{
				if ((*it).exist)
				{
					int off_x = it->col1;
					int off_y = it->row1;
					int rect_w = it->col2 - off_x;
					int rect_h = it->row2 - off_y;
					if (/*off_x < 0 || off_x + rect_w > width || off_y < 0 || off_y + rect_h > height ||*/ rect_w <= 0.5*min_size || rect_h <= 0.5*min_size)
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

			thirdBbox.resize(o_count);

			if (thread_num <= 1)
			{
				for (int pp = 0; pp < o_count; pp++)
				{
					ncnn::Mat task_onet_images;
					ncnn::Mat tempIm;
					copy_cut_border(input, tempIm, src_off_y[pp], input.h - src_off_y[pp] - src_rect_h[pp], src_off_x[pp],
						input.w - src_off_x[pp] - src_rect_w[pp]);
					resize_bilinear(tempIm, task_onet_images, 48, 48);
					ncnn::Extractor ex = onet[0].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[0]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[0]);
					ex.set_num_threads(1);
					ex.input("data", task_onet_images);
					ncnn::Mat score, bbox, keyPoint;
					ex.extract("prob1", score);
					ex.extract("conv6-2", bbox);
					if ((float)score[1] > thresh[2])
					{
						for (int j = 0; j < 4; j++)
							thirdBbox[pp].regreCoord[j] = (float)bbox[j];
						thirdBbox[pp].area = src_rect_w[pp] * src_rect_h[pp];
						thirdBbox[pp].score = (float)score[1];
					}
					else
					{
						thirdBbox[pp].exist = false;
					}
				}
			}
			else
			{
#pragma omp parallel for num_threads(thread_num) schedule(dynamic,1)
				for (int pp = 0; pp < o_count; pp++)
				{
					int thread_id = omp_get_thread_num();
					ncnn::Mat task_onet_images;
					ncnn::Mat tempIm;
					copy_cut_border(input, tempIm, src_off_y[pp], input.h - src_off_y[pp] - src_rect_h[pp], src_off_x[pp],
						input.w - src_off_x[pp] - src_rect_w[pp]);
					resize_bilinear(tempIm, task_onet_images, 48, 48);
					ncnn::Extractor ex = onet[thread_id].create_extractor();
					ex.set_light_mode(true);
					ex.set_blob_allocator(&g_blob_pool_allocator[thread_id]);
					ex.set_workspace_allocator(&g_workspace_pool_allocator[thread_id]);
					ex.set_num_threads(1);
					ex.input("data", task_onet_images);
					ncnn::Mat score, bbox, keyPoint;
					ex.extract("prob1", score);
					ex.extract("conv6-2", bbox);
					if ((float)score[1] > thresh[2])
					{
						for (int j = 0; j < 4; j++)
							thirdBbox[pp].regreCoord[j] = (float)bbox[j];
						thirdBbox[pp].area = src_rect_w[pp] * src_rect_h[pp];
						thirdBbox[pp].score = (float)score[1];
					}
					else
					{
						thirdBbox[pp].exist = false;
					}
				}
			}

			for (int i = thirdBbox.size() - 1; i >= 0; i--)
			{
				if (!thirdBbox[i].exist)
					thirdBbox.erase(thirdBbox.begin() + i);
			}

			int count = thirdBbox.size();
			thirdScore.resize(count);
			for (int i = 0; i < count; i++)
			{
				thirdScore[i].score = thirdScore[i].score;
				thirdScore[i].oriOrder = i;
			}

			_nms(thirdBbox, thirdScore, nms_thresh[2], "Min");
			_refine_and_square_bbox(thirdBbox, width, height, true);
			count = thirdBbox.size();

			double t4 = omp_get_wtime();
			if (show_debug_info)
				printf("run Onet [%d] times, candidate after nms: %d \n", o_count, count);
			if (show_debug_info)
				printf("stage 3: cost %.3f ms\n", 1000 * (t4 - t3));

			return true;
		}

		void _select(std::vector<ZQ_CNN_BBox>& bbox, int limit_num, int width, int height)
		{
			int in_num = bbox.size();
			if (limit_num >= in_num)
				return;
			bbox.resize(limit_num);
		}
	};
}
#endif
