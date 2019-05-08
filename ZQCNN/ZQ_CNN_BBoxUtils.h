#ifndef _ZQ_CNN_BBOX_UTILS_H_
#define _ZQ_CNN_BBOX_UTILS_H_
#pragma once

#include "ZQ_CNN_BBox.h"
#include <string>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

namespace ZQ
{
	class ZQ_CNN_BBoxUtils
	{
	public:
		enum PriorBoxCodeType
		{
			PriorBoxCodeType_CORNER = 0,
			PriorBoxCodeType_CORNER_SIZE,
			PriorBoxCodeType_CENTER_SIZE
		};

		static bool _cmp_score(const ZQ_CNN_OrderScore& lsh, const ZQ_CNN_OrderScore& rsh)
		{
			return lsh.score < rsh.score;
		}

		static void _nms(std::vector<ZQ_CNN_BBox> &boundingBox, std::vector<ZQ_CNN_OrderScore> &bboxScore, const float overlap_threshold, 
			const std::string& modelname = "Union", int overlap_count_thresh = 0, int thread_num = 1)
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
				if (thread_num == 1)
				{
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
				}
				else
				{
					int chunk_size = ceil(box_num / thread_num);
#pragma omp parallel for schedule(static, chunk_size) num_threads(thread_num)
					for (int num = 0; num < box_num; num++)
					{
						if (boundingBox.at(num).exist)
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
								boundingBox.at(num).exist = false;
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
				overlap_num.push_back(cur_overlap);
			}
			for (int i = 0; i < heros.size(); i++)
			{
				if(!boundingBox[heros[i]].need_check_overlap_count 
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
						float scale_h = h*it->scale_y;
						float scale_w = w*it->scale_x;
						bboxSize = (scale_h > scale_w) ? scale_h : scale_w;
						y1 = y1 + h*0.5 - bboxSize/it->scale_y*0.5;
						x1 = x1 + w*0.5 - bboxSize/it->scale_x*0.5;
						(*it).row2 = round(y1 + bboxSize / it->scale_y - 1);
						(*it).col2 = round(x1 + bboxSize / it->scale_x - 1);
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
					h = (*it).row2 - (*it).row1 + 1;
					w = (*it).col2 - (*it).col1 + 1;
					y1 = (*it).row1;
					x1 = (*it).col1;
					float scale_h = h*it->scale_y;
					float scale_w = w*it->scale_x;
					bboxSize = (scale_h > scale_w) ? scale_h : scale_w;
					y1 = y1 + h*0.5 - bboxSize / it->scale_y*0.5;
					x1 = x1 + w*0.5 - bboxSize / it->scale_x*0.5;
					(*it).row2 = round(y1 + bboxSize / it->scale_y - 1);
					(*it).col2 = round(x1 + bboxSize / it->scale_x - 1);
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

		static bool DecodeBBoxesAll(const std::vector<ZQ_CNN_LabelBBox>& all_loc_preds,
			const std::vector<ZQ_CNN_NormalizedBBox>& prior_bboxes,
			const std::vector<std::vector<float> >& prior_variances,
			const int num, const bool share_location,
			const int num_loc_classes, const int background_label_id,
			const PriorBoxCodeType code_type, const bool variance_encoded_in_target,
			const bool clip, std::vector<ZQ_CNN_LabelBBox>* all_decode_bboxes) 
		{
			if (all_loc_preds.size() != num)
				return false;
			all_decode_bboxes->clear();
			all_decode_bboxes->resize(num);
			for (int i = 0; i < num; ++i) {
				// Decode predictions into bboxes.
				ZQ_CNN_LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
				for (int c = 0; c < num_loc_classes; ++c) 
				{
					int label = share_location ? -1 : c;
					if (label == background_label_id) {
						// Ignore background class.
						continue;
					}
					if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) 
					{
						// Something bad happened if there are no predictions for current label.
						//LOG(FATAL) << "Could not find location predictions for label " << label;
					}
					const std::vector<ZQ_CNN_NormalizedBBox>& label_loc_preds =	all_loc_preds[i].find(label)->second;
					if (!DecodeBBoxes(prior_bboxes, prior_variances,
						code_type, variance_encoded_in_target, clip,
						label_loc_preds, &(decode_bboxes[label])))
						return false;
				}
			}
			return true;
		}

		static bool DecodeBBoxes(
			const std::vector<ZQ_CNN_NormalizedBBox>& prior_bboxes,
			const std::vector<std::vector<float> >& prior_variances,
			const PriorBoxCodeType code_type, const bool variance_encoded_in_target,
			const bool clip_bbox, const std::vector<ZQ_CNN_NormalizedBBox>& bboxes,
			std::vector<ZQ_CNN_NormalizedBBox>* decode_bboxes) 
		{
			if (prior_bboxes.size() != prior_variances.size())
				return false;
			if (prior_bboxes.size() != bboxes.size())
				return false;
			int num_bboxes = prior_bboxes.size();
			if (num_bboxes >= 1) 
			{
				if (prior_variances[0].size() != 4)
					return false;
			}
			decode_bboxes->clear();
			for (int i = 0; i < num_bboxes; ++i) 
			{
				ZQ_CNN_NormalizedBBox decode_bbox;
				if (!DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
					variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox))
					return false;
				decode_bboxes->push_back(decode_bbox);
			}
			return true;
		}

		static bool DecodeBBox(
			const ZQ_CNN_NormalizedBBox& prior_bbox, const std::vector<float>& prior_variance,
			const PriorBoxCodeType code_type, const bool variance_encoded_in_target,
			const bool clip_bbox, const ZQ_CNN_NormalizedBBox& bbox,
			ZQ_CNN_NormalizedBBox* decode_bbox) 
		{
			if (code_type == PriorBoxCodeType_CORNER) 
			{
				if (variance_encoded_in_target) 
				{
					// variance is encoded in target, we simply need to add the offset
					// predictions.
					decode_bbox->col1 = prior_bbox.col1 + bbox.col1;
					decode_bbox->col2 = prior_bbox.col2 + bbox.col2;
					decode_bbox->row1 = prior_bbox.row1 + bbox.row1;
					decode_bbox->row2 = prior_bbox.row2 + bbox.row2;
				}
				else 
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decode_bbox->col1 =	prior_bbox.col1 + prior_variance[0] * bbox.col1;
					decode_bbox->row1 = prior_bbox.row1 + prior_variance[1] * bbox.row1;
					decode_bbox->col2 = prior_bbox.col2 + prior_variance[2] * bbox.col2;
					decode_bbox->row2 = prior_bbox.row2 + prior_variance[3] * bbox.row2;
				}
			}
			else if (code_type == PriorBoxCodeType_CENTER_SIZE) 
			{
				float prior_width = prior_bbox.col2 - prior_bbox.col1;
				if (prior_width < 0)
				{
				//	return false;
					printf("x = [%f , %f]\n", prior_bbox.col1, prior_bbox.col2);
				}
				float prior_height = prior_bbox.row2 - prior_bbox.row1;
				if (prior_height < 0)
				{
					//return false;
					printf("y = [%f , %f]\n", prior_bbox.row1, prior_bbox.row2);
				}
				float prior_center_x = (prior_bbox.col1 + prior_bbox.col2) / 2.;
				float prior_center_y = (prior_bbox.row1 + prior_bbox.row2) / 2.;

				float decode_bbox_center_x, decode_bbox_center_y;
				float decode_bbox_width, decode_bbox_height;
				if (variance_encoded_in_target) 
				{
					// variance is encoded in target, we simply need to retore the offset
					// predictions.
					decode_bbox_center_x = bbox.col1 * prior_width + prior_center_x;
					decode_bbox_center_y = bbox.row1 * prior_height + prior_center_y;
					decode_bbox_width = exp(bbox.col2) * prior_width;
					decode_bbox_height = exp(bbox.row2) * prior_height;
				}
				else 
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decode_bbox_center_x =	prior_variance[0] * bbox.col1 * prior_width + prior_center_x;
					decode_bbox_center_y =	prior_variance[1] * bbox.row1 * prior_height + prior_center_y;
					decode_bbox_width =	exp(prior_variance[2] * bbox.col2) * prior_width;
					decode_bbox_height = exp(prior_variance[3] * bbox.row2) * prior_height;
				}

				decode_bbox->col1 = decode_bbox_center_x - decode_bbox_width / 2.;
				decode_bbox->row1 = decode_bbox_center_y - decode_bbox_height / 2.;
				decode_bbox->col2 = decode_bbox_center_x + decode_bbox_width / 2.;
				decode_bbox->row2 = decode_bbox_center_y + decode_bbox_height / 2.;
			}
			else if (code_type == PriorBoxCodeType_CORNER_SIZE) 
			{
				float prior_width = prior_bbox.col2 - prior_bbox.col1;
				if (prior_width < 0)
				{
				//return false;
					printf("x = [%f , %f]\n", prior_bbox.col1, prior_bbox.col2);
				}
				float prior_height = prior_bbox.row2 - prior_bbox.row1;
				if (prior_height < 0)
				{
				//	return false;
					printf("y = [%f , %f]\n", prior_bbox.row1, prior_bbox.row2);
				}
				if (variance_encoded_in_target) 
				{
					// variance is encoded in target, we simply need to add the offset
					// predictions.
					decode_bbox->col1 = prior_bbox.col1 + bbox.col1 * prior_width;
					decode_bbox->row1 = prior_bbox.row1 + bbox.row1 * prior_height;
					decode_bbox->col2 = prior_bbox.col2 + bbox.col2 * prior_width;
					decode_bbox->row2 = prior_bbox.row2 + bbox.row2 * prior_height;
				}
				else 
				{
					// variance is encoded in bbox, we need to scale the offset accordingly.
					decode_bbox->col1 = prior_bbox.col1 + prior_variance[0] * bbox.col1 * prior_width;
					decode_bbox->row1 = prior_bbox.row1 + prior_variance[1] * bbox.row1 * prior_height;
					decode_bbox->col2 = prior_bbox.col2 + prior_variance[2] * bbox.col2 * prior_width;
					decode_bbox->row2 = prior_bbox.row2 + prior_variance[3] * bbox.row2 * prior_height;
				}
			}
			else 
			{
				printf("unknown code type\n");
				return false;
			}

			float bbox_size = BBoxSize(*decode_bbox, true);
			decode_bbox->size = bbox_size;
			if (clip_bbox) 
			{
				ClipBBox(*decode_bbox, decode_bbox);
			}
			return true;
		}

		static float BBoxSize(const ZQ_CNN_NormalizedBBox& bbox, const bool normalized)
		{
			if (bbox.col2 < bbox.col1 || bbox.row2 < bbox.row1) 
			{
				// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
				return 0;
			}
			else 
			{

				float width = bbox.col2 - bbox.col1;
				float height = bbox.row2 - bbox.row1;
				if (normalized) 
				{
					return width * height;
				}
				else 
				{
					// If bbox is not within range [0, 1].
					return (width + 1) * (height + 1);
				}
			}
		}

		static void ClipBBox(const ZQ_CNN_NormalizedBBox& bbox, ZQ_CNN_NormalizedBBox* clip_bbox) 
		{
			clip_bbox->col1 = __max(__min(bbox.col1, 1.f), 0.f);
			clip_bbox->row1 = __max(__min(bbox.row1, 1.f), 0.f);
			clip_bbox->col2 = __max(__min(bbox.col2, 1.f), 0.f);
			clip_bbox->row2 = __max(__min(bbox.row2, 1.f), 0.f);
			clip_bbox->size = BBoxSize(*clip_bbox, true);
			clip_bbox->difficult = bbox.difficult;
		}

		static bool GetLocPredictions(const float* loc_data, const int num,
			const int num_preds_per_class, const int num_loc_classes,
			const bool share_location, std::vector<ZQ_CNN_LabelBBox>* loc_preds) 
		{
			loc_preds->clear();
			if (share_location) 
			{
				if (num_loc_classes != 1)
					return false;
			}
			loc_preds->resize(num);
			for (int i = 0; i < num; i++) 
			{
				ZQ_CNN_LabelBBox& label_bbox = (*loc_preds)[i];
				for (int p = 0; p < num_preds_per_class; p++) 
				{
					int start_idx = p * num_loc_classes * 4;
					for (int c = 0; c < num_loc_classes; c++) 
					{
						int label = share_location ? -1 : c;
						if (label_bbox.find(label) == label_bbox.end()) 
						{
							label_bbox[label].resize(num_preds_per_class);
						}
						label_bbox[label][p].col1 = loc_data[start_idx + c * 4];
						label_bbox[label][p].row1 = loc_data[start_idx + c * 4 + 1];
						label_bbox[label][p].col2 = loc_data[start_idx + c * 4 + 2];
						label_bbox[label][p].row2 = loc_data[start_idx + c * 4 + 3];
					}
				}
				loc_data += num_preds_per_class * num_loc_classes * 4;
			}
			return true;
		}

		static void TransformLocations_MXNET(float *out, const float *anchors,
			const float *loc_pred, const bool clip,
			const float vx, const float vy,	const float vw, const float vh) 
		{
			// transform predictions to detection results
			float al = anchors[0];
			float at = anchors[1];
			float ar = anchors[2];
			float ab = anchors[3];
			float aw = ar - al;
			float ah = ab - at;
			float ax = (al + ar) / 2.f;
			float ay = (at + ab) / 2.f;
			float px = loc_pred[0];
			float py = loc_pred[1];
			float pw = loc_pred[2];
			float ph = loc_pred[3];
			float ox = px * vx * aw + ax;
			float oy = py * vy * ah + ay;
			float ow = exp(pw * vw) * aw / 2;
			float oh = exp(ph * vh) * ah / 2;
			out[0] = clip ? __max(0, __min(1, ox - ow)) : (ox - ow);
			out[1] = clip ? __max(0, __min(1, oy - oh)) : (oy - oh);
			out[2] = clip ? __max(0, __min(1, ox + ow)) : (ox + ow);
			out[3] = clip ? __max(0, __min(1, oy + oh)) : (oy + oh);
		}

		static void GetConfidenceScores(const float* conf_data, const int num,
			const int num_preds_per_class, const int num_classes,
			std::vector<std::map<int, std::vector<float> > >* conf_preds) 
		{
			conf_preds->clear();
			conf_preds->resize(num);
			for (int i = 0; i < num; ++i) 
			{
				std::map<int, std::vector<float> >& label_scores = (*conf_preds)[i];
				for (int p = 0; p < num_preds_per_class; ++p) 
				{
					int start_idx = p * num_classes;
					for (int c = 0; c < num_classes; ++c) 
					{
						label_scores[c].push_back(conf_data[start_idx + c]);
					}
				}
				conf_data += num_preds_per_class * num_classes;
			}
		}

		static void GetConfidenceScores(const float* conf_data, const int num,
			const int num_preds_per_class, const int num_classes,
			const bool class_major, std::vector<std::map<int, std::vector<float> > >* conf_preds) 
		{
			conf_preds->clear();
			conf_preds->resize(num);
			for (int i = 0; i < num; ++i) 
			{
				std::map<int, std::vector<float> >& label_scores = (*conf_preds)[i];
				if (class_major) 
				{
					for (int c = 0; c < num_classes; ++c) 
					{
						label_scores[c].assign(conf_data, conf_data + num_preds_per_class);
						conf_data += num_preds_per_class;
					}
				}
				else 
				{
					for (int p = 0; p < num_preds_per_class; ++p) 
					{
						int start_idx = p * num_classes;
						for (int c = 0; c < num_classes; ++c) 
						{
							label_scores[c].push_back(conf_data[start_idx + c]);
						}
					}
					conf_data += num_preds_per_class * num_classes;
				}
			}
		}

		static void GetPriorBBoxes(const float* prior_data, const int num_priors,
			std::vector<ZQ_CNN_NormalizedBBox>* prior_bboxes,
			std::vector<std::vector<float> >* prior_variances) 
		{
			prior_bboxes->clear();
			prior_variances->clear();
			for (int i = 0; i < num_priors; ++i) 
			{
				int start_idx = i * 4;
				ZQ_CNN_NormalizedBBox bbox;
				bbox.col1 = prior_data[start_idx];
				bbox.row1 = prior_data[start_idx + 1];
				bbox.col2 = prior_data[start_idx + 2];
				bbox.row2 = prior_data[start_idx + 3];
				float bbox_size = BBoxSize(bbox, true);
				bbox.size = bbox_size;
				prior_bboxes->push_back(bbox);
			}

			for (int i = 0; i < num_priors;i++) 
			{
				int start_idx = (num_priors + i) * 4;
				std::vector<float> var;
				for (int j = 0; j < 4; ++j) 
				{
					var.push_back(prior_data[start_idx + j]);
				}
				prior_variances->push_back(var);
			}
		}

		static bool ApplyNMSFast(const std::vector<ZQ_CNN_NormalizedBBox>& bboxes,
			const std::vector<float>& scores, const float score_threshold,
			const float nms_threshold, const float eta, const int top_k,
			std::vector<int>* indices) 
		{
			// Sanity check.
			if (bboxes.size() != scores.size())
			{
				printf("bboxes and scores have different size.\n");
				return false;
			}
			
			// Get top_k scores (with corresponding indices).
			std::vector<std::pair<float, int> > score_index_vec;
			GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

			// Do nms.
			float adaptive_threshold = nms_threshold;
			indices->clear();
			while (score_index_vec.size() != 0) 
			{
				const int idx = score_index_vec.front().second;
				bool keep = true;
				for (int k = 0; k < indices->size(); ++k) 
				{
					if (keep) 
					{
						const int kept_idx = (*indices)[k];
						float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx], true);
						keep = overlap <= adaptive_threshold;
					}
					else 
					{
						break;
					}
				}
				if (keep) 
				{
					indices->push_back(idx);
				}
				score_index_vec.erase(score_index_vec.begin());
				if (keep && eta < 1 && adaptive_threshold > 0.5) 
				{
					adaptive_threshold *= eta;
				}
			}
			return true;
		}


		static void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,
			const int top_k, std::vector<std::pair<float, int> >* score_index_vec) 
		{
			// Generate index score pairs.
			for (int i = 0; i < scores.size(); ++i) 
			{
				if (scores[i] > threshold) 
				{
					score_index_vec->push_back(std::make_pair(scores[i], i));
				}
			}

			// Sort the score pair according to the scores in descending order
			std::stable_sort(score_index_vec->begin(), score_index_vec->end(), SortScorePairDescend<int>);

			// Keep top_k scores if needed.
			if (top_k > -1 && top_k < score_index_vec->size()) 
			{
				score_index_vec->resize(top_k);
			}
		}

		template <typename T>
		static bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) 
		{
			return pair1.first > pair2.first;
		}

		static float JaccardOverlap(const ZQ_CNN_NormalizedBBox& bbox1, const ZQ_CNN_NormalizedBBox& bbox2,
			const bool normalized) {
			ZQ_CNN_NormalizedBBox intersect_bbox;
			IntersectBBox(bbox1, bbox2, &intersect_bbox);
			float intersect_width, intersect_height;
			if (normalized) 
			{
				intersect_width = intersect_bbox.col2 - intersect_bbox.col1;
				intersect_height = intersect_bbox.row2 - intersect_bbox.row1;
			}
			else 
			{
				intersect_width = intersect_bbox.col2 - intersect_bbox.col1 + 1;
				intersect_height = intersect_bbox.row2 - intersect_bbox.row1 + 1;
			}
			if (intersect_width > 0 && intersect_height > 0) 
			{
				float intersect_size = intersect_width * intersect_height;
				float bbox1_size = BBoxSize(bbox1, true);
				float bbox2_size = BBoxSize(bbox2, true);
				return intersect_size / (bbox1_size + bbox2_size - intersect_size);
			}
			else 
			{
				return 0.;
			}
		}

		static void IntersectBBox(const ZQ_CNN_NormalizedBBox& bbox1, const ZQ_CNN_NormalizedBBox& bbox2, ZQ_CNN_NormalizedBBox* intersect_bbox) 
		{
			if (bbox2.col1 > bbox1.col2 || bbox2.col2 < bbox1.col1 ||
				bbox2.row1 > bbox1.row2 || bbox2.row2 < bbox1.row1) 
			{
				// Return [0, 0, 0, 0] if there is no intersection.
				intersect_bbox->col1 = 0;
				intersect_bbox->row1 = 0;
				intersect_bbox->col2 = 0;
				intersect_bbox->row2 = 0;
			}
			else 
			{
				intersect_bbox->col1 = __max(bbox1.col1, bbox2.col1);
				intersect_bbox->row1 = __max(bbox1.row1, bbox2.row1);
				intersect_bbox->col2 = __min(bbox1.col2, bbox2.col2);
				intersect_bbox->row2 = __min(bbox1.row2, bbox2.row2);
			}
		}
	};
}
#endif
