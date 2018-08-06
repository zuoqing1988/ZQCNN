#ifndef _ZQ_FACE_RECOGNIZER_H_
#define _ZQ_FACE_RECOGNIZER_H_
#pragma once
#include "ZQ_PixelFormat.h"
#include <vector>
#include <string>
#include <stdlib.h>
namespace ZQ
{
	class ZQ_FaceRecognizer
	{
	public:
		virtual bool Init(const std::string model_name, 
			const std::string prototxt_file = "", const std::string caffemodel_file = "",
			const std::string out_blob_name = ""
		) = 0;

		virtual int GetFeatDim() const = 0;
		virtual int GetCropWidth() const = 0;
		virtual int GetCropHeight() const = 0;
		virtual bool CropImage(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y,
			unsigned char* crop_img, int crop_widthStep) const = 0;
		
		virtual bool ExtractFeature(const unsigned char* in_img, int in_width, int in_height, int in_widthStep,
			ZQ_PixelFormat pixFmt, const float* face5point_x, const float* face5point_y, float* feat, bool normalize) = 0;

		virtual bool ExtractFeature(const unsigned char* img, int widthStep, ZQ_PixelFormat pixFmt, float* feat, bool normalize) = 0;
		virtual float CalSimilarity(const float* feat1, const float* feat2) const = 0;	
	public:
		virtual bool Clustering(int nPts, int dim, const float* pts,  std::vector<std::vector<int>>& unions, 
			int* idx_for_unions = NULL, int* idx_for_conquers = NULL, float union_thresh = 0.8f, float conquer_thresh = 0.9f, 
			bool shuffle = false,  int MAX_PT_NUM = 0)
		{
			if (nPts <= 0 || pts == 0 || GetFeatDim() != dim)
				return false;
			std::vector<int> rest_set(nPts);
			if (MAX_PT_NUM <= 0 || MAX_PT_NUM > nPts)
				MAX_PT_NUM = nPts;
			int rest_num = MAX_PT_NUM;
			for (int i = 0; i < nPts; i++)
			{
				rest_set[i] = i;
			}

			if (shuffle || MAX_PT_NUM < nPts)
			{
				for (int i = 0; i < MAX_PT_NUM; i++)
				{
					int id = rand() % (nPts - i) + i;
					if (id != i)
					{
						int tmp = rest_set[i];
						rest_set[i] = rest_set[id];
						rest_set[id] = tmp;
					}
				}
			}

			// First stage: find the masters conquering a range of conquer_thresh
			std::vector<int> conquers;
			std::vector<std::vector<int>> slaves;
			const int max_search_len = 10000;
			while (rest_num > 0)
			{
				//printf("rest_num = %d\n", rest_num);
				std::vector<int> cur_slaves;
				int cur_conquer_pt_id = rest_set[0];
				conquers.push_back(cur_conquer_pt_id);
				rest_set[0] = rest_set[rest_num - 1];
				rest_num--;
				cur_slaves.push_back(cur_conquer_pt_id);
				
				for (int i = 0; i < rest_num && i < max_search_len; )
				{
					int cur_slave_pt_id = rest_set[i];
					float score = CalSimilarity(pts + cur_conquer_pt_id*dim, pts + cur_slave_pt_id*dim);
					if (score >= conquer_thresh)
					{
						cur_slaves.push_back(cur_slave_pt_id);
						rest_set[i] = rest_set[rest_num - 1];
						rest_num--;
					}
					else
					{
						i++;
					}
				}
				slaves.push_back(cur_slaves);
			}

			//Second stage: connect the conquers
			
			int rest_conquer_num = conquers.size();
			std::vector<int> rest_conquers(rest_conquer_num);
			for (int i = 0; i < rest_conquer_num; i++)
				rest_conquers[i] = i;
			std::vector<std::vector<int>> conquer_id_unions;
			while (rest_conquer_num > 0)
			{
				//printf("rest_conquer_num = %d, union_num = %d\n", rest_conquer_num,(int)conquer_id_unions.size());
				std::vector<int> tmp_union;
				int cur_conquer_id = rest_conquers[0];
				int cur_conquer_pt_id = conquers[cur_conquer_id];
				tmp_union.push_back(cur_conquer_id);
				rest_conquers[0] = rest_conquers[rest_conquer_num - 1];
				rest_conquer_num--;
				int last_num = 0;
				bool should_end = false;
				while (!should_end)
				{
					int cur_num = tmp_union.size();
					should_end = true;
					for (int i = 0; i < rest_conquer_num;)
					{
						int cur_rest_conquer_id = rest_conquers[i];
						int cur_rest_pt_id = conquers[cur_rest_conquer_id];
						bool has_found = false;
						for (int j = last_num; j < cur_num; j++)
						{
							int cur_union_member_id = tmp_union[j];
							int cur_union_member_pt_id = conquers[cur_union_member_id];
							float score = CalSimilarity(pts + cur_rest_pt_id*dim, pts + cur_union_member_pt_id*dim);
							if (score >= union_thresh)
							{
								tmp_union.push_back(cur_rest_conquer_id);
								rest_conquers[i] = rest_conquers[rest_conquer_num - 1];
								rest_conquer_num--;
								has_found = true;
								break;
							}
						}
						
						if (has_found)
						{
							should_end = false;	
						}
						else
						{
							i++;
						}
					}
					last_num = cur_num;
				}
				conquer_id_unions.push_back(tmp_union);
			}

			//output
			int union_num = conquer_id_unions.size();
			unions.resize(union_num);
			for (int i = 0; i < union_num; i++)
			{
				int member_num = conquer_id_unions[i].size();
				unions[i].resize(member_num);
				for (int j = 0; j < member_num; j++)
				{
					int conquer_id = conquer_id_unions[i][j];
					unions[i][j] = conquers[conquer_id];
				}
			}

			if (idx_for_conquers != NULL)
			{
				for (int i = 0; i < nPts; i++)
					idx_for_conquers[i] = -1;
				for (int i = 0; i < slaves.size(); i++)
				{
					for (int j = 0; j < slaves[i].size(); j++)
					{
						idx_for_conquers[slaves[i][j]] = i;
					}
				}
			}

			if (idx_for_unions != NULL)
			{
				for (int i = 0; i < nPts; i++)
					idx_for_unions[i] = -1;
				for (int i = 0; i < conquer_id_unions.size(); i++)
				{
					for (int j = 0; j < conquer_id_unions[i].size(); j++)
					{
						int conquer_id = conquer_id_unions[i][j];
						for (int k = 0; k < slaves[conquer_id].size(); k++)
						{
							idx_for_unions[slaves[conquer_id][k]] = i;
						}
					}
				}
			}

			return true;
		}
	};
}
#endif
