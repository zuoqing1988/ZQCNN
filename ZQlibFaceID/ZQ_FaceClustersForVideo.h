#ifndef _ZQ_FACE_CLUSTERS_FOR_VIDEO_H_
#define _ZQ_FACE_CLUSTERS_FOR_VIDEO_H_
#pragma once

#include "ZQ_FaceFeature.h"
#include "ZQ_FaceContainerForVideo.h"
#include "ZQ_Kmeans.h"
#include "ZQ_MathBase.h"
#include "ZQ_MergeSort.h"
#include <vector>
#include <map>
#include <omp.h>

namespace ZQ
{
	class ZQ_FaceClustersForVideo
	{
	public:
		class Rect
		{
		public:
			int col1, row1, col2, row2;
		};
	public:
		int video_frames;
		std::vector<ZQ_FaceFeature> face_clusters;
		std::vector<int> pivot_frame_ids;
		std::vector<Rect> pivot_rects;	
		std::vector<std::vector<int> > appear_frame_ids;

	public:
		ZQ_FaceClustersForVideo()
		{
			video_frames = 0;
		}
		~ZQ_FaceClustersForVideo()
		{
		}

		void Clear()
		{
			video_frames = 0;
			face_clusters.clear();
			appear_frame_ids.clear();
			pivot_frame_ids.clear();
			pivot_rects.clear();
		}

		bool SaveToFile(const std::string& file) const
		{
			if (face_clusters.size() != appear_frame_ids.size())
				return false;
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
			fwrite(&video_frames, sizeof(int), 1, out);
			int cluster_num = face_clusters.size();
			fwrite(&cluster_num, sizeof(int), 1, out);
			for (int i = 0; i < cluster_num; i++)
			{
				fwrite(&face_clusters[i].length, sizeof(int), 1, out);
				if (face_clusters[i].length > 0)
					fwrite(face_clusters[i].pData, sizeof(float), face_clusters[i].length, out);
			}
			for (int i = 0; i < cluster_num; i++)
			{
				fwrite(&pivot_frame_ids[i], sizeof(int), 1, out);
				fwrite(&pivot_rects[i].col1, sizeof(int), 1, out);
				fwrite(&pivot_rects[i].row1, sizeof(int), 1, out);
				fwrite(&pivot_rects[i].col2, sizeof(int), 1, out);
				fwrite(&pivot_rects[i].row2, sizeof(int), 1, out);
			}
			for (int i = 0; i < cluster_num; i++)
			{
				int fr_num = appear_frame_ids[i].size();
				fwrite(&fr_num, sizeof(int), 1, out);
			}	
			for (int i = 0; i < cluster_num; i++)
			{
				int fr_num = appear_frame_ids[i].size();
				if(fr_num > 0)
					fwrite(&appear_frame_ids[i][0], sizeof(int), fr_num, out);
			}
			fclose(out);
			return true;
		}

		bool LoadFromFile(const std::string& file)
		{
			Clear();
			FILE* in = 0;
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;
			if (fread(&video_frames, sizeof(int), 1, in) != 1)
			{
				fclose(in);
				Clear();
				return false;
			}
			int cluster_num;
			if (fread(&cluster_num, sizeof(int), 1, in) != 1 || cluster_num < 0)
			{
				fclose(in);
				Clear();
				return false;
			}
			face_clusters.resize(cluster_num);
			appear_frame_ids.resize(cluster_num);
			pivot_frame_ids.resize(cluster_num);
			pivot_rects.resize(cluster_num);
			for (int i = 0; i < cluster_num; i++)
			{
				int feat_dim;
				if (fread(&feat_dim, sizeof(int), 1, in) != 1 || feat_dim < 0)
				{
					fclose(in);
					Clear();
					return false;
				}
				if (feat_dim > 0)
				{
					face_clusters[i].ChangeSize(feat_dim);
					if (fread(face_clusters[i].pData, sizeof(float), feat_dim, in) != feat_dim)
					{
						fclose(in);
						Clear();
						return false;
					}
				}
			}
			for (int i = 0; i < cluster_num; i++)
			{
				if (fread(&pivot_frame_ids[i], sizeof(int), 1, in) != 1
					|| fread(&pivot_rects[i].col1, sizeof(int), 1, in) != 1
					|| fread(&pivot_rects[i].row1, sizeof(int), 1, in) != 1
					|| fread(&pivot_rects[i].col2, sizeof(int), 1, in) != 1
					|| fread(&pivot_rects[i].row2, sizeof(int), 1, in) != 1)
				{
					fclose(in);
					Clear();
					return false;
				}	
			}
			for (int i = 0; i < cluster_num; i++)
			{
				int fr_num;
				if (fread(&fr_num, sizeof(int), 1, in) != 1 || fr_num < 0)
				{
					fclose(in);
					Clear();
					return false;
				}
				appear_frame_ids[i].resize(fr_num);
			}
			for (int i = 0; i < cluster_num; i++)
			{
				int fr_num = appear_frame_ids[i].size();
				if (fr_num > 0)
				{
					if (fread(&appear_frame_ids[i][0], sizeof(int), fr_num, in) != fr_num)
					{
						fclose(in);
						Clear();
						return false;
					}
				}
			}
			fclose(in);
			return true;
		}

	public:
		bool ConvertFromContainer(ZQ_FaceRecognizer& recognizer, const ZQ_FaceContainerForVideo& container, float union_thresh = 0.8f, float conquer_thresh = 0.9f, bool shuffle = false)
		{
			Clear();
			int fr_num = container.frames.size();
			if (fr_num == 0)
			{
				printf("no frame in container\n");
				return false;
			}

			int skip = container.skip;
			int feat_dim = container.frames[0].feat_dim;
			std::vector<std::vector<bool> > use_flag(fr_num);
			float face_pose_thresh = 0.8f;
			int valid_face_num = 0;
			for (int i = 0; i < fr_num; i++)
			{
				int cur_num = container.frames[i].face_feats.size();
				use_flag[i].resize(cur_num);
				for (int j = 0; j < cur_num; j++)
				{
					use_flag[i][j] = false;
					if (container.frames[i].face_feats[j].length == feat_dim)
					{
						use_flag[i][j] = true;
						valid_face_num++;
					}
				}
			}

			printf("valid_face_num = %d\n", valid_face_num);
			if (valid_face_num == 0)
			{
				printf("no face in container has feat_dim = %d\n", feat_dim);
				return false;
			}

			// Clustering
			int nPts = valid_face_num;
			int dim = feat_dim;
			std::vector<float> points(nPts*dim + 8);
			float* points_ptr = (float*)((long long)(&points[0] + 7) & 0xFFFFFFFFFFFFFFE0);
			std::vector<int> pts_fr_id(nPts);
			std::vector<int> pts_box_id(nPts);
			int cur_idx = 0;

			for (int i = 0; i < fr_num; i++)
			{
				int cur_num = container.frames[i].face_feats.size();
				for (int j = 0; j < cur_num; j++)
				{
					if (use_flag[i][j])
					{
						memcpy(points_ptr + cur_idx* dim, container.frames[i].face_feats[j].pData, sizeof(float)*dim);
						pts_fr_id[cur_idx] = i;
						pts_box_id[cur_idx] = j;
						cur_idx++;
					}
				}
			}

			std::vector<int> idx_for_unions(nPts);
			std::vector<int> idx_for_conquers(nPts);
			std::vector<std::vector<int> > unions;
			double t1 = omp_get_wtime();
			if (!recognizer.Clustering(nPts, dim, points_ptr, unions, &idx_for_unions[0], &idx_for_conquers[0],
				union_thresh, conquer_thresh, shuffle, 0))
			{
				printf("failed to run Clustring\n");
				return false;
			}
			double t2 = omp_get_wtime();
			printf("clustering costs : %.3f s\n", t2 - t1);

			//
			int union_num = unions.size();
			std::vector<int> counts(union_num);
			for (int i = 0; i < union_num; i++)
				counts[i] = 0;
			for (int i = 0; i < nPts; i++)
			{
				if (idx_for_unions[i] >= 0)
					counts[idx_for_unions[i]]++;
			}
			std::vector<int> indices(union_num);
			for (int i = 0; i < union_num; i++)
			{
				indices[i] = i;
			}
			ZQ_MergeSort::MergeSort<int>(&counts[0], &indices[0], union_num, false);

			
			video_frames = fr_num;
			face_clusters.resize(union_num);
			appear_frame_ids.resize(union_num);
			pivot_frame_ids.resize(union_num);
			pivot_rects.resize(union_num);
			
			std::vector<std::map<int, int> > tmp_appear_fr_ids(union_num);
			std::vector<std::vector<float> > center(union_num);
			printf("union_num = %d\n", union_num);
			for (int i = 0; i < union_num; i++)
			{
				int union_id = indices[i];
				center[i].resize(dim);
				for (int d = 0; d < dim; d++)
					center[i][d] = 0;
				for (int j = 0; j < unions[union_id].size(); j++)
				{
					int pt_id = unions[union_id][j];
					for (int d = 0; d < dim; d++)
						center[i][d] += points_ptr[pt_id*dim + d];
				}

				ZQ_MathBase::Normalize(dim, &center[i][0]);
				/*face_clusters[i].ChangeSize(dim);
				memcpy(face_clusters[i].pData, &center[i][0], sizeof(float)*dim);*/
			}

			std::vector<int> pivot_pt_ids(union_num);
			std::vector<float> max_scores(union_num);
			for (int i = 0; i < union_num; i++)
			{
				pivot_pt_ids[i] = -1;
				max_scores[i] = -FLT_MAX;
			}
			for (int pp = 0; pp < nPts; pp++)
			{
				int i = idx_for_unions[pp];
				float cur_score = recognizer.CalSimilarity(&center[i][0], points_ptr + pp*dim);
				if (cur_score > max_scores[i])
				{
					pivot_pt_ids[i] = pp;
					max_scores[i] = cur_score;
				}
			}
			for (int i = 0; i < union_num; i++)
			{
				face_clusters[i].ChangeSize(dim);
				memcpy(face_clusters[i].pData, points_ptr + pivot_pt_ids[i]*dim, sizeof(float)*dim);
			}

			for(int i = 0;i < union_num;i++)
			{
				pivot_frame_ids[i] = pts_fr_id[pivot_pt_ids[i]];
				ZQ_CNN_BBox cur_box = container.frames[pts_fr_id[pivot_pt_ids[i]]].face_boxes[pts_box_id[pivot_pt_ids[i]]];
				pivot_rects[i].col1 = cur_box.col1;
				pivot_rects[i].row1 = cur_box.row1;
				pivot_rects[i].col2 = cur_box.col2;
				pivot_rects[i].row2 = cur_box.row2;
			}

			for (int i = 0; i < nPts; i++)
			{
				int cur_union_id = idx_for_unions[i];
				int fr_id = pts_fr_id[i];
				if (tmp_appear_fr_ids[cur_union_id].find(fr_id) == tmp_appear_fr_ids[cur_union_id].end())
				{
					tmp_appear_fr_ids[cur_union_id][fr_id] = 1;
				}
			}

			for (int i = 0; i < union_num; i++)
			{
				int union_id = indices[i];
				std::map<int, int>::iterator it = tmp_appear_fr_ids[union_id].begin();
				for (; it != tmp_appear_fr_ids[union_id].end(); ++it)
				{
					appear_frame_ids[i].push_back(it->first);
				}
			}
			return true;
		}
	};
}
#endif
