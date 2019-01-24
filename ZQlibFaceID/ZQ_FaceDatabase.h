#ifndef _ZQ_FACE_DATABASE_H_
#define _ZQ_FACE_DATABASE_H_
#pragma once
#include <vector>
#include <string>
#include "ZQ_FaceFeature.h"
#include "ZQ_FaceRecognizerSphereFace.h"
#include "ZQ_MathBase.h"
#include "ZQ_MergeSort.h"
#include <omp.h>

namespace ZQ
{
	class ZQ_FaceDatabase
	{
	public:
		class Person
		{
		public:
			std::vector<ZQ_FaceFeature> features;
			std::vector<std::string> filenames;
		};

		friend class ZQ_FaceDatabaseMaker;
	private:
		std::vector<Person> persons;
		std::vector<std::string> names;

	public:
		bool Search(const std::vector<ZQ_FaceFeature>& feat, std::vector<int>& out_ids, std::vector<float>& out_scores, std::vector<std::string>& out_names,
			std::vector<std::string>& out_filenames, int max_num = 3, int max_thread_num = 1) const
		{
			return _find_the_best_matches(feat, *this, out_ids, out_scores, out_names, out_filenames, max_num, max_thread_num);
		}

		bool ExportSimilarityForAllPairs(const std::string& out_score_file, const std::string& out_flag_file, 
			__int64& all_pair_num, __int64& same_pair_num, __int64& notsame_pair_num, int max_thread_num, bool quantization) const
		{
			return _export_similarity_for_all_pairs(out_score_file, out_flag_file, all_pair_num,
				same_pair_num, notsame_pair_num, max_thread_num, quantization);
		}

		bool SelectSubset(const std::string& out_file, int max_thread_num, int num_image_thresh = 10, float similarity_thresh = 0.5) const
		{
			return _select_subset(out_file, max_thread_num, similarity_thresh, num_image_thresh);
		}

		bool SelectSubsetDesiredNum(const std::string& out_file, int desired_person_num, int min_image_num_per_person, int max_image_num_per_person,
			int max_thread_num, float similarity_thresh = 0.5) const
		{
			return _select_subset_desired_num(out_file, desired_person_num, min_image_num_per_person, max_image_num_per_person,
				max_thread_num, similarity_thresh);
		}

		bool DetectRepeatPerson(const std::string& out_file, int max_thread_num, float similarity_thresh = 0.5) const
		{
			return _detect_repeat_person(out_file, max_thread_num, similarity_thresh);
		}

		bool DetectLowestPair(const std::string& out_file, int max_thread_num, float similarity_thresh = 0.5) const
		{
			return _detect_lowest_pair(out_file, max_thread_num, similarity_thresh);
		}

		void Clear()
		{
			persons.clear();
			names.clear();
		}

		bool LoadFromFileBinay(const std::string& feats_file, const std::string& names_file)
		{
			Clear();
			if (!_load_feats_binary(feats_file))
			{
				Clear();
				return false;
			}
			if (!_load_names(names_file))
			{
				Clear();
				return false;
			}
			if (persons.size() != names.size())
			{
				Clear();
				return false;
			}	
			return true;
		}

		bool SaveToFileBinary(const std::string& feats_file, const std::string& names_file)
		{
			if (!_check_valid())
			{
				printf("not a valid database\n");
				return false;
			}
			if (!_write_feats_binary(feats_file))
			{
				printf("failed to save %s\n", feats_file.c_str());
				return false;
			}
			if (!_write_names(names_file))
			{
				printf("failed to save %s\n", names_file.c_str());
				return false;
			}
			return true;
		}

		bool SaveToFileBinaryCompact(const std::string& feats_file, const std::string& names_file)
		{
			if (!_check_valid())
			{
				printf("not a valid database\n");
				return false;
			}

			if (!_write_feats_binary_compact(feats_file))
			{
				printf("failed to save %s\n", feats_file.c_str());
				return false;
			}

			if (!_write_names(names_file))
			{
				printf("failed to save %s\n", names_file.c_str());
				return false;
			}
			return true;
		}

	private:
		bool _check_valid()
		{
			int person_num = persons.size();
			if (person_num == 0)
				return false;
			if (person_num != names.size())
				return false;
			for (int i = 0; i < person_num; i++)
			{
				int feat_num = persons[i].features.size();
				if (feat_num == 0)
					return false;
				if (feat_num != persons[i].filenames.size())
					return false;
			}
			int feat_dim = persons[0].features[0].length;
			if (feat_dim == 0)
				return false;
			for (int i = 0; i < person_num; i++)
			{
				int feat_num = persons[i].features.size();
				for (int j = 0; j < feat_num; j++)
				{
					if (feat_dim != persons[i].features[j].length)
						return false;
				}
			}
			return true;
		}

		bool _write_feats_binary(const std::string& file)
		{
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;

			int person_num = persons.size();
			int feat_dim = persons[0].features[0].length;
			
			if (1 != fwrite(&feat_dim, sizeof(int), 1, out))
			{
				fclose(out);
				return false;
			}
			if (1 != fwrite(&person_num, sizeof(int), 1, out))
			{
				fclose(out);
				return false;
			}

			char end_c = '\0';
			for (int i = 0; i < person_num; i++)
			{
				int feat_num = persons[i].features.size();
				if (1 != fwrite(&feat_num, sizeof(int), 1, out))
				{
					fclose(out);
					return false;
				}
				
				for (int j = 0; j < feat_num; j++)
				{
					const char* str = persons[i].filenames[j].c_str();
					int len = strlen(str) + 1;
					if (1 != fwrite(&len, sizeof(int), 1, out))
					{
						fclose(out);
						return false;
					}
					
					if ((len-1) != fwrite(str, 1, len-1, out))
					{
						fclose(out);
						return false;
					}
					if (1 != fwrite(&end_c, sizeof(char), 1, out))
					{
						fclose(out);
						return false;
					}
					if (feat_dim != fwrite(persons[i].features[j].pData, sizeof(float), feat_dim, out))
					{
						fclose(out);
						return false;
					}
				}
			}
			fclose(out);
			return true;
		}

		bool _write_feats_binary_compact(const std::string& file)
		{
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;

			int person_num = persons.size();
			int feat_dim = persons[0].features[0].length;
			if (1 != fwrite(&feat_dim, sizeof(int), 1, out))
			{
				fclose(out);
				return false;
			}
			if (1 != fwrite(&person_num, sizeof(int), 1, out))
			{
				fclose(out);
				return false;
			}
			for (int i = 0; i < person_num; i++)
			{
				int feat_num = persons[i].features.size();
				if (1 != fwrite(&feat_num, sizeof(int), 1, out))
				{
					fclose(out);
					return false;
				}
			}

			for (int i = 0; i < person_num; i++)
			{
				int feat_num = persons[i].features.size();
				for (int j = 0; j < feat_num; j++)
				{
					if (feat_dim != fwrite(persons[i].features[j].pData, sizeof(float), feat_dim, out))
					{
						fclose(out);
						return false;
					}
				}
			}
			fclose(out);
			return true;
		}

		bool _load_feats_binary(const std::string& file)
		{
			FILE* in = 0;
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;

			int person_num = 0;
			int feat_dim = 0;
			
			if (1 != fread(&feat_dim, sizeof(int), 1, in))
			{
				fclose(in);
				return false;
			}
			if (1 != fread(&person_num, sizeof(int), 1, in))
			{
				fclose(in);
				return false;
			}
			if (person_num <= 0 || feat_dim <= 0)
			{
				fclose(in);
				return false;
			}
			std::vector<char> buf;
			persons.resize(person_num);
			for (int i = 0; i < person_num; i++)
			{
				int feat_num = 0;
				if (1 != fread(&feat_num, sizeof(int), 1, in))
				{
					fclose(in);
					return false;
				}
				if (feat_num <= 0)
				{
					fclose(in);
					return false;
				}
				persons[i].features.resize(feat_num);
				persons[i].filenames.resize(feat_num);
				for (int j = 0; j < feat_num; j++)
				{
					int len;
					if (1 != fread(&len, sizeof(int), 1, in))
					{
						fclose(in);
						return false;
					}
					buf.resize(len);
					
					if (len > 0)
					{
						if (len != fread(&buf[0], 1, len, in) || buf[len-1] != '\0')
						{
							fclose(in);
							return false;
						}
						persons[i].filenames[j] = &buf[0];
					}
					persons[i].features[j].ChangeSize(feat_dim);
					if (feat_dim != fread(persons[i].features[j].pData, sizeof(float), feat_dim, in))
					{
						fclose(in);
						return false;
					}
				}
			}
			fclose(in);
			return true;
		}

		bool _write_names(const std::string& file)
		{
			FILE* out = 0;
			if (0 != fopen_s(&out, file.c_str(), "w"))
				return false;
			int person_num = names.size();
			for (int i = 0; i < person_num; i++)
			{
				fprintf(out, "%s\n", names[i].c_str());
			}
			fclose(out);
			return true;
		}

		bool _load_names(const std::string& file)
		{
			FILE* in = 0;
			if (0 != fopen_s(&in, file.c_str(), "r"))
				return false;
			char line[200] = { 0 };
			while (true)
			{
				line[0] = '\0';
				fgets(line, 199, in);
				if (line[0] == '\0')
					break;
				int len = strlen(line);
				if (line[len - 1] == '\n')
					line[--len] = '\0';
				names.push_back(std::string(line));
			}
			
			fclose(in);
			return true;
		}

		static bool _find_the_best_matches(const std::vector<ZQ_FaceFeature>& feat, const ZQ_FaceDatabase& database, std::vector<int>& out_ids,
			std::vector<float>& out_scores, std::vector<std::string>& out_names, std::vector<std::string>& out_filenames, int max_num, int max_thread_num)
		{
			int feat_num = feat.size();
			if (feat_num == 0)
				return false;
			double t1 = omp_get_wtime();
			int person_num = database.persons.size();
			std::vector<int> person_j(person_num);
			std::vector<float> scores(person_num);
			std::vector<int> ids(person_num);
			int num_procs = omp_get_num_procs();
			int real_threads = __max(1, __min(max_thread_num, num_procs - 1));
			//printf("real_threads = %d\n", real_threads);
#pragma omp parallel for schedule(dynamic) num_threads(real_threads)
			for (int i = 0; i < person_num; i++)
			{
				ids[i] = i;
				float max_score = -FLT_MAX;
				int max_id = -1;
				for (int j = 0; j < database.persons[i].features.size(); j++)
				{
					float tmp_score = -FLT_MAX;
					for (int k = 0; k < feat_num; k++)
					{
						if (feat[k].length == database.persons[i].features[j].length)
						{
							tmp_score = ZQ_FaceRecognizerSphereFace::CalSimilarity(feat[k].length, feat[k].pData, database.persons[i].features[j].pData);
						}

						if (max_id < 0)
						{
							max_id = 0;
							max_score = tmp_score;
						}
						else
						{
							if (max_score < tmp_score)
							{
								max_id = j;
								max_score = tmp_score;
							}
						}
					}
					
				}
				person_j[i] = max_id;
				scores[i] = max_score;
			}

			double t2 = omp_get_wtime();

			out_ids.clear();
			out_scores.clear();
			out_names.clear();
			out_filenames.clear();
			for (int i = 0; i < __min(max_num, person_num); i++)
			{
				float max_score = scores[i];
				int max_id = i;
				for (int j = i + 1; j < person_num; j++)
				{
					if (max_score < scores[j])
					{
						max_id = j;
						max_score = scores[j];
					}
				}
				int tmp_id = ids[i];
				ids[i] = ids[max_id];
				ids[max_id] = tmp_id;
				float tmp_score = scores[i];
				scores[i] = scores[max_id];
				scores[max_id] = tmp_score;

				out_ids.push_back(ids[i]);
				out_scores.push_back(scores[i]);
				out_names.push_back(database.names[ids[i]]);
				out_filenames.push_back(database.persons[ids[i]].filenames[person_j[ids[i]]]);
			}

			double t3 = omp_get_wtime();
			//printf("part1 = %.3f, part2 = %.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

		bool _export_similarity_for_all_pairs(const std::string& out_score_file, const std::string& out_flag_file, 
			__int64& all_pair_num, __int64& same_pair_num, __int64& notsame_pair_num, int max_thread_num, bool quantization) const
		{
			FILE* out1 = 0;
			if (0 != fopen_s(&out1, out_score_file.c_str(), "wb"))
			{
				printf("failed to create file %s\n", out_score_file.c_str());
				return false;
			}

			FILE* out2 = 0;
			if (0 != fopen_s(&out2, out_flag_file.c_str(), "wb"))
			{
				printf("failed to create file %s\n", out_flag_file.c_str());
				fclose(out1);
				return false;
			}

			int dim = persons[0].features[0].length;
			__int64 person_num = persons.size();
			__int64 total_face_num = 0;
			std::vector<__int64> cur_face_offset(person_num);
			for (int pp = 0; pp < person_num; pp++)
			{
				cur_face_offset[pp] = total_face_num;
				__int64 cur_face_num = persons[pp].features.size();
				total_face_num += cur_face_num;
			}
			
			all_pair_num = total_face_num *(total_face_num - 1) / 2;
			
			int real_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
			if (real_thread_num == 1)
			{
				for (int pp = 0; pp < person_num; pp++)
				{
					__int64 cur_face_num = persons[pp].features.size();
					__int64 max_pair_num = (total_face_num - cur_face_offset[pp] - 1);
					std::vector<float> scores(max_pair_num);
					std::vector<char> flags(max_pair_num);
					for (__int64 i = 0; i < cur_face_num; i++)
					{
						float* cur_i_feat = persons[pp].features[i].pData;
						float* cur_j_feat;
						int idx = 0;
						for (__int64 j = i + 1; j < cur_face_num; j++)
						{
							cur_j_feat = persons[pp].features[j].pData;
							scores[idx] = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							flags[idx] = 1;
							same_pair_num++;
							idx++;
						}
						for (__int64 qq = pp + 1; qq < person_num; qq++)
						{
							for (__int64 j = 0; j < persons[qq].features.size(); j++)
							{
								cur_j_feat = persons[qq].features[j].pData;
								scores[idx] = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
								flags[idx] = 0;
								notsame_pair_num++;
								idx++;
							}
						}
						
						if (idx > 0)
						{
							if (quantization)
							{
								std::vector<short> short_scores(idx);
								for (int j = 0; j < idx; j++)
									short_scores[j] = __min(SHRT_MAX, __max(-SHRT_MAX, scores[j] * SHRT_MAX));
								fwrite(&short_scores[0], sizeof(short), idx, out1);
							}
							else
							{
								fwrite(&scores[0], sizeof(float), idx, out1);
							}
							fwrite(&flags[0], 1, idx, out2);
						}
					}
					printf("%d/%d handled\n", pp + 1, person_num);
				}
			}
			else
			{
				int chunk_size = 100;
				int handled[1] = { 0 };
				__int64 tmp_same_pair_num[1] = { 0 };
				printf("real_thread_num = %d\n", real_thread_num);
#pragma omp parallel for schedule(dynamic,chunk_size) num_threads(real_thread_num) shared(handled)
				for (int pp = 0; pp < person_num; pp++)
				{
					__int64 cur_face_num = persons[pp].features.size();
					__int64 max_pair_num = (total_face_num - cur_face_offset[pp] - 1);
					std::vector<float> scores(max_pair_num);
					std::vector<char> flags(max_pair_num);
					for (__int64 i = 0; i < cur_face_num; i++)
					{
						float* cur_i_feat = persons[pp].features[i].pData;
						float* cur_j_feat;
						int idx = 0;
						for (__int64 j = i + 1; j < cur_face_num; j++)
						{
							cur_j_feat = persons[pp].features[j].pData;
							scores[idx] = ZQ::ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							flags[idx] = 1;
							same_pair_num++;
							idx++;
						}
						for (__int64 qq = pp + 1; qq < person_num; qq++)
						{
							for (__int64 j = 0; j < persons[qq].features.size(); j++)
							{
								cur_j_feat = persons[qq].features[j].pData;
								scores[idx] = ZQ::ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
								flags[idx] = 0;
								notsame_pair_num++;
								idx++;
							}
						}
#pragma omp critical
						{
							if (idx > 0)
							{
								for (int kk = 0; kk < idx; kk++)
								{
									(*tmp_same_pair_num) += flags[kk];
								}
								if (quantization)
								{
									std::vector<short> short_scores(idx);
									for (int j = 0; j < idx; j++)
										short_scores[j] = __min(SHRT_MAX, __max(-SHRT_MAX, scores[j] * SHRT_MAX));
									fwrite(&short_scores[0], sizeof(short), idx, out1);
								}
								else
								{
									fwrite(&scores[0], sizeof(float), idx, out1);
								}
								fwrite(&flags[0], 1, idx, out2);
							}
						}
					}
#pragma omp critical
					{
						(*handled)++;
						printf("%d/%d\n", *handled, person_num);
					}
				}
				same_pair_num = tmp_same_pair_num[0];
				notsame_pair_num = all_pair_num - same_pair_num;
			}

			fclose(out1);
			fclose(out2);
			return true;
		}

		bool _select_subset_desired_num(const std::string& out_file, int desired_person_num, 
			int min_image_num_per_person, int max_image_num_per_person,
			int max_thread_num, float similarity_thresh) const
		{
			std::vector<int> person_ids, pivot_ids;
			std::vector<std::vector<int> > other_good_ids;
			if (!_select_subset(person_ids, pivot_ids, other_good_ids, max_thread_num, similarity_thresh, 
				min_image_num_per_person))
			{
				return false;
			}
			FILE* out = 0;
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}

			std::vector<int> select_ids;
			int person_num = person_ids.size();
			for (int i = 0; i < person_num; i++)
				select_ids.push_back(i);
			if (person_num > desired_person_num)
			{
				for (int i = 0; i < desired_person_num; i++)
				{
					int rand_id = rand() % (person_num - i) + i;
					if (rand_id != i)
					{
						int tmp_id = select_ids[i];
						select_ids[i] = select_ids[rand_id];
						select_ids[rand_id] = tmp_id;
					}
				}
			}
			else
			{
				desired_person_num = person_num;
			}
			
			for (int i = 0; i < desired_person_num; i++)
			{
				int select_id = select_ids[i];
				int p_id = person_ids[select_id];
				fprintf(out, "%s\n", persons[p_id].filenames[pivot_ids[select_id]].c_str());
				if (min_image_num_per_person > 1)
				{
					int good_num = other_good_ids[select_id].size();
					std::vector<int> select_good_id(good_num);
					for (int j = 0; j < good_num; j++)
						select_good_id[j] = j;
					int desired_image_num_per_person = __min(max_image_num_per_person, good_num + 1);
					for (int j = 0; j < desired_image_num_per_person - 1; j++)
					{
						int rand_id = rand() % (desired_image_num_per_person - 1 - j) + j;
						if (rand_id != j)
						{
							int tmp_id = select_good_id[j];
							select_good_id[j] = select_good_id[rand_id];
							select_good_id[rand_id] = tmp_id;
						}
					}
					for (int j = 0; j < desired_image_num_per_person - 1; j++)
					{
						fprintf(out, "%s\n", persons[p_id].filenames[other_good_ids[select_id][select_good_id[j]]].c_str());
					}
				}
				
			}
			fclose(out);
			return true;
		}

		bool _select_subset(const std::string& out_file, int max_thread_num, float similarity_thresh, int num_image_thresh) const
		{
			std::vector<int> person_ids, pivot_ids;
			std::vector<std::vector<int> > other_good_ids;
			if (!_select_subset(person_ids, pivot_ids, other_good_ids, max_thread_num, similarity_thresh, num_image_thresh))
			{
				return false;
			}
			FILE* out = 0;
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}

			for (int i = 0; i < person_ids.size(); i++)
			{
				int p_id = person_ids[i];
				fprintf(out, "%s\n", persons[p_id].filenames[pivot_ids[i]].c_str());
				for (int j = 0; j < other_good_ids[i].size(); j++)
				{
					fprintf(out, "%s\n", persons[p_id].filenames[other_good_ids[i][j]].c_str());
				}
			}
			fclose(out);
			return true;
		}
		
		bool _select_subset(std::vector<int>& person_ids, std::vector<int>& pivot_ids, std::vector<std::vector<int> >& other_good_ids,
			int max_thread_num, float similarity_thresh, int num_image_thresh) const
		{
			int person_num = persons.size();
			if (person_num == 0 || persons[0].features.size() == 0)
				return false;
			int dim = persons[0].features[0].length;
			if (dim == 0)
				return false;
			
			person_ids.clear();
			pivot_ids.clear();
			other_good_ids.clear();

			if (max_thread_num <= 1)
			{
				for (int p = 0; p < person_num; p++)
				{
					int cur_num = persons[p].features.size();
					
					std::vector<float> scores(cur_num*cur_num);
					for (int i = 0; i < cur_num; i++)
					{
						scores[i*cur_num + i] = 1;
						const float* cur_i_feat = persons[p].features[i].pData;
						const float* cur_j_feat;
						for (int j = i + 1; j < cur_num; j++)
						{
							cur_j_feat = persons[p].features[j].pData;
							float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							scores[i*cur_num + j] = tmp_score;
							scores[j*cur_num + i] = tmp_score;
						}
					}
					int pivot_id = -1;
					float sum_score = -FLT_MAX;
					for (int i = 0; i < cur_num; i++)
					{
						float tmp_sum = 0;
						for (int j = 0; j < cur_num; j++)
							tmp_sum += scores[i*cur_num + j];
						if (sum_score < tmp_sum)
						{
							pivot_id = i;
							sum_score = tmp_sum;
						}
					}

					std::vector<int> ids;
					for (int i = 0; i < cur_num; i++)
					{
						if (scores[pivot_id*cur_num + i] >= similarity_thresh && i != pivot_id)
						{
							ids.push_back(i);
						}
					}
					int id_num = ids.size();
					if (id_num + 1 >= num_image_thresh)
					{
						person_ids.push_back(p);
						pivot_ids.push_back(pivot_id);
						other_good_ids.push_back(ids);
					}
				}
			}
			else
			{
				int chunk_size = (person_num + max_thread_num - 1) / max_thread_num;
#pragma omp parallel for schedule(static,chunk_size) num_threads(max_thread_num)
				for (int p = 0; p < person_num; p++)
				{
					int cur_num = persons[p].features.size();

					std::vector<float> scores(cur_num*cur_num);
					for (int i = 0; i < cur_num; i++)
					{
						scores[i*cur_num + i] = 1;
						const float* cur_i_feat = persons[p].features[i].pData;
						const float* cur_j_feat;
						for (int j = i + 1; j < cur_num; j++)
						{
							cur_j_feat = persons[p].features[j].pData;
							float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							scores[i*cur_num + j] = tmp_score;
							scores[j*cur_num + i] = tmp_score;
						}
					}
					int pivot_id = -1;
					float sum_score = -FLT_MAX;
					for (int i = 0; i < cur_num; i++)
					{
						float tmp_sum = 0;
						for (int j = 0; j < cur_num; j++)
							tmp_sum += scores[i*cur_num + j];
						if (sum_score < tmp_sum)
						{
							pivot_id = i;
							sum_score = tmp_sum;
						}
					}

					std::vector<int> ids;
					for (int i = 0; i < cur_num; i++)
					{
						if (scores[pivot_id*cur_num + i] >= similarity_thresh && i != pivot_id)
						{
							ids.push_back(i);
						}
					}
					int id_num = ids.size();
					if (id_num + 1 >= num_image_thresh)
					{
#pragma omp critical
						{
							person_ids.push_back(p);
							pivot_ids.push_back(pivot_id);
							other_good_ids.push_back(ids);
						}
					}
				}
			}
			return true;
		}

		bool _detect_repeat_person(const std::string& out_file, int max_thread_num, float similarity_thresh) const
		{
			std::vector<std::pair<int, int> > repeat_pairs;
			std::vector<float> scores;
			if (!_detect_repeat_person(repeat_pairs, scores, max_thread_num, similarity_thresh))
			{
				return false;
			}

			int num = scores.size();
			if (num > 0)
			{
				ZQ_MergeSort::MergeSortWithData(&scores[0], &repeat_pairs[0], sizeof(std::pair<int, int>), num, false);
			}

			FILE* out = 0;
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}
			for (int i = 0; i < num; i++)
			{
				fprintf(out, "%.3f %s %s\n", scores[i], names[repeat_pairs[i].first].c_str(), names[repeat_pairs[i].second].c_str());
			}
			fclose(out);
			return true;
		}

		bool _detect_repeat_person(std::vector<std::pair<int,int> >& repeat_pairs, std::vector<float>& repeat_scores,
			int max_thread_num, float similarity_thresh) const
		{
			int person_num = persons.size();
			if (person_num == 0 || persons[0].features.size() == 0)
				return false;
			int dim = persons[0].features[0].length;
			if (dim == 0)
				return false;
			
			repeat_pairs.clear();
			repeat_scores.clear();

			std::vector<int> pivot_ids(person_num);
			
			if (max_thread_num <= 1)
			{
				for (int p = 0; p < person_num; p++)
				{
					int cur_num = persons[p].features.size();

					std::vector<float> scores(cur_num*cur_num);
					for (int i = 0; i < cur_num; i++)
					{
						scores[i*cur_num + i] = 1;
						const float* cur_i_feat = persons[p].features[i].pData;
						const float* cur_j_feat;
						for (int j = i + 1; j < cur_num; j++)
						{
							cur_j_feat = persons[p].features[j].pData;
							float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							scores[i*cur_num + j] = tmp_score;
							scores[j*cur_num + i] = tmp_score;
						}
					}
					int pivot_id = -1;
					float sum_score = -FLT_MAX;
					for (int i = 0; i < cur_num; i++)
					{
						float tmp_sum = 0;
						for (int j = 0; j < cur_num; j++)
							tmp_sum += scores[i*cur_num + j];
						if (sum_score < tmp_sum)
						{
							pivot_id = i;
							sum_score = tmp_sum;
						}
					}
					pivot_ids[p] = pivot_id;
				}

				//
				for (int i = 0; i < person_num; i++)
				{
					for (int j = i + 1; j < person_num; j++)
					{
						const float* cur_i_feat = persons[i].features[pivot_ids[i]].pData;
						const float* cur_j_feat = persons[j].features[pivot_ids[j]].pData;
						float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
						if (tmp_score >= similarity_thresh)
						{
							repeat_pairs.push_back(std::make_pair(i, j));
							repeat_scores.push_back(tmp_score);
						}
					}
				}
			}
			else
			{
				int chunk_size = (person_num + max_thread_num - 1) / max_thread_num;
#pragma omp parallel for schedule(static,chunk_size) num_threads(max_thread_num)
				for (int p = 0; p < person_num; p++)
				{
					int cur_num = persons[p].features.size();

					std::vector<float> scores(cur_num*cur_num);
					for (int i = 0; i < cur_num; i++)
					{
						scores[i*cur_num + i] = 1;
						const float* cur_i_feat = persons[p].features[i].pData;
						const float* cur_j_feat;
						for (int j = i + 1; j < cur_num; j++)
						{
							cur_j_feat = persons[p].features[j].pData;
							float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							scores[i*cur_num + j] = tmp_score;
							scores[j*cur_num + i] = tmp_score;
						}
					}
					int pivot_id = -1;
					float sum_score = -FLT_MAX;
					for (int i = 0; i < cur_num; i++)
					{
						float tmp_sum = 0;
						for (int j = 0; j < cur_num; j++)
							tmp_sum += scores[i*cur_num + j];
						if (sum_score < tmp_sum)
						{
							pivot_id = i;
							sum_score = tmp_sum;
						}
					}
					pivot_ids[p] = pivot_id;
				}

#pragma omp parallel for schedule(static,chunk_size) num_threads(max_thread_num)
				for (int i = 0; i < person_num; i++)
				{
					for (int j = i + 1; j < person_num; j++)
					{
						const float* cur_i_feat = persons[i].features[pivot_ids[i]].pData;
						const float* cur_j_feat = persons[j].features[pivot_ids[j]].pData;
						float tmp_score = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
						if (tmp_score >= similarity_thresh)
						{
#pragma omp critical
							{
								repeat_pairs.push_back(std::make_pair(i, j));
								repeat_scores.push_back(tmp_score);
							}
						}
					}
				}
			}
			return true;
		}

		bool _detect_lowest_pair(const std::string& out_file, int max_thread_num, float similarity_thresh) const
		{
			std::vector<float> scores;
			std::vector<std::pair<std::string, std::string> > pairs;
			std::vector<std::pair<std::string, std::string>* > pair_ptr;
			if (!_detect_lowest_pair(scores, pairs, max_thread_num, similarity_thresh))
			{
				return false;
			}
			__int64 num = scores.size();
			printf("num = %lld\n", num);
			if (num > 0)
			{
				for (__int64 i = 0; i < num; i++)
					pair_ptr.push_back(&pairs[i]);
				ZQ_MergeSort::MergeSortWithData(&scores[0], &pair_ptr[0], sizeof(std::pair<std::string, std::string>*), num, true);
			}

			FILE* out = 0;
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}

			for (__int64 i = 0; i < num; i++)
			{
				fprintf(out, "%8.3f %s %s\n", scores[i], pair_ptr[i]->first.c_str(), pair_ptr[i]->second.c_str());
			}

			fclose(out);
			return true;
		}

		bool _detect_lowest_pair(std::vector<float>& scores, std::vector<std::pair<std::string, std::string> >& pairs, 
			int max_thread_num, float similarity_thresh) const
		{
			scores.clear();
			pairs.clear();
			int person_num = persons.size();
			if (person_num == 0 || persons[0].features.size() == 0)
				return false;
			int dim = persons[0].features[0].length;

			if (max_thread_num <= 1)
			{
				
				for (int p = 0; p < person_num; p++)
				{
					int num = persons[p].features.size();
					float out_min_score = FLT_MAX;
					int out_i, out_j;
					for (int i = 0; i < num; i++)
					{
						for (int j = i + 1; j < num; j++)
						{
							float tmp_score = ZQ_MathBase::DotProduct(dim, persons[p].features[i].pData, 
								persons[p].features[j].pData);
							if (tmp_score <= out_min_score)
							{
								out_min_score = tmp_score;
								out_i = i;
								out_j = j;
							}
						}
					}
					if (out_min_score <= similarity_thresh)
					{
						scores.push_back(out_min_score);
						pairs.push_back(std::make_pair(persons[p].filenames[out_i], persons[p].filenames[out_j]));
					}
				}
			}
			else
			{
				int chunk_size = 100;
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(max_thread_num)
				for (int p = 0; p < person_num; p++)
				{
					int num = persons[p].features.size();
					float out_min_score = FLT_MAX;
					int out_i, out_j;
					for (int i = 0; i < num; i++)
					{
						for (int j = i + 1; j < num; j++)
						{
							float tmp_score = ZQ_MathBase::DotProduct(dim, persons[p].features[i].pData,
								persons[p].features[j].pData);
							if (tmp_score <= out_min_score)
							{
								out_min_score = tmp_score;
								out_i = i;
								out_j = j;
							}
						}
					}
					if (out_min_score <= similarity_thresh)
					{
#pragma omp critical
						{
							scores.push_back(out_min_score);
							pairs.push_back(std::make_pair(persons[p].filenames[out_i], persons[p].filenames[out_j]));
						}
					}
				}
			}
			return true;
		}
	};

	
}
#endif
