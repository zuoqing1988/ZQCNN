#ifndef _ZQ_FACE_DATABASE_H_
#define _ZQ_FACE_DATABASE_H_
#pragma once
#include <vector>
#include <string>
#include "ZQ_FaceFeature.h"
#include "ZQ_FaceRecognizerSphereFace.h"
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
	};

	
}
#endif
