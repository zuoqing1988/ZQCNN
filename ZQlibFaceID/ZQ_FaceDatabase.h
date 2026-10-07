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
		// 审计修复 2026-10-02（附录 BL.3）：offset 用 __int64, Windows 的 fseek
		// 收的是 32 位 long, 超过 2 GB 的库会定位到错的地方。LP64 的 Linux 上
		// long 本来就是 64 位, 直接 fseek 即可 —— 与 ZQ_FaceGroup::LoadFromFile
		// 里 _ftelli64 / fseek 那个 #ifdef 是同一套做法。
		static bool _fseek64(FILE* f, __int64 pos)
		{
#if defined(_WIN32)
			return 0 == _fseeki64(f, pos, SEEK_SET);
#else
			return 0 == fseek(f, (long)pos, SEEK_SET);
#endif
		}

		// 审计修复 2026-10-02（附录 BL.1）：空库 / 零特征 / 零维一律干净返回 false。
		// 四个分析入口（SelectSubset / DetectLowestPair / DetectRepeatPerson /
		// ExportSimilarityForAllPairs）原来各自写了一份 person_num==0 的检查，
		// 只有 ExportSimilarityForAllPairs 漏了, 现在统一走这一个。
		// _check_valid() 里本来就有 feat_dim 一致性检查，但只有保存路径调它。
		static bool _check_analyzable(const std::vector<Person>& persons, int& out_dim)
		{
			if (persons.size() == 0 || persons[0].features.size() == 0)
				return false;
			int dim = persons[0].features[0].length;
			if (dim <= 0)
				return false;
			out_dim = dim;
			return true;
		}

		// 审计修复 2026-10-02（附录 BL.2）：下面 4 处都是
		//     std::vector<float> scores(cur_num*cur_num);
		// 而 cur_num 是 **int**。加载器允许单人最多 1e7 个特征
		// （_load_feats_binary 的 feat_num 上界），于是：
		//   * cur_num >= 46341 时 cur_num*cur_num 在 int 里回绕成负数
		//     -> vector<float>(负) 先抛 length_error
		//   * cur_num == 65536 时正好回绕成 **0**
		//     -> vector<float>(0) 分配"成功"，紧接着 scores[i*cur_num+i] = 1
		//        就是 4 字节**堆越界写**（ASan 抓到的是 SEGV on address 0）
		// 必须在**算乘法之前**查，所以这里查的是 cur_num 本身而不是它的平方。
		// 上限 16384 = 1 GB / sizeof(float)：一个人不可能有 16384 张脸，
		// 真有这么多的话该换算法，而不是先吃 1 GB 内存。
		static bool _check_pivot_square_size(__int64 cur_num)
		{
			const __int64 max_root = 16384;
			return cur_num >= 0 && cur_num <= max_root;
		}

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
#if defined(_WIN32)
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
#else
			out = fopen(file.c_str(), "wb");
			if (out == NULL)
				return false;
#endif
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
#if defined(_WIN32)
			if (0 != fopen_s(&out, file.c_str(), "wb"))
				return false;
#else
			out = fopen(file.c_str(), "wb");
			if (out == NULL)
				return false;
#endif

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
#if defined(_WIN32)
			if (0 != fopen_s(&in, file.c_str(), "rb"))
				return false;
#else
			in = fopen(file.c_str(), "rb");
			if (in == NULL)
				return false;
#endif

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
			if (person_num <= 0 || person_num > 10000000 || feat_dim <= 0 || feat_dim > 4096)
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
				if (feat_num <= 0 || feat_num > 10000000)
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

					if (len < 0 || len > 65536)
					{
						fclose(in);
						return false;
					}

					if (len > 0)
					{
						buf.resize(len);
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
#if defined(_WIN32)
			if (0 != fopen_s(&out, file.c_str(), "w"))
				return false;
#else
			out = fopen(file.c_str(), "w");
			if (out == NULL)
				return false;
#endif
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
#if defined(_WIN32)
			if (0 != fopen_s(&in, file.c_str(), "r"))
				return false;
#else
			in = fopen(file.c_str(), "r");
			if (in == NULL)
				return false;
#endif
			char line[200] = { 0 };
			while (true)
			{
				line[0] = '\0';
				fgets(line, 199, in);
				if (line[0] == '\0')
					break;
				int len = strlen(line);
				while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
					line[--len] = '\0';
				names.push_back(std::string(line));
				// 附录 JY：与 ZQ_FaceDatabaseCompact::_load_names 同型补齐 ——
				// 行数上界对齐加载器的人数上界（1000 万），超大畸形文件在
				// vector 撑爆前即被拒。
				if (names.size() > 10000000)
				{
					fclose(in);
					return false;
				}
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
					}
					// 维度对不上就整对跳过。原来的 "if (max_id < 0)" 首元素判断
					// 写在 k 循环**里面**: 不匹配时 tmp_score 恒为 -FLT_MAX, 第一个
					// k 命中该分支把 max_id 钉死成 0, 之后 -FLT_MAX < -FLT_MAX 恒假
					// 于是永不更新 —— 拿 512 维去查 128 维的库时, Search 会返回
					// true 并给出一整份 person_j 全是 0 的"结果"。
					if (tmp_score == -FLT_MAX)
						continue;
					if (max_id < 0 || max_score < tmp_score)
					{
						max_id = j;
						max_score = tmp_score;
					}
				}
				person_j[i] = max_id;    // 一个都没匹配上时保持 -1
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
				// person_j 为 -1 表示这个人与查询特征维度对不上 (见上面的 continue),
				// 拿它去索引 filenames 就是 filenames[-1]。维度不匹配的排在
				// 分数最低的一端, 只有匹配的人不足 max_num 时才会落到这里。
				int pj = person_j[ids[i]];
				// 审计修复 2026-10-02（附录 BL.4）：四路输出必须**一起**推或一起
				// 不推。原来只 continue 掉 filenames, 于是 ids/scores/names 有 3 个
				// 而 filenames 只有 0 个 —— 调用方按 ids 的下标去取 filenames[i]
				// 就是越界。上一轮修"维度不匹配就跳过"只修了一半, 这里补齐。
				if (pj < 0 || pj >= (int)database.persons[ids[i]].filenames.size())
				{
					out_ids.pop_back();
					out_scores.pop_back();
					out_names.pop_back();
					continue;
				}
				out_filenames.push_back(database.persons[ids[i]].filenames[pj]);
			}

			double t3 = omp_get_wtime();
			//printf("part1 = %.3f, part2 = %.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
			return true;
		}

		bool _export_similarity_for_all_pairs(const std::string& out_score_file, const std::string& out_flag_file,
			__int64& all_pair_num, __int64& same_pair_num, __int64& notsame_pair_num, int max_thread_num, bool quantization) const
		{
			// 审计修复 2026-10-02（附录 BL.1）：校验提到**开文件之前**。
			// 原来 persons[0].features[0].length 在两个 fopen 之后才取，
			// 于是空库上不但越界，还顺手留下两个 0 字节的产物文件。
			int dim = 0;
			if (!_check_analyzable(persons, dim))
			{
				printf("not a valid database\n");
				return false;
			}
			int person_num = persons.size();

			FILE* out1 = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&out1, out_score_file.c_str(), "wb"))
			{
				printf("failed to create file %s\n", out_score_file.c_str());
				return false;
			}
#else
			out1 = fopen(out_score_file.c_str(), "wb");
			if (out1 == NULL)
			{
				printf("failed to create file %s\n", out_score_file.c_str());
				return false;
			}
#endif

			FILE* out2 = 0;
#if defined(_WIN32)
			if (0 != fopen_s(&out2, out_flag_file.c_str(), "wb"))
			{
				printf("failed to create file %s\n", out_flag_file.c_str());
				fclose(out1);
				return false;
			}
#else
			out2 = fopen(out_flag_file.c_str(), "wb");
			if (out2 == NULL)
			{
				printf("failed to create file %s\n", out_flag_file.c_str());
				fclose(out1);
				return false;
			}
#endif

			__int64 total_face_num = 0;
			std::vector<__int64> cur_face_offset(person_num);
			for (int pp = 0; pp < person_num; pp++)
			{
				cur_face_offset[pp] = total_face_num;
				__int64 cur_face_num = persons[pp].features.size();
				total_face_num += cur_face_num;
			}

			all_pair_num = total_face_num *(total_face_num - 1) / 2;

			// 审计修复 2026-10-02（附录 BL.3）：为了让并行分支的产物与单线程分支
			// **逐字节一致**（原来 fwrite 全在 omp critical 里，落到文件里的顺序
			// 由线程调度决定），先算出每个人在两个文件里各占多少字节：
			//   第 pp 个人对第 i 张脸产生的记录条数 = (cur_face_num-1-i) + F
			//   其中 F = pp 之后所有人的脸数
			//   rec_num[pp]       = cur*(cur-1)/2 + cur*F
			//   score_begin[pp]   = 前面所有人的记录数 * 单条字节数
			//   flag_begin[pp]    = 前面所有人的记录数
			// 之后并行时各自 fseek 到自己那一段写，于是：
			//   * 字节布局与单线程分支完全相同（与线程数、调度策略都无关）
			//   * critical 里只剩"定位 + 写"，点积仍然全并行
			const __int64 score_elem = quantization ? (__int64)sizeof(short) : (__int64)sizeof(float);
			std::vector<__int64> rec_num(person_num, 0);
			{
				__int64 faces_after = 0;
				for (int pp = person_num - 1; pp >= 0; pp--)
				{
					__int64 cur = persons[pp].features.size();
					rec_num[pp] = cur * (cur - 1) / 2 + cur * faces_after;
					faces_after += cur;
				}
			}
			std::vector<__int64> score_begin(person_num, 0), flag_begin(person_num, 0);
			{
				__int64 s = 0, f = 0;
				for (int pp = 0; pp < person_num; pp++)
				{
					score_begin[pp] = s;  s += rec_num[pp] * score_elem;
					flag_begin[pp] = f;   f += rec_num[pp];
				}
			}

			int real_thread_num = __max(1, __min(max_thread_num, omp_get_num_procs() - 1));
			if (real_thread_num == 1)
			{
				// 审计修复 2026-10-02（附录 BL.5）：原来**只有并行分支**会去写
				// same_pair_num / notsame_pair_num, 单线程分支压根不管 ——
				// 调用方（SamplesZQlibFaceID/SampleFaceDatabase*.cpp）拿到的是
				// 自己传进去的初值。compact 版的单线程分支里是有的, 两边不对称。
				__int64 tmp_same_pair_num = 0;
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
							tmp_same_pair_num++;
							idx++;
						}
						for (__int64 qq = pp + 1; qq < person_num; qq++)
						{
							for (__int64 j = 0; j < persons[qq].features.size(); j++)
							{
								cur_j_feat = persons[qq].features[j].pData;
								scores[idx] = ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
								flags[idx] = 0;
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
				same_pair_num = tmp_same_pair_num;
				notsame_pair_num = all_pair_num - same_pair_num;
			}
			else
			{
				int chunk_size = 100;
				int handled[1] = { 0 };
				__int64 tmp_same_pair_num[1] = { 0 };
				int write_failed[1] = { 0 };
				printf("real_thread_num = %d\n", real_thread_num);
#pragma omp parallel for schedule(dynamic,chunk_size) num_threads(real_thread_num) shared(handled, tmp_same_pair_num, write_failed)
				for (int pp = 0; pp < person_num; pp++)
				{
					__int64 cur_face_num = persons[pp].features.size();
					__int64 faces_after = total_face_num - cur_face_offset[pp] - cur_face_num;
					__int64 max_pair_num = (total_face_num - cur_face_offset[pp] - 1);
					std::vector<float> scores(max_pair_num);
					std::vector<char> flags(max_pair_num);
					for (__int64 i = 0; i < cur_face_num; i++)
					{
						float* cur_i_feat = persons[pp].features[i].pData;
						float* cur_j_feat;
						int idx = 0;
						__int64 same_cnt = 0;
						for (__int64 j = i+1; j < cur_face_num; j++)
						{
							cur_j_feat = persons[pp].features[j].pData;
							scores[idx] = ZQ::ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
							flags[idx] = 1;
							same_cnt++;
							idx++;
						}
						for (__int64 qq = pp + 1; qq < person_num; qq++)
						{
							for (__int64 j = 0; j < persons[qq].features.size(); j++)
							{
								cur_j_feat = persons[qq].features[j].pData;
								scores[idx] = ZQ::ZQ_MathBase::DotProduct(dim, cur_i_feat, cur_j_feat);
								flags[idx] = 0;
								idx++;
							}
						}
						if (idx > 0)
						{
							// 第 i 张脸之前已经写掉的记录数（上面 rec_num 的闭式解）
							__int64 rec_before = i * (cur_face_num - 1 + faces_after) - i * (i - 1) / 2;
							// 点积在上面就全算完了, critical 里只有"定位 + 写"。
							// fseek 与 fwrite 必须**一起**在 critical 里: glibc 的
							// fseek 会先把写缓冲刷出去, 与别的线程的 fwrite 并发
							// 会把对方的数据写到错的位置。
#pragma omp critical
							{
								if (!_fseek64(out1, score_begin[pp] + rec_before * score_elem)
									|| !_fseek64(out2, flag_begin[pp] + rec_before))
								{
									(*write_failed) = 1;
								}
								else
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
									(*tmp_same_pair_num) += same_cnt;
								}
							}
						}
					}
#pragma omp critical
					{
						(*handled)++;
						printf("%d/%d\n", *handled, person_num);
					}
				}
				if ((*write_failed) != 0)
				{
					fclose(out1);
					fclose(out2);
					return false;
				}
				// 并行区里的自增已删(非原子的共享写, 且结果本来就会被这里整体覆盖)
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
#if defined(_WIN32)
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}
#else
			out = fopen(out_file.c_str(), "w");
			if (out == NULL)
			{
				return false;
			}
#endif

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
#if defined(_WIN32)
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}
#else
			out = fopen(out_file.c_str(), "w");
			if (out == NULL)
			{
				return false;
			}
#endif

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
			int dim = 0;
			if (!_check_analyzable(persons, dim))
				return false;
			int person_num = persons.size();
			// 审计修复 2026-10-02（附录 BL.2）：cur_num*cur_num 是在下面四个地方
			// 各算一遍的 int 乘法，单人的脸数一旦 >= 46341 就回绕。查放在**并行区
			// 之外**的串行一遍里，于是并行分支里可以放心地 return/continue，
			// 代价只是 person_num 次整数比较。
			for (int p = 0; p < person_num; p++)
			{
				if (!_check_pivot_square_size(persons[p].features.size()))
				{
					printf("person %d has too many features (%d) for the pivot search\n",
						p, (int)persons[p].features.size());
					return false;
				}
			}

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
				// 审计修复 2026-10-02（附录 BL.3）：原来这里用 omp critical 往三个
				// 共享 vector 里 push_back，输出顺序由线程调度决定 —— 同一个库、
				// 同一个线程数，两次跑出来的 SelectSubset 文件内容不同，而单线程
				// 分支是**按 p 升序**的。改成"每个人一个自己的槽位，并行填，
				// 串行按 p 升序收"：索引互不相同，天然无竞争，连 critical 都
				// 不需要，同时让输出与线程数彻底无关。
				std::vector<int> sel_flag(person_num, 0);
				std::vector<int> sel_pivot(person_num, 0);
				std::vector<std::vector<int> > sel_good(person_num);
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
						sel_flag[p] = 1;
						sel_pivot[p] = pivot_id;
						sel_good[p] = ids;
					}
				}
				for (int p = 0; p < person_num; p++)
				{
					if (sel_flag[p] == 0)
						continue;
					person_ids.push_back(p);
					pivot_ids.push_back(sel_pivot[p]);
					other_good_ids.push_back(sel_good[p]);
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
#if defined(_WIN32)
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}
#else
			out = fopen(out_file.c_str(), "w");
			if (out == NULL)
			{
				return false;
			}
#endif
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
			int dim = 0;
			if (!_check_analyzable(persons, dim))
				return false;
			int person_num = persons.size();
			// 审计修复 2026-10-02（附录 BL.2）：同 _select_subset, 串行一遍拦住
			// cur_num*cur_num 的 int 回绕（65536 时回绕成 0 -> 堆越界写）。
			for (int p = 0; p < person_num; p++)
			{
				if (!_check_pivot_square_size(persons[p].features.size()))
				{
					printf("person %d has too many features (%d) for the pivot search\n",
						p, (int)persons[p].features.size());
					return false;
				}
			}

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

				// 审计修复 2026-10-02（附录 BL.3）：原来第二个 parallel for 用
				// omp critical 往共享 vector push_back，收集顺序由线程调度决定。
				// 上层虽然会按分数 MergeSort 一遍，但**分数完全相同**的对之间
				// 顺序仍然是任意的，产物逐字节不可复现。改成"按 i 分槽位，
				// 并行填，串行按 i 升序收"：索引互不相同，不需要 critical。
				std::vector<std::vector<std::pair<int, int> > > per_i_pair(person_num);
				std::vector<std::vector<float> > per_i_score(person_num);
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
							per_i_pair[i].push_back(std::make_pair(i, j));
							per_i_score[i].push_back(tmp_score);
						}
					}
				}
				for (int i = 0; i < person_num; i++)
				{
					for (size_t k = 0; k < per_i_pair[i].size(); k++)
					{
						repeat_pairs.push_back(per_i_pair[i][k]);
						repeat_scores.push_back(per_i_score[i][k]);
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
#if defined(_WIN32)
			if (0 != fopen_s(&out, out_file.c_str(), "w"))
			{
				return false;
			}
#else
			out = fopen(out_file.c_str(), "w");
			if (out == NULL)
			{
				return false;
			}
#endif

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
			int dim = 0;
			if (!_check_analyzable(persons, dim))
				return false;
			int person_num = persons.size();

			if (max_thread_num <= 1)
			{
				
				for (int p = 0; p < person_num; p++)
				{
					int num = persons[p].features.size();
					float out_min_score = FLT_MAX;
					// 特征数 < 2 时下面的内层循环一次都不跑, out_i/out_j 会是栈上
					// 未定义值, 再拿 filenames[out_i] 索引就是越界读。
					int out_i = 0, out_j = 0;
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
				// 审计修复 2026-10-02（附录 BL.3）：原来用 omp critical 往两个共享
				// vector 里 push_back，收集顺序由线程调度决定。chunk_size=100，
				// 所以人少于 100 时**碰巧**只有一个 chunk、一个线程在干活，输出是
				// 确定的 —— 也就是说"人数少时看着正常，人数一多就不可复现"。
				// 改成按 p 分槽位：并行填，串行按 p 升序收，不需要 critical。
				std::vector<float> sel_score(person_num, 0.f);
				std::vector<char> sel_flag(person_num, 0);
				std::vector<std::pair<std::string, std::string> > sel_pair(person_num);
				int chunk_size = 100;
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(max_thread_num)
				for (int p = 0; p < person_num; p++)
				{
					int num = persons[p].features.size();
					float out_min_score = FLT_MAX;
					// 特征数 < 2 时下面的内层循环一次都不跑, out_i/out_j 会是栈上
					// 未定义值, 再拿 filenames[out_i] 索引就是越界读。
					int out_i = 0, out_j = 0;
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
						sel_flag[p] = 1;
						sel_score[p] = out_min_score;
						sel_pair[p] = std::make_pair(persons[p].filenames[out_i], persons[p].filenames[out_j]);
					}
				}
				for (int p = 0; p < person_num; p++)
				{
					if (sel_flag[p] == 0)
						continue;
					scores.push_back(sel_score[p]);
					pairs.push_back(sel_pair[p]);
				}
			}
			return true;
		}
	};

	
}
#endif
