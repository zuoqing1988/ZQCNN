#ifndef _ZQ_FACEID_PRECISION_EVALUATION_H_
#define _ZQ_FACEID_PRECISION_EVALUATION_H_
#pragma once
// 审计修复 2026-10-03（附录 EG）：这一行必须放在**最前面**。
// 本文件（以及下面 include 的 ZQ_MathBase.h）用了 __min / __max / __int64，
// 这三个在 ZQ_CNN_CompileConfig.h 里为非 MSVC 提供了可移植定义。
// 之前这里 include 的是 <opencv2\opencv.hpp>（**反斜杠**），加上 MSVC 专有写法，
// 整份头在 Linux/gcc 上**根本编不过** —— 而全仓没有任何 .cpp include 它，
// 所以这个缺陷在两边构建里都不会暴露。
#include "ZQ_CNN_CompileConfig.h"
#include "ZQ_FaceRecognizer.h"
#include "ZQ_FaceFeature.h"
#include "ZQ_MathBase.h"
#include "ZQ_MergeSort.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdlib.h>
#include <string>
#include <errno.h>
#include <omp.h>
namespace ZQ
{
	class ZQ_FaceIDPrecisionEvaluation
	{
		class EvaluationPair
		{
		public:
			std::string fileL;
			std::string nameL;
			int idL;
			std::string fileR;
			std::string nameR;
			int idR;
			int flag; //-1 or 1
			ZQ_FaceFeature featL;
			ZQ_FaceFeature featR;
			bool valid;

			// 审计修复 2026-10-06（附录 IH.11）：原来**没有默认构造函数**，
			// 于是 `EvaluationPair cur_pair;`（`_parse_lfw_list` 里两个分支都有）
			// 之后 idL/idR/flag/valid 全是**未定值**；
			// 紧接着 `pairs[i].push_back(cur_pair)` 走**隐式拷贝构造**，
			// 那一步就把未定值读了一遍 —— UBSan 报得很直白：
			//     ZQ_FaceIDPrecisionEvaluation.h:25:9: runtime error: load of value 37,
			//     which is not a valid value for type 'bool'
			// （值每次都不一样，因为它读的是栈垃圾。）
			// 影响面：valid/flag 在**所有**使用点之前都会被重新赋值，所以结果碰巧是对的；
			// 但「拷贝未定值」本身就是 UB，编译器有权基于它做任何假设。
			// 而且这条是 `zq_lfw_eval` 门禁在 **UBSan** 下第一次跑出来的 ——
			// ASan 那一轴完全看不见（附录 CA.3 说的「观测手段决定你看得见什么」）。
			EvaluationPair()
			{
				idL = 0; idR = 0; flag = 0; valid = false;
			}
		};

		class EvaluationSingle
		{
		public:
			std::string name;
			int id;
			ZQ_FaceFeature feat;

			EvaluationSingle& operator = (const EvaluationSingle& v2)
			{
				name = v2.name;
				id = v2.id;
				feat.CopyData(v2.feat);
				return *this;
			}

			bool operator < (const EvaluationSingle& v2) const
			{
#if defined(_WIN32)
				int cmp_v = _strcmpi(name.c_str(), v2.name.c_str());
#else
				int cmp_v = strcmp(name.c_str(), v2.name.c_str());
#endif
				if (cmp_v < 0)
					return true;
				else if (cmp_v > 0)
					return false;
				else
				{
					return id < v2.id;
				}
			}

			bool operator > (const EvaluationSingle& v2) const
			{
#if defined(_WIN32)
				int cmp_v = _strcmpi(name.c_str(), v2.name.c_str());
#else
				int cmp_v = strcmp(name.c_str(), v2.name.c_str());
#endif
				if (cmp_v > 0)
					return true;
				else if (cmp_v < 0)
					return false;
				else
				{
					return id > v2.id;
				}
			}

			bool operator == ( const EvaluationSingle& v2) const
			{
#if defined(_WIN32)
				int cmp_v = _strcmpi(name.c_str(), v2.name.c_str());
#else
				int cmp_v = strcmp(name.c_str(), v2.name.c_str());
#endif
				return cmp_v == 0 && id == v2.id;
			}

			bool SameName(const EvaluationSingle& v2) const
			{
#if defined(_WIN32)
				int cmp_v = _strcmpi(name.c_str(), v2.name.c_str());
#else
				int cmp_v = strcmp(name.c_str(), v2.name.c_str());
#endif
				return cmp_v == 0;
			}
		};

	public:
		// 审计修复 2026-10-06（附录 IH.3 / IH.4）：list 文件头里的两个数来自
		// **不可信输入**，必须和同族解析器一样有上界。
		// `ZQ_FaceClustersForVideo.h:122` 对 cluster_num 的判据是
		// `> 0 && <= 1000000`，`ZQ_FaceGroup.h` 对 num 是 `< 1000000` ——
		// 这里原来**只有 `> 0`**，于是：
		//   * `pairs.resize(part_num)` 会被一个 20 亿的头变成 48 GB 分配，
		//     分配失败时抛出的 bad_alloc 没有任何人接（函数不是 noexcept，
		//     异常会一路穿到 main）；
		//   * `2 * half_pair_num` 在 half_pair_num > 2^30 时**int 回绕成负**，
		//     `for (j = 0; j < 2*half_pair_num; j++)` 一次都不跑，
		//     fgets 的 NULL 检查（它在这个循环体**里面**）也就一次都不执行，
		//     于是"文件里一行数据都没有"被当成解析成功返回 true。
		// 取 1000000 是为了和 ZQ_FaceClustersForVideo.h 的 cluster_num 对齐；
		// LFW 官方是 10 折，十万折的 list 文件本来就不存在。
		static const int MAX_LFW_PART_NUM = 1000000;
		static const int MAX_LFW_HALF_PAIR_NUM = 1000000;

	public:
		static bool EvaluationOnLFW(std::vector<ZQ_FaceRecognizer*>& recognizers, const std::string& list_file, const std::string& folder, bool use_flip)
		{
			int recognizer_num = recognizers.size();
			if (recognizer_num == 0)
				return false;
			int real_num_threads = __max(1, __min(recognizer_num, omp_get_num_procs() - 1));

			// 审计修复 2026-10-06（附录 IH.4）：原来只有 recognizer_num == 0 一道门。
			// GetFeatDim() 若是 0 或负，real_dim 就是 0/负，后面
			// `featL.pData + feat_dim` 与 `ChangeSize(0)` 全部失去意义。
			int feat_dim = recognizers[0]->GetFeatDim();
			if (feat_dim <= 0 || feat_dim > (1 << 20))
			{
				printf("invalid feat_dim = %d from recognizer\n", feat_dim);
				return false;
			}
			int real_dim = use_flip ? (feat_dim * 2) : feat_dim;
			printf("feat_dim = %d, real_dim = %d\n", feat_dim, real_dim);
			std::vector<std::vector<EvaluationPair> > pairs;
			if (!_parse_lfw_list(list_file, folder, pairs))
			{
				printf("failed to parse list file %s\n", list_file.c_str());
				// 审计修复 2026-10-06（附录 IH.5）：原来写的是 `return EXIT_FAILURE`。
				// 这个函数返回 **bool**，而 EXIT_FAILURE == 1 —— 于是
				// "list 文件根本打不开"被报告成 **true**。
				// 七个 SampleEvaluationOnLFW* 全都是 `return EvaluationOnLFW(...)`
				// 直接当进程退出码，调用方 `if (!...)` 的失败分支**永远进不去**。
				return false;
			}

			printf("parse list file %s done!\n", list_file.c_str());
			int part_num = pairs.size();
			std::vector<std::pair<int, int> > pair_list;
			for (int i = 0; i < part_num; i++)
			{
				for (int j = 0; j < pairs[i].size(); j++)
				{
					pair_list.push_back(std::make_pair(i, j));
				}
			}

			double t1 = omp_get_wtime();
			if (real_num_threads == 1)
			{
				int handled_num = 0;
				for (int nn = 0; nn < pair_list.size(); nn++)
				{
					handled_num++;
					if (handled_num % 100 == 0)
						printf("%d handled\n", handled_num);
					int i = pair_list[nn].first;
					int j = pair_list[nn].second;
					pairs[i][j].featL.ChangeSize(real_dim);
					pairs[i][j].featR.ChangeSize(real_dim);
					cv::Mat imgL = cv::imread(pairs[i][j].fileL);
					if (imgL.empty())
					{
						printf("failed to load image %s\n", pairs[i][j].fileL.c_str());
						pairs[i][j].valid = false;
						continue;
					}
					cv::Mat imgR = cv::imread(pairs[i][j].fileR);
					if (imgR.empty())
					{
						printf("failed to load image %s\n", pairs[i][j].fileR.c_str());
						pairs[i][j].valid = false;
						continue;
					}
					if (!recognizers[0]->ExtractFeature(imgL.data, imgL.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featL.pData, true))
					{
						printf("failed to extract feature for image %s\n", pairs[i][j].fileL.c_str());
						pairs[i][j].valid = false;
						continue;
					}
					if (!recognizers[0]->ExtractFeature(imgR.data, imgR.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featR.pData, true))
					{
						printf("failed to extract feature for image %s\n", pairs[i][j].fileR.c_str());
						pairs[i][j].valid = false;
						continue;
					}
					if (use_flip)
					{
						cv::flip(imgL, imgL, 1);
						cv::flip(imgR, imgR, 1);
						if (!recognizers[0]->ExtractFeature(imgL.data, imgL.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featL.pData+feat_dim, true))
						{
							printf("failed to extract feature for image %s\n", pairs[i][j].fileL.c_str());
							pairs[i][j].valid = false;
							continue;
						}
						if (!recognizers[0]->ExtractFeature(imgR.data, imgR.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featR.pData+feat_dim, true))
						{
							printf("failed to extract feature for image %s\n", pairs[i][j].fileR.c_str());
							pairs[i][j].valid = false;
							continue;
						}
					}
					pairs[i][j].valid = true;
				}
			}
			else
			{
				int handled_num = 0;
#pragma omp parallel for  schedule(dynamic, 10) num_threads(real_num_threads)
				for (int nn = 0; nn < pair_list.size(); nn++)
				{
#pragma omp critical
					{
						handled_num++;
						if (handled_num % 100 == 0)
						{
							printf("%d handled\n", handled_num);
						}
					}
					int thread_id = omp_get_thread_num();
					int i = pair_list[nn].first;
					int j = pair_list[nn].second;
					pairs[i][j].featL.ChangeSize(real_dim);
					pairs[i][j].featR.ChangeSize(real_dim);
					cv::Mat imgL = cv::imread(pairs[i][j].fileL);
					if (imgL.empty())
					{
#pragma omp critical
						{
							printf("failed to load image %s\n", pairs[i][j].fileL.c_str());

						}
						pairs[i][j].valid = false;
						continue;
					}
					cv::Mat imgR = cv::imread(pairs[i][j].fileR);
					if (imgR.empty())
					{
#pragma omp critical
						{
							printf("failed to load image %s\n", pairs[i][j].fileR.c_str());

						}
						pairs[i][j].valid = false;
						continue;
					}
					if (!recognizers[thread_id]->ExtractFeature(imgL.data, imgL.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featL.pData, true))
					{
#pragma omp critical
						{
							printf("failed to extract feature for image %s\n", pairs[i][j].fileL.c_str());

						}
						pairs[i][j].valid = false;
						continue;
					}
					if (!recognizers[thread_id]->ExtractFeature(imgR.data, imgR.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featR.pData, true))
					{
#pragma omp critical
						{
							printf("failed to extract feature for image %s\n", pairs[i][j].fileR.c_str());

						}
						pairs[i][j].valid = false;
						continue;
					}
					if (use_flip)
					{
						cv::flip(imgL, imgL, 1);
						cv::flip(imgR, imgR, 1);
						if (!recognizers[thread_id]->ExtractFeature(imgL.data, imgL.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featL.pData + feat_dim, true))
						{
#pragma omp critical
							{
								printf("failed to extract feature for image %s\n", pairs[i][j].fileL.c_str());
							}
							pairs[i][j].valid = false;
							continue;
						}
						if (!recognizers[thread_id]->ExtractFeature(imgR.data, imgR.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, pairs[i][j].featR.pData + feat_dim, true))
						{
#pragma omp critical
							{
								printf("failed to extract feature for image %s\n", pairs[i][j].fileR.c_str());
							}
							pairs[i][j].valid = false;
							continue;
						}
					}
					
					pairs[i][j].valid = true;
				}
			}
			printf("extract feature done!");
			double t2 = omp_get_wtime();
			printf("extract features cost: %.3f secs\n", t2 - t1);

			int erased_num = 0;
			for (int i = 0; i < part_num; i++)
			{
				for (int j = pairs[i].size() - 1; j >= 0; j--)
				{
					if (!pairs[i][j].valid)
					{
						pairs[i].erase(pairs[i].begin() + j);
						erased_num++;
					}
					else
					{
						ZQ_MathBase::Normalize(real_dim, pairs[i][j].featL.pData);
						ZQ_MathBase::Normalize(real_dim, pairs[i][j].featR.pData);
					}
				}
			}
			printf("%d pairs haved been erased\n", erased_num);

			std::vector<EvaluationSingle> singles;
			for (int i = 0; i < part_num; i++)
			{
				for (int j = 0; j < pairs[i].size(); j++)
				{
					EvaluationSingle cur_single;
					cur_single.name = pairs[i][j].nameL;
					cur_single.id = pairs[i][j].idL;
					cur_single.feat.CopyData(pairs[i][j].featL);
					singles.push_back(cur_single);
					cur_single.name = pairs[i][j].nameR;
					cur_single.id = pairs[i][j].idR;
					cur_single.feat.CopyData(pairs[i][j].featR);
					singles.push_back(cur_single);
				}
			}

			float ACC = _compute_accuracy(pairs);
			_compute_far_tar(singles, real_num_threads);
			return true;
		}


	private:
		static float _compute_accuracy(const std::vector<std::vector<EvaluationPair> >& pairs)
		{
			int part_num = pairs.size();
			std::vector<float> ACCs(part_num);
			float ACC = 0;
			int done_num = 0;
			for (int i = 0; i < part_num; i++)
			{
				std::vector<EvaluationPair> val_pairs;
				for (int j = 0; j < part_num; j++)
				{
					if (j != i)
						val_pairs.insert(val_pairs.end(), pairs[j].begin(), pairs[j].end());
				}

				// 审计修复 2026-10-06（附录 IH.8）：留一法在 part_num == 1 时
				// 把**唯一那一折也留掉了**，val_pairs 必然是空的。
				// 原来不管三七二十一接着算：_compute_mu 提前 return false、
				// mu 保持 length==0，于是 _compute_scores 的 feat_dim 成了 0，
				// **所有 test score 恒为 0**，阈值也退化成 0，
				// 最后打印一行 "0  0.00% (threshold = 0.000000)" ——
				// 一份看起来正常、实际毫无意义的准确率。
				// 现在明说跳过，并且不算进平均值。
				if (val_pairs.empty())
					continue;

				ZQ_FaceFeature mu;
				if (!_compute_mu(val_pairs, mu))
					continue;
				if (mu.length <= 0)
					continue;
				std::vector<double> val_scores, test_scores;
				if (!_compute_scores(val_pairs, mu, val_scores))
					continue;
				if (!_compute_scores(pairs[i], mu, test_scores))
					continue;
				double threshold = _get_threshold(val_pairs, val_scores, 10000);
				ACCs[i] = _get_accuracy(pairs[i], test_scores, threshold);
				ACC += ACCs[i];
				done_num++;
				printf("%d\t%2.2f%% (threshold = %f)\n", i, ACCs[i] * 100, threshold);

				/*const static int BUF_LEN = 50;
				char file[BUF_LEN];
				sprintf_s(file, BUF_LEN, "%d_mu.txt", i);
				FILE* out = 0;
				fopen_s(&out, file, "w");
				for (int k = 0; k < mu.length; k++)
					fprintf(out, "%12.6f\n", mu.pData[k]);
				fclose(out);
				sprintf_s(file, BUF_LEN, "%d_validscores.txt", i);
				fopen_s(&out, file, "w");
				for (int k = 0; k < val_scores.size(); k++)
					fprintf(out, "%12.6f\n", val_scores[k]);
				fclose(out);
				sprintf_s(file, BUF_LEN, "%d_testscores.txt", i);
				fopen_s(&out, file, "w");
				for (int k = 0; k < test_scores.size(); k++)
					fprintf(out, "%12.6f\n", test_scores[k]);
				fclose(out);*/
			}

			printf("----------------\n");
			if (done_num > 0)
				printf("AVE\t%2.2f%%\n", ACC / done_num * 100);
			else
				printf("AVE\tn/a (no fold had a non-empty leave-one-out set)\n");
			return ACC;
		}

		// 审计修复 2026-10-06（附录 IH.6）：原来直接 `atoi`。
		// atoi 对超出 int 范围的串是**未定义行为**（glibc 的实现是 UB，不是饱和），
		// 而 list 文件是不可信输入。改成 strtol + 显式夹取，
		// 顺便让 `%04i` 的输出有确定宽度 —— num2str 是 200 字节，
		// 原先不会溢出，但 out-of-range 时的输出是不可预测的。
		static int _safe_id(const std::string& s)
		{
			errno = 0;
			char* endp = 0;
			long v = strtol(s.c_str(), &endp, 10);
			if (endp == s.c_str()) return 0;          // 一位都没解析出来
			if (errno == ERANGE || v > 2147483647L) return 2147483647;
			if (v < -2147483647L - 1) return -2147483647 - 1;
			return (int)v;
		}

		static bool _parse_lfw_list(const std::string& list_file, const std::string& folder, std::vector<std::vector<EvaluationPair> >& pairs)
		{
			FILE* in = 0;
#if defined(_WIN32)
			if(0 != fopen_s(&in, list_file.c_str(), "r"))
				return false;
#else
			in = fopen(list_file.c_str(), "r");
			if (in == NULL)
				return false;
#endif

			int part_num = 0, half_pair_num = 0;
			const static int BUF_LEN = 200;
			char line[BUF_LEN];
			if (NULL == fgets(line, BUF_LEN, in))
			{
				fclose(in);
				return false;
			}
#if defined(_WIN32)
			if (2 != sscanf_s(line, "%d%d", &part_num, &half_pair_num))
#else
			if (2 != sscanf(line, "%d%d", &part_num, &half_pair_num))
#endif
			{
				fclose(in);
				return false;
			}
			// 审计修复 2026-10-06（附录 IH.3）：原来只有 `part_num <= 0 || half_pair_num <= 0`。
			// 上界与 EvaluationOnLFW 里那两个常量对齐（见那里的说明）。
			if (part_num <= 0 || part_num > MAX_LFW_PART_NUM
				|| half_pair_num <= 0 || half_pair_num > MAX_LFW_HALF_PAIR_NUM)
			{
				fclose(in);
				return false;
			}
			pairs.resize(part_num);

			// 审计修复 2026-10-06（附录 IH.3）：`2 * half_pair_num` 在 int 里会回绕。
			// 即使有了上面的上界（1e6），乘积本身仍然在 int 范围外时不该依赖
			// "回绕成负所以循环不跑"这种行为 —— 那是 UB，不是拒绝。
			// 显式用 __int64 算**每折的行数**（外层那个 i 已经按 part_num 走了）。
			__int64 row_num_per_part = (__int64)2 * (__int64)half_pair_num;
			__int64 bad_row_num = 0;

			std::vector<std::string> strings;
			for (int i = 0; i < part_num; i++)
			{
				for (__int64 j = 0; j < row_num_per_part; j++)
				{
					if (NULL == fgets(line, 199, in))
					{
						fclose(in);
						return false;
					}
					int len = strlen(line);
					while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
						line[--len] = '\0';
					std::string input = line;
					_split_string(input, std::string("\t"), strings);
					if (strings.size() == 3)
					{
						EvaluationPair cur_pair;
						cur_pair.nameL = strings[0];
						cur_pair.nameR = strings[0];
						int idL = _safe_id(strings[1]);
						int idR = _safe_id(strings[2]);
						cur_pair.idL = idL;
						cur_pair.idR = idR;
						char num2str[BUF_LEN];
#if defined(_WIN32)
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", idL);
						cur_pair.fileL = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", idR);
						cur_pair.fileR = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
#else
						sprintf(num2str, "_%04i.jpg", idL);
						cur_pair.fileL = folder + "/" + strings[0] + "/" + strings[0] + std::string(num2str);
						sprintf(num2str, "_%04i.jpg", idR);
						cur_pair.fileR = folder + "/" + strings[0] + "/" + strings[0] + std::string(num2str);
#endif
						cur_pair.flag = 1;
						pairs[i].push_back(cur_pair);
					}
					else if (strings.size() == 4)
					{
						EvaluationPair cur_pair;
						cur_pair.nameL = strings[0];
						cur_pair.nameR = strings[2];
						int idL = _safe_id(strings[1]);
						int idR = _safe_id(strings[3]);
						cur_pair.idL = idL;
						cur_pair.idR = idR;
						char num2str[BUF_LEN];
#if defined(_WIN32)
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", idL);
						cur_pair.fileL = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", idR);
						cur_pair.fileR = folder + "\\" + strings[2] + "\\" + strings[2] + std::string(num2str);
#else
						sprintf(num2str, "_%04i.jpg", idL);
						cur_pair.fileL = folder + "/" + strings[0] + "/" + strings[0] + std::string(num2str);
						sprintf(num2str, "_%04i.jpg", idR);
						cur_pair.fileR = folder + "/" + strings[2] + "/" + strings[2] + std::string(num2str);
#endif
						cur_pair.flag = -1;
						pairs[i].push_back(cur_pair);
					}
					else
					{
						// 审计修复 2026-10-06（附录 IH.7）：原来是**静默丢掉**这一行。
						// 结果是"头里说 10 折、实际只解析出 3 对"这种情况**不可见** ——
						// 后面的留一法照跑，出一份看起来正常的准确率。
						// 现在计数并在结尾报告；一行都没解析出来时**判失败**，
						// 因为那种情况下 pairs 全空，正是 IH.1 那个空 vector 越界的入口。
						bad_row_num++;
					}
				}
			}
			fclose(in);
			__int64 got = (__int64)part_num * row_num_per_part - bad_row_num;
			if (bad_row_num > 0)
				printf("WARNING: %lld of %lld rows in %s are malformed and were skipped\n",
					(long long)bad_row_num, (long long)((__int64)part_num * row_num_per_part),
					list_file.c_str());
			if (got <= 0)
			{
				printf("no usable pair parsed from %s\n", list_file.c_str());
				return false;
			}
			return true;
		}

		static bool _compute_mu(const std::vector<EvaluationPair>& val_pairs, ZQ_FaceFeature& mu)
		{
			if (val_pairs.size() == 0)
				return false;
			int feat_dim = val_pairs[0].featL.length;
			mu.ChangeSize(feat_dim);
			std::vector<double> sum(feat_dim);
			for (int dd = 0; dd < feat_dim; dd++)
				sum[dd] = 0;
			for (int i = 0; i < val_pairs.size(); i++)
			{
				for (int dd = 0; dd < feat_dim; dd++)
				{
					sum[dd] += val_pairs[i].featL.pData[dd];
					sum[dd] += val_pairs[i].featR.pData[dd];
				}
			}
			for (int dd = 0; dd < feat_dim; dd++)
			{
				mu.pData[dd] = sum[dd] / (2 * val_pairs.size());
			}
			return true;
		}

		static bool _compute_scores(const std::vector<EvaluationPair>& pairs, const ZQ_FaceFeature& mu, std::vector<double>& scores)
		{
			int num = pairs.size();
			if (num == 0)
				return false;
			scores.resize(num);

			int feat_dim = mu.length;
			std::vector<double> featL(feat_dim), featR(feat_dim);

			for (int i = 0; i < num; i++)
			{
				for (int j = 0; j < feat_dim; j++)
				{
					featL[j] = pairs[i].featL.pData[j] - mu.pData[j];
					featR[j] = pairs[i].featR.pData[j] - mu.pData[j];
				}
				double lenL = 0, lenR = 0;
				for (int j = 0; j < feat_dim; j++)
				{
					lenL += featL[j] * featL[j];
					lenR += featR[j] * featR[j];
				}
				lenL = sqrt(lenL);
				lenR = sqrt(lenR);
				if (lenL != 0)
				{
					for (int j = 0; j < feat_dim; j++)
						featL[j] /= lenL;
				}
				if (lenR != 0)
				{
					for (int j = 0; j < feat_dim; j++)
						featR[j] /= lenR;
				}
				double sco = 0;

				for (int j = 0; j < feat_dim; j++)
					sco += featL[j] * featR[j];
				scores[i] = sco;
			}
			return true;
		}

		static float _get_threshold(const std::vector<EvaluationPair>& pairs, const std::vector<double>& scores, int thrNum)
		{
			std::vector<double> accurarys(2 * thrNum + 1);
			for (int i = 0; i < 2 * thrNum + 1; i++)
			{
				double threshold = (double)i / thrNum - 1;
				accurarys[i] = _get_accuracy(pairs, scores, threshold);
			}
			double max_acc = accurarys[0];
			for (int j = 1; j < 2 * thrNum + 1; j++)
				max_acc = __max(max_acc, accurarys[j]);

			double sum_threshold = 0;
			int sum_num = 0;
			for (int i = 0; i < 2 * thrNum + 1; i++)
			{
				if (max_acc == accurarys[i])
				{
					sum_threshold += (double)i / thrNum - 1;
					sum_num++;
				}
			}
			return sum_threshold / sum_num;
		}

		static float _get_accuracy(const std::vector<EvaluationPair>& pairs, const std::vector<double>& scores, double threshold)
		{
			if (pairs.size() == 0 || pairs.size() != scores.size())
				return 0;

			double sum = 0;
			for (int i = 0; i < pairs.size(); i++)
			{
				if (pairs[i].flag > 0 && scores[i] > threshold || pairs[i].flag < 0 && scores[i] < threshold)
					sum++;
			}
			return sum / pairs.size();
		}

		static void _split_string(const std::string& s, const std::string& delim, std::vector< std::string >& ret)
		{
			size_t last = 0;
			size_t index = s.find_first_of(delim, last);
			ret.clear();
			while (index != std::string::npos)
			{
				ret.push_back(s.substr(last, index - last));
				last = index + 1;
				index = s.find_first_of(delim, last);
			}
			if (index - last>0)
			{
				ret.push_back(s.substr(last, index - last));
			}
		}


		static void _compute_far_tar(std::vector<EvaluationSingle>& singles, int real_num_threads)
		{
			printf("compute far tar begin\n");
			// 审计修复 2026-10-06（附录 IH.1）：这里原来**没有空表守卫**。
			// EvaluationOnLFW 末尾无条件调它，而 singles 是由
			// "所有还活着的 pair" 堆出来的 —— list 文件本身合法、
			// 只是**一张图都读不到**（目录写错、图片被挪走、格式不对）时，
			// 每一对都被标成 invalid 并 erase 掉，singles 变成**空 vector**。
			// 紧接着的 `&singles[0]` 之后，`int dim = singles[0].feat.length;`
			// 是**真的解引用**空 vector 的第 0 号元素 —— 实测 ASan:
			//   SEGV on unknown address 0x28 ... in _compute_far_tar
			//   ZQ_FaceIDPrecisionEvaluation.h:648
			// 也就是说"路径配错了"这种最常见的用户错误会**直接崩掉进程**，
			// 而不是给出"没读到任何图片"的提示。
			// image_num < 2 同理：只剩 0/1 张图时下面那个 O(N^2) 的双重循环
			// 一对都凑不出来，all_num == 0，后面 `&all_scores[0]` 也是空 vector。
			if (singles.empty())
			{
				printf("no valid image left after feature extraction, skip FAR/TAR\n");
				return;
			}
			ZQ_MergeSort::MergeSort(&singles[0], (__int64)singles.size(), true);
			int removed_num = 0;
			for (int i = (int)singles.size() - 2; i >= 0; i--)
			{
				if (singles[i] == singles[i + 1])
				{
					singles.erase(singles.begin() + i + 1);
					removed_num++;
				}
			}
			int image_num = (int)singles.size();
			printf("%d removed, remain %d\n", removed_num, image_num);
			if (image_num < 2)
			{
				printf("only %d image left, FAR/TAR needs at least 2, skip\n", image_num);
				return;
			}

			// 审计修复 2026-10-06（附录 IH.9）：原来写的是
			//   int all_num = (int)((long long)image_num*(image_num - 1)/2);
			// 那个 (long long) 说明作者**意识到**会溢出，但外面那层 (int) 又把它掐回去了：
			// image_num 到 65536 时乘积就是 2.1e9 > INT_MAX，
			// `std::vector<float> all_num(负数)` 抛 length_error，
			// 而这个函数在 main 的调用栈上，异常没人接 -> terminate。
			// 顺带：O(N^2) 的分数表在 LFW 全量（11480 张）上已经是
			// 6.6e7 * (4+4+4+1+4) ≈ 1.1 GB，再大就不现实了。
			// 这里显式判上限并**说明为什么跳过**，而不是让它在分配里炸掉。
			const __int64 max_pair_num = 200000000LL;   // ~3.2 GB 的分数表
			__int64 all_num64 = (__int64)image_num * (__int64)(image_num - 1) / 2;
			if (all_num64 > max_pair_num)
			{
				printf("image_num = %d would need %lld pair scores (> %lld), skip FAR/TAR\n",
					image_num, (long long)all_num64, (long long)max_pair_num);
				return;
			}
			int all_num = (int)all_num64;
			std::vector<float> all_scores(all_num);
			std::vector<int> all_idx_i(all_num), all_idx_j(all_num);
			std::vector<bool> all_flag(all_num);
			std::vector<int> sort_indices(all_num);
			int idx = 0;
			int same_num = 0;
			for (int i = 0; i < image_num; i++)
			{
				for (int j = i + 1; j < image_num; j++)
				{
					all_idx_i[idx] = i;
					all_idx_j[idx] = j;
					bool is_same = singles[i].SameName(singles[j]);
					all_flag[idx] = is_same;
					if (is_same)
						same_num++;
					sort_indices[idx] = idx;
					idx++;
				}
			}
			int notsame_num = all_num - same_num;
			printf("all_num = %d, same_num = %d, notsame_num = %d\n", all_num, same_num, notsame_num);
			// 审计修复 2026-10-06（附录 IH.10）：原来只在 printf 里除。
			// notsame_num == 0（所有图都同名，比如一个 list 里全是同一个人的照片）
			// 时 `cur_far_num / notsame_num` 是整数除零之外的浮点除零 —— 打印 inf，
			// 不崩，但输出是垃圾；而 far_num[stage] 全是 0 还会让 cur_stage
			// 每轮都自增，第一轮就走完。两种情况都该明说，而不是打一串 inf。
			if (notsame_num <= 0 || same_num <= 0)
			{
				printf("same_num = %d, notsame_num = %d: no usable target/far set, skip FAR/TAR curve\n",
					same_num, notsame_num);
				return;
			}

			double t1 = omp_get_wtime();
			// 审计修复 2026-10-06（附录 IH.1）：这里就是那个 SEGV 点。
			// 上面 image_num < 2 已经挡掉了空表；再加一道 feat.length 的检查，
			// 因为 singles 里的特征是 CopyData 过来的，理论上恒为 feat_dim，
			// 但真为 0 的话 DotProduct(0, 0, 0) 之后 all_scores 全 0，排序也没意义。
			int dim = singles[0].feat.length;
			if (dim <= 0 || singles[0].feat.pData == 0)
			{
				printf("invalid feature (length = %d), skip FAR/TAR\n", dim);
				return;
			}
			if (real_num_threads == 1)
			{
				for (int n = 0; n < all_num; n++)
				{
					int i = all_idx_i[n];
					int j = all_idx_j[n];
					all_scores[n] = ZQ_MathBase::DotProduct(dim, singles[i].feat.pData, singles[j].feat.pData);
				}
			}
			else
			{
				int chunk_size = (all_num + real_num_threads - 1) / real_num_threads;
#pragma omp parallel for schedule(static, chunk_size) num_threads(real_num_threads)
				for (int n = 0; n < all_num; n++)
				{
					int i = all_idx_i[n];
					int j = all_idx_j[n];
					all_scores[n] = ZQ_MathBase::DotProduct(dim, singles[i].feat.pData, singles[j].feat.pData);
				}
			}
			double t2 = omp_get_wtime();
			printf("compute all scores cost: %.3f secs\n", t2 - t1);
			ZQ_MergeSort::MergeSort(&all_scores[0], &sort_indices[0], all_num, false);
			double t3 = omp_get_wtime();
			printf("sort all scores cost: %.3f secs\n", t3 - t2);

			const int stage_num = 4;
			double far_num[stage_num] =
			{
				1e-6 * notsame_num,
				1e-5 * notsame_num,
				1e-4 * notsame_num,
				1e-3 * notsame_num
			};

			int cur_far_num = 0;
			int cur_tar_num = 0;
			int cur_stage = 0;
			for (int i = 0; i < all_num; i++)
			{
				if (cur_stage >= stage_num)
					break;
				int sort_id = sort_indices[i];
				if (all_flag[sort_id])
				{
					cur_tar_num++;
				}
				else
				{
					cur_far_num++;
				}
				if (cur_far_num > far_num[cur_stage])
				{
					printf("thresh = %.5f far = %15e, tar = %15f\n", all_scores[i],
						(double)cur_far_num / notsame_num, (double)cur_tar_num / same_num);
					cur_stage++;
				}
			}
		}
	};
}

#endif
