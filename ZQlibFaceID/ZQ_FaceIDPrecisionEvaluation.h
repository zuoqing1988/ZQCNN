#ifndef _ZQ_FACEID_PRECISION_EVALUATION_H_
#define _ZQ_FACEID_PRECISION_EVALUATION_H_
#pragma once
#include "ZQ_FaceRecognizer.h"
#include "ZQ_FaceFeature.h"
#include "ZQ_MathBase.h"
#include <opencv2\opencv.hpp>
#include <vector>
#include <stdlib.h>
#include <string>
#include <omp.h>
namespace ZQ
{
	class ZQ_FaceIDPrecisionEvaluation
	{
		class EvaluationPair
		{
		public:
			std::string fileL;
			std::string fileR;
			int flag; //-1 or 1
			ZQ_FaceFeature featL;
			ZQ_FaceFeature featR;
			bool valid;
		};

	public:
		static bool EvaluationOnLFW(std::vector<ZQ_FaceRecognizer*>& recognizers, const std::string& list_file, const std::string& folder, bool use_flip)
		{
			int recognizer_num = recognizers.size();
			if (recognizer_num == 0)
				return false;
			int real_num_threads = __max(1, __min(recognizer_num, omp_get_num_procs() - 1));

			int feat_dim = recognizers[0]->GetFeatDim();
			int real_dim = use_flip ? (feat_dim * 2) : feat_dim;
			printf("feat_dim = %d, real_dim = %d\n", feat_dim, real_dim);
			std::vector<std::vector<EvaluationPair>> pairs;
			if (!_parse_lfw_list(list_file, folder, pairs))
			{
				printf("failed to parse list file %s\n", list_file.c_str());
				return EXIT_FAILURE;
			}

			printf("parse list file %s done!\n", list_file.c_str());
			int part_num = pairs.size();
			std::vector<std::pair<int, int>> pair_list;
			for (int i = 0; i < part_num; i++)
			{
				for (int j = 0; j < pairs[i].size(); j++)
				{
					pair_list.push_back(std::make_pair(i, j));
				}
			}
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

			float ACC = _compute_accuracy(pairs);
			return true;
		}


	private:
		static float _compute_accuracy(const std::vector<std::vector<EvaluationPair>>& pairs)
		{
			int part_num = pairs.size();
			std::vector<float> ACCs(part_num);
			float ACC = 0;
			for (int i = 0; i < part_num; i++)
			{
				std::vector<EvaluationPair> val_pairs;
				for (int j = 0; j < part_num; j++)
				{
					if (j != i)
						val_pairs.insert(val_pairs.end(), pairs[j].begin(), pairs[j].end());
				}

				ZQ_FaceFeature mu;
				_compute_mu(val_pairs, mu);
				std::vector<double> val_scores, test_scores;
				_compute_scores(val_pairs, mu, val_scores);
				_compute_scores(pairs[i], mu, test_scores);
				double threshold = _get_threshold(val_pairs, val_scores, 10000);
				ACCs[i] = _get_accuracy(pairs[i], test_scores, threshold);
				ACC += ACCs[i];
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
			printf("AVE\t%2.2f%%\n", ACC / part_num * 100);
			return ACC;
		}

		static bool _parse_lfw_list(const std::string& list_file, const std::string& folder, std::vector<std::vector<EvaluationPair>>& pairs)
		{
			FILE* in = 0;
			if(0 != fopen_s(&in, list_file.c_str(), "r"))
				return false;

			int part_num, half_pair_num;
			const static int BUF_LEN = 200;
			char line[BUF_LEN];
			fgets(line, BUF_LEN, in);
			sscanf_s(line, "%d%d", &part_num, &half_pair_num);
			pairs.resize(part_num);

			std::vector<std::string> strings;
			for (int i = 0; i < part_num; i++)
			{
				for (int j = 0; j < 2 * half_pair_num; j++)
				{
					fgets(line, 199, in);
					int len = strlen(line);
					if (line[len - 1] == '\n')
						line[--len] = '\0';
					std::string input = line;
					_split_string(input, std::string("\t"), strings);
					if (strings.size() == 3)
					{
						EvaluationPair cur_pair;
						char num2str[BUF_LEN];
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", atoi(strings[1].c_str()));
						cur_pair.fileL = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", atoi(strings[2].c_str()));
						cur_pair.fileR = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
						cur_pair.flag = 1;
						pairs[i].push_back(cur_pair);
					}
					else if (strings.size() == 4)
					{
						EvaluationPair cur_pair;
						char num2str[BUF_LEN];
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", atoi(strings[1].c_str()));
						cur_pair.fileL = folder + "\\" + strings[0] + "\\" + strings[0] + std::string(num2str);
						sprintf_s(num2str, BUF_LEN, "_%04i.jpg", atoi(strings[3].c_str()));
						cur_pair.fileR = folder + "\\" + strings[2] + "\\" + strings[2] + std::string(num2str);
						cur_pair.flag = -1;
						pairs[i].push_back(cur_pair);
					}
				}
			}
			fclose(in);
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
	};
}

#endif
