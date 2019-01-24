#ifndef _ZQ_FACE_DATABASE_MAKER_H_
#define _ZQ_FACE_DATABASE_MAKER_H_
#pragma once
#include <sstream>
#include <direct.h>
#include <windows.h>
#include <io.h>
#include <omp.h>
#include <opencv2\opencv.hpp>
#include "ZQ_FaceDetector.h"
#include "ZQ_FaceRecognizer.h"
#include "ZQ_FaceDatabase.h"
#include "ZQ_FaceDatabaseCompact.h"
#include "ZQ_FaceRecognizerSphereFace.h"
#include "ZQ_MergeSort.h"

namespace ZQ
{
	class ZQ_FaceDatabaseMaker
	{
	public:
		enum ErrorCode
		{
			ERR_WARNING = 0,
			ERR_FATAL = 1
		};
		enum MakeDatabaseType
		{
			ONLY_MERGE_FEATS = 0,
			UPDATE_WHO_NOT_HAVE_FEATS = 1,
			FORCE_UPDATE_ALL = 2
		};
	public:
		static bool MakeDatabase(std::vector<ZQ_FaceDetector*>& detectors, std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1)
		{
			for (int i = 0; i < detectors.size(); i++)
				if (detectors[i] == 0)
					return false;
			for (int i = 0; i < recognizers.size(); i++)
				if (recognizers[i] == 0)
					return false;
			return _make_database(detectors, recognizers, database_root, database_featsfile, database_namesfile, type, show_face, 
				max_thread_num, false);
		}

		static bool MakeDatabaseAlreadyCropped(std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1)
		{
			for (int i = 0; i < recognizers.size(); i++)
				if (recognizers[i] == 0)
					return false;
			return _make_database_already_cropped(recognizers, database_root, database_featsfile, database_namesfile, type, show_face,
				max_thread_num, false);
		}

		static bool MakeDatabaseCompact(std::vector<ZQ_FaceDetector*>& detectors, std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1)
		{
			for (int i = 0; i < detectors.size(); i++)
				if (detectors[i] == 0)
					return false;
			for (int i = 0; i < recognizers.size(); i++)
				if (recognizers[i] == 0)
					return false;
			return _make_database(detectors, recognizers, database_root, database_featsfile, database_namesfile, type, show_face,
				max_thread_num, true);
		}

		static bool MakeDatabaseCompactAlreadyCropped(std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1)
		{
			for (int i = 0; i < recognizers.size(); i++)
				if (recognizers[i] == 0)
					return false;
			return _make_database_already_cropped(recognizers, database_root, database_featsfile, database_namesfile, type, show_face,
				max_thread_num, true);
		}

		static bool CropImagesForDatabase(const std::vector<ZQ_FaceDetector*>& detectors, const std::vector<ZQ_FaceRecognizer*>& recognizers,
			const std::string& src_root, const std::string& dst_root, int max_thread_num = 4, bool strict_check = true, 
			std::string err_logfile = "err_log.txt", bool only_for_high_quality = false)
		{
			return _crop_images_for_database(detectors, recognizers, src_root, dst_root, max_thread_num, strict_check, err_logfile, only_for_high_quality);
		}

		/*must be cropped image*/
		static bool DetectOutliersInDatabase(const std::vector<ZQ_FaceRecognizer*>& recognizers, const std::string& src_root, int max_thread_num = 4,
			const std::string out_file = "outlier_score.txt")
		{
			return _detect_outliers_in_database(recognizers, src_root, max_thread_num, out_file);
		}
		
	private:
	
		static bool _make_database(std::vector<ZQ_FaceDetector*>& detectors, std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1, bool compact = false)
		{
			if (type != ONLY_MERGE_FEATS && type != UPDATE_WHO_NOT_HAVE_FEATS && type != FORCE_UPDATE_ALL)
			{
				printf("type must be : ONLY_MERGE_FEATS(%d), UPDATE_WHO_NOT_HAVE_FEATS(%d), FORCE_UPDATE_ALL(%d)\n", 
					ONLY_MERGE_FEATS, UPDATE_WHO_NOT_HAVE_FEATS, FORCE_UPDATE_ALL);
				return false;
			}
			int num_detectors = detectors.size();
			int num_recognizers = recognizers.size();
			if (num_detectors == 0 || num_recognizers == 0)
			{
				printf("You should use at least one detector and one recognizer\n");
				return false;
			}

			ZQ_FaceDatabase database;
			std::vector<ErrorCode> ErrorCodes;
			std::vector<std::string> error_messages;
			std::string err_logfile = "err_log.txt";
			std::ostringstream oss;

			std::vector<std::string> person_names;
			std::vector<std::vector<std::string> > filenames;
			std::vector<std::vector<ZQ_CNN_BBox> > boxes;
			std::vector<std::vector<bool> > fail_flag;

			_auto_detect_database(database_root, person_names, filenames);
			
			int num_cores = omp_get_num_procs() - 1;

			int real_thread_num = __min(max_thread_num, __min(num_cores, __min(num_detectors, num_recognizers)));
			printf("real_thread_num = %d\n", real_thread_num);
			int feat_dim = 0;

			feat_dim = recognizers[0]->GetFeatDim();

			/****************************************/
			int person_num = person_names.size();
			database.persons.resize(person_num);
			database.names = person_names;

			double start_time = omp_get_wtime();
			printf("begin\n");
			std::vector<std::pair<int, int> > pairs;
			for (int i = 0; i < person_num; i++)
			{
				for (int j = 0; j < filenames[i].size(); j++)
					pairs.push_back(std::make_pair(i, j));
			}

#pragma omp parallel for schedule(dynamic, 100) num_threads(real_thread_num)
			for (int p = 0; p < pairs.size(); p++)
			{
				int i = pairs[p].first;
				int j = pairs[p].second;
				int id = omp_get_thread_num();
				std::ostringstream oss;
				ZQ_FaceFeature feat;
				cv::Mat crop;
				ErrorCode err_code;
				std::string err_msg;

				bool has_feat = false;
				bool need_detect = false;
				if (type == ONLY_MERGE_FEATS)
				{
					if (!_load_feature_from_file(filenames[i][j], feat))
					{
						need_detect = false;
						has_feat = false;
					}
					else
						has_feat = true;
				}
				else if (type == UPDATE_WHO_NOT_HAVE_FEATS)
				{
					if (!_load_feature_from_file(filenames[i][j], feat))
					{
						need_detect = true;
						has_feat = false;
					}
					else
						has_feat = true;
				}
				else if (type == FORCE_UPDATE_ALL)
					need_detect = true;


				bool need_write = false;
				bool ret = true;
				if (need_detect)
				{
					if (!_extract_feature_from_img(*detectors[id], *recognizers[id], filenames[i][j], feat, crop, err_code, err_msg, false))
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					need_write = true;
					has_feat = true;
					if (show_face)
					{
						if (id == 0)
						{
							cv::namedWindow("crop");
							cv::imshow("crop", crop);
							cv::waitKey(5);
						}
					}
				}

#pragma omp critical
				{
					if (has_feat)
					{
						database.persons[i].features.push_back(feat);
						database.persons[i].filenames.push_back(filenames[i][j]);
					}
				}

				if (need_write)
					_write_feature_to_file(filenames[i][j], feat);

			}

			double end_time = omp_get_wtime();
			printf("detect_and_extract total_cost:%.3f s\n", (end_time - start_time));

			/*******************/
			for (int i = person_num - 1; i >= 0; i--)
			{
				if (database.persons[i].features.size() == 0)
				{
					printf("person [%d]: %s has no data\n", i, person_names[i].c_str());
					oss.str("");
					oss << "person [" << i << "]: " << person_names[i] << " has no data";
					ErrorCodes.push_back(ERR_WARNING);
					error_messages.push_back(oss.str());
					database.persons.erase(database.persons.begin() + i);
					person_names.erase(person_names.begin() + i);
				}
			}

			database.names = person_names;

			if (compact)
			{
				if (!database.SaveToFileBinaryCompact(database_featsfile, database_namesfile))
				{
					printf("failed to save database\n");
					return EXIT_FAILURE;
				}
			}
			else
			{
				if (!database.SaveToFileBinary(database_featsfile, database_namesfile))
				{
					printf("failed to save database\n");
					return EXIT_FAILURE;
				}
			}

			printf("all done\n");
			_write_error_messages(err_logfile, ErrorCodes, error_messages);
			return EXIT_SUCCESS;
		}

		static bool _make_database_already_cropped(std::vector<ZQ_FaceRecognizer*> recognizers,
			const std::string& database_root, const std::string& database_featsfile, const std::string& database_namesfile,
			MakeDatabaseType type = ONLY_MERGE_FEATS, bool show_face = false, int max_thread_num = 1, bool compact = false)
		{
			if (type != ONLY_MERGE_FEATS && type != UPDATE_WHO_NOT_HAVE_FEATS && type != FORCE_UPDATE_ALL)
			{
				printf("type must be : ONLY_MERGE_FEATS(%d), UPDATE_WHO_NOT_HAVE_FEATS(%d), FORCE_UPDATE_ALL(%d)\n",
					ONLY_MERGE_FEATS, UPDATE_WHO_NOT_HAVE_FEATS, FORCE_UPDATE_ALL);
				return false;
			}
			int num_recognizers = recognizers.size();
			if (num_recognizers == 0)
			{
				printf("You should use at least one recognizer\n");
				return false;
			}

			ZQ_FaceDatabase database;
			std::vector<ErrorCode> ErrorCodes;
			std::vector<std::string> error_messages;
			std::string err_logfile = "err_log.txt";
			std::ostringstream oss;

			std::vector<std::string> person_names;
			std::vector<std::vector<std::string> > filenames;
			std::vector<std::vector<ZQ_CNN_BBox> > boxes;
			std::vector<std::vector<bool> > fail_flag;

			_auto_detect_database(database_root, person_names, filenames);

			int num_cores = omp_get_num_procs() - 1;

			int real_thread_num = __min(max_thread_num, __min(num_cores, num_recognizers));
			printf("real_thread_num = %d\n", real_thread_num);
			int feat_dim = 0;

			feat_dim = recognizers[0]->GetFeatDim();

			/****************************************/
			int person_num = person_names.size();
			database.persons.resize(person_num);
			database.names = person_names;

			double start_time = omp_get_wtime();
			printf("begin\n");
			std::vector<std::pair<int, int> > pairs;
			for (int i = 0; i < person_num; i++)
			{
				for (int j = 0; j < filenames[i].size(); j++)
					pairs.push_back(std::make_pair(i, j));
			}

#pragma omp parallel for schedule(dynamic, 100) num_threads(real_thread_num)
			for (int p = 0; p < pairs.size(); p++)
			{
				int i = pairs[p].first;
				int j = pairs[p].second;
				int id = omp_get_thread_num();
				std::ostringstream oss;
				ZQ_FaceFeature feat;
				cv::Mat crop;
				ErrorCode err_code;
				std::string err_msg;

				bool has_feat = false;
				bool need_detect = false;
				if (type == ONLY_MERGE_FEATS)
				{
					if (!_load_feature_from_file(filenames[i][j], feat))
					{
						need_detect = false;
						has_feat = false;
					}
					else
						has_feat = true;
				}
				else if (type == UPDATE_WHO_NOT_HAVE_FEATS)
				{
					if (!_load_feature_from_file(filenames[i][j], feat))
					{
						need_detect = true;
						has_feat = false;
					}
					else
						has_feat = true;
				}
				else if (type == FORCE_UPDATE_ALL)
					need_detect = true;


				bool need_write = false;
				bool ret = true;
				if (need_detect)
				{
					if (!_extract_feature_from_cropped_image(*recognizers[id], filenames[i][j], feat, crop, err_code, err_msg))
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					need_write = true;
					has_feat = true;
					if (show_face)
					{
						if (id == 0)
						{
							cv::namedWindow("crop");
							cv::imshow("crop", crop);
							cv::waitKey(5);
						}
					}
				}

#pragma omp critical
				{
					if (has_feat)
					{
						database.persons[i].features.push_back(feat);
						database.persons[i].filenames.push_back(filenames[i][j]);
					}
				}

				if (need_write)
					_write_feature_to_file(filenames[i][j], feat);

			}

			double end_time = omp_get_wtime();
			printf("detect_and_extract total_cost:%.3f s\n", (end_time - start_time));

			/*******************/
			for (int i = person_num - 1; i >= 0; i--)
			{
				if (database.persons[i].features.size() == 0)
				{
					printf("person [%d]: %s has no data\n", i, person_names[i].c_str());
					oss.str("");
					oss << "person [" << i << "]: " << person_names[i] << " has no data";
					ErrorCodes.push_back(ERR_WARNING);
					error_messages.push_back(oss.str());
					database.persons.erase(database.persons.begin() + i);
					person_names.erase(person_names.begin() + i);
				}
			}

			database.names = person_names;

			if (compact)
			{
				if (!database.SaveToFileBinaryCompact(database_featsfile, database_namesfile))
				{
					printf("failed to save database\n");
					return EXIT_FAILURE;
				}
			}
			else
			{
				if (!database.SaveToFileBinary(database_featsfile, database_namesfile))
				{
					printf("failed to save database\n");
					return EXIT_FAILURE;
				}
			}

			printf("all done\n");
			_write_error_messages(err_logfile, ErrorCodes, error_messages);
			return EXIT_SUCCESS;
		}

		static bool _crop_images_for_database(const std::vector<ZQ_FaceDetector*>& detectors, const std::vector<ZQ_FaceRecognizer*>& recognizers,
			const std::string& src_root, const std::string& dst_root, int max_thread_num = 4, bool strict_check = true, std::string err_logfile = "err_log.txt",
			bool only_for_high_quality = false)
		{
			int num_detector = detectors.size();
			int num_recognizer = recognizers.size();
			if (num_detector == 0 || num_recognizer == 0)
				return false;

			std::vector<ErrorCode> ErrorCodes;
			std::vector<std::string> error_messages;

			int num_cores = omp_get_num_procs();
			int real_thread_num = __max(1, __min(num_cores - 1, max_thread_num));
			real_thread_num = __min(real_thread_num, __min(detectors.size(), recognizers.size()));

			std::vector<std::string> person_names;
			std::vector<std::vector<std::string> > filenames;

			_auto_detect_database(src_root, person_names, filenames);

			int person_num = person_names.size();
			_mkdir(dst_root.c_str());
			for (int i = 0; i < person_num; i++)
			{
				std::string path = dst_root + "\\" + person_names[i];
				_mkdir(path.c_str());
			}

			clock_t start_time = clock();
			//double start_time = omp_get_wtime();
			printf("begin\n");
			std::vector<std::pair<int, int> > pairs;
			for (int i = 0; i < person_num; i++)
			{
				for (int j = 0; j < filenames[i].size(); j++)
					pairs.push_back(std::make_pair(i, j));
			}

			clock_t start = clock();

			if (real_thread_num == 1)
			{
				for (int p = 0; p < pairs.size(); p++)
				{
					int i = pairs[p].first;
					int j = pairs[p].second;
					int id = omp_get_thread_num();
					int crop_width = recognizers[id]->GetCropWidth();
					int crop_height = recognizers[id]->GetCropHeight();
					std::ostringstream oss;
					cv::Mat crop(crop_height, crop_width, CV_MAKETYPE(8, 3));
					ErrorCode err_code;
					std::string err_msg;

					bool ret = true;
					cv::Mat image = cv::imread(filenames[i][j]);
					if (image.empty())
					{
						printf("failed to read image: %s\n", filenames[i][j].c_str());
						oss.str("");
						oss << "failed to read image: " << filenames[i][j];
						err_code = ERR_WARNING;
						err_msg = oss.str();
						ret = false;
					}

					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					ZQ_CNN_BBox box;
					ret = _get_face5point_from_img(*detectors[id], filenames[i][j], image, box, err_code, err_msg, strict_check, only_for_high_quality);
					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					float facial5point[10] =
					{
						box.ppoint[0],box.ppoint[5],
						box.ppoint[1],box.ppoint[6],
						box.ppoint[2],box.ppoint[7],
						box.ppoint[3],box.ppoint[8],
						box.ppoint[4],box.ppoint[9]
					};

					ret = recognizers[id]->CropImage(image.data, image.cols, image.rows, image.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, box.ppoint, box.ppoint + 5, crop.data, crop.step[0]);

					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					size_t pos = filenames[i][j].find_last_of('\\');
					if (pos != std::string::npos)
					{
						std::string real_name(filenames[i][j].c_str() + pos + 1);
						std::string fullname = dst_root + "\\" + person_names[i] + "\\" + real_name;
						cv::imwrite(fullname, crop);
#pragma omp critical
						{
							//std::cout << fullname << "\n";
						}
					}
				}
			}
			else
			{
#pragma omp parallel for  schedule(dynamic, 10) num_threads(real_thread_num)
				for (int p = 0; p < pairs.size(); p++)
				{
					int i = pairs[p].first;
					int j = pairs[p].second;
					int id = omp_get_thread_num();
					int crop_width = recognizers[id]->GetCropWidth();
					int crop_height = recognizers[id]->GetCropHeight();
					std::ostringstream oss;
					cv::Mat crop(crop_height, crop_width, CV_MAKETYPE(8, 3));
					ErrorCode err_code;
					std::string err_msg;

					bool ret = true;
					cv::Mat image = cv::imread(filenames[i][j]);
					if (image.empty())
					{
						printf("failed to read image: %s\n", filenames[i][j].c_str());
						oss.str("");
						oss << "failed to read image: " << filenames[i][j];
						err_code = ERR_WARNING;
						err_msg = oss.str();
						ret = false;
					}

					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					ZQ_CNN_BBox box;
					ret = _get_face5point_from_img(*detectors[id], filenames[i][j], image, box, err_code, err_msg, strict_check, only_for_high_quality);
					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					float facial5point[10] =
					{
						box.ppoint[0],box.ppoint[5],
						box.ppoint[1],box.ppoint[6],
						box.ppoint[2],box.ppoint[7],
						box.ppoint[3],box.ppoint[8],
						box.ppoint[4],box.ppoint[9]
					};

					ret = recognizers[id]->CropImage(image.data, image.cols, image.rows, image.step[0], ZQ_PixelFormat::ZQ_PIXEL_FMT_BGR, box.ppoint, box.ppoint + 5, crop.data, crop.step[0]);

					if (!ret)
					{
#pragma omp critical
						{
							ErrorCodes.push_back(err_code);
							error_messages.push_back(err_msg);
						}
						continue;
					}

					size_t pos = filenames[i][j].find_last_of('\\');
					if (pos != std::string::npos)
					{
						std::string real_name(filenames[i][j].c_str() + pos + 1);
						std::string fullname = dst_root + "\\" + person_names[i] + "\\" + real_name;
						cv::imwrite(fullname, crop);
#pragma omp critical
						{
							//std::cout << fullname << "\n";
						}
					}
				}
			}


			

			clock_t end = clock();
			printf("time: %f\n", 0.001*(end - start));
			_write_error_messages(err_logfile, ErrorCodes, error_messages);

			return true;
		}


		static bool _extract_feature_from_box(ZQ_FaceRecognizer& recognizer, const std::string& imgfile, const cv::Mat& image, const ZQ_CNN_BBox& box,
			ZQ_FaceFeature& feat, cv::Mat& crop, ErrorCode& err_code, std::string& err_msg)
		{
			int nChannels = image.channels();
			ZQ_PixelFormat pixFmt = (nChannels == 1) ? ZQ_PIXEL_FMT_GRAY : ZQ_PIXEL_FMT_BGR;
			int width = recognizer.GetCropWidth();
			int height = recognizer.GetCropHeight();
			int feat_dim = recognizer.GetFeatDim();
			crop = cv::Mat(cv::Size(width, height), CV_MAKETYPE(8U, 3));
			std::ostringstream oss;
			if (!recognizer.CropImage(image.data, image.cols, image.rows, image.step[0], pixFmt, box.ppoint, box.ppoint + 5, crop.data, crop.step[0]))
			{
				printf("failed to crop face in image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to crop face in image: " << imgfile;
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}
			feat.ChangeSize(feat_dim);


			if (!recognizer.ExtractFeature(crop.data, crop.step[0], pixFmt, feat.pData, true))
			{
				printf("failed to extract feature in image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to extract feature in image: " << imgfile.c_str();
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}
			return true;
		}

		static bool _extract_feature_from_cropped_image(ZQ_FaceRecognizer& recognizer, const std::string& imgfile, const cv::Mat& image, 
			ZQ_FaceFeature& feat, ErrorCode& err_code, std::string& err_msg)
		{
			int nChannels = image.channels();
			ZQ_PixelFormat pixFmt = (nChannels == 1) ? ZQ_PIXEL_FMT_GRAY : ZQ_PIXEL_FMT_BGR;
			int width = recognizer.GetCropWidth();
			int height = recognizer.GetCropHeight();
			
			int feat_dim = recognizer.GetFeatDim();
			std::ostringstream oss;
			feat.ChangeSize(feat_dim);
			if (image.cols != width || image.rows != height
				|| !recognizer.ExtractFeature(image.data, image.step[0], pixFmt, feat.pData, true))
			{
				printf("failed to extract feature in image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to extract feature in image: " << imgfile.c_str();
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}
			return true;
		}

		static bool _extract_feature_from_img(ZQ_FaceDetector& detector, ZQ_FaceRecognizer& recognizer,
			const std::string& imgfile, ZQ_FaceFeature& feat, cv::Mat& crop, ErrorCode& err_code, std::string& err_msg,
			bool only_for_high_quality)
		{
			std::ostringstream oss;
			cv::Mat image = cv::imread(imgfile);
			if (image.empty())
			{
				printf("failed to read image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to read image: " << imgfile;
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}
			double t1 = omp_get_wtime();
			ZQ_CNN_BBox box;
			if (!_get_face5point_from_img(detector, imgfile, image, box, err_code, err_msg, only_for_high_quality))
				return false;
			double t2 = omp_get_wtime();
			if (!_extract_feature_from_box(recognizer, imgfile, image, box, feat, crop, err_code, err_msg))
				return false;
			double t3 = omp_get_wtime();
			printf("image: %s done! findface: %.3f, extract: %.3f\n", imgfile.c_str(), t2 - t1, t3 - t2);

			return true;
		}

		static bool _extract_feature_from_cropped_image(ZQ_FaceRecognizer& recognizer,
			const std::string& imgfile, ZQ_FaceFeature& feat, cv::Mat& crop, ErrorCode& err_code, std::string& err_msg)
		{
			std::ostringstream oss;
			cv::Mat image = cv::imread(imgfile);
			if (image.empty())
			{
				printf("failed to read image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to read image: " << imgfile;
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}
			double t1 = omp_get_wtime();
			if (!_extract_feature_from_cropped_image(recognizer, imgfile, image, feat, err_code, err_msg))
				return false;
			double t2 = omp_get_wtime();
			printf("image: %s done! extract: %.3f\n", imgfile.c_str(), t2 - t1);
			crop = image;
			return true;
		}

		static bool _load_feature_from_file(const std::string& imgfile, ZQ_FaceFeature& feat)
		{
			std::string feat_file = imgfile + ".imgfeat";
			FILE* in = 0;
			if( 0 != fopen_s(&in, feat_file.c_str(), "rb"))
				return false;
			int feat_dim = 0;
			fread(&feat_dim, sizeof(int), 1, in);
			feat.ChangeSize(feat_dim);
			fread(feat.pData, sizeof(float), feat_dim, in);
			fclose(in);
			return true;
		}

		static bool _write_feature_to_file(const std::string& imgfile, const ZQ_FaceFeature& feat)
		{
			std::string feat_file = imgfile + ".imgfeat";
			FILE* out = 0;
			if (0 != fopen_s(&out, feat_file.c_str(), "wb"))
				return false;
			int feat_dim = feat.length;
			fwrite(&feat_dim, sizeof(int), 1, out);
			fwrite(feat.pData, sizeof(float), feat_dim, out);
			fclose(out);
			return true;
		}

		static bool _write_error_messages(const std::string& file, const std::vector<ErrorCode>& ErrorCodes, const std::vector<std::string>& error_messages)
		{
			FILE* out = 0;
			if(0 != fopen_s(&out, file.c_str(), "w"))
				return false;
			int num = ErrorCodes.size();
			if (num != error_messages.size())
				return false;
			for (int i = 0; i < num; i++)
			{
				fprintf(out, "err_code: %d: msg: %s\n", ErrorCodes[i], error_messages[i].c_str());
			}
			fclose(out);
			return true;
		}

		static bool _auto_detect_database(const std::string& root_path, std::vector<std::string>& person_names, std::vector<std::vector<std::string> >& filenames)
		{
			std::string dir(root_path);
			dir.append("\\*.*");
			_finddata_t fileDir;
			intptr_t lfDir;

			person_names.clear();
			filenames.clear();

			if ((lfDir = _findfirst(dir.c_str(), &fileDir)) == -1l)
			{
				//printf("No file is found\n");
			}
			else
			{
				do {

					std::string str(fileDir.name);
					if (fileDir.attrib & _A_SUBDIR && 0 != strcmp(str.c_str(), ".") && 0 != strcmp(str.c_str(), ".."))
						person_names.push_back(str);


				} while (_findnext(lfDir, &fileDir) == 0);
			}
			_findclose(lfDir);

			int person_num = person_names.size();
			filenames.resize(person_num);
			for (int i = 0; i < person_num; i++)
			{
				dir = root_path + "\\" + person_names[i] + "\\*.jpg";
				if ((lfDir = _findfirst(dir.c_str(), &fileDir)) == -1l)
				{
					//printf("No file is found\n");
				}
				else
				{
					do {
						std::string str(fileDir.name);
						filenames[i].push_back(root_path + "\\" + person_names[i] + "\\" + str);
					} while (_findnext(lfDir, &fileDir) == 0);
				}
				_findclose(lfDir);
			}
			return true;
		}

		static bool _write_database_txt(const std::string& data_base_file, const std::vector<std::vector<std::string> >& filenames)
		{
			int num = filenames.size();
			FILE* out = 0;
			if(0 != fopen_s(&out, data_base_file.c_str(), "w"))
			{
				return false;
			}

			fprintf(out, "%d\n", num);
			for (int i = 0; i < num; i++)
			{
				for (int j = 0; j < filenames[i].size(); j++)
				{
					fprintf(out, "%d\n%s\n", i, filenames[i][j].c_str());
				}
			}

			fprintf(out, "\n");
			fclose(out);

			return true;
		}

		static bool _get_face5point_from_img(ZQ_FaceDetector& detector, const std::string& imgfile, const cv::Mat& image, ZQ_CNN_BBox& box,
			ErrorCode& err_code, std::string& err_msg, bool strict_check = true, bool only_for_high_quality = false)
		{
			std::ostringstream oss;
			bool use_cuda = false;

			std::vector<ZQ_CNN_BBox> bbox;
			//first try
			bool has_found = false;
			ZQ_PixelFormat pixFmt = image.channels() == 1 ? ZQ_PIXEL_FMT_GRAY : ZQ_PIXEL_FMT_BGR;
			if (!detector.FindFace(image.data, image.cols, image.rows, image.step[0], pixFmt, 60, 0.709, bbox) || bbox.size() == 0)
			{
				/*printf("failed to find face in image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to find face in image: " << imgfile;
				err_code = ERR_WARNING;
				err_msg = oss.str();*/
				has_found = false;
			}
			else
				has_found = true;

			if (!only_for_high_quality)
			{
				//second try
				if (!has_found)
				{
					printf("second try\n");
					if (!detector.FindFace(image.data, image.cols, image.rows, image.step[0], pixFmt, 40, 0.709, bbox) || bbox.size() == 0)
					{
						/*printf("failed to find face in image: %s\n", imgfile.c_str());
						oss.str("");
						oss << "failed to find face in image: " << imgfile;
						err_code = ERR_WARNING;
						err_msg = oss.str();*/
						has_found = false;
					}
					else
						has_found = true;
				}

				//third try
				if (!has_found)
				{
					printf("third try\n");
					if (!detector.FindFace(image.data, image.cols, image.rows, image.step[0], pixFmt, 30, 0.8, bbox) || bbox.size() == 0)
					{
						/*printf("failed to find face in image: %s\n", imgfile.c_str());
						oss.str("");
						oss << "failed to find face in image: " << imgfile;
						err_code = ERR_WARNING;
						err_msg = oss.str();*/
						has_found = false;
					}
					else
						has_found = true;
				}

				//fourth try
				if (!has_found)
				{
					printf("fourth try\n");
					if (!detector.FindFace(image.data, image.cols, image.rows, image.step[0], pixFmt, 20, 0.85, bbox) || bbox.size() == 0)
					{
						/*printf("failed to find face in image: %s\n", imgfile.c_str());
						oss.str("");
						oss << "failed to find face in image: " << imgfile;
						err_code = ERR_WARNING;
						err_msg = oss.str();*/
						has_found = false;
					}
					else
						has_found = true;
				}

				//fifth try
				if (!has_found)
				{

					printf("fifth try\n");
					if (!detector.FindFace(image.data, image.cols, image.rows, image.step[0], pixFmt, 12, 0.9, bbox) || bbox.size() == 0)
					{
						/*printf("failed to find face in image: %s\n", imgfile.c_str());
						oss.str("");
						oss << "failed to find face in image: " << imgfile;
						err_code = ERR_WARNING;
						err_msg = oss.str();*/
						has_found = false;
					}
					else
						has_found = true;
				}
			}
			if (!has_found)
			{
				printf("failed to find face in image: %s\n", imgfile.c_str());
				oss.str("");
				oss << "failed to find face in image: " << imgfile;
				err_code = ERR_WARNING;
				err_msg = oss.str();
				return false;
			}

			if (bbox.size() > 1)
			{
				if (strict_check)
				{
					printf("find more than one face in image: %s\n", imgfile.c_str());
					oss.str("");
					oss << "find more than one face in image: " << imgfile;
					err_code = ERR_WARNING;
					err_msg = oss.str();
					return false;
				}

				//pick the face closed to the center
				float center[2] = { image.cols*0.5,image.rows*0.5 };
				std::vector<float> distance(bbox.size());
				for (int i = 0; i < bbox.size(); i++)
				{
					float cx = 0.5*(bbox[i].col1 + bbox[i].col2);
					float cy = 0.5*(bbox[i].row1 + bbox[i].row2);
					distance[i] = (center[0] - cx)*(center[0] - cx) + (center[1] - cy)*(center[1] - cy);
				}
				float min_dis = distance[0];
				int min_id = 0;
				for (int i = 1; i < bbox.size(); i++)
				{
					if (min_dis > distance[i])
					{
						min_dis = distance[i];
						min_id = i;
					}
				}
				box = bbox[min_id];
				return true;
			}
			else
				box = bbox[0];
			return true;
		}

		static bool _detect_outliers_in_database(const std::vector<ZQ_FaceRecognizer*>& recognizers, const std::string& src_root, int max_thread_num,
			const std::string& out_file)
		{
			int num_recognizer = recognizers.size();
			if (num_recognizer == 0)
				return false;

			std::vector<ErrorCode> ErrorCodes;
			std::vector<std::string> error_messages;

			int num_cores = omp_get_num_procs();
			int real_thread_num = __max(1, __min(num_cores - 1, max_thread_num));
			real_thread_num = __min(real_thread_num, recognizers.size());

			std::vector<std::string> person_names;
			std::vector<float> person_min_scores;
			std::vector<int> person_min_scores_i;
			std::vector<int> person_min_scores_j;
			std::vector<std::vector<std::string> > filenames;

			_auto_detect_database(src_root, person_names, filenames);

			int person_num = person_names.size();
			if (person_num == 0)
			{
				printf("no person in %s\n", src_root.c_str());
				return false;
			}
			person_min_scores.resize(person_num);
			person_min_scores_i.resize(person_num);
			person_min_scores_j.resize(person_num);

			if (real_thread_num == 1)
			{
				for (int i = 0; i < person_num; i++)
				{
					float out_min_score;
					int out_i, out_j;
					if (!_detect_outlier_for_one_person(*(recognizers[0]), filenames[i], out_min_score, out_i, out_j))
					{
						printf("failed to detect outliter for %s\n", person_names[i].c_str());
						return false;
					}
					if (filenames[i].size() == 0)
						out_min_score = 100;
					else if (filenames[i].size() == 1)
						out_min_score = 10;
					person_min_scores[i] = out_min_score;
					person_min_scores_i[i] = out_i;
					person_min_scores_j[i] = out_j;
					//if ((i + 1) % 100 == 0)
					{
						printf("%d/%d handled\n", i + 1, person_num);
					}
				}
			}
			else
			{
				int handled[1] = { 0 };
#pragma omp parallel for num_threads(real_thread_num)
				for (int i = 0; i < person_num; i++)
				{
					int thread_id = omp_get_thread_num();
					float out_min_score;
					int out_i, out_j;
					if (!_detect_outlier_for_one_person(*(recognizers[thread_id]), filenames[i], out_min_score, out_i, out_j))
					{
						printf("failed to detect outliter for %s\n", person_names[i].c_str());	
						out_min_score = -1000;
					}
					if (filenames[i].size() == 0)
						out_min_score = 100;
					else if (filenames[i].size() == 1)
						out_min_score = 10;
					person_min_scores[i] = out_min_score;
					person_min_scores_i[i] = out_i;
					person_min_scores_j[i] = out_j;
#pragma omp critical
					{
						(*handled)++;
						//if (handled % 100 == 0)
						{
							printf("%d/%d handled\n", *handled,person_num);
						}
					}
				}
			}

			std::vector<int> sort_indices(person_num);
			for (int i = 0; i < person_num; i++)
				sort_indices[i] = i;

			ZQ_MergeSort::MergeSort(&person_min_scores[0], &sort_indices[0], person_num, true);

			FILE* out = 0;
			if (0 != fopen_s(&out,out_file.c_str(), "w"))
			{
				printf("failed to create file %s\n", out_file.c_str());
				return false;
			}

			for (int i = 0; i < person_num; i++)
			{
				int id = sort_indices[i];
				fprintf(out, "%12.3f %s ", person_min_scores[i], person_names[id].c_str());
				int img_num = filenames[id].size();
				int out_i = person_min_scores_i[id];
				int out_j = person_min_scores_j[id];
				if (img_num > 1 && out_i >= 0 && out_i < img_num
					&& out_j >= 0 && out_j < img_num)
				{
					fprintf(out, "%s %s\n", filenames[id][out_i].c_str(), filenames[id][out_j].c_str());
				}
				else
				{
					fprintf(out, "\n");
				}
			}
			fclose(out);
			return true;
		}

		static bool _detect_outlier_for_one_person(ZQ_FaceRecognizer& recognier, const std::vector<std::string>& filenames,
			float& out_min_score, int& out_i, int& out_j)
		{
			out_i = 0;
			out_j = 0;
			out_min_score = 1;
			int num = filenames.size();
			if (num <= 1)
				return true;

			std::vector<ZQ_FaceFeature> feats(num);
			int W = recognier.GetCropWidth();
			int H = recognier.GetCropHeight();
			int dim = recognier.GetFeatDim();
			for (int i = 0; i < num; i++)
			{
				//printf("%d/%d\n", i + 1, num);
				cv::Mat img = cv::imread(filenames[i]);
				if (img.empty())
					return false;
				if (img.rows != H || img.cols != W || img.channels() != 3)
					return false;
				feats[i].ChangeSize(dim);
				if (!recognier.ExtractFeature(img.data, img.step[0], ZQ_PIXEL_FMT_BGR, feats[i].pData, true))
					return false;
			}

			out_min_score = FLT_MAX;
			for (int i = 0; i < num - 1; i++)
			{
				for (int j = i + 1; j < num; j++)
				{
					float tmp_score = ZQ_MathBase::DotProduct(dim, feats[i].pData, feats[j].pData);
					if (tmp_score <= out_min_score)
					{
						out_min_score = tmp_score;
						out_i = i;
						out_j = j;
					}
				}
			}
			return true;
		}
	};
}
#endif
