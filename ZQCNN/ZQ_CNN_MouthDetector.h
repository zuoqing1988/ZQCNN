#ifndef _ZQ_CNN_MOUTH_DETECTOR_H_
#define _ZQ_CNN_MOUTH_DETECTOR_H_
#pragma once

#include <string>
#include "ZQ_CNN_MTCNN.h"
#include "ZQ_CNN_SSD.h"
#include "opencv2/opencv.hpp"
#include "ZQ_CNN_DetectorInterface.h"

namespace ZQ
{
	class ZQ_CNN_MouthDetector : ZQ_CNN_DetectorInterface
	{
	public:
		class InitialArgs
		{
		public:
			std::string mtcnn_pnet_proto;
			std::string mtcnn_pnet_model;
			std::string mtcnn_rnet_proto;
			std::string mtcnn_rnet_model;
			std::string mtcnn_onet_proto;
			std::string mtcnn_onet_model;
			//
			std::string ssd_proto;
			std::string ssd_model;
			std::string ssd_out_blob_name;
			std::string ssd_class_names_file;

			InitialArgs()
			{
				mtcnn_pnet_proto = "model/det1.zqparams";
				mtcnn_pnet_model = "model/det1.nchwbin";
				mtcnn_rnet_proto = "model/det2.zqparams";
				mtcnn_rnet_model = "model/det2.nchwbin";
				mtcnn_onet_proto = "model/det3.zqparams";
				mtcnn_onet_model = "model/det3.nchwbin";
				ssd_proto = "model/MobileNetSSD_deploy-face.zqparams";
				ssd_model = "model/MobileNetSSD_deploy-face.nchwbin";
				ssd_out_blob_name = "detection_out";
				ssd_class_names_file = "model/MobileNetSSD_deploy-face.names";
			}
		};

		class DetectArgs
		{
		public:
			bool enable_rot;
			int mtcnn_min_size/* = 60*/;
			float mtcnn_scale/* = 0.709 */;
			float mtcnn_thresh_p/* = 0.6*/;
			float mtcnn_thresh_r/* = 0.7*/;
			float mtcnn_thresh_o/* = 0.6*/;
			float mtcnn_thresh_nms_p/* = 0.7*/;
			float mtcnn_thresh_nms_r/* = 0.7*/;
			float mtcnn_thresh_nms_o/* = 0.7*/;
			float enlarge_border /*= 0.2*/;
			float ssd_mouth_thresh/* = 0.8*/;

			DetectArgs()
			{
				enable_rot = true;
				mtcnn_min_size = 60;
				mtcnn_scale = 0.709;
				mtcnn_thresh_p = 0.6;
				mtcnn_thresh_r = 0.7;
				mtcnn_thresh_o = 0.6;
				mtcnn_thresh_nms_p = 0.7;
				mtcnn_thresh_nms_r = 0.7;
				mtcnn_thresh_nms_o = 0.7;
				enlarge_border = 0.2;
				ssd_mouth_thresh = 0.8;
			}
		};

		class DetectedFace
		{
		public:
			int off_x, off_y;
			int width, height;
			float rot_in_rad;
			ZQ_CNN_BBox box;
			std::vector<ZQ_CNN_SSD::BBox> result_vec_mouth;
		};

		class DetectedResult
		{
		public:
			std::vector<DetectedFace> faces;
		};

		class SimpleFaceInfo
		{
		public:
			int face_off_x;
			int face_off_y;
			int face_width;
			int face_height;
			float mouth_prob;
			int mouth_off_x;
			int mouth_off_y;
			int mouth_width;
			int mouth_height;
		};

		class SimpleDetectedResult
		{
		public:
			std::vector<SimpleFaceInfo> faces;
		};


	public:
		ZQ_CNN_MouthDetector()
		{
		}
		~ZQ_CNN_MouthDetector()
		{
			_clear();
		}
	private:
		InitialArgs args;
		ZQ_CNN_MTCNN mtcnn_detector;
		ZQ_CNN_SSD ssd_detector;
		std::vector<std::string> mouth_objnames;

	public:
		bool Initialize(const void* input_args)
		{
			if (input_args == 0)
				return false;

			args = *((const InitialArgs*)input_args);

			_clear();
			if (!mtcnn_detector.Init(args.mtcnn_pnet_proto, args.mtcnn_pnet_model, args.mtcnn_rnet_proto, args.mtcnn_rnet_model,
				args.mtcnn_onet_proto, args.mtcnn_onet_model))
			{
				printf("failed to init MTCNN!\n");
				return false;
			}

			if (!ssd_detector.Init(args.ssd_proto, args.ssd_model, args.ssd_out_blob_name))
			{
				printf("failed to init SSD detector!\n");
				return false;
			}
			//const char* kClassNames[] = { "__background__", "eye", "nose", "mouth", "face" };
			mouth_objnames = _objects_names_from_file(args.ssd_class_names_file);
			return true;
		}

		bool DetectSimpleResult(const unsigned char* rgb_image, int width, int height, int widthStep, const void* detect_arg, void* output_detected_data)
		{
			DetectedResult tmp_result;
			const DetectArgs* d_arg = (const DetectArgs*)detect_arg;
			if (!Detect(rgb_image, width, height, widthStep, detect_arg, &tmp_result))
				return false;


			SimpleDetectedResult* result = (SimpleDetectedResult*)output_detected_data;
			result->faces.clear();
			for (int i = 0; i < tmp_result.faces.size(); i++)
			{
				std::vector<ZQ_CNN_SSD::BBox> result_vec;

				_trans_boxes(
					tmp_result.faces[i].off_x,
					tmp_result.faces[i].off_y,
					tmp_result.faces[i].width,
					tmp_result.faces[i].height,
					tmp_result.faces[i].rot_in_rad,
					tmp_result.faces[i].result_vec_mouth,
					result_vec);

				for (int j = 0; j < result_vec.size(); j++)
				{
					if (mouth_objnames[result_vec[j].label] == "mouth")
					{
						SimpleFaceInfo info;
						info.face_off_x = tmp_result.faces[i].off_x;
						info.face_off_y = tmp_result.faces[i].off_y;
						info.face_width = tmp_result.faces[i].width;
						info.face_height = tmp_result.faces[i].height;
						info.mouth_prob = tmp_result.faces[i].result_vec_mouth[j].score;
						info.mouth_off_x = result_vec[j].col1;
						info.mouth_off_y = result_vec[j].row1;
						info.mouth_width = result_vec[j].col2 - result_vec[j].col1;
						info.mouth_height = result_vec[j].row2 - result_vec[j].row1;
						if(info.mouth_prob >= d_arg->ssd_mouth_thresh)
							result->faces.push_back(info);
					}
				}
			}

			return true;
		}

		bool Detect(const unsigned char* bgr_image, int width, int height, int widthStep, const void* detect_arg, void* output_detected_data)
		{
			if (bgr_image == 0 || detect_arg == 0 || output_detected_data == 0)
				return false;

			const DetectArgs* d_arg = (DetectArgs*)detect_arg;
			std::vector<ZQ_CNN_BBox> thirdBbox;
			double t1 = omp_get_wtime();
			//printf("min_size = %d\n", d_arg->mtcnn_min_size);

			mtcnn_detector.SetPara(width, height, d_arg->mtcnn_min_size, d_arg->mtcnn_thresh_p, d_arg->mtcnn_thresh_r, d_arg->mtcnn_thresh_o,
				d_arg->mtcnn_thresh_nms_p, d_arg->mtcnn_thresh_nms_r, d_arg->mtcnn_thresh_nms_o, d_arg->mtcnn_scale);
			if (!mtcnn_detector.Find(bgr_image, width, height, widthStep, thirdBbox))
			{
				double t2 = omp_get_wtime();
				//printf("find no face: %.3f ms\n", 1000*(t2 - t1));
				return false;
			}
			double t2 = omp_get_wtime();
			//printf("find face: %.3f ms\n", 1000*(t2 - t1));

			DetectedResult* out_result = (DetectedResult*)output_detected_data;
			out_result->faces.clear();


			cv::Mat image(height, width, CV_MAKE_TYPE(8, 3));
			for (int h = 0; h < height; h++)
			{
				memcpy(image.data + h*image.step[0], bgr_image + h*widthStep, sizeof(char)*width * 3);
			}

			int num = thirdBbox.size();
			for (int i = 0; i < num; i++)
			{
				DetectedFace one_face;
				one_face.off_x = thirdBbox[i].col1;
				one_face.off_y = thirdBbox[i].row1;
				one_face.width = thirdBbox[i].col2 - thirdBbox[i].col1;
				one_face.height = thirdBbox[i].row2 - thirdBbox[i].row1;
				one_face.box = thirdBbox[i];
				int border_x = one_face.width*d_arg->enlarge_border;
				int border_y = one_face.height*d_arg->enlarge_border;
				int real_border_x = __min(border_x, __min(one_face.off_x, width - one_face.off_x - one_face.width));
				int real_border_y = __min(border_y, __min(one_face.off_y, height - one_face.off_y - one_face.height));
				cv::Rect rect(cv::Point(thirdBbox[i].col1 - real_border_x, thirdBbox[i].row1 - real_border_y), cv::Point(thirdBbox[i].col2 + real_border_x, thirdBbox[i].row2 + real_border_y));
				cv::Mat tmp_img(image, rect);

				if (d_arg->enable_rot)
				{
					float eye_cx = (thirdBbox[i].ppoint[0] + thirdBbox[i].ppoint[1]) / 2;
					float eye_cy = (thirdBbox[i].ppoint[5] + thirdBbox[i].ppoint[6]) / 2;
					float mouth_cx = (thirdBbox[i].ppoint[3] + thirdBbox[i].ppoint[4]) / 2;
					float mouth_cy = (thirdBbox[i].ppoint[8] + thirdBbox[i].ppoint[9]) / 2;
					float cur_angle = _get_rot_angle(eye_cx, eye_cy, mouth_cx, mouth_cy);
					float standard_angle = _get_standard_angle();
					//printf("cur_angle = %f, std = %f\n", cur_angle, standard_angle);
					cv::Mat tmp_img2;
					one_face.rot_in_rad = cur_angle - standard_angle;
					_warp_image_to_standard(tmp_img, cur_angle, standard_angle, tmp_img2);
					//printf("border=%d,%d\n", real_border_x,real_border_y);

					//cv::Rect tmp_rect(cv::Point(real_border_x, real_border_y), cv::Point(tmp_img2.cols - real_border_x, tmp_img2.rows - real_border_y));
					//tmp_img = tmp_img2(tmp_rect);
					/*tmp_img = tmp_img2;
					cv::imshow("tmp", tmp_img);
					cv::waitKey(10);*/
					//printf("%d,%d\n", tmp_img.cols, tmp_img.rows);

				}
				else
				{
					one_face.rot_in_rad = 0;
				}

				ssd_detector.Detect(one_face.result_vec_mouth, tmp_img.data, tmp_img.cols, tmp_img.rows, tmp_img.step[0], d_arg->ssd_mouth_thresh, false);
				int mouth_num = one_face.result_vec_mouth.size();
				//printf("mouth_num = %d\n", mouth_num);
				for (int j = mouth_num - 1; j >= 0; j--)
				{
					ZQ_CNN_SSD::BBox& cur_box = one_face.result_vec_mouth[j];

					if (cur_box.col1 < real_border_x || cur_box.col2 > one_face.width + real_border_x
						|| cur_box.row1 < real_border_y || cur_box.row2 > one_face.height + real_border_y)
						one_face.result_vec_mouth.erase(one_face.result_vec_mouth.begin() + j);
					else
					{
						cur_box.col1 -= real_border_x;
						cur_box.col2 -= real_border_x;
						cur_box.row1 -= real_border_y;
						cur_box.row2 -= real_border_y;
					}
				}

				out_result->faces.push_back(one_face);
			}

			return true;
		}

		bool DrawResult(unsigned char* rgb_image, int width, int height, int widthStep, const void* detected_data)
		{
			if (rgb_image == 0 || detected_data == 0)
				return false;
			cv::Mat image(height, width, CV_MAKETYPE(8, 3), rgb_image);
			DetectedResult* detected_args = (DetectedResult*)detected_data;

			for (int i = 0; i < detected_args->faces.size(); i++)
			{
				int off_x = detected_args->faces[i].off_x;
				int off_y = detected_args->faces[i].off_y;
				int roi_width = detected_args->faces[i].width;
				int roi_height = detected_args->faces[i].height;
				float rot_in_rad = detected_args->faces[i].rot_in_rad;
				_draw_boxes(image, off_x, off_y, roi_width, roi_height, rot_in_rad,
					detected_args->faces[i].result_vec_mouth, mouth_objnames);
			}

			std::vector<ZQ_CNN_BBox> face_boxes;
			for (int i = 0; i < detected_args->faces.size(); i++)
				face_boxes.push_back(detected_args->faces[i].box);
			return true;
		}

		bool DrawSimpleResult(unsigned char* rgb_image, int width, int height, int widthStep, const void* detected_data)
		{
			if (rgb_image == 0 || detected_data == 0)
				return false;
			cv::Mat image(height, width, CV_MAKETYPE(8, 3), rgb_image);
			SimpleDetectedResult* detected_args = (SimpleDetectedResult*)detected_data;

			const std::vector<SimpleFaceInfo>& faces = detected_args->faces;
			cv::Scalar color1(0, 0, 255);
			cv::Scalar color2(0, 255, 0);
			for (int i = 0; i < detected_args->faces.size(); i++)
			{
				cv::rectangle(image, cv::Rect(faces[i].face_off_x, faces[i].face_off_y, faces[i].face_width, faces[i].face_height), color1, 3);
				cv::rectangle(image, cv::Rect(faces[i].mouth_off_x, faces[i].mouth_off_y, faces[i].mouth_width, faces[i].mouth_height), color2, 3);
			}


			return true;
		}

	private:

		void _clear()
		{
		}


		std::vector<std::string> _objects_names_from_file(std::string const filename)
		{
			std::ifstream file(filename);
			std::vector<std::string> file_lines;
			if (!file.is_open()) return file_lines;
			for (std::string line; getline(file, line);) file_lines.push_back(line);
			//std::cout << "object names loaded \n";
			return file_lines;
		}

		static void _trans_boxes(int off_x, int off_y, int roi_width, int roi_height, float rot_in_rad,
			const std::vector<ZQ_CNN_SSD::BBox>& input_vec, std::vector<ZQ_CNN_SSD::BBox>& output_vec)
		{
			output_vec = input_vec;
			if (rot_in_rad == 0)
			{
				for (int i = 0; i < output_vec.size(); i++)
				{
					output_vec[i].col1 += off_x;
					output_vec[i].col2 += off_x;
					output_vec[i].row1 += off_y;
					output_vec[i].row2 += off_y;
				}
			}
			else
			{
				float rot_mat[4] =
				{
					cos(rot_in_rad), -sin(rot_in_rad),
					sin(rot_in_rad), cos(rot_in_rad)
				};

				float out_corners[10];

				for (int i = 0; i < output_vec.size(); i++)
				{
					float x = output_vec[i].col1;
					float y = output_vec[i].row1;
					float w = output_vec[i].col2 - x;
					float h = output_vec[i].row2 - y;
					//printf("%f,%f,%f,%f\n", x,y,w,h);
					float corners[15] =
					{
						x,x + w,x + w,x, x + 0.5*w,
						y,y,y + h,y + h, y + 0.5*h,
						1,1,1,1,1
					};

					for (int j = 0; j < 5; j++)
					{
						corners[j] -= roi_width*0.5f;
						corners[5 + j] -= roi_height*0.5f;
					}
					memset(out_corners, 0, sizeof(float) * 10);
					for (int s = 0; s < 2; s++)
					{
						for (int t = 0; t < 5; t++)
						{
							for (int m = 0; m < 2; m++)
								out_corners[s * 5 + t] += rot_mat[s * 2 + m] * corners[m * 5 + t];
						}
					}

					for (int j = 0; j < 5; j++)
					{
						out_corners[j] += roi_width*0.5f;
						out_corners[5 + j] += roi_height*0.5f;
					}

					float min_x = out_corners[0];
					float max_x = out_corners[0];
					float min_y = out_corners[5];
					float max_y = out_corners[5];
					for (int j = 1; j < 5; j++)
					{
						min_x = __min(min_x, out_corners[j]);
						max_x = __max(max_x, out_corners[j]);
						min_y = __min(min_y, out_corners[5 + j]);
						max_y = __max(max_y, out_corners[5 + j]);
					}

					w = __max(0, max_x - min_x);
					h = __max(0, max_y - min_y);

					output_vec[i].col1 = __max(0, out_corners[4] - w*0.5f + off_x);
					output_vec[i].row1 = __max(0, out_corners[9] - h*0.5f + off_y);
					output_vec[i].col2 = output_vec[i].col1 + w;
					output_vec[i].row2 = output_vec[i].row1 + h;
				}
			}
		}

		void _draw_boxes(cv::Mat& mat_img, int off_x, int off_y, int roi_width, int roi_height, float rot_in_rad,
			const std::vector<ZQ_CNN_SSD::BBox>& result_vec, const std::vector<std::string>& obj_names,
			unsigned int wait_msec = 0, int current_det_fps = -1, int current_cap_fps = -1)
		{
			std::vector<ZQ_CNN_SSD::BBox> vecs;
			_trans_boxes(off_x, off_y, roi_width, roi_height, rot_in_rad, result_vec, vecs);

			int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };


			for (int i = 0; i < vecs.size(); i++)
			{
				int const offset = vecs[i].label * 123457 % 6;
				int const color_scale = 150 + (vecs[i].label * 123457) % 100;
				cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
				color *= color_scale;
				cv::rectangle(mat_img, cv::Point2f(vecs[i].col1, vecs[i].row1), cv::Point2f(vecs[i].col2, vecs[i].row2), color, 4);
				if (obj_names.size() > vecs[i].label)
				{
					std::string obj_name = obj_names[vecs[i].label];
					cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0);
					float w = vecs[i].col2 - vecs[i].col1;
					int const max_width = (text_size.width > w + 2) ? text_size.width : (w + 2);
					cv::rectangle(mat_img, cv::Point2f(__max((int)vecs[i].col1 - 3, 0), __max((int)vecs[i].row1 - 30, 0)),
						cv::Point2f(__min((int)vecs[i].col1 + max_width, mat_img.cols - 1), __min((int)vecs[i].row1, mat_img.rows - 1)),
						color, 1, 8, 0);
					putText(mat_img, obj_name, cv::Point2f(vecs[i].col1, vecs[i].row1 - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1);
				}
			}

		}

		static float _get_rot_angle(float eye_cx, float eye_cy, float mouth_cx, float mouth_cy)
		{
			float dir_x = mouth_cx - eye_cx;
			float dir_y = mouth_cy - eye_cy;
			return  atan2(dir_y, dir_x);
		}

		static float _get_standard_angle()
		{
			return atan2(1.0f, 0.0f);
		}

		static void _warp_image_to_standard(const cv::Mat& input, float cur_angle, float standard_angle, cv::Mat& output)
		{
			int width = input.cols;
			int height = input.rows;
			float rot_angle = cur_angle - standard_angle;
			const float m_pi = 3.1415926535f;
			cv::Mat affine_mat = cv::getRotationMatrix2D(cv::Point2f(width*0.5f, height*0.5f), rot_angle*180.0f / m_pi, 1);
			cv::warpAffine(input, output, affine_mat, cv::Size(width, height));
			/*cv::imshow("1", input);
			cv::imshow("2", output);
			cv::waitKey(0);*/
		}
	};
}

#endif
