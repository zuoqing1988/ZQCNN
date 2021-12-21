#ifndef _ZQ_NVM_LOADER_H_
#define _ZQ_NVM_LOADER_H_
#pragma once
#include <vector>
#include <map>
#include <string>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>

/*loader of nvm file
VisualSFM : A Visual Structure from Motion System, Changchang Wu
*/

namespace ZQ
{
	class ZQ_NVM_Model
	{
	public:
		class Camera
		{
			/*
			The internal camera model has 8 parameters(7 if radial distortion is disabled)
			Given camera K[R T], K = [f, 0 0; 0 f 0; 0 0 1], radial distortion r, and a 3D point X.
			The reprojection in the image is[x, y, z]' = K (RX + T) -> (x/z, y/z)'
			Let the measurement be(mx, my), which is relative to principal point(typically image center)
			The distortion factor is r2 = r * (mx * mx + my * my)
			The undistorted measurement is(1 + r2) * (mx, my)
			Then, the reprojection error is(x / z - (1 + r2) mx, y / z - (1 + r2) my)

			NOTE that the parameters saved in NVM file is slightly different with the internal representation.Instead, NVM saves the following for each camera:
			f, R(as quaternion), C = -R'T, rn = r * f * f.
			*/
		public:
			std::string image_name;
			int width, height;
			double focal;
			double qw, qx, qy, qz;
			double tx, ty, tz;
			double r;
		};

		class Measure
		{
		public:
			int cam_id;
			int feat_id;
			double x, y;
		};

		class Point
		{
		public:
			double x, y, z;
			double r, g, b;
			std::vector<Measure> measure_list;
		};
	public:
		std::vector<Camera> camera_list;
		std::vector<Point> point_list;
	};

	class ZQ_NVM_Container
	{
		/*
		NVM_V3 [optional calibration]                        # file version header
		<Model1> <Model2> ...                                # multiple reconstructed models
		<Empty Model containing the unregistered Images>     # number of camera > 0, but number of points = 0
		<0>                                                  # 0 camera to indicate the end of model section
		<Some comments describing the PLY section>
		<Number of PLY files> <List of indices of models that have associated PLY>

		The [optional calibration] exists only if you use "Set Fixed Calibration" Function
		FixedK fx cx fy cy

		Each reconstructed <model> contains the following
		<Number of cameras>   <List of cameras>
		<Number of 3D points> <List of points>

		The cameras and 3D points are saved in the following format
		<Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
		<Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
		<Measurement> = <Image index> <Feature Index> <xy>
		*/

		class _tmp_point
		{
		public:
			int id;
			double r, g, b;
			double pt2Dx, pt2Dy;
			double pt3Dx, pt3Dy, pt3Dz;
		};

	public:
		std::vector<ZQ_NVM_Model> model_list;

	public:
		bool LoadFromFile(const char* file)
		{
			std::fstream in_ptr(file, std::ios::in);
			if (!in_ptr.is_open())
				return false;
			
			model_list.clear();
			std::string buffer;
			std::getline(in_ptr, buffer);
			int len = buffer.length();
			if (len != 0)
			{
				int pos = len - 1;
				while (pos >= 0 && (buffer[pos] == '\n' || buffer[pos] == ' ' || buffer[pos] == '\t'))
					pos--;
				if (pos != len-1)
					buffer.erase(buffer.begin() + len - 1, buffer.end());
			}
			
#if defined(_WIN32)
			if (strncmp(buffer.c_str(), "NVM_V3",6) != 0)
#else
			if (strncmp(buffer.c_str(), "NVM_V3",6) != 0)
#endif
			{
				return false;
			}

			while (true)
			{
				int cam_num = 0;
				in_ptr >> cam_num;
				if (cam_num <= 0 || cam_num > 65535)
					break;
				int old_size = model_list.size();
				model_list.resize(old_size + 1);
				model_list[old_size].camera_list.resize(cam_num);
				for (int i = 0; i < cam_num; i++)
				{
					double end_flag = -1;
					in_ptr >> model_list[old_size].camera_list[i].image_name
						>> model_list[old_size].camera_list[i].focal
						>> model_list[old_size].camera_list[i].qw
						>> model_list[old_size].camera_list[i].qx
						>> model_list[old_size].camera_list[i].qy
						>> model_list[old_size].camera_list[i].qz
						>> model_list[old_size].camera_list[i].tx
						>> model_list[old_size].camera_list[i].ty
						>> model_list[old_size].camera_list[i].tz
						>> model_list[old_size].camera_list[i].r
						>> end_flag;
					
					if (end_flag != 0)
					{
						model_list.clear();
						return false;
					}
				}


				int point_num = -1;
				in_ptr >> point_num;
				if (point_num < 0 || point_num > 1e8)
				{
					model_list.clear();
					return false;
				}
				model_list[old_size].point_list.resize(point_num);
				for (int i = 0; i < point_num; i++)
				{
					in_ptr >> model_list[old_size].point_list[i].x
						>> model_list[old_size].point_list[i].y
						>> model_list[old_size].point_list[i].z
						>> model_list[old_size].point_list[i].r
						>> model_list[old_size].point_list[i].g
						>> model_list[old_size].point_list[i].b;
					int measure_num;
					in_ptr >> measure_num;
					if (measure_num < 0 || measure_num > cam_num)
					{
						model_list.clear();
						return false;
					}
					model_list[old_size].point_list[i].measure_list.resize(measure_num);
					for (int j = 0; j < measure_num; j++)
					{
						in_ptr >> model_list[old_size].point_list[i].measure_list[j].cam_id
							>> model_list[old_size].point_list[i].measure_list[j].feat_id
							>> model_list[old_size].point_list[i].measure_list[j].x
							>> model_list[old_size].point_list[i].measure_list[j].y;
					}
				}
			}
			return true;
		}

		bool ExportPointsOfOneImage(const char* img_name_ptr, const char* file_name_ptr) const
		{
			std::string img_name(img_name_ptr);
			std::string file_name(file_name_ptr);

			bool has_found = false;
			int model_idx = -1;
			int cam_idx = -1;
			for (int i = 0; i < model_list.size(); i++)
			{
				const ZQ_NVM_Model& cur_model = model_list[i];
				for (int j = 0; j < cur_model.camera_list.size(); j++)
				{
					if (img_name == cur_model.camera_list[j].image_name)
					{
						has_found = true;
						model_idx = i;
						cam_idx = j;
						break;
					}
				}
				if (has_found)
					break;
			}
			if (!has_found)
				return false;

			FILE* out = NULL;
#if defined(_WIN32)
			if (0 != fopen_s(&out, file_name.c_str(), "w"))
				return false;
#else
			out = fopen(file_name.c_str(), "w");
			if (out == NULL)
				return false;
#endif
			
			std::map<int, _tmp_point> tmp_points;
			const ZQ_NVM_Model& cur_model = model_list[model_idx];
			for (int i = 0; i < cur_model.point_list.size(); i++)
			{
				const ZQ_NVM_Model::Point& cur_point = cur_model.point_list[i];
				for (int j = 0; j < cur_point.measure_list.size(); j++)
				{
					const ZQ_NVM_Model::Measure& cur_measure = cur_point.measure_list[j];
					if (cur_measure.cam_id == cam_idx)
					{
						_tmp_point tmp_pt;
						int feat_id = cur_measure.feat_id;
						tmp_pt.r = cur_point.r;
						tmp_pt.g = cur_point.g;
						tmp_pt.b = cur_point.b;
						tmp_pt.id = feat_id;
						tmp_pt.pt2Dx = cur_measure.x;
						tmp_pt.pt2Dy = cur_measure.y;
						tmp_pt.pt3Dx = cur_point.x;
						tmp_pt.pt3Dy = cur_point.y;
						tmp_pt.pt3Dz = cur_point.z;
						if (tmp_points.find(feat_id) == tmp_points.end())
						{
							tmp_points.insert(std::make_pair(feat_id, tmp_pt));
						}
						else
						{
							printf("warining: repeat point detected, cam_id: %d, feat_id: %d\n", cam_idx, feat_id);
						}
					}
				}
			}

			std::map<int, _tmp_point>::const_iterator it = tmp_points.begin();
			for (; it != tmp_points.end(); ++it)
			{
				int feat_id = it->first;
				const _tmp_point& tmp_pt = it->second;
				fprintf(out, "%d %f %f %f %f %f %.0f %.0f %.0f\n",
					tmp_pt.id,
					tmp_pt.pt2Dx, tmp_pt.pt2Dy,
					tmp_pt.pt3Dx, tmp_pt.pt3Dy, tmp_pt.pt3Dz,
					tmp_pt.r, tmp_pt.g, tmp_pt.b);
			}
			fclose(out);
			return true;
		}


		void DeleteBadPoints(int seen_less_than_n_cam) 
		{
			for (int i = 0; i < model_list.size(); i++)
			{
				ZQ_NVM_Model& cur_model = model_list[i];
				for (int j = cur_model.point_list.size() - 1; j >= 0; j--)
				{
					if (cur_model.point_list[j].measure_list.size() < seen_less_than_n_cam)
					{
						cur_model.point_list.erase(cur_model.point_list.begin() + j);
					}
				}
			}
		}
	};


}

#endif