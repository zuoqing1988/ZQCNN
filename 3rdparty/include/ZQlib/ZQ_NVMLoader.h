#ifndef _ZQ_NVM_LOADER_H_
#define _ZQ_NVM_LOADER_H_
#pragma once
#include <vector>
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
			
			if (_strcmpi(buffer.c_str(), "NVM_V3") != 0)
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
	};


}

#endif