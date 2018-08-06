#ifndef _ZQ_BINARY_IMAGE_CONTOUR_H_
#define _ZQ_BINARY_IMAGE_CONTOUR_H_

#pragma once

#include <vector>
#include "ZQ_Vec2D.h"
#include "ZQ_BinaryImageProcessing.h"

namespace ZQ
{
	class ZQ_BinaryImageContour
	{
	public:
		class Contour
		{
		public:
			/*child ids are corresponding to the hole polygon, 
			child_id = -1 means no other contour in that hole.*/
			std::vector<ZQ_Vec2D> outer_polygon;
			std::vector<std::vector<ZQ_Vec2D>> hole_polygon;
			std::vector<int> child_ids;
			int father_id;
		};

		static bool GetBinaryImageContour(const bool* image, int width, int height, std::vector<Contour>& contours)
		{
			if (image == 0)
				return false;

			int imWidth = width + 2;
			int imHeight = height + 2;
			bool* padding_image = new bool[imWidth*imHeight];
			memset(padding_image, 0, sizeof(bool)*imWidth*imHeight);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					padding_image[(i + 1)*imWidth + (j + 1)] = image[i*width + j];
				}
			}
			int* label1 = new int[imWidth*imHeight];
			int* label2 = new int[imWidth*imHeight];
			std::vector<int> area_size1, area_size2;
			ZQ_BinaryImageProcessing::BWlabel(padding_image, imWidth, imHeight, label1, area_size1, 8);
			for (int i = 0; i < imWidth*imHeight; i++)
				padding_image[i] = !padding_image[i];
			ZQ_BinaryImageProcessing::BWlabel(padding_image, imWidth, imHeight, label2, area_size2, 8);
			
			int label_num_for_true = area_size1.size();
			int label_num_for_false = area_size2.size();
			std::vector<std::vector<ZQ_Vec2D>> contours_for_true(label_num_for_true);
			std::vector<std::vector<ZQ_Vec2D>> contours_for_false(label_num_for_false);
			std::vector<int> father_ids_for_true(label_num_for_true);
			std::vector<int> father_ids_for_false(label_num_for_false);
			std::vector<int> has_child_for_true(label_num_for_true);
			std::vector<int> has_child_for_false(label_num_for_false);

			int border_label_id = label2[0] - 1;
			
			for (int i = 0; i < label_num_for_true; i++)
			{
				father_ids_for_true[i] = -1;
				for (int h = 0; h < imHeight; h++)
				{
					int w = 0;
					bool has_found = false;
					for (; w < imWidth; w++)
					{
						if (label1[h*imWidth + w] == i + 1)
						{
							_findContour(label1, imWidth, imHeight, i + 1, w, h, contours_for_true[i]);
							has_found = true;
							break;
						}
					}
					
					if (has_found)
					{
						father_ids_for_true[i] = label2[h*imWidth + w - 1] - 1;
						has_child_for_false[label2[h*imWidth + w - 1] - 1] = true;
						break;
					}
				}
			}

			for (int i = 0; i < label_num_for_false; i++)
			{
				father_ids_for_false[i] = -1;
				if (i== border_label_id)
					continue; 

				for (int h = 0; h < imHeight; h++)
				{
					int w = 0;
					bool has_found = false;
					for (; w < imWidth; w++)
					{
						if (label2[h*imWidth + w] == i + 1)
						{
							_findContour(label2, imWidth, imHeight, i + 1, w, h, contours_for_false[i]);
							has_found = true;
							break;
						}
					}
					if (has_found)
					{
						father_ids_for_false[i] = label1[h*imWidth + w - 1] - 1;
						has_child_for_true[label1[h*imWidth + w - 1] - 1] = true;
						break;
					}
				}
			}

			contours.resize(label_num_for_true);
			for (int i = 0; i < label_num_for_false; i++)
			{
				int cur_father_id_for_false = father_ids_for_false[i];
				if (cur_father_id_for_false >= 0)
				{
					contours[cur_father_id_for_false].child_ids.push_back(i+label_num_for_true);
					contours[cur_father_id_for_false].hole_polygon.push_back(contours_for_false[i]);
				}
			}

			for (int i = 0; i < label_num_for_true; i++)
			{
				contours[i].outer_polygon = contours_for_true[i];
				int cur_father_id_for_true = father_ids_for_true[i];
				if (cur_father_id_for_true >= 0)
				{
					int cur_father_id_for_false = father_ids_for_false[cur_father_id_for_true];
					if (cur_father_id_for_false >= 0)
					{
						contours[i].father_id = cur_father_id_for_false;
						for (int j = 0; j < contours[cur_father_id_for_false].child_ids.size(); j++)
						{
							if (contours[cur_father_id_for_false].child_ids[j] >= label_num_for_true)
							{
								if (contours[cur_father_id_for_false].child_ids[j] - label_num_for_true == cur_father_id_for_true)
								{
									contours[cur_father_id_for_false].child_ids[j] = cur_father_id_for_true;
								}
							}
						}
					}
				}
			}

			for (int i = 0; i < label_num_for_true; i++)
			{
				for (int j = 0; j < contours[i].child_ids.size(); j++)
				{
					if (contours[i].child_ids[j] >= label_num_for_true)
					{
						contours[i].child_ids[j] = -1;
					}
				}
			}
			delete[]padding_image;
			delete[]label1;
			delete[]label2;

			return true;
		}

		/*Only return outer polygons (hole polygons are omitted)*/
		static bool GetBinaryImageContour(const bool* image, int width, int height, std::vector<std::vector<ZQ_Vec2D>>& contours)
		{
			if (image == 0)
				return false;

			int imWidth = width + 2;
			int imHeight = height + 2;
			bool* padding_image = new bool[imWidth*imHeight];
			memset(padding_image, 0, sizeof(bool)*imWidth*imHeight);

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					padding_image[(i + 1)*imWidth + (j + 1)] = image[i*width + j];
				}
			}

			bool* used_flag = new bool[imWidth*imHeight];
			memset(used_flag, 0, sizeof(bool)*imWidth*imHeight);
			bool* cur_filled = new bool[imWidth*imHeight];

			while (true)
			{
				int x, y;
				
				bool flag = _findSeedAndSeedFilling(padding_image, used_flag, cur_filled, imWidth, imHeight, x, y);
				if (flag == false)
					break;

				std::vector<ZQ_Vec2D> contour;
				_findContour(padding_image, imWidth, imHeight, x, y, contour);
				contours.push_back(contour);
				contour.clear();
			}

			for (int i = 0; i < contours.size(); i++)
			{
				for (int j = 0; j < contours[i].size(); j++)
				{
					contours[i][j].x -= 1;
					contours[i][j].y -= 1;
				}
			}

			delete[]used_flag;
			delete[]cur_filled;
			delete[]padding_image;
			return true;
		}

		static bool GetBinaryImageMaxContour(const bool* image, int width, int height, std::vector<ZQ_Vec2D>& contour)
		{
			std::vector<std::vector<ZQ_Vec2D>> contours;
			if (!GetBinaryImageContour(image, width, height, contours))
				return false;

			if (contours.size() == 0)
				return false;
			int len = contours[0].size();
			int idx = 0;
			for (int i = 0; i < contours.size(); i++)
			{
				if (len < contours[i].size())
				{
					idx = i;
					len = contours[i].size();
				}
			}
			contour = contours[idx];
			return true;
		}

	private:
		static bool _has_hole(const bool* cur_filled, int width, int height)
		{
			for (int i = 0; i < height; i++)
			{
				int j = 0;
				for (; j < width && !cur_filled[i*width + j]; j++);
				if (j == width)
					continue;
				for (; j < width && cur_filled[i*width + j]; j++);
				if (j == width)
					continue;
				for (; j < width && !cur_filled[i*width + j]; j++);
				if (j != width)
					return true;
			}
			return false;
		}

		static bool _findSeedAndSeedFilling(const bool* image, bool* used_flag, bool* cur_filled, int width, int height,int& x, int& y)
		{
			bool flag = false;
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					if (image[i*width + j] && !used_flag[i*width + j])
					{
						flag = true;
						x = j;
						y = i;
					}
				}
			}

			if (flag)
			{
				int* idx_x = new int[width*height];
				int* idx_y = new int[width*height];
				memset(cur_filled, 0, sizeof(bool)*width*height);

				int head = 0;
				int tail = 1;
				idx_x[head] = x;
				idx_y[head] = y;
				cur_filled[y*width + x] = true;

				do
				{
					int cur_x = idx_x[head];
					int cur_y = idx_y[head];
					head++;

					if (image[(cur_y - 1)*width + cur_x] && !cur_filled[(cur_y - 1)*width + cur_x])
					{
						idx_x[tail] = cur_x;
						idx_y[tail] = cur_y - 1;
						tail++;
						cur_filled[(cur_y - 1)*width + cur_x] = true;
					}
					if (image[(cur_y + 1)*width + cur_x] && !cur_filled[(cur_y + 1)*width + cur_x])
					{
						idx_x[tail] = cur_x;
						idx_y[tail] = cur_y + 1;
						tail++;
						cur_filled[(cur_y + 1)*width + cur_x] = true;
					}
					if (image[cur_y*width + cur_x - 1] && !cur_filled[cur_y*width + cur_x - 1])
					{
						idx_x[tail] = cur_x - 1;
						idx_y[tail] = cur_y;
						tail++;
						cur_filled[cur_y*width + cur_x - 1] = true;
					}
					if (image[cur_y*width + cur_x + 1] && !cur_filled[cur_y*width + cur_x + 1])
					{
						idx_x[tail] = cur_x + 1;
						idx_y[tail] = cur_y;
						tail++;
						cur_filled[cur_y*width + cur_x + 1] = true;
					}
				} while (head < tail);


				for (int i = 0; i < tail; i++)
				{
					used_flag[idx_y[i] * width + idx_x[i]] = true;
				}

				delete[]idx_x;
				delete[]idx_y;

				return true;

			}
			else
			{
				return false;
			}
		}

		static void _findContour(const bool* image, int width, int height, int x, int y, std::vector<ZQ_Vec2D>& contour)
		{
			int cur_x = x;
			int cur_y = y;
			while (image[cur_y*width + cur_x - 1])
				cur_x--;
			ZQ_Vec2D pt;
			pt.x = cur_x - 0.5;
			pt.y = cur_y + 0.5;

			contour.push_back(pt);
			pt.x = cur_x - 0.5;
			pt.y = cur_y - 0.5;
			contour.push_back(pt);

			//DOWN:[0,-1],UP:[0,1],LEFT:[-1,0],RIGHT:[1,0]
			const int DOWN_DIR = 0, UP_DIR = 1, LEFT_DIR = 2, RIGHT_DIR = 3;

			int cur_dir = DOWN_DIR;

			while (true)
			{
				switch (cur_dir)
				{
				case DOWN_DIR:
					if (!image[(cur_y - 1)*width + cur_x])
					{
						pt.x = cur_x + 0.5;
						pt.y = cur_y - 0.5;
						contour.push_back(pt);
						cur_dir = RIGHT_DIR;
					}
					else
					{
						if (image[(cur_y - 1)*width + cur_x - 1])
						{
							cur_x = cur_x - 1;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = LEFT_DIR;
						}
						else
						{
							cur_x = cur_x;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = DOWN_DIR;
						}
					}
					break;
				case UP_DIR:
					if (!image[(cur_y + 1)*width + cur_x])
					{
						pt.x = cur_x - 0.5;
						pt.y = cur_y + 0.5;
						contour.push_back(pt);
						cur_dir = LEFT_DIR;
					}
					else
					{
						if (image[(cur_y + 1)*width + cur_x + 1])
						{
							cur_x = cur_x + 1;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = RIGHT_DIR;
						}
						else
						{
							cur_x = cur_x;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = UP_DIR;
						}
					}
					break;
				case LEFT_DIR:
					if (!image[cur_y*width + cur_x - 1])
					{
						pt.x = cur_x - 0.5;
						pt.y = cur_y - 0.5;
						contour.push_back(pt);
						cur_dir = DOWN_DIR;
					}
					else
					{
						if (image[(cur_y + 1)*width + cur_x - 1])
						{
							cur_x = cur_x - 1;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = UP_DIR;
						}
						else
						{
							cur_x = cur_x - 1;
							cur_y = cur_y;
							pt.x = cur_x - 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = LEFT_DIR;
						}
					}
					break;
				case RIGHT_DIR:
					if (!image[cur_y*width + cur_x + 1])
					{
						pt.x = cur_x + 0.5;
						pt.y = cur_y + 0.5;
						contour.push_back(pt);
						cur_dir = UP_DIR;
					}
					else
					{
						if (image[(cur_y - 1)*width + cur_x + 1])
						{
							cur_x = cur_x + 1;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = DOWN_DIR;
						}
						else
						{
							cur_x = cur_x + 1;
							cur_y = cur_y;
							pt.x = cur_x + 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = RIGHT_DIR;
						}
					}
					break;
				}

				int len = contour.size();
				if (contour[0].x == contour[len - 1].x && contour[0].y == contour[len - 1].y)
					break;

			}

			contour.pop_back();
		}

		static void _findContour(const int* label_img, int width, int height, int cur_label, int x, int y, std::vector<ZQ_Vec2D>& contour)
		{
			int cur_x = x;
			int cur_y = y;
			while (label_img[cur_y*width + cur_x - 1] == cur_label)
				cur_x--;
			ZQ_Vec2D pt;
			pt.x = cur_x - 0.5;
			pt.y = cur_y + 0.5;

			contour.push_back(pt);
			pt.x = cur_x - 0.5;
			pt.y = cur_y - 0.5;
			contour.push_back(pt);

			//DOWN:[0,-1],UP:[0,1],LEFT:[-1,0],RIGHT:[1,0]
			const int DOWN_DIR = 0, UP_DIR = 1, LEFT_DIR = 2, RIGHT_DIR = 3;

			int cur_dir = DOWN_DIR;

			while (true)
			{
				switch (cur_dir)
				{
				case DOWN_DIR:
					if (label_img[(cur_y - 1)*width + cur_x] != cur_label)
					{
						pt.x = cur_x + 0.5;
						pt.y = cur_y - 0.5;
						contour.push_back(pt);
						cur_dir = RIGHT_DIR;
					}
					else
					{
						if (label_img[(cur_y - 1)*width + cur_x - 1] == cur_label)
						{
							cur_x = cur_x - 1;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = LEFT_DIR;
						}
						else
						{
							cur_x = cur_x;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = DOWN_DIR;
						}
					}
					break;
				case UP_DIR:
					if (label_img[(cur_y + 1)*width + cur_x] != cur_label)
					{
						pt.x = cur_x - 0.5;
						pt.y = cur_y + 0.5;
						contour.push_back(pt);
						cur_dir = LEFT_DIR;
					}
					else
					{
						if (label_img[(cur_y + 1)*width + cur_x + 1] == cur_label)
						{
							cur_x = cur_x + 1;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = RIGHT_DIR;
						}
						else
						{
							cur_x = cur_x;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = UP_DIR;
						}
					}
					break;
				case LEFT_DIR:
					if (label_img[cur_y*width + cur_x - 1] != cur_label)
					{
						pt.x = cur_x - 0.5;
						pt.y = cur_y - 0.5;
						contour.push_back(pt);
						cur_dir = DOWN_DIR;
					}
					else
					{
						if (label_img[(cur_y + 1)*width + cur_x - 1] == cur_label)
						{
							cur_x = cur_x - 1;
							cur_y = cur_y + 1;
							pt.x = cur_x + 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = UP_DIR;
						}
						else
						{
							cur_x = cur_x - 1;
							cur_y = cur_y;
							pt.x = cur_x - 0.5;
							pt.y = cur_y + 0.5;
							contour.push_back(pt);
							cur_dir = LEFT_DIR;
						}
					}
					break;
				case RIGHT_DIR:
					if (label_img[cur_y*width + cur_x + 1] != cur_label)
					{
						pt.x = cur_x + 0.5;
						pt.y = cur_y + 0.5;
						contour.push_back(pt);
						cur_dir = UP_DIR;
					}
					else
					{
						if (label_img[(cur_y - 1)*width + cur_x + 1] == cur_label)
						{
							cur_x = cur_x + 1;
							cur_y = cur_y - 1;
							pt.x = cur_x - 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = DOWN_DIR;
						}
						else
						{
							cur_x = cur_x + 1;
							cur_y = cur_y;
							pt.x = cur_x + 0.5;
							pt.y = cur_y - 0.5;
							contour.push_back(pt);
							cur_dir = RIGHT_DIR;
						}
					}
					break;
				}

				int len = contour.size();
				if (contour[0].x == contour[len - 1].x && contour[0].y == contour[len - 1].y)
					break;

			}

			contour.pop_back();
		}
	};

}

#endif