#ifndef _ZQ_BINARY_IMAGE_PROCESSING_H_
#define _ZQ_BINARY_IMAGE_PROCESSING_H_
#pragma once
#include <string.h>
#include <stdlib.h>
#include <vector>
#ifdef ZQLIB_USE_OPENMP
#include <omp.h>
#endif

namespace ZQ
{
	class ZQ_BinaryImageProcessing
	{
	public:
		static bool Dilate(const bool* input, bool* output, int width, int height, const bool* pfilter2D, int xfsize, int yfsize, bool use_omp = false)
		{
			if (input == 0 || output == 0 || pfilter2D == 0)
				return false;

			memset(output, 0, sizeof(bool)*width*height);

			int XSIZE = 2 * xfsize + 1;
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
#pragma omp parallel for schedule(dynamic)
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (!input[i*width + j])
							continue;
						for (int yy = -yfsize; yy <= yfsize; yy++)
						{
							for (int xx = -xfsize; xx <= xfsize; xx++)
							{
								int ii = i + yy;
								int jj = j + xx;
								if (ii < 0 || ii >= height || jj < 0 || jj >= width)
									continue;
								if (pfilter2D[(yy + yfsize)*XSIZE + xx + xfsize])
									output[ii*width + jj] = true;
							}
						}
					}
				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (!input[i*width + j])
							continue;
						for (int yy = -yfsize; yy <= yfsize; yy++)
						{
							for (int xx = -xfsize; xx <= xfsize; xx++)
							{
								int ii = i + yy;
								int jj = j + xx;
								if (ii < 0 || ii >= height || jj < 0 || jj >= width)
									continue;
								if (pfilter2D[(yy + yfsize)*XSIZE + xx + xfsize])
									output[ii*width + jj] = true;
							}
						}
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
			return true;
		}

		static bool Erode(const bool* input, bool* output, int width, int height, const bool* pfilter2D, int xfsize, int yfsize, bool use_omp = false)
		{
			if (input == 0 || output == 0 || pfilter2D == 0)
				return false;

			memset(output, 1, sizeof(bool)*width*height);

			int XSIZE = 2 * xfsize + 1;
#ifdef ZQLIB_USE_OPENMP
			if (use_omp)
			{
#pragma omp parallel for schedule(dynamic)
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (input[i*width + j])
							continue;
						for (int yy = -yfsize; yy <= yfsize; yy++)
						{
							for (int xx = -xfsize; xx <= xfsize; xx++)
							{
								int ii = i + yy;
								int jj = j + xx;
								if (ii < 0 || ii >= height || jj < 0 || jj >= width)
									continue;
								if (pfilter2D[(yy + yfsize)*XSIZE + xx + xfsize])
									output[ii*width + jj] = false;
							}
						}
					}
				}
			}
			else
			{
#endif
				for (int i = 0; i < height; i++)
				{
					for (int j = 0; j < width; j++)
					{
						if (input[i*width + j])
							continue;
						for (int yy = -yfsize; yy <= yfsize; yy++)
						{
							for (int xx = -xfsize; xx <= xfsize; xx++)
							{
								int ii = i + yy;
								int jj = j + xx;
								if (ii < 0 || ii >= height || jj < 0 || jj >= width)
									continue;
								if (pfilter2D[(yy + yfsize)*XSIZE + xx + xfsize])
									output[ii*width + jj] = false;
							}
						}
					}
				}
#ifdef ZQLIB_USE_OPENMP
			}
#endif
			return true;
		}

		static bool BWlabel_naive(const bool* input, int width, int height, int* label, std::vector<int>& area_size, int connect_N = 8)
		{
			if (connect_N != 4 && connect_N != 8 || input == 0 || label == 0)
				return false;

			int connect_dir[8][2] = 
			{
				{ 1, 0 }, { -1, 0 }, { 0, -1 }, {0,1},
				{ 1, 1 }, { 1, -1 }, { -1, -1 }, {-1,1}
			};
			area_size.clear();
			int* queue_x = new int[width*height];
			int* queue_y = new int[width*height];
			bool* visited = new bool[width*height];
			memset(visited, 0, sizeof(bool)*width*height);
			memset(label, 0, sizeof(int)*width*height);
			int area_id = 1;
			while (true)
			{
				//find a seed
				int seed_x = -1;
				int seed_y = -1;
				bool has_find_seed = false;
				for (int w = 0; w < width; w++)
				{
					for (int h = 0; h < height; h++)
					{
						int offset = h*width + w;
						if (!visited[offset] && input[offset])
						{
							seed_x = w;
							seed_y = h;
							has_find_seed = true;
							break;
						}
					}
					if (has_find_seed)
						break;
				}

				if (!has_find_seed)
					break;
				//
				int head = 0;
				int tail = 0;
				queue_x[tail] = seed_x;
				queue_y[tail] = seed_y;
				visited[seed_y*width + seed_x] = true;
				label[seed_y*width + seed_x] = area_id;
				tail++;
				while (head < tail)
				{
					int cur_x = queue_x[head];
					int cur_y = queue_y[head];
					head++;
					for (int dd = 0; dd < connect_N; dd++)
					{
						int tmp_x = cur_x + connect_dir[dd][0];
						int tmp_y = cur_y + connect_dir[dd][1];
						if (tmp_x >= 0 && tmp_x < width && tmp_y >= 0 && tmp_y < height && !visited[tmp_y*width + tmp_x] && input[tmp_y*width + tmp_x])
						{
							queue_x[tail] = tmp_x;
							queue_y[tail] = tmp_y;
							tail++;
							visited[tmp_y*width + tmp_x] = true;
							label[tmp_y*width + tmp_x] = area_id;
						}
					}
				}
				area_size.push_back(tail);
				area_id++;
			}
			delete[]queue_x;
			delete[]queue_y;
			delete[]visited;
			return true;
		}

		static bool BWlabel(const bool* input, int width, int height, int* label, std::vector<int>& area_size, int connect_N = 8)
		{
			if (connect_N != 4 && connect_N != 8 || input == 0 || label == 0)
				return false;

			std::vector<int> start_row, end_row, start_col, label_for_each_run;
			std::vector<int> pair_i, pair_j;
			
			_bwlabel1_label_for_each_run(input, width, height, start_row, end_row, start_col, label_for_each_run, pair_i, pair_j, connect_N);
			
			int max_run_id = -1;
			for (int i = 0; i < label_for_each_run.size(); i++)
			{
				if (max_run_id < label_for_each_run[i])
					max_run_id = label_for_each_run[i];
			}

			if (max_run_id == -1)
			{
				memset(label, 0, sizeof(int)*width*height);
				area_size.clear();
				return true;
			}
		
			std::vector<std::vector<int>> graphs(max_run_id);
			
			for (int tt = 0; tt < pair_i.size(); tt++)
			{
				int id1 = pair_i[tt] - 1;
				int id2 = pair_j[tt] - 1;
				_bwlabel1_setvalue(graphs[id1], id2);
				_bwlabel1_setvalue(graphs[id2], id1);
			}

			bool converged = false;
			while (!converged)
			{
				converged = true;
				for (int c = 0; c < max_run_id; c++)
				{
					for (int j = 0; j < graphs[c].size(); j++)
					{
						int id1 = graphs[c][j];
						for (int k = j + 1; k < graphs[c].size(); k++)
						{
							int id2 = graphs[c][k];
							if (!_bwlabel1_find(graphs[id1], id2))
							{
								_bwlabel1_setvalue(graphs[id1], id2);
								converged = false;
							}
							if (!_bwlabel1_find(graphs[id2], id1))
							{
								_bwlabel1_setvalue(graphs[id2], id1);
								converged = false;
							}
						}
					}
				}
			}
	
			memset(label, 0, sizeof(int)*width*height);
			int* handled_id = new int[max_run_id];
			memset(handled_id, 0, sizeof(int)*max_run_id);
			bool done = false;
			int area_id = 1;
			while (!done)
			{
				done = true;
				for (int i = 0; i < max_run_id; i++)
				{
					if (handled_id[i] == 0)
					{
						handled_id[i] = area_id;
						for (int j = 0; j < graphs[i].size(); j++)
						{
							int tmp_id = graphs[i][j];
							handled_id[tmp_id] = area_id;
						}
						area_id++;
						done = false;
					}
				}
			}

			int area_num = area_id - 1;
			for (int i = 0; i < start_row.size(); i++)
			{
				int cur_st_r = start_row[i];
				int cur_ed_r = end_row[i];
				int cur_c = start_col[i];
				int run_id = label_for_each_run[i];
				int real_area_id = handled_id[run_id - 1];
				for (int r = cur_st_r; r <= cur_ed_r; r++)
					label[r*width + cur_c] = real_area_id;
			}
			
			area_size.resize(area_num);
			for (int i = 0; i < area_num; i++)
				area_size[i] = 0;

			for (int i = 0; i < width*height; i++)
			{
				if (label[i] > 0)
					area_size[label[i]-1]++;
			}
			delete[] handled_id;
			return true;
		}

		static bool ComputeDistance(const bool* flag, int width, int height, int* distance, int connect_N = 8)
		{
			if (connect_N != 4 && connect_N != 8 || flag == 0 || distance == 0)
			{
				return false;
			}
			int connect_dir[8][2] =
			{
				{ 1, 0 },{ -1, 0 },{ 0, -1 },{ 0,1 },
				{ 1, 1 },{ 1, -1 },{ -1, -1 },{ -1,1 }
			};
			int* queue_x = new int[width*height];
			int* queue_y = new int[width*height];
			bool* visited = new bool[width*height];
			memset(visited, 0, sizeof(bool)*width*height);
			memset(distance, 0, sizeof(int)*width*height);
			
			int head = 0;
			int tail = 0;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					if (flag[offset])
					{
						queue_x[tail] = w;
						queue_y[tail] = h;
						tail++;
						visited[offset] = true;
						distance[offset] = 0;
					}
				}
			}
			
			while (head < tail)
			{
				int cur_x = queue_x[head];
				int cur_y = queue_y[head];
				head++;
				int cur_dis = distance[cur_y*width + cur_x];
				for (int dd = 0; dd < connect_N; dd++)
				{
					int tmp_x = cur_x + connect_dir[dd][0];
					int tmp_y = cur_y + connect_dir[dd][1];
					if (tmp_x >= 0 && tmp_x < width && tmp_y >= 0 && tmp_y < height && !visited[tmp_y*width + tmp_x])
					{
						queue_x[tail] = tmp_x;
						queue_y[tail] = tmp_y;
						tail++;
						visited[tmp_y*width + tmp_x] = true;
						distance[tmp_y*width + tmp_x] = cur_dis + 1;
					}
				}
			}

			delete[]queue_x;
			delete[]queue_y;
			delete[]visited;
			return true;
		}

	private:

		static void _bwlabel1_label_for_each_run(const bool* input, int width, int height, std::vector<int>& start_row, std::vector<int>& end_row,
			std::vector<int>& start_col, std::vector<int>& label_for_each_run, std::vector<int>& pair_i, std::vector<int>& pair_j, int connect_N)
		{
			start_row.clear();
			end_row.clear();
			start_col.clear();
			label_for_each_run.clear();
			pair_i.clear();
			pair_j.clear();

			int* area_id = new int[width*height];
			memset(area_id, 0, sizeof(int)*width*height);
			int run_id = 1;
			for (int c = 0; c < width; c++)
			{
				int r = 0;
				while (true)
				{
					bool has_found = false;
					int i, start_r, end_r;

					for (i = r; i < height; i++)
					{
						if (input[i*width + c])
						{
							has_found = true;
							start_r = i;
							break;
						}
					}

					if (has_found)
					{
						for (i = start_r; input[i*width+c] && i < height; i++);
						end_r = i - 1;
						start_row.push_back(start_r);
						end_row.push_back(end_r);
						start_col.push_back(c);
						int pre_run_id = -1;
						if (connect_N == 4)
						{
							if (c >= 1)
							{
								for (int iii = start_r; iii <= end_r; iii++)
								{
									if (area_id[iii*width + c - 1] > 0)
									{
										pre_run_id = area_id[iii*width + c - 1];
										break;
									}
								}
							}
							
						}
						else
						{
							if (c >= 1)
							{
								for (int iii = __max(0, start_r - 1); iii <= __min(end_r + 1, height - 1); iii++)
								{
									if (area_id[iii*width + c - 1] > 0)
									{
										pre_run_id = area_id[iii*width + c - 1];
										break;
									}
								}
							}
						}

						if (pre_run_id > 0)
						{
							for (int iii = start_r; iii <= end_r; iii++)
							{
								area_id[iii*width + c] = pre_run_id;
							}
							label_for_each_run.push_back(pre_run_id);
						}
						else
						{
							for (int iii = start_r; iii <= end_r; iii++)
							{
								area_id[iii*width + c] = run_id;
							}
							label_for_each_run.push_back(run_id);
							run_id++;
						}
						
						r = i;
					}
					else
					{
						break;
					}
				}
			}
			
			///

			int num = start_row.size();
			for (int nn = 0; nn < num; nn++)
			{
				int cur_c = start_col[nn];
				int cur_st_r = start_row[nn];
				int cur_ed_r = end_row[nn];
				int cur_run_id = label_for_each_run[nn];
				if (cur_c != width - 1)
				{
					std::vector<int> tmp_area_ids;
					if (connect_N == 4)
					{
						for (int r = cur_st_r; r <= cur_ed_r; r++)
						{
							int tmp_area_id1 = area_id[r*width + cur_c + 1];
							if (tmp_area_id1 > 0)
							{
								if (!_bwlabel1_is_in_vec(tmp_area_ids,tmp_area_id1))
									tmp_area_ids.push_back(tmp_area_id1);
							}
						}
					}
					else
					{
						for (int r = __max(0,cur_st_r-1); r <= __min(height-1,cur_ed_r+1); r++)
						{
							int tmp_area_id1 = area_id[r*width + cur_c + 1];
							if (tmp_area_id1 > 0)
							{
								if (!_bwlabel1_is_in_vec(tmp_area_ids, tmp_area_id1))
									tmp_area_ids.push_back(tmp_area_id1);
							}
						}
					}

					for (int tt = 0; tt < tmp_area_ids.size(); tt++)
					{
						if (cur_run_id != tmp_area_ids[tt])
						{
							pair_i.push_back(cur_run_id);
							pair_j.push_back(tmp_area_ids[tt]);
						}
						
					}
				}
			}

			delete[]area_id;
		}

		static bool _bwlabel1_is_in_vec(const std::vector<int>& list, int v)
		{
			for (int i = 0; i < list.size(); i++)
			{
				if (v == list[i])
					return true;
			}
			return false;
		}

		static bool _bwlabel1_find(std::vector<int>& one_col, int row)
		{
			int size = one_col.size();
			if (size == 0)
			{
				return false;
			}
			int low = 0;
			int high = size - 1;
			int mid = size / 2;
			bool find_flag = false;

			do
			{
				if (one_col[mid] == row)
				{
					find_flag = true;
					break;
				}
				else if (one_col[mid] < row)
				{
					low = mid + 1;
					mid = (low + high) / 2;
				}
				else
				{
					high = mid - 1;
					mid = (low + high) / 2;
				}
			} while (low <= high);

			return find_flag;

		}

		
		static void _bwlabel1_setvalue(std::vector<int>& one_col, int row)
		{
			int size = one_col.size();
			
			if (size == 0)
			{
				one_col.push_back(row);
				return;
			}

			int low = 0;
			int high = size - 1;
			int mid = (low + high) / 2;
			bool find_flag = false;

			do
			{
				if (one_col[mid] == row)
				{
					find_flag = true;
					break;
				}
				else if (one_col[mid] < row)
				{
					low = mid + 1;
					mid = (low + high) / 2;
				}
				else
				{
					high = mid - 1;
					mid = (low + high) / 2;
				}
			} while (low <= high);

			if (find_flag)
			{
				return;
			}
			else
			{
				if (one_col[0] > row)
				{
					one_col.insert(one_col.begin(), row);
				}
				else if (one_col[size - 1] < row)
				{
					one_col.push_back(row);
				}
				else
				{
					if (one_col[high] < row)
					{
						do
						{
							if (one_col[high + 1] > row)
							{
								one_col.insert(one_col.begin() + high + 1, row);
								break;
							}
							high++;
							if (high + 1 >= size)
							{
								one_col.push_back(row);
								break;
							}
						} while (true);
					}
					else
					{
						do
						{
							if (one_col[high - 1] < row)
							{
								one_col.insert(one_col.begin() + high, row);
								break;
							}
							high--;
							if (high < 0)
							{
								one_col.insert(one_col.begin(), row);
								break;
							}
						} while (true);
					}
				}
			}
		}
	};
}

#endif