#ifndef _ZQ_MIN_INDEPENDENT_SETS_H_
#define _ZQ_MIN_INDEPENDENT_SETS_H_
#pragma once

#include <string.h>
#include <stdio.h>
#include <vector>

namespace ZQ
{
	class ZQ_RawGraph
	{
	public:
		int* idx_of_each_row; // 
		int* row_of_each_idx; //

		int vert_num;
		int* edge_num_of_vert;
		int** edges;

	public:
		ZQ_RawGraph()
		{
			vert_num = 0;
			edge_num_of_vert = 0;
			edges = 0;
			idx_of_each_row = 0;
			row_of_each_idx = 0;
		}

		~ZQ_RawGraph()
		{
			for (int i = 0; i < vert_num; i++)
			{
				if (edges[i])
				{
					delete[](edges[i]);
					edges[i] = 0;
				}
			}
			delete[]edges;
			edges = 0;
			delete[]edge_num_of_vert;
			edge_num_of_vert = 0;

		}
		
		bool Sort()
		{
			if (vert_num <= 0 || edge_num_of_vert == 0 || edges == 0)
				return false;

			int len = 1;
			int K = 0;
			while (len < vert_num)
			{
				len *= 2;
				K++;
			}
			int* idx = new int[vert_num];
			int* val = new int[vert_num];
			for (int i = 0; i < vert_num; i++)
			{
				idx[i] = i;
				val[i] = edge_num_of_vert[i];
			}

			for (int i = 0; i < K; i++)
			{
				_mergeSort(vert_num, idx, val, i);
			}
			int* old_idx = new int[vert_num];

			for (int i = 0; i < vert_num; i++)
				old_idx[idx[i]] = i;

			delete[]val;
			val = 0;

			int* new_idx_of_each_row = new int[vert_num];
			int* new_row_of_each_idx = new int[vert_num];
			for (int i = 0; i < vert_num; i++)
			{
				int real_idx = idx_of_each_row[idx[i]];
				new_idx_of_each_row[i] = real_idx;
				new_row_of_each_idx[real_idx] = i;
			}
			int* tmp_idx_ptr = idx_of_each_row;
			idx_of_each_row = new_idx_of_each_row;
			delete[]tmp_idx_ptr;
			int* tmp_row_ptr = row_of_each_idx;
			row_of_each_idx = new_row_of_each_idx;
			delete[]tmp_row_ptr;


			delete[]idx;
			idx = 0;
			delete[]old_idx;
			old_idx = 0;

			return true;
		}

		static void _mergeSort(int N, int* idx, int* val, int k)
		{
			int * tmpIdx = new int[N];
			int * tmpVal = new int[N];
			memcpy(tmpIdx, idx, sizeof(int)*N);
			memcpy(tmpVal, val, sizeof(int)*N);

			int len = 1;
			for (int i = 0; i < k; i++)
				len *= 2;

			for (int i = 0, j = len; i < N && j < N; i += 2 * len, j += 2 * len)
			{
				int x = i, y = j, z = i;
				while (x < i + len && y < j + len && y < N)
				{
					if (tmpVal[x] > tmpVal[y])
					{
						idx[z] = tmpIdx[x];
						val[z++] = tmpVal[x++];
					}
					else
					{
						idx[z] = tmpIdx[y];
						val[z++] = tmpVal[y++];
					}
				}
				if (x == i + len)
				{

				}
				else
				{
					while (x < i + len)
					{
						idx[z] = tmpIdx[x];
						val[z++] = tmpVal[x++];
					}
				}
			}
			delete[]tmpIdx;
			delete[]tmpVal;

		}
	};

	class ZQ_RawSets
	{
	public:
		ZQ_RawSets()
		{
			set_num = 0;
			element_num_of_set = 0;
			elements = 0;
		}

		~ZQ_RawSets()
		{
			if (element_num_of_set)
			{
				delete[]element_num_of_set;
				element_num_of_set = 0;
			}
			if (elements)
			{
				for (int i = 0; i < set_num; i++)
				{
					if (elements[i])
					{
						delete[](elements[i]);
					}
				}
				delete[]elements;
				elements = 0;
			}
		}

		int set_num;
		int* element_num_of_set;
		int** elements;
	public:
		bool Print(const char* name)
		{
			if (strcmp(name, "stdout") == 0 || strcmp(name, "stderr") == 0)
			{
				for (int i = 0; i < set_num; i++)
				{
					printf("[S%3d]:", i);
					for (int j = 0; j < element_num_of_set[i]; j++)
					{
						printf("%d ", elements[i][j]);
					}
					printf("\n");
				}
				return true;
			}
			else
			{
				FILE* out = 0;
				if (0 != fopen_s(&out, name, "w"))
				{
					printf("failed to open file %s\n", name);
					return false;
				}
				for (int i = 0; i < set_num; i++)
				{
					fprintf(out, "[S%3d]:", i);
					for (int j = 0; j < element_num_of_set[i]; j++)
					{
						fprintf(out, "%d ", elements[i][j]);
					}
					fprintf(out, "\n");
				}
				fclose(out);
				return true;
			}
		}
	};

	class ZQ_MinIndependentSets
	{
	public:
		static bool ZQ_MinIndependentSetsWelshPowell(const ZQ_RawGraph* graph, ZQ_RawSets** sets)
		{
			if (graph == 0 || graph->edge_num_of_vert == 0 || graph->edges == 0)
				return false;
			std::vector< std::vector<int> > mySets;
			int num_sets = 0;
			int vert_num = graph->vert_num;
			int* vert_color = new int[vert_num];
			for (int i = 0; i < vert_num; i++)
				vert_color[i] = -1;

			int cur_color_num = 0;
			int start_color = 0;
			for (int i = 0; i < vert_num; i++)
			{
				int real_idx = graph->idx_of_each_row[i];
				if (cur_color_num == 0)
				{
					std::vector<int> one_set;
					one_set.push_back(real_idx);
					mySets.push_back(one_set);
					vert_color[i] = 0;
					cur_color_num = 1;
					start_color = 0;
				}
				else
				{
					bool* conflict = new bool[cur_color_num];
					memset(conflict, false, sizeof(bool)*cur_color_num);
					for (int j = 0; j < graph->edge_num_of_vert[real_idx]; j++)
					{
						int neighbor_idx = graph->edges[real_idx][j];
						int neighbor_row = graph->row_of_each_idx[neighbor_idx];
						if (neighbor_row < i)
						{
							conflict[vert_color[neighbor_row]] = true;
						}
					}
					int mycolor = cur_color_num;
					for (int j = 0; j < cur_color_num; j++)
					{
						int real_j = j + start_color + 1;
						if (real_j >= cur_color_num)
							real_j %= cur_color_num;
						if (!conflict[real_j])
						{
							mycolor = real_j;
							break;
						}
					}
					vert_color[i] = mycolor;
					start_color = mycolor;
					if (mycolor == cur_color_num)
					{
						std::vector<int> one_set;
						one_set.push_back(real_idx);
						mySets.push_back(one_set);

						cur_color_num++;
					}
					else
					{
						mySets[mycolor].push_back(real_idx);

					}
					delete[]conflict;
				}
			}

			*sets = new ZQ_RawSets;
			(*sets)->set_num = cur_color_num;
			(*sets)->element_num_of_set = new int[cur_color_num];
			(*sets)->elements = (int**)malloc(sizeof(int*)*cur_color_num);
			for (int i = 0; i < cur_color_num; i++)
			{
				int elmt_num_of_set = mySets[i].size();
				(*sets)->element_num_of_set[i] = elmt_num_of_set;
				(*sets)->elements[i] = new int[elmt_num_of_set];
				for (int j = 0; j < elmt_num_of_set; j++)
					(*sets)->elements[i][j] = mySets[i][j];
			}
			delete[]vert_color;
			mySets.clear();
			return true;

		}
	};
}



#endif