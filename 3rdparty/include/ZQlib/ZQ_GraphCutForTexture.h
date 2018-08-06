#ifndef _ZQ_GRAPHCUT_FOR_TEXTURE_H_
#define _ZQ_GRAPHCUT_FOR_TEXTURE_H_
#pragma once

#include "ZQ_GraphCut.h"
#include <math.h>

#ifndef ZQ_GRAPHCUT_FOR_TEXTURE_N_WORST
#define ZQ_GRAPHCUT_FOR_TEXTURE_N_WORST 3
#endif

namespace ZQ
{
	template<class T>
	class ZQ_GraphCutForTexture
	{
	public:
		ZQ_GraphCutForTexture(int patch_size = 32, int boarder_size = 8, float probability = 0.05);
		~ZQ_GraphCutForTexture();

	private:

		int in_width,in_height;
		int nChannels;
		T* input;

		int patch_size;
		int boarder_size;
		float probability;
		int optimize_iterations;

		int out_width,out_height;
		T* output;
		bool* is_empty;
		int* xpos_in_input;
		int* ypos_in_input;
		T* up_weight;
		T* down_weight;
		T* left_weight;
		T* right_weight;

	public:
		bool SetInputTexture(T* input, int in_width, int in_height, int nChannels);

		bool TextureSynthesis(T* input, int in_width, int in_height, T* output, int out_width, int out_height, int nChannels);

		bool TextureSynthesis(int out_width, int out_height, int patch_size = 32, int boarder_size = 8, float probability = 1.0);

		bool RandomSynthesis(int out_width, int out_height, int patch_size = 32, int num_patchs = 1000, float probability = 1.0);

		int Optimize(int iterations, int patch_size = 32);

		bool ExportOutputTexture(T* output);

		bool ExportCutPath(bool* path);

	private:
		T _distance(int nChannels, T* ptr1, T* ptr2);

		void _random_select_patch(int width, int height, int pathch_size, int& offsetx, int& offsety);

		void _select_patch(int in_width, int in_height, T* input, int out_width, int out_height, T* output, int nChannels,
			int xsize, int ysize, bool* mask, int match_xoff, int match_yoff, int& offsetx, int& offsety);

		void _graphcut_and_copy(int in_width, int height, T* input, int out_width, int out_height, T* output, int nChannels,
			int xsize, int ysize, bool* mask, int match_xoff, int match_yoff, int offsetx, int offsety);

		void _select_constraints(int xsize, int ysize, bool* mask, bool* outer_mask, bool* inner_mask);

		void _clear();

		bool _random_select_patch(int xsize, int ysize, int& input_offx, int& input_offy);

		bool _select_patch(int xsize, int ysize,int output_offx, int output_offy, int& input_offx, int& input_offy);

		bool _splat_to(int xsize, int ysize, int input_offx, int input_offy, int output_offx, int output_offy);

		int _optimizeOnce(int patch_size);

		bool _find_worst_area(int xsize, int ysize, int n_worst, int* output_offx, int* output_offy);
	};



	/****************************  definitions  ************************************/

	static const double ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val = 1e16;

	template<class T>
	ZQ_GraphCutForTexture<T>::ZQ_GraphCutForTexture(int patch_size, int boarder_size, float probability)
	{
		this->patch_size = patch_size;
		this->boarder_size = boarder_size;
		this->probability = probability;
		this->input = 0;
		this->output = 0;
		this->xpos_in_input = 0;
		this->ypos_in_input = 0;
		this->left_weight = 0;
		this->right_weight = 0;
		this->up_weight = 0;
		this->down_weight = 0;
		this->is_empty = 0;
	}

	template<class T>
	ZQ_GraphCutForTexture<T>::~ZQ_GraphCutForTexture()
	{

		_clear();
		if (input)
		{
			delete[]input;
			input = 0;
		}
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::SetInputTexture(T* input, int in_width, int in_height, int nChannels)
	{
		if (input == 0)
		{
			return false;
		}

		if (this->input)
		{
			delete[]this->input;
		}
		this->in_width = in_width;
		this->in_height = in_height;
		this->nChannels = nChannels;
		this->input = new T[in_width*in_height*nChannels];
		memcpy(this->input, input, sizeof(T)*nChannels*in_width*in_height);
		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::TextureSynthesis(T* input, int in_width, int in_height, T* output, int out_width, int out_height, int nChannels)
	{
		int offsetx = 0;
		int offsety = 0;

		int match_offx = 0;
		int match_offy = 0;

		while (true)
		{
			if (match_offx == 0 && match_offy == 0)
			{
				_random_select_patch(in_width, in_height, patch_size, offsetx, offsety);
				for (int y = 0; y < patch_size; y++)
				{
					for (int x = 0; x < patch_size; x++)
					{
						for (int c = 0; c < nChannels; c++)
						{
							output[(y*out_width + x)*nChannels + c] = input[(y*in_width + x)*nChannels + c];
						}
					}
				}
			}
			else
			{
				int xsize = __min(out_width - match_offx, patch_size);
				int ysize = __min(out_height - match_offy, patch_size);
				bool* mask = new bool[xsize*ysize];
				memset(mask, 0, sizeof(bool)*xsize*ysize);

				if (match_offx == 0 && match_offy > 0)
				{
					for (int y = 0; y < boarder_size; y++)
					{
						for (int x = 0; x < xsize; x++)
						{
							mask[y*xsize + x] = true;
						}
					}
				}
				else if (match_offx != 0 && match_offy == 0)
				{
					for (int y = 0; y < ysize; y++)
					{
						for (int x = 0; x < boarder_size; x++)
						{
							mask[y*xsize + x] = true;
						}
					}
				}
				else
				{
					for (int y = 0; y < boarder_size; y++)
					{
						for (int x = 0; x < xsize; x++)
						{
							mask[y*xsize + x] = true;
						}
					}
					for (int y = 0; y < ysize; y++)
					{
						for (int x = 0; x < boarder_size; x++)
						{
							mask[y*xsize + x] = true;
						}
					}
				}
				_select_patch(in_width, in_height, input, out_width, out_height, output, nChannels, xsize, ysize, mask, match_offx, match_offy, offsetx, offsety);
				_graphcut_and_copy(in_width, in_height, input, out_width, out_height, output, nChannels, xsize, ysize, mask, match_offx, match_offy, offsetx, offsety);

				delete[]mask;
			}

			match_offx += patch_size - boarder_size;

			if (match_offx + boarder_size >= out_width)
			{
				match_offy += patch_size - boarder_size;
				match_offx = 0;
			}

			if (match_offy + boarder_size >= out_height)
			{
				break;
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::TextureSynthesis(int out_width, int out_height, int patch_size, int boarder_size, float probability)
	{
		_clear();

		this->out_width = out_width;
		this->out_height = out_height;
		this->patch_size = patch_size;
		this->boarder_size = boarder_size;
		this->probability = probability;

		this->is_empty = new bool[out_width*out_height];
		this->left_weight = new T[out_width*out_height];
		this->right_weight = new T[out_width*out_height];
		this->up_weight = new T[out_width*out_height];
		this->down_weight = new T[out_width*out_height];
		this->xpos_in_input = new int[out_width*out_height];
		this->ypos_in_input = new int[out_width*out_height];
		this->output = new T[out_width*out_height*nChannels];

		memset(this->output, 0, sizeof(T)*out_width*out_height*nChannels);
		for (int i = 0; i < out_width*out_height; i++)
		{
			is_empty[i] = true;
			left_weight[i] = 0;
			right_weight[i] = 0;
			up_weight[i] = 0;
			down_weight[i] = 0;
			xpos_in_input[i] = 0;
			ypos_in_input[i] = 0;
		}



		int input_offx = 0;
		int input_offy = 0;
		int output_offx = 0;
		int output_offy = 0;

		while (true)
		{
			if (output_offx == 0 && output_offy == 0)
			{
				int xsize = __min(out_width - output_offx, patch_size);
				int ysize = __min(out_height - output_offy, patch_size);

				_random_select_patch(xsize, ysize, input_offx, input_offy);
				_splat_to(xsize, ysize, input_offx, input_offy, output_offx, output_offy);
			}
			else
			{
				int xsize = __min(out_width - output_offx, patch_size);
				int ysize = __min(out_height - output_offy, patch_size);

				_select_patch(xsize, ysize, output_offx, output_offy, input_offx, input_offy);
				_splat_to(xsize, ysize, input_offx, input_offy, output_offx, output_offy);
			}

			output_offx += patch_size - boarder_size;

			if (output_offx + boarder_size >= out_width)
			{
				output_offy += patch_size - boarder_size;
				output_offx = 0;
			}

			if (output_offy + boarder_size >= out_height)
			{
				break;
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::RandomSynthesis(int out_width, int out_height, int patch_size, int num_patchs, float probability)
	{
		_clear();

		this->out_width = out_width;
		this->out_height = out_height;
		this->patch_size = patch_size;
		this->probability = probability;

		this->is_empty = new bool[out_width*out_height];
		this->left_weight = new T[out_width*out_height];
		this->right_weight = new T[out_width*out_height];
		this->up_weight = new T[out_width*out_height];
		this->down_weight = new T[out_width*out_height];
		this->xpos_in_input = new int[out_width*out_height];
		this->ypos_in_input = new int[out_width*out_height];
		this->output = new T[out_width*out_height*nChannels];

		memset(this->output, 0, sizeof(T)*out_width*out_height*nChannels);
		for (int i = 0; i < out_width*out_height; i++)
		{
			is_empty[i] = true;
			left_weight[i] = 0;
			right_weight[i] = 0;
			up_weight[i] = 0;
			down_weight[i] = 0;
			xpos_in_input[i] = 0;
			ypos_in_input[i] = 0;
		}



		int input_offx = 0;
		int input_offy = 0;
		int output_offx = 0;
		int output_offy = 0;

		for (output_offy = 0; output_offy < out_height; output_offy += patch_size)
		{
			for (output_offx = 0; output_offx < out_width; output_offx += patch_size)
			{
				int xsize = __min(patch_size, out_width - output_offx);
				int ysize = __min(patch_size, out_height - output_offy);
				_random_select_patch(xsize, ysize, input_offx, input_offy);
				_splat_to(xsize, ysize, input_offx, input_offy, output_offx, output_offy);
			}
		}

		for (int nn = 0; nn < num_patchs; nn++)
		{
			int xsize = patch_size;
			int ysize = patch_size;

			int output_offx = rand() % (out_width - xsize);
			int output_offy = rand() % (out_height - ysize);

			_select_patch(xsize, ysize, output_offx, output_offy, input_offx, input_offy);
			_splat_to(xsize, ysize, input_offx, input_offy, output_offx, output_offy);
		}

		return true;

	}

	template<class T>
	int ZQ_GraphCutForTexture<T>::Optimize(int iterations, int patch_size)
	{
		for (int i = 0; i < iterations; i++)
		{
			if (1 == _optimizeOnce(patch_size))
				return 1;
		}
		return 0;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::ExportOutputTexture(T* output)
	{
		if (output == 0 || this->output == 0)
			return false;

		memcpy(output, this->output, sizeof(T)*out_width*out_height*nChannels);
		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::ExportCutPath(bool* path)
	{
		if (path == 0)
		{
			return false;
		}

		memset(path, 0, sizeof(bool)*out_width*out_height);

		for (int y = 0; y < out_height; y++)
		{
			for (int x = 0; x < out_width - 1; x++)
			{
				if (!(xpos_in_input[y*out_width + x] + 1 == xpos_in_input[y*out_width + x + 1] && ypos_in_input[y*out_width + x] == ypos_in_input[y*out_width + x]))
				{
					path[y*out_width + x] = true;
				}
			}
		}
		for (int y = 0; y < out_height - 1; y++)
		{
			for (int x = 0; x < out_width; x++)
			{
				if (!(xpos_in_input[y*out_width + x] == xpos_in_input[(y + 1)*out_width + x] && ypos_in_input[y*out_width + x] + 1 == ypos_in_input[(y + 1)*out_width + x]))
				{
					path[y*out_width + x] = true;
				}
			}
		}
		return true;
	}

	template<class T>
	T ZQ_GraphCutForTexture<T>::_distance(int nChannels, T* ptr1, T* ptr2)
	{
		T result = 0;
		for (int i = 0; i < nChannels; i++)
			result += (ptr1[i] - ptr2[i])*(ptr1[i] - ptr2[i]);
		return result;
	}

	template<class T>
	void ZQ_GraphCutForTexture<T>::_random_select_patch(int width, int height, int pathch_size, int& offsetx, int& offsety)
	{
		offsetx = rand() % (width - pathch_size);
		offsety = rand() % (height - pathch_size);
	}

	template<class T>
	void ZQ_GraphCutForTexture<T>::_select_patch(int in_width, int in_height, T* input, int out_width, int out_height, T* output, int nChannels,
		int xsize, int ysize, bool* mask, int match_xoff, int match_yoff, int& offsetx, int& offsety)
	{

		int xrange = in_width - xsize;
		int yrange = in_height - ysize;
		int total_range = xrange*yrange;

		const double max_double_val = 1e16;
		double max_distance = max_double_val;
		offsetx = 0;
		offsety = 0;

		for (int pp = 0; pp < total_range*probability; pp++)
		{
			int cur_x = rand() % xrange;
			int cur_y = rand() % yrange;

			double distance = 0;
			for (int y = 0; y < ysize; y++)
			{
				for (int x = 0; x < xsize; x++)
				{
					if (mask[y*xsize + x])
						distance += _distance(nChannels, input + nChannels*((cur_y + y)*in_width + (cur_x + x)), output + nChannels*((match_yoff + y)*out_width + (match_xoff + x)));
				}
			}
			if (distance < max_distance)
			{
				max_distance = distance;
				offsetx = cur_x;
				offsety = cur_y;
			}
		}
	}

	template<class T>
	void ZQ_GraphCutForTexture<T>::_graphcut_and_copy(int in_width, int height, T* input, int out_width, int out_height, T* output, int nChannels,
		int xsize, int ysize, bool* mask, int match_xoff, int match_yoff, int offsetx, int offsety)
	{

		const double max_val = 1e16;

		T* dist = new T[xsize*ysize];
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (mask[y*xsize + x])
				{
					dist[y*xsize + x] = _distance(nChannels, input + nChannels*((offsety + y)*in_width + (offsetx + x)), output + nChannels*((match_yoff + y)*out_width + (match_xoff + x)));
				}
				else
				{
					dist[y*xsize + x] = 0;
				}
			}
		}

		bool* outer_mask = new bool[xsize*ysize];
		bool* inner_mask = new bool[xsize*ysize];
		memset(outer_mask, 0, sizeof(bool)*xsize*ysize);
		memset(inner_mask, 0, sizeof(bool)*xsize*ysize);

		_select_constraints(xsize, ysize, mask, outer_mask, inner_mask);

		int* node_id = new int[xsize*ysize];
		int cur_node_id = 0;
		for (int i = 0; i < xsize*ysize; i++)
		{
			if (mask[i])
				node_id[i] = cur_node_id++;
			else
				node_id[i] = -1;
		}

		typedef Graph<double, double, double> GraphType;
		GraphType* graph = new GraphType(cur_node_id, 4 * cur_node_id);

		graph->add_node(cur_node_id);
		for (int i = 0; i < cur_node_id; i++)
		{
			if (outer_mask[i])
			{
				graph->add_tweights(node_id[i], max_val, 0);
			}
		}
		for (int i = 0; i < cur_node_id; i++)
		{
			if (inner_mask[i])
			{
				graph->add_tweights(node_id[i], 0, max_val);
			}
		}

		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (mask[y*xsize + x])
				{
					if (x > 0 && mask[y*xsize + x - 1])
					{
						double dis = dist[y*xsize + x] + dist[y*xsize + x - 1];
						graph->add_edge(node_id[y*xsize + x], node_id[y*xsize + x - 1], dis, dis);
					}
					if (x < xsize - 1 && mask[y*xsize + x + 1])
					{
						double dis = dist[y*xsize + x] + dist[y*xsize + x + 1];
						graph->add_edge(node_id[y*xsize + x], node_id[y*xsize + x + 1], dis, dis);
					}
					if (y > 0 && mask[(y - 1)*xsize + x])
					{
						double dis = dist[y*xsize + x] + dist[(y - 1)*xsize + x];
						graph->add_edge(node_id[y*xsize + x], node_id[(y - 1)*xsize + x], dis, dis);
					}
					if (y < ysize - 1 && mask[(y + 1)*xsize + x])
					{
						double dis = dist[y*xsize + x] + dist[(y + 1)*xsize + x];
						graph->add_edge(node_id[y*xsize + x], node_id[(y + 1)*xsize + x], dis, dis);
					}
				}
			}
		}

		graph->maxflow();

		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (mask[y*xsize + x] && graph->what_segment(node_id[y*xsize + x]) == GraphType::SOURCE)
					;
				else
				{
					for (int c = 0; c < nChannels; c++)
					{
						output[((match_yoff + y)*out_width + (match_xoff + x))*nChannels + c] = input[((offsety + y)*in_width + (offsetx + x))*nChannels + c];
					}
				}
			}
		}
		delete[]outer_mask;
		delete[]inner_mask;
		delete[]node_id;
		delete dist;
		delete graph;
	}

	template<class T>
	void ZQ_GraphCutForTexture<T>::_select_constraints(int xsize, int ysize, bool* mask, bool* outer_mask, bool* inner_mask)
	{
		bool check_top = true;
		bool check_bottom = true;
		bool check_left = true;
		bool check_right = true;
		for (int x = 0; x < xsize; x++)
		{
			if (!mask[0 * xsize + x])
			{
				check_top = false;
				break;
			}
		}
		for (int x = 0; x < xsize; x++)
		{
			if (!mask[(ysize - 1)*xsize + x])
			{
				check_bottom = false;
				break;
			}
		}
		for (int y = 0; y < ysize; y++)
		{
			if (!mask[y*xsize + 0])
			{
				check_left = false;
				break;
			}
		}
		for (int y = 0; y < ysize; y++)
		{
			if (!mask[y*xsize + xsize - 1])
			{
				check_right = false;
				break;
			}
		}

		if (check_top)
		{
			for (int x = 0; x < xsize; x++)
				outer_mask[0 * xsize + x] = true;
		}
		if (check_bottom)
		{
			for (int x = 0; x < xsize; x++)
				outer_mask[(ysize - 1)*xsize + x] = true;
		}
		if (check_left)
		{
			for (int y = 0; y < ysize; y++)
				outer_mask[y*xsize + 0] = true;
		}
		if (check_right)
		{
			for (int y = 0; y < ysize; y++)
				outer_mask[y*xsize + xsize - 1] = true;
		}

		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (mask[y*xsize + x])
				{
					if ((x > 1 && !mask[y*xsize + x - 1])
						|| (x < xsize - 1 && !mask[y*xsize + x + 1])
						|| (y > 1 && !mask[(y - 1)*xsize + x])
						|| (y < ysize - 1 && !mask[(y + 1)*xsize + x]))
						inner_mask[y*xsize + x] = true;
				}
			}
		}
	}

	template<class T>
	void ZQ_GraphCutForTexture<T>::_clear()
	{
		if (is_empty)
		{
			delete[]is_empty;
			is_empty = 0;
		}
		if (xpos_in_input)
		{
			delete[]xpos_in_input;
			xpos_in_input = 0;
		}
		if (ypos_in_input)
		{
			delete[]ypos_in_input;
			ypos_in_input = 0;
		}
		if (left_weight)
		{
			delete[]left_weight;
			left_weight = 0;
		}
		if (right_weight)
		{
			delete[]right_weight;
			right_weight = 0;
		}
		if (up_weight)
		{
			delete[]up_weight;
			up_weight = 0;
		}
		if (down_weight)
		{
			delete[]down_weight;
			down_weight = 0;
		}
		if (output)
		{
			delete[]output;
			output = 0;
		}
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::_random_select_patch(int xsize, int ysize, int& input_offx, int& input_offy)
	{
		if (xsize > in_width || ysize > in_height)
			return false;

		input_offx = rand() % (in_width - xsize);
		input_offy = rand() % (in_height - ysize);
		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::_select_patch(int xsize, int ysize, int output_offx, int output_offy, int& input_offx, int& input_offy)
	{
		if (output_offx + xsize > out_width || output_offy + ysize > out_height)
			return false;

		bool* mask = new bool[xsize*ysize];
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				int out_x = output_offx + x;
				int out_y = output_offy + y;
				mask[y*xsize + x] = !is_empty[out_y*out_width + out_x];
			}
		}

		int xrange = in_width - xsize;
		int yrange = in_height - ysize;
		int total_range = xrange*yrange;

		double max_distance = ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val;
		input_offx = 0;
		input_offy = 0;



		if (probability >= 1)
		{
			for (int cur_x = 0; cur_x < xrange; cur_x++)
			{
				for (int cur_y = 0; cur_y < yrange; cur_y++)
				{
					double distance = 0;
					for (int y = 0; y < ysize; y++)
					{
						for (int x = 0; x < xsize; x++)
						{
							int in_x = cur_x + x;
							int in_y = cur_y + y;
							int out_x = output_offx + x;
							int out_y = output_offy + y;
							if (mask[y*xsize + x])
								distance += _distance(nChannels, input + nChannels*(in_y*in_width + in_x), output + nChannels*(out_y*out_width + out_x));
						}
					}
					if (distance < max_distance)
					{
						max_distance = distance;
						input_offx = cur_x;
						input_offy = cur_y;
					}

				}
			}
		}
		else
		{
			for (int pp = 0; pp < total_range*probability; pp++)
			{
				int cur_x = rand() % xrange;
				int cur_y = rand() % yrange;

				double distance = 0;
				for (int y = 0; y < ysize; y++)
				{
					for (int x = 0; x < xsize; x++)
					{
						int in_x = cur_x + x;
						int in_y = cur_y + y;
						int out_x = output_offx + x;
						int out_y = output_offy + y;
						if (mask[y*xsize + x])
							distance += _distance(nChannels, input + nChannels*(in_y*in_width + in_x), output + nChannels*(out_y*out_width + out_x));
					}
				}
				if (distance < max_distance)
				{
					max_distance = distance;
					input_offx = cur_x;
					input_offy = cur_y;
				}
			}
		}

		delete[]mask;

		return true;
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::_splat_to(int xsize, int ysize, int input_offx, int input_offy, int output_offx, int output_offy)
	{
		if (input_offx + xsize > in_width || input_offy + ysize > in_height || output_offx + xsize > out_width || output_offy + ysize > out_height)
			return false;

		bool* outer_mask = new bool[xsize*ysize];
		bool* inner_mask = new bool[xsize*ysize];
		memset(outer_mask, 0, sizeof(bool)*xsize*ysize);
		memset(inner_mask, 0, sizeof(bool)*xsize*ysize);

		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (y == 0 || y == ysize - 1 || x == 0 || x == xsize - 1)
				{
					if (!is_empty[(output_offy + y)*out_width + (output_offx + x)])
						outer_mask[y*xsize + x] = true;
				}
			}
		}

		// at least one node for inner
		bool has_one_for_inner = false;
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				if (is_empty[(output_offy + y)*out_width + (output_offx + x)])
				{
					inner_mask[y*xsize + x] = true;
					has_one_for_inner = true;
				}
			}
		}
		// in fact not needed
		/*if(!has_one_for_inner)
		{
		inner_mask[ysize/2*xsize+xsize/2] = true;
		}*/

		// graph cut
		typedef Graph<double, double, double> GraphType;

		GraphType* g = new GraphType(2 * xsize*ysize, 8 * xsize*ysize);
		g->add_node(xsize*ysize);
		for (int nn = 0; nn < xsize*ysize; nn++)
		{
			if (outer_mask[nn])
			{
				g->add_tweights(nn, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val, 0);
			}
		}
		for (int nn = 0; nn < xsize*ysize; nn++)
		{
			if (inner_mask[nn])
			{
				g->add_tweights(nn, 0, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val);
			}
		}

		T* dist = new T[xsize*ysize];
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				int in_x = x + input_offx;
				int in_y = y + input_offy;

				if (is_empty[out_y*out_width + out_x])
				{
					dist[y*xsize + x] = 0;
				}
				else
				{
					dist[y*xsize + x] = _distance(nChannels, input + nChannels*(in_y*in_width + in_x), output + nChannels*(out_y*out_width + out_x));
				}
			}
		}


		// left to right edge
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize - 1; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				int in_x = x + input_offx;
				int in_y = y + input_offy;

				if (is_empty[out_y*out_width + out_x])
				{
					if (is_empty[out_y*out_width + out_x + 1])
						g->add_edge(y*xsize + x, y*xsize + x + 1, 0, 0);
					else
					{
						// do not permit the cut at the boarder: 
						g->add_edge(y*xsize + x, y*xsize + x + 1,/*2*dist[y*xsize+x+1],2*dist[y*xsize+x+1]*/ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val);
					}
				}
				else
				{

					if (is_empty[out_y*out_width + out_x + 1])
					{
						// do not permit the cut at the boarder: 
						g->add_edge(y*xsize + x, y*xsize + x + 1,/*2*dist[y*xsize+x],2*dist[y*xsize+x]*/ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val);
					}
					else
					{
						if (right_weight[out_y*out_width + out_x] > 0)
						{
							int cur_node_id = g->add_node();
							g->add_tweights(cur_node_id, 0, right_weight[out_y*out_width + out_x]);

							int xpos_ = xpos_in_input[out_y*out_width + out_x];
							int ypos_ = ypos_in_input[out_y*out_width + out_x];
							float cur_dis = dist[y*xsize + x] + _distance(nChannels, input + nChannels*(ypos_*in_width + xpos_ + 1), input + nChannels*(in_y*in_width + in_x + 1));
							g->add_edge(y*xsize + x, cur_node_id, cur_dis, cur_dis);

							xpos_ = xpos_in_input[out_y*out_width + out_x + 1];
							ypos_ = ypos_in_input[out_y*out_width + out_x + 1];
							cur_dis = dist[y*xsize + x + 1] + _distance(nChannels, input + nChannels*(ypos_*in_width + xpos_ - 1), input + nChannels*(in_y*in_width + in_x));
							g->add_edge(y*xsize + x + 1, cur_node_id, cur_dis, cur_dis);
						}
						else
						{
							g->add_edge(y*xsize + x, y*xsize + x + 1, dist[y*xsize + x] + dist[y*xsize + x + 1], dist[y*xsize + x] + dist[y*xsize + x + 1]);
						}
					}
				}
			}
		}

		//up to down edge
		for (int y = 0; y < ysize - 1; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				int in_x = x + input_offx;
				int in_y = y + input_offy;

				if (is_empty[out_y*out_width + out_x])
				{
					if (is_empty[(out_y + 1)*out_width + out_x])
						g->add_edge(y*xsize + x, (y + 1)*xsize + x, 0, 0);
					else
					{
						// do not permit the cut at the boarder: 
						g->add_edge(y*xsize + x, (y + 1)*xsize + x,/*2*dist[(y+1)*xsize+x],2*dist[(y+1)*xsize+x]*/ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val);
					}
				}
				else
				{
					if (is_empty[(out_y + 1)*out_width + out_x])
					{
						// do not permit the cut at the boarder: 
						g->add_edge(y*xsize + x, (y + 1)*xsize + x,/*2*dist[y*xsize+x],2*dist[y*xsize+x]*/ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val, ZQ_GRAPHCUT_FOR_TEXTURE_max_weight_val);
					}
					else
					{
						if (down_weight[out_y*out_width + out_x] > 0)
						{
							int cur_node_id = g->add_node();
							g->add_tweights(cur_node_id, 0, down_weight[out_y*out_width + out_x]);

							int xpos_ = xpos_in_input[out_y*out_width + out_x];
							int ypos_ = ypos_in_input[out_y*out_width + out_x];
							float cur_dis = dist[y*xsize + x] + _distance(nChannels, input + nChannels*((ypos_ + 1)*in_width + xpos_), input + nChannels*((in_y + 1)*in_width + in_x));
							g->add_edge(y*xsize + x, cur_node_id, cur_dis, cur_dis);

							xpos_ = xpos_in_input[(out_y + 1)*out_width + out_x];
							ypos_ = ypos_in_input[(out_y + 1)*out_width + out_x];
							cur_dis = dist[(y + 1)*xsize + x] + _distance(nChannels, input + nChannels*((ypos_ - 1)*in_width + xpos_), input + nChannels*(in_y*in_width + in_x));
							g->add_edge((y + 1)*xsize + x, cur_node_id, cur_dis, cur_dis);
						}
						else
						{
							g->add_edge(y*xsize + x, (y + 1)*xsize + x, dist[y*xsize + x] + dist[(y + 1)*xsize + x], dist[y*xsize + x] + dist[(y + 1)*xsize + x]);
						}

					}
				}
			}
		}

		g->maxflow();

		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				int in_x = x + input_offx;
				int in_y = y + input_offy;

				is_empty[out_y*out_width + out_x] = false;

				if (g->what_segment(y*xsize + x) == GraphType::SOURCE)
					;
				else
				{
					for (int c = 0; c < nChannels; c++)
					{
						output[(out_y*out_width + out_x)*nChannels + c] = input[(in_y*in_width + in_x)*nChannels + c];
					}
					xpos_in_input[out_y*out_width + out_x] = in_x;
					ypos_in_input[out_y*out_width + out_x] = in_y;
				}
			}
		}

		//update old weights
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x <= xsize; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				if (out_x == 0)
					left_weight[out_y*out_width + out_x] = 0;
				else if (out_x == out_width)
					right_weight[out_y*out_width + out_x - 1] = 0;
				else
				{
					int x_pos = xpos_in_input[out_y*out_width + out_x];
					int y_pos = ypos_in_input[out_y*out_width + out_x];

					int x_pos_left = xpos_in_input[out_y*out_width + out_x - 1];
					int y_pos_left = ypos_in_input[out_y*out_width + out_x - 1];
					if (x_pos_left + 1 == x_pos && y_pos_left == y_pos)
					{
						right_weight[out_y*out_width + out_x - 1] = 0;
						left_weight[out_y*out_width + out_x] = 0;
					}
					else
					{
						if (x_pos_left == in_width - 1 && x_pos == 0)
						{
							right_weight[out_y*out_width + out_x - 1] = 0;
							left_weight[out_y*out_width + out_x] = 0;
						}
						else if (x_pos == 0)
						{
							right_weight[out_y*out_width + out_x - 1] = left_weight[out_y*out_width + out_x]
								= _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos), input + nChannels*(y_pos_left*in_width + x_pos_left + 1));
						}
						else if (x_pos_left == in_width - 1)
						{
							right_weight[out_y*out_width + out_x - 1] = left_weight[out_y*out_width + out_x]
								= _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos - 1), input + nChannels*(y_pos_left*in_width + x_pos_left));
						}
						else
						{
							float cur_dist = _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos), input + nChannels*(y_pos_left*in_width + x_pos_left + 1))
								+ _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos - 1), input + nChannels*(y_pos_left*in_width + x_pos_left));
							right_weight[out_y*out_width + out_x - 1] = cur_dist;
							left_weight[out_y*out_width + out_x] = cur_dist;
						}
					}
				}
			}
		}

		for (int y = 0; y <= ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				int out_x = x + output_offx;
				int out_y = y + output_offy;
				if (out_y == 0)
					up_weight[out_y*out_width + out_x] = 0;
				else if (out_y == out_height)
					down_weight[(out_y - 1)*out_width + out_x] = 0;
				else
				{
					int x_pos = xpos_in_input[out_y*out_width + out_x];
					int y_pos = ypos_in_input[out_y*out_width + out_x];

					int x_pos_up = xpos_in_input[(out_y - 1)*out_width + out_x];
					int y_pos_up = ypos_in_input[(out_y - 1)*out_width + out_x];
					if (x_pos_up == x_pos && y_pos_up + 1 == y_pos)
					{
						down_weight[(out_y - 1)*out_width + out_x] = 0;
						up_weight[out_y*out_width + out_x] = 0;
					}
					else
					{
						if (y_pos_up == in_height - 1 && y_pos == 0)
						{
							down_weight[(out_y - 1)*out_width + out_x] = 0;
							up_weight[out_y*out_width + out_x] = 0;
						}
						else if (y_pos == 0)
						{
							down_weight[(out_y - 1)*out_width + out_x] = up_weight[out_y*out_width + out_x]
								= _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos), input + nChannels*((y_pos_up + 1)*in_width + x_pos_up));
						}
						else if (y_pos_up == in_height - 1)
						{
							down_weight[(out_y - 1)*out_width + out_x] = up_weight[out_y*out_width + out_x]
								= _distance(nChannels, input + nChannels*((y_pos - 1)*in_width + x_pos), input + nChannels*(y_pos_up*in_width + x_pos_up));
						}
						else
						{
							float cur_dist = _distance(nChannels, input + nChannels*(y_pos*in_width + x_pos), input + nChannels*((y_pos_up + 1)*in_width + x_pos_up))
								+ _distance(nChannels, input + nChannels*((y_pos - 1)*in_width + x_pos), input + nChannels*(y_pos_up*in_width + x_pos_up));
							down_weight[(out_y - 1)*out_width + out_x] = cur_dist;
							up_weight[out_y*out_width + out_x] = cur_dist;
						}
					}
				}
			}
		}

		delete g;
		delete[]dist;
		delete[]inner_mask;
		delete[]outer_mask;

		return true;
	}

	template<class T>
	int ZQ_GraphCutForTexture<T>::_optimizeOnce(int patch_size)
	{
		int xsize = patch_size;
		int ysize = patch_size;
		int input_offx, input_offy;
		int* old_xpos = new int[out_width*out_height];
		int* old_ypos = new int[out_width*out_height];
		memcpy(old_xpos, xpos_in_input, sizeof(int)*out_width*out_height);
		memcpy(old_ypos, ypos_in_input, sizeof(int)*out_width*out_height);

		int n_worst = ZQ_GRAPHCUT_FOR_TEXTURE_N_WORST;
		int output_offx[ZQ_GRAPHCUT_FOR_TEXTURE_N_WORST], output_offy[ZQ_GRAPHCUT_FOR_TEXTURE_N_WORST];
		bool check_convergence = true;

		_find_worst_area(xsize, ysize, n_worst, output_offx, output_offy);
		for (int nn = 0; nn < n_worst; nn++)
		{
			_select_patch(xsize, ysize, output_offx[nn], output_offy[nn], input_offx, input_offy);
			_splat_to(xsize, ysize, input_offx, input_offy, output_offx[nn], output_offy[nn]);

			for (int i = 0; i < out_width*out_height; i++)
			{
				if (xpos_in_input[i] != old_xpos[i] || ypos_in_input[i] != old_ypos[i])
				{
					check_convergence = false;
					break;
				}
			}
			if (!check_convergence)
				break;
		}

		delete[]old_xpos;
		delete[]old_ypos;
		if (check_convergence)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	template<class T>
	bool ZQ_GraphCutForTexture<T>::_find_worst_area(int xsize, int ysize, int n_worst, int* output_offx, int* output_offy)
	{
		if (xsize > out_width || ysize > out_height)
			return false;

		int xrange = out_width - xsize;
		int yrange = out_height - ysize;

		int xskip = 1;
		int yskip = 1;

		T* gaussian_weights = new T[xsize*ysize];
		float radius = (xsize + ysize) / 2.0;
		for (int y = 0; y < ysize; y++)
		{
			for (int x = 0; x < xsize; x++)
			{
				gaussian_weights[y*xsize + x] = exp(-0.5*((x - xsize)*(x - xsize) + (y - ysize)*(y - ysize)) / (radius*radius));
			}
		}

		double* max_error = new double[n_worst];
		for (int nn = 0; nn < n_worst; nn++)
		{
			max_error[nn] = -1;
			output_offx[nn] = 0;
			output_offy[nn] = 0;
		}

		if (probability >= 1)
		{
			for (int cur_y = 0; cur_y < yrange; cur_y += xskip)
			{
				for (int cur_x = 0; cur_x < xrange; cur_x += yskip)
				{
					double sum_dis = 0;
					for (int y = 0; y < ysize; y++)
					{
						for (int x = 0; x < xsize; x++)
						{
							sum_dis += gaussian_weights[y*xsize + x] * (right_weight[(cur_y + y)*out_width + (cur_x + x)] + down_weight[(cur_y + y)*out_width + (cur_x + x)]);
						}
					}
					if (sum_dis > max_error[n_worst - 1])
					{
						max_error[n_worst - 1] = sum_dis;
						output_offx[n_worst - 1] = cur_x;
						output_offy[n_worst - 1] = cur_y;

						if (n_worst >= 2)
						{
							int pos = n_worst - 2;
							for (; pos >= 0 && max_error[pos] < sum_dis; pos--);
							if (pos < 0)
							{
								for (int nn = n_worst - 1; nn > 0; nn--)
								{
									max_error[nn] = max_error[nn - 1];
									output_offx[nn] = output_offx[nn - 1];
									output_offy[nn] = output_offy[nn - 1];
								}
								max_error[0] = sum_dis;
								output_offx[0] = cur_x;
								output_offy[0] = cur_y;
							}
							else
							{
								for (int nn = n_worst - 1; nn > pos + 1; nn--)
								{
									max_error[nn] = max_error[nn - 1];
									output_offx[nn] = output_offx[nn - 1];
									output_offy[nn] = output_offy[nn - 1];
								}
								max_error[pos + 1] = sum_dis;
								output_offx[pos + 1] = cur_x;
								output_offy[pos + 1] = cur_y;
							}
						}
					}
				}
			}
		}
		else
		{
			for (int pp = 0; pp < xrange*yrange*probability; pp++)
			{
				int cur_x = rand() % xrange;
				int cur_y = rand() % yrange;
				double sum_dis = 0;
				for (int y = 0; y < ysize; y++)
				{
					for (int x = 0; x < xsize; x++)
					{
						sum_dis += gaussian_weights[y*xsize + x] * (right_weight[(cur_y + y)*out_width + (cur_x + x)] + down_weight[(cur_y + y)*out_width + (cur_x + x)]);
					}
				}
				if (sum_dis > max_error[n_worst - 1])
				{
					max_error[n_worst - 1] = sum_dis;
					output_offx[n_worst - 1] = cur_x;
					output_offy[n_worst - 1] = cur_y;

					if (n_worst >= 2)
					{
						int pos = n_worst - 2;
						for (; pos >= 0 && max_error[pos] < sum_dis; pos--);
						if (pos < 0)
						{
							for (int nn = n_worst - 1; nn > 0; nn--)
							{
								max_error[nn] = max_error[nn - 1];
								output_offx[nn] = output_offx[nn - 1];
								output_offy[nn] = output_offy[nn - 1];
							}
							max_error[0] = sum_dis;
							output_offx[0] = cur_x;
							output_offy[0] = cur_y;
						}
						else
						{
							for (int nn = n_worst - 1; nn > pos + 1; nn--)
							{
								max_error[nn] = max_error[nn - 1];
								output_offx[nn] = output_offx[nn - 1];
								output_offy[nn] = output_offy[nn - 1];
							}
							max_error[pos + 1] = sum_dis;
							output_offx[pos + 1] = cur_x;
							output_offy[pos + 1] = cur_y;
						}
					}
				}
			}
		}

		delete[]gaussian_weights;
		delete[]max_error;

		return true;
	}
}


#endif