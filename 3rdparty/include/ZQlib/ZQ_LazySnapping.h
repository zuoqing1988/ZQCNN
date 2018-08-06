#ifndef _ZQ_LAZY_SNAPPING_H_
#define _ZQ_LAZY_SNAPPING_H_
#pragma once

#include "ZQ_LazySnappingOptions.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_Vec2D.h"
#include "ZQ_GraphCut.h"
#include <vector>
#include "ZQ_Kmeans.h"
#include "ZQ_BinaryImageProcessing.h"

namespace ZQ
{
	template<class T, const int MAX_CLUSTER_NUM = 32>
	class ZQ_LazySnapping
	{
		typedef Graph<T, T, T> GraphType;
	public:
		static bool LazySnapping(const ZQ_DImage<T>& image, const int* back_pts, int back_num, const int* fore_pts, int fore_num, ZQ_DImage<bool>& fore_mask, const ZQ_LazySnappingOptions& opt);

		static bool LazySnapping(const T* image, int width, int height, int nChannels, const int* back_pts, int back_num, const int* fore_pts, int fore_num, bool* fore_mask, const ZQ_LazySnappingOptions& opt);

		static bool FilterMask(const bool* mask, bool* out_mask, int width, int height, int area_size_thresh, int dilate_erode_size);

	private:
		static T _colorDistance(const T* color1, const T* color2, int nChannels);

		static bool _getClusterColors(const T* image, int width, int height, int nChannels, int ifore_num, const int* ifore_pts, int iback_num, const int* iback_pts,
			int& ifore_k, int*& ifore_k_id_num, T ifore_colors[], int& iback_k, int*& iback_k_id_num, T iback_colors[]);

		static void _getE1(const T* cur_val, int ifore_k, const int* ifore_k_id_num, const T* ifore_colors, int iback_k, const int* iback_k_id_num, const T* iback_colors, int nChannels, T e1[2]);

		static T _getE2(const T* color1, const T* color2, float lambda, float color_scale);

		static bool _isInVector(int x, int y, const std::vector<int>& pts);

	public:
		ZQ_LazySnapping(int width, int height)
		{
			graph = new GraphType(width*height, width*height * 4);
			this->width = width;
			this->height = height;
			has_image = false;
			FLOW_MAX_VALUE = 0;
			fore_mask.allocate(width, height, 1);
			tweights_source.allocate(width, height, 1);
			tweights_sink.allocate(width, height, 1);
			enabled_E3 = false;
			tweights_E3_source.allocate(width, height, 1);
			tweights_E3_source.reset();
			tweights_E3_sink.allocate(width, height, 1);
			tweights_E3_sink.reset();
			lambda_for_E3 = 0;
			sigma_for_E3 = 1;
			cur_fore_k = 0;
			cur_back_k = 0;
		}
		~ZQ_LazySnapping()
		{
			if (graph)
			{
				delete graph;
			}
		}
	private:
		int width, height;
		GraphType* graph;
		bool has_image;
		ZQ_DImage<T> image;
		T FLOW_MAX_VALUE;
		std::vector<int> ifore_pts;
		std::vector<int> iback_pts;
		ZQ_DImage<bool> fore_mask;
		ZQ_DImage<T> tweights_source;
		ZQ_DImage<T> tweights_sink;
		bool enabled_E3;
		ZQ_DImage<T> tweights_E3_source;
		ZQ_DImage<T> tweights_E3_sink;
		float lambda_for_E3;
		float sigma_for_E3;
		int cur_fore_k;
		int cur_back_k;
		T fore_colors[MAX_CLUSTER_NUM * 3];
		T back_colors[MAX_CLUSTER_NUM * 3];
	public:
		bool SetEnableE3(bool b)
		{
			bool old_b = enabled_E3;
			enabled_E3 = b;
			if (has_image)
			{
				if (!_update())
				{
					enabled_E3 = old_b;
					return false;
				}
			}
			
			return true;
		}

		bool GetEnabledE3() const { return enabled_E3; }

		const bool*& GetForegroundMaskPtr() const { return fore_mask.data(); }
		bool SetImage(const ZQ_DImage<T>& image, float lambda_for_E2, float color_scale_for_E2, float lambda_for_E3, float sigma_for_E3)
		{
			if (image.width() != width || image.height() != height || image.nchannels() == 0)
				return false;
			this->image = image;
			has_image = true;
			ifore_pts.clear();
			iback_pts.clear();
			int nChannels = image.nchannels();
			const T*& image_data = image.data();
			graph->reset();
			graph->add_node(width*height);

			T max_edge_val = 0;
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					const T* cur_val = image_data + offset*nChannels;
					if (w > 0 && h > 0)
					{
						const T* val2 = image_data + (offset - 1)*nChannels;
						T e2 = _getE2(cur_val, val2, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset, offset - 1, e2, e2);
						const T* val3 = image_data + (offset - width)*nChannels;
						e2 = _getE2(cur_val, val3, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset, offset - width, e2, e2);
						const T* val4 = image_data + (offset - width - 1)*nChannels;
						e2 = _getE2(cur_val, val4, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset, offset - width - 1, e2, e2);
						e2 = _getE2(val2, val3, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset - 1, offset - width, e2, e2);
					}
					else if (w > 0)
					{
						const T* val2 = image_data + (offset - 1)*nChannels;
						T e2 = _getE2(cur_val, val2, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset, offset - 1, e2, e2);
					}
					else if (h > 0)
					{
						const T* val3 = image_data + (offset - width)*nChannels;
						T e2 = _getE2(cur_val, val3, lambda_for_E2, color_scale_for_E2);
						max_edge_val = __max(e2, max_edge_val);
						graph->add_edge(offset, offset - width, e2, e2);
					}
				}
			}
			FLOW_MAX_VALUE = max_edge_val * 80 + 1;
			this->lambda_for_E3 = lambda_for_E3;
			this->sigma_for_E3 = sigma_for_E3;

			////
			T*& tweights_source_data = tweights_source.data();
			T*& tweights_sink_data = tweights_sink.data();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					const T* cur_val = image_data + offset*nChannels;
					T e1[2] = { 0, FLOW_MAX_VALUE };
					graph->add_tweights(offset, e1[0], e1[1]);
					tweights_source_data[offset] = e1[0];
					tweights_sink_data[offset] = e1[1];
				}
			}
			graph->maxflow();
			bool*& fore_mask_data = fore_mask.data();
			for (int i = 0; i < height*width; i++)
			{
				fore_mask_data[i] = graph->what_segment(i) == GraphType::SINK;
			}
			return true;
		}

		bool FirstSnapping(int fore_num, const int* fore_pts, int back_num, const int* back_pts)
		{
			if (!has_image)
				return false;
		
			ifore_pts.clear();
			iback_pts.clear();
			for (int i = 0; i < fore_num; i++)
			{
				int cur_x = fore_pts[i * 2];
				int cur_y = fore_pts[i * 2 + 1];
				if (cur_x >= 0 && cur_x < width && cur_y >= 0 && cur_y < height && !_isInVector(cur_x, cur_y, ifore_pts))
				{
					ifore_pts.push_back(cur_x);
					ifore_pts.push_back(cur_y);
				}
			}

			for (int i = 0; i < back_num; i++)
			{
				int cur_x = back_pts[i * 2];
				int cur_y = back_pts[i * 2 + 1];
				if (cur_x >= 0 && cur_x < width && cur_y >= 0 && cur_y < height && !_isInVector(cur_x, cur_y, iback_pts))
				{
					iback_pts.push_back(cur_x);
					iback_pts.push_back(cur_y);
				}
			}

			int ifore_num = ifore_pts.size() / 2;
			int iback_num = iback_pts.size() / 2;
			if (ifore_num == 0 || iback_num == 0)
				return false;

			/************/
			int nChannels = image.nchannels();
			T*& image_data = image.data();
			int* ifore_k_id_num = 0;
			int* iback_k_id_num = 0;
			if (!_getClusterColors(image_data, width, height, nChannels, ifore_num, ifore_num == 0 ? 0 : &ifore_pts[0], iback_num, iback_num == 0 ? 0 : &iback_pts[0], 
				cur_fore_k, ifore_k_id_num, fore_colors, cur_back_k, iback_k_id_num, back_colors))
			{
				return false;
			}

			/*******************/

			bool* fore_flag = new bool[width*height];
			bool* back_flag = new bool[width*height];
			memset(fore_flag, 0, sizeof(bool)*width*height);
			memset(back_flag, 0, sizeof(bool)*width*height);
			for (int i = 0; i < ifore_num; i++)
			{
				int x = ifore_pts[i * 2];
				int y = ifore_pts[i * 2 + 1];
				fore_flag[y*width + x] = true;
			}
			for (int i = 0; i < iback_num; i++)
			{
				int x = iback_pts[i * 2];
				int y = iback_pts[i * 2 + 1];
				back_flag[y*width + x] = true;
			}

			////////////
			if (enabled_E3)
			{
				_computeE3();
			}
			_recomputeFlow(cur_fore_k, ifore_k_id_num, fore_colors, cur_back_k, iback_k_id_num, back_colors);
			delete[]ifore_k_id_num;
			delete[]iback_k_id_num;
			return true;
		}

		bool EditSnappingAddForeground(int fore_num, const int* fore_pts)
		{
			std::vector<int> old_ifore_pts(ifore_pts);
			bool has_changed = false;
			for (int i = 0; i < fore_num; i++)
			{
				int x = fore_pts[i * 2];
				int y = fore_pts[i * 2 + 1];
				if (x >= 0 && x < width && y >= 0 && y < height && !_isInVector(x, y, ifore_pts))
				{
					ifore_pts.push_back(x);
					ifore_pts.push_back(y);
					has_changed = true;
				}
			}

			if (!has_changed)
				return true;

			if (!_update())
			{
				ifore_pts = old_ifore_pts;
				return false;
			}
			return true;
		}

		bool EditSnappingAddBackground(int back_num, const int* back_pts)
		{
			std::vector<int> old_iback_pts(iback_pts);
			bool has_changed = false;
			for (int i = 0; i < back_num; i++)
			{
				int x = back_pts[i * 2];
				int y = back_pts[i * 2 + 1];
				if (x >= 0 && x < width && y >= 0 && y < height && !_isInVector(x, y, iback_pts))
				{
					iback_pts.push_back(x);
					iback_pts.push_back(y);
					has_changed = true;
				}
			}

			if (!has_changed)
				return true;

			if (!_update())
			{
				iback_pts = old_iback_pts;
				return false;
			}
			return true;
		}

		bool EditSnappingEraseForeground(int fore_num, const int* fore_pts)
		{
			std::vector<int> old_ifore_pts(ifore_pts);
			bool has_changed = false;
			ifore_pts.clear();
			for (int i = 0; i < old_ifore_pts.size() / 2; i++)
			{
				int x = old_ifore_pts[i * 2];
				int y = old_ifore_pts[i * 2 + 1];
				bool erase_flag = false;
				for (int j = 0; j < fore_num; j++)
				{
					if (x == fore_pts[j * 2] && y == fore_pts[j * 2 + 1])
					{
						erase_flag = true;
						has_changed = true;
					}
				}
				if (!erase_flag)
				{
					ifore_pts.push_back(x);
					ifore_pts.push_back(y);
				}
			}
			

			if (!has_changed)
				return true;

			if (!_update())
			{
				ifore_pts = old_ifore_pts;
				return false;
			}
			return true;
		}

		bool EditSnappingEraseBackground(int back_num, const int* back_pts)
		{
			std::vector<int> old_iback_pts(iback_pts);
			bool has_changed = false;
			iback_pts.clear();
			for (int i = 0; i < old_iback_pts.size() / 2; i++)
			{
				int x = old_iback_pts[i * 2];
				int y = old_iback_pts[i * 2 + 1];
				bool erase_flag = false;
				for (int j = 0; j < back_num; j++)
				{
					if (x == back_pts[j * 2] && y == back_pts[j * 2 + 1])
					{
						erase_flag = true;
						has_changed = true;
					}
				}
				if (!erase_flag)
				{
					iback_pts.push_back(x);
					iback_pts.push_back(y);
				}
			}


			if (!has_changed)
				return true;

			if (!_update())
			{
				iback_pts = old_iback_pts;
				return false;
			}
			return true;
		}

	private:

		bool _update()
		{
			int ifore_num = ifore_pts.size() / 2;
			int iback_num = iback_pts.size() / 2;

			int nChannels = image.nchannels();
			T*& image_data = image.data();
			int* ifore_k_id_num = 0;
			int* iback_k_id_num = 0;
			int ifore_k, iback_k;
			if (!_getClusterColors(image_data, width, height, nChannels, ifore_num, ifore_num == 0 ? 0 : &ifore_pts[0], iback_num, iback_num == 0 ? 0 : &iback_pts[0],
				ifore_k, ifore_k_id_num, fore_colors, iback_k, iback_k_id_num, back_colors))
			{
				return false;
			}

			/*******************/
			if (enabled_E3)
			{
				_computeE3();
			}
			_recomputeFlow(ifore_k, ifore_k_id_num, fore_colors, iback_k, iback_k_id_num, back_colors);

			if (ifore_k_id_num) delete[]ifore_k_id_num;
			if (iback_k_id_num) delete[]iback_k_id_num;
			return true;
		}

		void _recomputeFlow(int ifore_k, const int* ifore_k_id_num, const T* ifore_colors, int iback_k, const int* iback_k_id_num, const T* iback_colors)
		{
			int ifore_num = ifore_pts.size() / 2;
			int iback_num = iback_pts.size() / 2;
			T* image_data = image.data();
			int nChannels = image.nchannels();

			bool* fore_flag = new bool[width*height];
			bool* back_flag = new bool[width*height];
			memset(fore_flag, 0, sizeof(bool)*width*height);
			memset(back_flag, 0, sizeof(bool)*width*height);
			for (int i = 0; i < ifore_num; i++)
			{
				int x = ifore_pts[i * 2];
				int y = ifore_pts[i * 2 + 1];
				fore_flag[y*width + x] = true;
			}
			for (int i = 0; i < iback_num; i++)
			{
				int x = iback_pts[i * 2];
				int y = iback_pts[i * 2 + 1];
				back_flag[y*width + x] = true;
			}

			////////////
			clock_t t1 = clock();
			T*& tweights_source_data = tweights_source.data();
			T*& tweights_sink_data = tweights_sink.data();
			T*& E3_source_data = tweights_E3_source.data();
			T*& E3_sink_data = tweights_E3_sink.data();
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					int offset = h*width + w;
					const T* cur_val = image_data + offset*nChannels;
					T e1[2] = { 0, 0 };

					if (fore_flag[offset])
					{
						e1[0] = 0;
						e1[1] = FLOW_MAX_VALUE;
					}
					else if (back_flag[offset])
					{
						e1[0] = FLOW_MAX_VALUE;
						e1[1] = 0;
					}
					else
					{
						_getE1(cur_val, ifore_k, ifore_k_id_num, ifore_colors, iback_k, iback_k_id_num, iback_colors, nChannels, e1);
					}

					T cur_source, cur_sink;
					if (enabled_E3)
					{
						cur_source = e1[0] + E3_source_data[offset];
						cur_sink = e1[1] + E3_sink_data[offset];
					}
					else
					{
						cur_source = e1[0];
						cur_sink = e1[1];
					}
					T add_weights_source = cur_source - tweights_source_data[offset];
					T add_weights_sink = cur_sink - tweights_sink_data[offset];
					graph->add_tweights(offset, add_weights_source, add_weights_sink);
					tweights_source_data[offset] = cur_source;
					tweights_sink_data[offset] = cur_sink;
					
					graph->mark_node(offset);
				}
			}

			clock_t t2 = clock();
			graph->maxflow();
			clock_t t3 = clock();
			

			bool*& fore_mask_data = fore_mask.data();
			for (int i = 0; i < height*width; i++)
			{
				fore_mask_data[i] = graph->what_segment(i) == GraphType::SINK;
			}

			clock_t t4 = clock();

			printf("maxflow/recompute cost: %.3f/%.3f sec\n", 0.001*(t3 - t2), 0.001*(t4-t1));
		}

		void _computeE3()
		{
			if (ifore_pts.size() == 0 || iback_pts.size() == 0)
				return;
			int ifore_num = ifore_pts.size() / 2;
			int iback_num = iback_pts.size() / 2;
			ZQ_DImage<int> fore_distance(width, height, 1);
			ZQ_DImage<int> back_distance(width, height, 1);
			int*& fore_distance_data = fore_distance.data();
			int*& back_distance_data = back_distance.data();

			ZQ_DImage<bool> landmark_map(width, height, 1);
			landmark_map.reset();
			bool*& fore_map_data = landmark_map.data();
			for (int i = 0; i < ifore_num; i++)
			{
				int x = ifore_pts[i * 2];
				int y = ifore_pts[i * 2 + 1];
				fore_map_data[y*width + x] = true;
			}

			ZQ_BinaryImageProcessing::ComputeDistance(fore_map_data, width, height, fore_distance_data, 8);
			landmark_map.reset();
			bool*& back_map_data = landmark_map.data();
			for (int i = 0; i < iback_num; i++)
			{
				int x = iback_pts[i * 2];
				int y = iback_pts[i * 2 + 1];
				back_map_data[y*width + x] = true;
			}
			ZQ_BinaryImageProcessing::ComputeDistance(back_map_data, width, height, back_distance_data, 8);

			float E3_cigma = sigma_for_E3;
			float E3_coeff = lambda_for_E3;

			T*& E3_source_data = tweights_E3_source.data();
			T*& E3_sink_data = tweights_E3_sink.data();
			for (int i = 0; i < height*width; i++)
			{
				float distance_to_fore = fore_distance_data[i];
				float distance_to_back = back_distance_data[i];
				float ratio = distance_to_fore / (distance_to_fore + distance_to_back + 1e-6);
				float df = exp(E3_cigma*(-ratio));
				float db = exp(E3_cigma*(ratio-1));
				df /= (df + db);
				db = 1 - df;
				E3_sink_data[i] = E3_coeff * df;
				E3_source_data[i] = E3_coeff * db;
			}
		}

	};

	/*******************************************************************/
	template<class T, const int MAX_CLUSTER_NUM>
	bool ZQ_LazySnapping<T, MAX_CLUSTER_NUM>::LazySnapping(const ZQ_DImage<T>& image, const int* back_pts, int back_num, const int* fore_pts, int fore_num, ZQ_DImage<bool>& fore_mask, const ZQ_LazySnappingOptions& opt)
	{
		int width = image.width();
		int height = image.height();
		int nChannels = image.nchannels();
		if (!fore_mask.matchDimension(width, height, 1))
			fore_mask.allocate(width, height);
		return LazySnapping(image.data(), width, height, nChannels, fore_pts, fore_num, back_pts, back_num, fore_mask.data(), opt);
	}

	template<class T, const int MAX_CLUSTER_NUM>
	bool ZQ_LazySnapping<T,MAX_CLUSTER_NUM>::LazySnapping(const T* image, int width, int height, int nChannels, const int* back_pts, int back_num, const int* fore_pts, int fore_num, bool* fore_mask, const ZQ_LazySnappingOptions& opt)
	{
		clock_t t1 = clock();
		std::vector<int> ifore_pts;
		std::vector<int> iback_pts;

		for (int i = 0; i < fore_num; i++)
		{
			int cur_x = fore_pts[i * 2];
			int cur_y = fore_pts[i * 2 + 1];
			if (cur_x >= 0 && cur_x < width && cur_y >= 0 && cur_y < height && !_isInVector(cur_x, cur_y, ifore_pts))
			{
				ifore_pts.push_back(cur_x);
				ifore_pts.push_back(cur_y);
			}
		}

		for (int i = 0; i < back_num; i++)
		{
			int cur_x = back_pts[i * 2];
			int cur_y = back_pts[i * 2 + 1];
			if (cur_x >= 0 && cur_x < width && cur_y >= 0 && cur_y < height && !_isInVector(cur_x, cur_y, iback_pts))
			{
				iback_pts.push_back(cur_x);
				iback_pts.push_back(cur_y);
			}
		}

		int ifore_num = ifore_pts.size() / 2;
		int iback_num = iback_pts.size() / 2;
		if (ifore_num == 0 || iback_num == 0)
			return false;

		/************/
		int ifore_k = 0;
		int iback_k = 0;
		int* ifore_k_id_num = 0;
		int* iback_k_id_num = 0;
		T* ifore_colors = new T[MAX_CLUSTER_NUM * 3];
		T* iback_colors = new T[MAX_CLUSTER_NUM * 3];
		if (!_getClusterColors(image, width, height, nChannels, ifore_num, &ifore_pts[0], iback_num, &iback_pts[0], ifore_k, ifore_k_id_num, ifore_colors, iback_k, iback_k_id_num, iback_colors))
		{
			return false;
		}
		
		clock_t t2 = clock();
		/*******************/

		bool* fore_flag = new bool[width*height];
		bool* back_flag = new bool[width*height];
		memset(fore_flag, 0, sizeof(bool)*width*height);
		memset(back_flag, 0, sizeof(bool)*width*height);
		for (int i = 0; i < ifore_num; i++)
		{
			int x = ifore_pts[i * 2];
			int y = ifore_pts[i * 2 + 1];
			fore_flag[y*width + x] = true;
		}
		for (int i = 0; i < iback_num; i++)
		{
			int x = iback_pts[i * 2];
			int y = iback_pts[i * 2 + 1];
			back_flag[y*width + x] = true;
		}


		const double FLOW_MAX_VALUE = 1e10;
		GraphType graph(width*height,width*height*4);
		graph.add_node(width*height);
		float lambda = opt.lambda_for_E2;
		float color_scale = opt.color_scale_for_E2;
		for (int h = 0; h < height; h++)
		{
			for (int w = 0; w < width; w++)
			{
				int offset = h*width + w;
				const T* cur_val = image + offset*nChannels;
				T e1[2] = { 0, 0 };
				if (fore_flag[offset])
				{
					e1[0] = 0;
					e1[1] = FLOW_MAX_VALUE;
				}
				else if (back_flag[offset])
				{
					e1[0] = FLOW_MAX_VALUE;
					e1[1] = 0;
				}
				else
				{
					_getE1(cur_val, ifore_k, ifore_k_id_num, ifore_colors, iback_k, iback_k_id_num, iback_colors, nChannels, e1);
				}
				graph.add_tweights(offset, e1[0], e1[1]);

				if (w > 0 && h > 0)
				{
					const T* val2 = image + (offset - 1)*nChannels;
					double e2 = _getE2(cur_val, val2, lambda, color_scale);
					graph.add_edge(offset, offset - 1, e2, e2);
					const T* val3 = image + (offset - width)*nChannels;
					e2 = _getE2(cur_val, val3, lambda, color_scale);
					graph.add_edge(offset, offset - width, e2, e2);
					const T* val4 = image + (offset - width - 1)*nChannels;
					e2 = _getE2(cur_val, val4, lambda, color_scale);
					graph.add_edge(offset, offset - width - 1, e2, e2);
					e2 = _getE2(val2, val3, lambda, color_scale);
					graph.add_edge(offset - 1, offset - width, e2, e2);
				}
				else if (w > 0)
				{
					const T* val2 = image + (offset - 1)*nChannels;
					T e2 = _getE2(cur_val, val2, lambda, color_scale);
					graph.add_edge(offset, offset - 1, e2, e2);
				}
				else if (h > 0)
				{
					const T* val3 = image + (offset - width)*nChannels;
					T e2 = _getE2(cur_val, val3, lambda, color_scale);
					graph.add_edge(offset, offset - width, e2, e2);
				}
			}
		}

		delete[]fore_flag;
		delete[]back_flag;
		delete[]ifore_colors;
		delete[]iback_colors;
		delete[]ifore_k_id_num;
		delete[]iback_k_id_num;

		clock_t t3 = clock();
		graph.maxflow();
		clock_t t4 = clock();
		for (int i = 0; i < height*width; i++)
		{
			fore_mask[i] = graph.what_segment(i) == GraphType::SINK;
		}

		ZQ_DImage<bool> tmp_mask(width,height,1);
		memcpy(tmp_mask.data(), fore_mask, sizeof(bool)*width*height);
		FilterMask(tmp_mask.data(), fore_mask, width, height, opt.area_thresh, opt.dilate_erode_size);
		
		clock_t t5 = clock();
		printf("prepare = %.3f, buildgraph = %.3f, maxflow = %.3f, after = %.3f sec\n", 0.001*(t2 - t1), 0.001*(t3 - t2), 0.001*(t4 - t3), 0.001*(t5 - t4));
		return true;
	}

	template<class T, const int MAX_CLUSTER_NUM>
	bool ZQ_LazySnapping<T,MAX_CLUSTER_NUM>::FilterMask(const bool* mask, bool* out_mask, int width, int height, int area_size_thresh, int dilate_erode_size)
	{
		memcpy(out_mask, mask, sizeof(bool)*width*height);
		std::vector<int> area_size;
		int* label = new int[width*height];
		if (!ZQ_BinaryImageProcessing::BWlabel(out_mask, width, height, label, area_size, 8))
		{
			delete[]label;
			return false;
		}
		for (int i = 0; i < width*height; i++)
		{
			if (out_mask[i])
			{
				if (area_size[label[i] - 1] < area_size_thresh)
					out_mask[i] = false;
			}
		}
		for (int i = 0; i < width*height; i++)
		{
			out_mask[i] = !out_mask[i];
		}
		if (!ZQ_BinaryImageProcessing::BWlabel(out_mask, width, height, label, area_size, 8))
		{
			delete[]label;
			return false;
		}
		for (int i = 0; i < width*height; i++)
		{
			if (out_mask[i])
			{
				if (area_size[label[i] - 1] < area_size_thresh)
					out_mask[i] = false;
			}
		}
		for (int i = 0; i < width*height; i++)
		{
			out_mask[i] = !out_mask[i];
		}

		delete[]label;

		if (dilate_erode_size > 0)
		{
			int XWIDTH = dilate_erode_size * 2 + 1;
			bool* pfilter2D = new bool[XWIDTH*XWIDTH];
			for (int i = 0; i < XWIDTH*XWIDTH; i++)
				pfilter2D[i] = true;
			bool* tmp_mask = new bool[width*height];
			ZQ_BinaryImageProcessing::Dilate(out_mask, tmp_mask, width, height, pfilter2D, dilate_erode_size, dilate_erode_size);
			ZQ_BinaryImageProcessing::Erode(tmp_mask, out_mask, width, height, pfilter2D, dilate_erode_size, dilate_erode_size);
			ZQ_BinaryImageProcessing::Erode(out_mask, tmp_mask, width, height, pfilter2D, dilate_erode_size, dilate_erode_size);
			ZQ_BinaryImageProcessing::Dilate(tmp_mask, out_mask, width, height, pfilter2D, dilate_erode_size, dilate_erode_size);
			delete[]tmp_mask;
			delete[]pfilter2D;
		}
		return true;
	}

	template<class T, const int MAX_CLUSTER_NUM>
	T ZQ_LazySnapping<T, MAX_CLUSTER_NUM>::_colorDistance(const T* color1, const T* color2, int nChannels)
	{
		double sum = 0;
		for (int i = 0; i < nChannels; i++)
		{
			double diff = color1[i] - color2[i];
			sum += diff*diff;
		}
		return sqrt(sum);
	}

	template<class T, const int MAX_CLUSTER_NUM>
	bool ZQ_LazySnapping<T,MAX_CLUSTER_NUM>::_getClusterColors(const T* image, int width, int height, int nChannels, int ifore_num, const int* ifore_pts, int iback_num, const int* iback_pts,
		int& ifore_k, int*& ifore_k_id_num, T ifore_colors[], int& iback_k, int*& iback_k_id_num, T iback_colors[])
	{
		bool fore_recompute_flag = false;
		bool back_recompute_flag = false;
		int new_ifore_k = __min(ifore_num, MAX_CLUSTER_NUM);
		int new_iback_k = __min(iback_num, MAX_CLUSTER_NUM);
		if (ifore_k != new_ifore_k)
			fore_recompute_flag = true;
		if (iback_k != new_iback_k)
			back_recompute_flag = true;
		ifore_k = new_ifore_k;
		iback_k = new_iback_k;
	
		if (ifore_k_id_num) delete[]ifore_k_id_num; ifore_k_id_num = 0;
		if (iback_k_id_num) delete[]iback_k_id_num; iback_k_id_num = 0;
		
		if (ifore_k > 0)
		{
			T* ifore_vals = new T[ifore_num*nChannels];
			for (int i = 0; i < ifore_num; i++)
			{
				int x = ifore_pts[i * 2];
				int y = ifore_pts[i * 2 + 1];
				int offset = y*width + x;
				memcpy(ifore_vals + i*nChannels, image + offset*nChannels, sizeof(T)*nChannels);
			}
			int* ifore_idx = new int[ifore_num];
			if (fore_recompute_flag)
			{
				if (!ZQ_Kmeans<T>::Kmeans(ifore_num, nChannels, ifore_k, ifore_vals, ifore_idx, ifore_colors))
				{
					delete[]ifore_vals;
					delete[]ifore_idx;
					return false;
				}
			}
			else
			{
				if (!ZQ_Kmeans<T>::Kmeans_with_init(ifore_num, nChannels, ifore_k, ifore_vals, ifore_colors, ifore_idx, ifore_colors))
				{
					delete[]ifore_vals;
					delete[]ifore_idx;
					return false;
				}
			}
			
			delete[]ifore_vals;
			ifore_k_id_num = new int[ifore_k];
			memset(ifore_k_id_num, 0, sizeof(int)*ifore_k);
			for (int i = 0; i < ifore_num; i++)
			{
				ifore_k_id_num[ifore_idx[i]]++;
			}
			delete[]ifore_idx;
		}

		if (iback_k > 0)
		{
			T* iback_vals = new T[iback_num*nChannels];
			for (int i = 0; i < iback_num; i++)
			{
				int x = iback_pts[i * 2];
				int y = iback_pts[i * 2 + 1];
				int offset = y*width + x;
				memcpy(iback_vals + i*nChannels, image + offset*nChannels, sizeof(T)*nChannels);
			}
			int* iback_idx = new int[iback_num];
			if (back_recompute_flag)
			{
				if (!ZQ_Kmeans<T>::Kmeans(iback_num, nChannels, iback_k, iback_vals, iback_idx, iback_colors))
				{
					delete[]iback_vals;
					delete[]iback_idx;
					return false;
				}
			}
			else
			{
				if (!ZQ_Kmeans<T>::Kmeans_with_init(iback_num, nChannels, iback_k, iback_vals, iback_colors, iback_idx, iback_colors))
				{
					delete[]iback_vals;
					delete[]iback_idx;
					return false;
				}
			}
			delete[]iback_vals;
			iback_k_id_num = new int[iback_k];
			memset(iback_k_id_num, 0, sizeof(int)*iback_k);
			for (int i = 0; i < iback_num; i++)
			{
				iback_k_id_num[iback_idx[i]]++;
			}
			delete[]iback_idx;
		}
		
		return true;
	}

	template<class T, const int MAX_CLUSTER_NUM>
	void ZQ_LazySnapping<T,MAX_CLUSTER_NUM>::_getE1(const T* cur_val, int ifore_k, const int* ifore_k_id_num, const T* ifore_colors, int iback_k, const int* iback_k_id_num, const T* iback_colors, int nChannels, T e1[2])
	{
		double df = 1e10;
		double db = 1e10;
		for (int i = 0; i < ifore_k; i++)
		{
			if (ifore_k_id_num[i] > 0)
			{
				T cur_dis = _colorDistance(cur_val, ifore_colors + i * nChannels, nChannels);
				if (df > cur_dis)
					df = cur_dis;
			}
		}
		for (int i = 0; i < iback_k; i++)
		{
			if (iback_k_id_num[i] > 0)
			{
				T cur_dis = _colorDistance(cur_val, iback_colors + i * nChannels, nChannels);
				if (db > cur_dis)
					db = cur_dis;
			}
		}

		e1[0] = df / (db + df);
		e1[1] = db / (db + df);
	}

	template<class T, const int MAX_CLUSTER_NUM>
	T ZQ_LazySnapping<T, MAX_CLUSTER_NUM>::_getE2(const T* color1, const T* color2, float lambda, float color_scale)
	{
		float color_scale2 = color_scale*color_scale;
		return lambda / (1 + color_scale2*((color1[0] - color2[0])*(color1[0] - color2[0]) +
			(color1[1] - color2[1])*(color1[1] - color2[1]) +
			(color1[2] - color2[2])*(color1[2] - color2[2])));
	}

	template<class T, const int MAX_CLUSTER_NUM>
	bool ZQ_LazySnapping<T,MAX_CLUSTER_NUM>::_isInVector(int x, int y, const std::vector<int>& pts)
	{
		for (int i = 0; i < pts.size() / 2; i++)
		{
			if (x == pts[i * 2] && y == pts[i * 2 + 1])
				return true;
		}
		return false;
	}
}

#endif