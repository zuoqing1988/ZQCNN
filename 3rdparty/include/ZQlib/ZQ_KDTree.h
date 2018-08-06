#ifndef _ZQ_KDTREE_H_
#define _ZQ_KDTREE_H_
#pragma once

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

namespace ZQ
{
	
	template<class T>
	class ZQ_KDTree
	{
		template<class T>
		class ZQ_KDTree_Node
		{
		public:
			ZQ_KDTree_Node():is_leaf(false),npts(0),ndim(0),pts_idx(0),pts(0),box_min(0),box_max(0),low_child(0),high_child(0){}
			ZQ_KDTree_Node(int _ndim):is_leaf(false),npts(0),ndim(_ndim),pts_idx(0),pts(0),low_child(0),high_child(0)
			{
				box_min = new T[ndim];
				box_max = new T[ndim];
				memset(box_min,0,sizeof(T)*ndim);
				memset(box_max,0,sizeof(T)*ndim);
			}
			~ZQ_KDTree_Node()
			{
				if(box_min)
					delete []box_min;
				if(box_max)
					delete []box_max;
			}

			bool is_leaf;
			int npts;
			int ndim;
			int* pts_idx;
			T** pts;
			T* box_min;
			T* box_max;
			ZQ_KDTree_Node* low_child;
			ZQ_KDTree_Node* high_child;
		};
	public:
		ZQ_KDTree():pts_raw(0),pts(0),pts_idx(0),tree(0){}
		~ZQ_KDTree(){_clear();}

	private:
		T* pts_raw;
		T** pts;
		int* pts_idx;
	
		ZQ_KDTree_Node<T>* tree;

	private:
		static void _swap(int& x, int& y){int tmp = x; x = y; y = tmp;}
		static T _box_distance_square(const T* pt,	const T* box_min, const T* box_max, int ndim);
		static void _find_min_max(int npts, const int* pts_idx, const T** pts, int d, T& min, T& max);
		static int _find_max_spread_dim(int npts, const int* pts_dix, const T** pts, int ndim);
		static void _median_split(int npts, int* pts_idx, const T** pts, int d, T& cv, int n_low);
		static void _recursive_subdivided(ZQ_KDTree_Node<T>* root, int max_leaf_npts);
		static void _recursive_free(ZQ_KDTree_Node<T>* root);
		static void _update_search_result(int* out_idx, T* out_dis2, int& cur_k, int k, int cur_idx, T cur_dis2, bool& updated);
		static T _distance2(const T* pt1, const T* pt2, int ndim);
		static void _recursive_ann_search(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double eps_plus_1_square);
		static void _recursive_ann_search_with_initial_radius(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double eps_plus_1_square, double radius2);
		static void _recursive_ann_fix_radius_search_count(const ZQ_KDTree_Node<T>* root, const T* pt, int& k, double radius2);
		static void _recursive_ann_fix_radius_search(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double radius2);
		static bool _recursive_check_box(const ZQ_KDTree_Node<T>* root);
		static bool _check_box(const ZQ_KDTree_Node<T>* root);
		void _clear();

	public:
		bool BuildKDTree(const T* data, int npts, int ndim, int max_leaf_npts = 10);
		bool Check() const;
		bool BruteForceSearch(const T* pt, int k, int* out_idx, T* out_dis2) const ;
		bool AnnSearch(const T* pt, int k, int* out_idx, T* out_dis2, double eps = 0.0) const ;
		bool AnnSearchWithInitalRadius(const T* pt, int k, int* out_idx, T* out_dis2, double radius, int& out_k, double eps = 0.0) const;
		bool AnnFixRadiusSearch(const T* pt, double radius, int k, int* out_idx, T* out_dis2) const;
		bool AnnFixRadiusSearchCountReturnNum(const T* pt, double radius, int& k) const;

	};

	/* compute distance from point to box,
	pt: the point
	box_min: box min
	box_max: box max
	ndim: dimensions
	*/
	template<class T>
	T ZQ_KDTree<T>::_box_distance_square(const T* pt,	const T* box_min, const T* box_max, int ndim)
	{
		T dist = 0;
		for(int d = 0;d < ndim;d++)
		{
			if(pt[d] < box_min[d])
			{
				T t = box_min[d] - pt[d];
				dist += t*t;
			}
			else if(pt[d] > box_max[d])
			{
				T t = pt[d] - box_max[d];
				dist += t*t;
			}
		}
		return dist;
	}

	template<class T>
	void ZQ_KDTree<T>::_find_min_max(int npts, const int* pts_idx, const T** pts, int d, T& min, T& max)
	{
		min = pts[pts_idx[0]][d];
		max = pts[pts_idx[0]][d];
		for(int i = 1;i < npts;i++)
		{
			if(min > pts[pts_idx[i]][d])
				min = pts[pts_idx[i]][d];
			if(max < pts[pts_idx[i]][d])
				max = pts[pts_idx[i]][d];
		}
	}

	template<class T>
	int ZQ_KDTree<T>::_find_max_spread_dim(int npts, const int* pts_dix, const T** pts, int ndim)
	{
		T min, max, spread;
		int ret_id = 0;
		_find_min_max(npts,pts_dix,pts,0,min,max);
		spread = max-min;
		for(int i = 1;i < ndim;i++)
		{
			T tmp_min,tmp_max,tmp_spread;
			_find_min_max(npts,pts_dix,pts,i,tmp_min,tmp_max);
			tmp_spread = tmp_max-tmp_min;
			if(spread < tmp_spread)
			{
				spread = tmp_spread;
				ret_id = i;
			}
		}
		return ret_id;
	}


	template<class T>
	void ZQ_KDTree<T>::_median_split(int npts, int* pts_idx, const T** pts, int d, T& cv, int n_low)
	{
		int l = 0;							// left end of current subarray
		int r = npts-1;						// right end of current subarray
		while (l < r) 
		{
			int i = (r+l)/2;		// select middle as pivot
			int k;

			if (pts[pts_idx[i]][d] > pts[pts_idx[r]][d])	// make sure last > pivot
				_swap(pts_idx[i],pts_idx[r]);
			_swap(pts_idx[l],pts_idx[i]);					// move pivot to first position

			T c = pts[pts_idx[l]][d];	// pivot value

			i = l;
			k = r;
			// pivot about c
			for(;;) 
			{						
				while (i < r && pts[pts_idx[i]][d] <= c) i++;
				while (k > l && pts[pts_idx[k]][d] > c) k--;
				if(i < k)
				{
					_swap(pts_idx[i],pts_idx[k]);
					i++;
					k--;
				}
				else
					break;
			}
			_swap(pts_idx[l],pts_idx[k]);	// pivot winds up in location k

			if (k > n_low)	   r = k-1;		// recurse on proper subarray
			else if (k < n_low) l = k+1;
			else break;						// got the median exactly
		}

		if (n_low > 0) // search for next smaller item
		{			
			T c = pts[pts_idx[0]][d];
			int k = 0;						
			for (int i = 1; i < n_low; i++) 
			{
				if (pts[pts_idx[i]][d] > c) 
				{
					c = pts[pts_idx[i]][d];
					k = i;
				}
			}
			_swap(pts_idx[n_low-1],pts_idx[k]);
		}
		// cut value is midpoint value
		cv = (pts[pts_idx[n_low-1]][d] + pts[pts_idx[n_low]][d])/2.0;
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_subdivided(ZQ_KDTree_Node<T>* root, int max_leaf_npts)
	{
		if(root == 0)
			return;
		
		//printf("%d\n",root->npts);
		if(root->npts <= max_leaf_npts)
		{
			root->is_leaf = true;
			return ;
		}

		int ndim = root->ndim;
		T spread = root->box_max[0] - root->box_min[0];
		int k = 0;
		for(int i = 1;i < ndim;i++)
		{
			T tmp_spread = root->box_max[i] - root->box_min[i];
			if(tmp_spread > spread)
			{
				spread = tmp_spread;
				k = i;
			}
		}
		int n_low = root->npts/2;
		T cv;
		_median_split(root->npts, root->pts_idx, (const T**)root->pts, k,  cv, n_low);
		ZQ_KDTree_Node<T>* left_child = new ZQ_KDTree_Node<T>(ndim);
		ZQ_KDTree_Node<T>* right_child = new ZQ_KDTree_Node<T>(ndim);
		root->low_child = left_child;
		root->high_child = right_child;

		left_child->npts = n_low;
		left_child->pts_idx = root->pts_idx;
		left_child->pts = root->pts;
		right_child->npts = root->npts - n_low;
		right_child->pts_idx = root->pts_idx + n_low;
		right_child->pts = root->pts;
		memcpy(left_child->box_min,root->box_min,sizeof(T)*ndim);
		memcpy(left_child->box_max,root->box_max,sizeof(T)*ndim);
		memcpy(right_child->box_min,root->box_min,sizeof(T)*ndim);
		memcpy(right_child->box_max,root->box_max,sizeof(T)*ndim);
		left_child->box_max[k] = cv;
		right_child->box_min[k] = cv;
		//if(!_check_box(root->low_child))
		//	printf("err\n");

		//if(!_check_box(root->high_child))
		//	printf("err\n");


		_recursive_subdivided(left_child,max_leaf_npts);
		_recursive_subdivided(right_child,max_leaf_npts);
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_free(ZQ_KDTree_Node<T>* root)
	{
		if(root != 0)
		{
			ZQ_KDTree_Node<T>* left_child = root->low_child;
			ZQ_KDTree_Node<T>* right_child = root->high_child;
			root->low_child = 0;
			root->high_child = 0;
			delete root;
			_recursive_free(left_child);
			_recursive_free(right_child);
		}
	}

	template<class T>
	void ZQ_KDTree<T>::_update_search_result(int* out_idx, T* out_dis2, int& cur_k, int k, int cur_idx, T cur_dis2, bool& updated)
	{
		updated = false;
		if(cur_k == k)
		{
			if(cur_dis2 < out_dis2[k-1])
			{
				int ii;
				for(ii = k-1; ii > 0;ii--)
				{
					if(out_dis2[ii-1] < cur_dis2)
					{
						out_dis2[ii] = cur_dis2;
						out_idx[ii] = cur_idx;
						break;
					}
					out_dis2[ii] = out_dis2[ii-1];
					out_idx[ii] = out_idx[ii-1];
				}
				if(ii == 0)
				{
					out_dis2[0] = cur_dis2;
					out_idx[0] = cur_idx;
				}
				updated = true;
			}
		}
		else
		{
			if(cur_k == 0)
			{
				out_dis2[0] = cur_dis2;
				out_idx[0] = cur_idx;
				cur_k = 1;
 			}
			else
			{
				int ii;
				for(ii = cur_k; ii > 0;ii--)
				{
					if(out_dis2[ii-1] < cur_dis2)
					{
						out_dis2[ii] = cur_dis2;
						out_idx[ii] = cur_idx;
						break;
					}
					out_dis2[ii] = out_dis2[ii-1];
					out_idx[ii] = out_idx[ii-1];
				}
				if(ii == 0)
				{
					out_dis2[0] = cur_dis2;
					out_idx[0] = cur_idx;
				}
				cur_k++;
			}
			updated = true;
		}
	}

	template<class T>
	T ZQ_KDTree<T>::_distance2(const T* pt1, const T* pt2, int ndim)
	{
		T cur_dis2 = 0;
		for(int i = 0;i < ndim;i++)
			cur_dis2 += (pt1[i]-pt2[i])*(pt1[i]-pt2[i]);
		return cur_dis2;
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_ann_search(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double eps_plus_1_square)
	{
		if(root->is_leaf)
		{
			for(int i = 0;i < root->npts;i++)
			{
				int cur_idx = root->pts_idx[i];
				T cur_dis2 = _distance2(pt,root->pts[cur_idx],root->ndim);
				bool updated;
				_update_search_result(out_idx,out_dis2,cur_k,k,cur_idx,cur_dis2,updated);
			}
		}
		else
		{
			const ZQ_KDTree_Node<T>* low_child = root->low_child;
			const ZQ_KDTree_Node<T>* high_child = root->high_child;

			const double tol_eps = 1.0+1e-6;
			if(low_child != 0 && high_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt,low_child->box_min,low_child->box_max,root->ndim);
				T box_dis2_high = _box_distance_square(pt,high_child->box_min,high_child->box_max,root->ndim);
				const ZQ_KDTree_Node<T>* order[2];
				T order_box_dis2[2];
				if(box_dis2_low < box_dis2_high)
				{
					order[0] = low_child; order[1] = high_child;
					order_box_dis2[0] = box_dis2_low; order_box_dis2[1] = box_dis2_high;
				}
				else
				{
					order[0] = high_child; order[1] = low_child;
					order_box_dis2[0] = box_dis2_high; order_box_dis2[1] = box_dis2_low;
				}

				if(cur_k < k || order_box_dis2[0]*eps_plus_1_square <= out_dis2[cur_k-1]*tol_eps)
					_recursive_ann_search(order[0],pt,cur_k,k,out_idx,out_dis2,eps_plus_1_square);
				if(cur_k < k || order_box_dis2[1]*eps_plus_1_square <= out_dis2[cur_k-1]*tol_eps)
					_recursive_ann_search(order[1],pt,cur_k,k,out_idx,out_dis2,eps_plus_1_square);
			}
			else if(low_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt,low_child->box_min,low_child->box_max,root->ndim);
				if(cur_k < k || box_dis2_low*eps_plus_1_square <= out_dis2[cur_k-1]*tol_eps)
					_recursive_ann_search(low_child,pt,cur_k,k,out_idx,out_dis2,eps_plus_1_square);
			}
			else if(high_child != 0)
			{
				T box_dis2_high = _box_distance_square(pt,high_child->box_min,high_child->box_max,root->ndim);
				if(cur_k < k || box_dis2_high*eps_plus_1_square <= out_dis2[cur_k-1]*tol_eps)
					_recursive_ann_search(high_child,pt,cur_k,k,out_idx,out_dis2,eps_plus_1_square);
			}
		}
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_ann_search_with_initial_radius(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double eps_plus_1_square, double radius2)
	{
		if (root->is_leaf)
		{
			for (int i = 0; i < root->npts; i++)
			{
				int cur_idx = root->pts_idx[i];
				T cur_dis2 = _distance2(pt, root->pts[cur_idx], root->ndim);
				if (cur_dis2 > radius2)
					continue;
				bool updated;
				_update_search_result(out_idx, out_dis2, cur_k, k, cur_idx, cur_dis2, updated);
			}
		}
		else
		{
			const ZQ_KDTree_Node<T>* low_child = root->low_child;
			const ZQ_KDTree_Node<T>* high_child = root->high_child;

			const double tol_eps = 1.0 + 1e-6;
			if (low_child != 0 && high_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt, low_child->box_min, low_child->box_max, root->ndim);
				T box_dis2_high = _box_distance_square(pt, high_child->box_min, high_child->box_max, root->ndim);
				const ZQ_KDTree_Node<T>* order[2];
				T order_box_dis2[2];
				if (box_dis2_low < box_dis2_high)
				{
					order[0] = low_child; order[1] = high_child;
					order_box_dis2[0] = box_dis2_low; order_box_dis2[1] = box_dis2_high;
				}
				else
				{
					order[0] = high_child; order[1] = low_child;
					order_box_dis2[0] = box_dis2_high; order_box_dis2[1] = box_dis2_low;
				}

				if (order_box_dis2[0] <= radius2 && (cur_k < k || order_box_dis2[0] * eps_plus_1_square <= out_dis2[cur_k - 1] * tol_eps))
					_recursive_ann_search_with_initial_radius(order[0], pt, cur_k, k, out_idx, out_dis2, eps_plus_1_square, radius2);
				if (order_box_dis2[1] <= radius2 && (cur_k < k || order_box_dis2[1] * eps_plus_1_square <= out_dis2[cur_k - 1] * tol_eps))
					_recursive_ann_search_with_initial_radius(order[1], pt, cur_k, k, out_idx, out_dis2, eps_plus_1_square, radius2);
			}
			else if (low_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt, low_child->box_min, low_child->box_max, root->ndim);
				if (box_dis2_low <= radius2 && (cur_k < k || box_dis2_low*eps_plus_1_square <= out_dis2[cur_k - 1] * tol_eps))
					_recursive_ann_search_with_initial_radius(low_child, pt, cur_k, k, out_idx, out_dis2, eps_plus_1_square,radius2);
			}
			else if (high_child != 0)
			{
				T box_dis2_high = _box_distance_square(pt, high_child->box_min, high_child->box_max, root->ndim);
				if (box_dis2_high <= radius2 && (cur_k < k || box_dis2_high*eps_plus_1_square <= out_dis2[cur_k - 1] * tol_eps))
					_recursive_ann_search_with_initial_radius(high_child, pt, cur_k, k, out_idx, out_dis2, eps_plus_1_square,radius2);
			}
		}
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_ann_fix_radius_search_count(const ZQ_KDTree_Node<T>* root, const T* pt, int& k, double radius2)
	{
		if(root->is_leaf)
		{
			for(int i = 0;i < root->npts;i++)
			{
				int cur_idx = root->pts_idx[i];
				T cur_dis2 = _distance2(pt,root->pts[cur_idx],root->ndim);
				if(cur_dis2 <= radius2)
					k++;
			}
		}
		else
		{
			const ZQ_KDTree_Node<T>* low_child = root->low_child;
			const ZQ_KDTree_Node<T>* high_child = root->high_child;

			const double tol_eps = 1.0+1e-6;
			
			if(low_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt,low_child->box_min,low_child->box_max,root->ndim);
				if(box_dis2_low <= radius2*tol_eps)
					_recursive_ann_fix_radius_search_count(low_child,pt,k,radius2);
			}
			if(high_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt,high_child->box_min,high_child->box_max,root->ndim);
				if(box_dis2_low <= radius2*tol_eps)
					_recursive_ann_fix_radius_search_count(high_child,pt,k,radius2);
			}
		}
	}

	template<class T>
	void ZQ_KDTree<T>::_recursive_ann_fix_radius_search(const ZQ_KDTree_Node<T>* root, const T* pt, int& cur_k, int k, int* out_idx, T* out_dis2, double radius2)
	{
		if(root->is_leaf)
		{
			for(int i = 0;i < root->npts;i++)
			{
				int cur_idx = root->pts_idx[i];
				T cur_dis2 = _distance2(pt,root->pts[cur_idx],root->ndim);
				if(cur_dis2 <= radius2)
				{
					out_idx[cur_k] = cur_idx;
					out_dis2[cur_k] = cur_dis2;
					cur_k++;
				}
			}
		}
		else
		{
			const ZQ_KDTree_Node<T>* low_child = root->low_child;
			const ZQ_KDTree_Node<T>* high_child = root->high_child;

			const double tol_eps = 1.0+1e-6;
			if(low_child != 0)
			{
				T box_dis2_low = _box_distance_square(pt,low_child->box_min,low_child->box_max,root->ndim);
				if(cur_k < k || box_dis2_low <= radius2*tol_eps)
					_recursive_ann_fix_radius_search(low_child,pt,cur_k,k,out_idx,out_dis2,radius2);
			}
			if(high_child != 0)
			{
				T box_dis2_high = _box_distance_square(pt,high_child->box_min,high_child->box_max,root->ndim);
				if(cur_k < k || box_dis2_high <= radius2*tol_eps)
					_recursive_ann_fix_radius_search(high_child,pt,cur_k,k,out_idx,out_dis2,radius2);
			}
		}
	}

	template<class T>
	bool ZQ_KDTree<T>::_recursive_check_box(const ZQ_KDTree_Node<T>* root)
	{
		if(!_check_box(root))
			return false;
		if(root->low_child != 0 && !_recursive_check_box(root->low_child))
			return false;
		if(root->high_child != 0 && !_recursive_check_box(root->high_child))
			return false;
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::_check_box(const ZQ_KDTree_Node<T>* root)
	{
		for(int i = 0;i < root->npts;i++)
		{
			T box_dis2 = _box_distance_square(root->pts[root->pts_idx[i]],root->box_min,root->box_max,root->ndim);
			if(box_dis2 != 0)
				return false;
		}
		return true;
	}

	template<class T>
	void ZQ_KDTree<T>::_clear()
	{
		if(pts_raw)
		{
			delete []pts_raw;
			pts_raw = 0;
		}
		if(pts)
		{
			delete []pts;
			pts = 0;
		}
		if(pts_idx)
		{
			delete []pts_idx;
			pts_idx = 0;
		}
		if(tree)
		{
			_recursive_free(tree);
			tree = 0;
		}
	}

	template<class T>
	bool ZQ_KDTree<T>::BuildKDTree(const T* data, int npts, int ndim, int max_leaf_npts /*= 3*/)
	{
		if(data == 0 || npts < 0 || ndim < 1 || max_leaf_npts < 1)
			return false;
		
		_clear();

		pts_raw = new T[npts*ndim];
		memcpy(pts_raw,data,sizeof(T)*npts*ndim);
		pts = new T*[npts];
		for(int i = 0;i < npts;i++)
			pts[i] = pts_raw+i*ndim;
		pts_idx = new int[npts];
		for(int i = 0;i < npts;i++)
			pts_idx[i] = i;

		//
		tree = new ZQ_KDTree_Node<T>(ndim);
		tree->npts = npts;
		tree->pts_idx = pts_idx;
		tree->pts = pts;
		tree->is_leaf = false;
		
		for(int i = 0;i < ndim;i++)
			_find_min_max(npts,pts_idx,(const T**)pts,i,tree->box_min[i],tree->box_max[i]);
		if(tree->npts > max_leaf_npts)
			_recursive_subdivided(tree,max_leaf_npts);
		else
			tree->is_leaf = true;
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::Check() const
	{
		if(tree == 0)
			return false;
		return _recursive_check_box(tree);
	}

	template<class T>
	bool ZQ_KDTree<T>::BruteForceSearch(const T* pt, int k, int* out_idx, T* out_dis2) const
	{
		if(pts == 0 || tree == 0 || tree->npts < k || pt == 0 || out_idx == 0 || out_dis2 == 0)
			return false;

		int cur_k = 0;
		for(int i = 0;i < tree->npts;i++)
		{
			T tmp_dis2 = _distance2(pt,pts[i],tree->ndim);
			bool updated;
			_update_search_result(out_idx,out_dis2,cur_k,k,i,tmp_dis2,updated);
		}
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::AnnSearch(const T* pt, int k, int* out_idx, T* out_dis2, double eps /* = 0.0 */) const
	{
		if(pts == 0 || tree == 0 || tree->npts < k || pt == 0 || out_idx == 0 || out_dis2 == 0)
			return false;

		int cur_k = 0;
		_recursive_ann_search(tree,pt,cur_k,k,out_idx,out_dis2,(1+eps)*(1+eps));
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::AnnSearchWithInitalRadius(const T* pt, int k, int* out_idx, T* out_dis2, double radius, int& out_k, double eps /*= 0.0*/) const
	{
		if (pts == 0 || tree == 0 || tree->npts < k || pt == 0 || out_idx == 0 || out_dis2 == 0)
			return false;

		int cur_k = 0;
		_recursive_ann_search_with_initial_radius(tree, pt, cur_k, k, out_idx, out_dis2, (1 + eps)*(1 + eps), radius*radius);
		out_k = cur_k;
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::AnnFixRadiusSearch(const T* pt, double radius, int k, int* out_idx, T* out_dis2) const
	{
		if(pts == 0 || tree == 0 || tree->npts < k || pt == 0 || out_idx == 0 || out_dis2 == 0)
			return false;

		int cur_k = 0;
		_recursive_ann_fix_radius_search(tree,pt,cur_k,k,out_idx,out_dis2,radius*radius);
		return true;
	}

	template<class T>
	bool ZQ_KDTree<T>::AnnFixRadiusSearchCountReturnNum(const T* pt, double radius, int& k) const
	{
		if(pts == 0 || tree == 0 || pt == 0)
			return false;
		k = 0;
		_recursive_ann_fix_radius_search_count(tree,pt,k,radius*radius);
		return true;
	}
}

#endif