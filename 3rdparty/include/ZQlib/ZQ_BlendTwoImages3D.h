#ifndef _ZQ_BLEND_TWO_IMAGES3D_H_
#define _ZQ_BLEND_TWO_IMAGES3D_H_
#pragma once 

#include "ZQ_ImageProcessing3D.h"
#include <vector>

namespace ZQ
{
	class ZQ_BlendTwoImages3D
	{
	public:
		template<class T>
		static void BlendTwoImages(const int width, const int height, const int depth, const int nChannels, const T* image1, const T* image2, 
			const T* u, const T* v, const T* w, const T weight1, const int skip, const T radius, const int iterations, T* out_image);

	private:
		
		template<class T>
		static T _kernel_square(const T dis2, const T radius2);

		template<class T>
		static void _distribute_bucket(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const T x_min, const T y_min, const T z_min,
			const T radius, const int bucket_width, const int bucket_height, const int bucket_depth, 
			int* bucket_stored_num, int* bucket_stored_offset, int* bucket_stored_index, int* coord_in_which_bucket);

		template<class T>
		static void _compute_neightbors(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const T radius, 
			const int bucket_width, const int bucket_height, const int bucket_depth,
			const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
			int* neighbor_num, int** neighbor_index, T** neighbor_weight);

		template<class T>
		static void _solve_coeffs(const int num, const int nChannels, const T* values, const int bucket_width, const int bucket_height, const int bucket_depth,
			const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
			const int* neighbor_num, const int** neighbor_index, const T** neighbor_weight, T* coeffs);

		template<class T>
		static void _splat_data(const int bucket_width, const int bucket_height, const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, 
			const int* bucket_stored_index, const T* coord_x, const T* coord_y, const T* coord_z, const T radius, const int nChannels, 
			const T* coeffs, const int out_width, const int out_height, const int out_depth, T* out_images);

		template<class T>
		static void _compute_coords_and_values(const int seed_width, const int seed_height, const int seed_depth, const int width, const int height, const int depth,
			const int skip, const T* u, const T* v, const T* w, const T weight1, 
			T* coord_x, T* coord_y, T* coord_z, T* values);

		template<class T>
		static void _warp_and_blend(const int width, const int height, const int depth, const int nChannels, const T* image1, const T* image2, 
			const T* vel_image, const T weight1, T* out);

		template<class T>
		static void _compute_boundingbox(const int num, const T* coord_x, const T* coord_y, const T* coord_z, T boxmin[3], T boxmax[3]);

		template<class T>
		static void _scattered_interpolation(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const int nChannels, const T* values, const T radius, 
			const int iterations,const int out_width, const int out_height, const int out_depth, T* out_images);	
	};


	template<class T>
	T ZQ_BlendTwoImages3D::_kernel_square(const T dis2, const T radius2)
	{
		//T d = dis2 / radius2;
		//return exp(-d*6);

		T d2 = dis2/radius2;
		T d = sqrt(d2);
		T tmp = 1.0 - d;
		tmp *= tmp;
		tmp *= tmp;
		return tmp*(4.0*d+1.0);
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_distribute_bucket(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const T x_min, const T y_min, const T z_min,
		const T radius, const int bucket_width, const int bucket_height, const int bucket_depth, 
		int* bucket_stored_num, int* bucket_stored_offset, int* bucket_stored_index, int* coord_in_which_bucket)
	{
		// firstly, compute how many coords each bucket has 
		for(int i = 0;i < num;i++)
		{
			int d_idx = (coord_z[i] - z_min)/radius;
			int h_idx = (coord_y[i] - y_min)/radius;
			int w_idx = (coord_x[i] - x_min)/radius;

			d_idx = __max(0,__min(bucket_depth-1,d_idx));
			h_idx = __max(0,__min(bucket_height-1,h_idx));
			w_idx = __max(0,__min(bucket_width-1,w_idx));
			int bucket_idx = d_idx*bucket_height*bucket_width+h_idx*bucket_width+w_idx;
			bucket_stored_num[bucket_idx] ++;
			coord_in_which_bucket[i] = bucket_idx;
		}

		// secondly, compute the offset for each  bucket
		bucket_stored_offset[0] = 0;
		for(int i = 1;i < bucket_depth*bucket_height*bucket_width;i++)
		{
			bucket_stored_offset[i] = bucket_stored_offset[i-1] + bucket_stored_num[i-1];
		}

		// finally, distribute the index to buckets
		for(int i = 0;i < bucket_depth*bucket_height*bucket_width;i++)
		{
			bucket_stored_num[i] = 0;
		}

		for(int i = 0;i < num;i++)
		{
			int d_idx = (coord_z[i] - z_min)/radius;
			int h_idx = (coord_y[i] - y_min)/radius;
			int w_idx = (coord_x[i] - x_min)/radius;
			int bucket_idx = d_idx*bucket_height*bucket_width+h_idx*bucket_width+w_idx;

			bucket_stored_index[bucket_stored_offset[bucket_idx]+bucket_stored_num[bucket_idx]] = i;
			bucket_stored_num[bucket_idx] ++;
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_compute_neightbors(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const T radius, 
		const int bucket_width, const int bucket_height, const int bucket_depth,
		const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index, const int* coord_in_which_bucket, 
		int* neighbor_num, int** neighbor_index, T** neighbor_weight)
	{
		for(int idx = 0;idx < num;idx++)
		{
			int bucket_idx = coord_in_which_bucket[idx];

			int bucket_slice = bucket_width*bucket_height;
			int d_idx = bucket_idx/bucket_slice;
			int rest_idx = bucket_idx%bucket_slice;
			int w_idx = rest_idx%bucket_width;
			int h_idx = rest_idx/bucket_width;

			T cur_x = coord_x[idx];
			T cur_y = coord_y[idx];
			T cur_z = coord_z[idx];
			int cur_neigh_num = 0;

			std::vector<int> neighbor_idx;
			std::vector<T> neighbor_wei;

			T radius2 = radius*radius;

			for(int cur_w_idx = __max(0,w_idx-1);cur_w_idx <= __min(bucket_width-1,w_idx+1);cur_w_idx++)
			{ 
				for(int cur_h_idx = __max(0,h_idx-1);cur_h_idx <= __min(bucket_height-1,h_idx+1);cur_h_idx++)
				{
					for(int cur_d_idx = __max(0,d_idx-1);cur_d_idx <= __min(bucket_depth-1,d_idx+1);cur_d_idx++)
					{
						int cur_bucket_idx = cur_d_idx*bucket_slice+cur_h_idx*bucket_width+cur_w_idx;
						int cur_offset = bucket_stored_offset[cur_bucket_idx];
						for(int iii = 0;iii < bucket_stored_num[cur_bucket_idx];iii++)
						{
							int cur_nei_idx = bucket_stored_index[cur_offset+iii];
							if(cur_nei_idx == idx)
								continue;
							T cur_nei_x = coord_x[cur_nei_idx];
							T cur_nei_y = coord_y[cur_nei_idx];
							T cur_nei_z = coord_z[cur_nei_idx];
							T cur_dis2 = (cur_x-cur_nei_x)*(cur_x-cur_nei_x)+(cur_y-cur_nei_y)*(cur_y-cur_nei_y)+(cur_z-cur_nei_z)*(cur_z-cur_nei_z);

							if(cur_dis2 <= radius2)
							{
								neighbor_idx.push_back(cur_neigh_num);
								neighbor_wei.push_back(_kernel_square(cur_dis2,radius2));
								cur_neigh_num++;
							}
						}
					}
				}
			}
			neighbor_num[idx] = cur_neigh_num;
			if(cur_neigh_num > 0)
			{
				neighbor_index[idx] = new int[cur_neigh_num];
				neighbor_weight[idx] = new T[cur_neigh_num];
				for(int iii = 0;iii < cur_neigh_num;iii++)
				{
					neighbor_index[idx][iii] = neighbor_idx[iii];
					neighbor_weight[idx][iii] = neighbor_wei[iii];
				}
			}
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_solve_coeffs(const int num, const int nChannels, const T* values, const int bucket_width, const int bucket_height, const int bucket_depth,
		const int* bucket_stored_num, const int* bucket_stored_offset, const int* bucket_stored_index,
		const int* neighbor_num, const int** neighbor_index, const T** neighbor_weight, T* coeffs)
	{
		for(int z = 0;z < bucket_depth;z++)
		{
			for(int y = 0;y < bucket_height;y++)
			{
				for(int x = 0;x < bucket_width;x++)
				{
					int bucket_idx = z*bucket_height*bucket_width+y*bucket_width+x;
					int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
					int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];
					for(int i = 0;i < cur_bucket_stored_num;i++)
					{
						int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
						int cur_nei_num = neighbor_num[cur_coord_index];
						for(int c = 0;c < nChannels; c++)
						{
							T cur_coeff = values[cur_coord_index*nChannels+c];
							for(int j = 0;j < cur_nei_num;j++)
							{
								int cur_nei_idx = neighbor_index[cur_coord_index][j];
								T cur_nei_weight = neighbor_weight[cur_coord_index][j];
								cur_coeff -= coeffs[cur_nei_idx*nChannels+c]*cur_nei_weight;
							}
							coeffs[cur_coord_index*nChannels+c] = cur_coeff;
						}
					}
				}
			}
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_splat_data(const int bucket_width, const int bucket_height, const int bucket_depth, const int* bucket_stored_num, const int* bucket_stored_offset, 
		const int* bucket_stored_index, const T* coord_x, const T* coord_y, const T* coord_z, const T radius, const int nChannels, 
		const T* coeffs, const int out_width, const int out_height, const int out_depth, T* out_images)
	{

		for(int z = 0;z < bucket_depth;z++)
		{
			for(int y = 0;y < bucket_height;y++)
			{
				for(int x = 0;x < bucket_width;x++)
				{
					int bucket_idx = z*bucket_height*bucket_width+y*bucket_width+x;
					int cur_bucket_stored_num = bucket_stored_num[bucket_idx];
					int cur_bucket_stored_offset = bucket_stored_offset[bucket_idx];

					T radius2 = radius*radius;

					for(int i = 0;i < cur_bucket_stored_num;i++)
					{
						int cur_coord_index = bucket_stored_index[cur_bucket_stored_offset+i];
						T cur_x = coord_x[cur_coord_index];
						T cur_y = coord_y[cur_coord_index];
						T cur_z = coord_z[cur_coord_index];

						for(int d = __max(0,cur_z-radius); d <= __min(out_depth-1,cur_z+radius); d++)
						{
							T radius_xy2 = radius2 - (d-cur_z)*(d-cur_z);
							T radius_xy = sqrt(radius_xy2);	
							for(int h = __max(0,cur_y-radius_xy); h <= __min(out_height-1,cur_y+radius_xy); h++)
							{
								T radius_x2 = radius_xy2 - (h-cur_y)*(h-cur_y);
								T radius_x = sqrt(radius_x2);
								for(int w = __max(0,cur_x-radius_x); w <= __min(out_width-1,cur_x+radius_x); w++)
								{
									T dis2 = (d-cur_z)*(d-cur_z)+(h-cur_y)*(h-cur_y)+(w-cur_x)*(w-cur_x);
									if(dis2 <= radius)
									{
										T cur_weight = _kernel_square(dis2,radius2);
										for(int c = 0;c < nChannels;c++)
										{
											out_images[(d*out_height*out_width+h*out_width+w)*nChannels+c] += cur_weight*coeffs[cur_coord_index*nChannels+c];
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_compute_coords_and_values(const int seed_width, const int seed_height, const int seed_depth, const int width, const int height, const int depth,
		const int skip, const T* u, const T* v, const T* w, const T weight1, T* coord_x, T* coord_y, T* coord_z, T* values)
	{
		for(int z = 0;z < seed_depth;z++)
		{
			for(int y = 0;y < seed_height;y++)
			{
				for(int x = 0;x < seed_width;x++)
				{
					int seed_offset = z*seed_height*seed_width+y*seed_width+x;
					int x_off = x*skip;
					int y_off = y*skip;
					int z_off = z*skip;
					int offset = z_off*height*width+y_off*width+x_off;

					coord_x[seed_offset] = x_off + u[offset]*(1-weight1);
					coord_y[seed_offset] = y_off + v[offset]*(1-weight1);
					coord_z[seed_offset] = z_off + w[offset]*(1-weight1);

					values[seed_offset*6+0] = -u[offset]*(1-weight1);
					values[seed_offset*6+1] = -v[offset]*(1-weight1);
					values[seed_offset*6+2] = -w[offset]*(1-weight1);
					values[seed_offset*6+3] = u[offset]*weight1;
					values[seed_offset*6+4] = v[offset]*weight1;
					values[seed_offset*6+5] = w[offset]*weight1;
				}
			}
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_warp_and_blend(const int width, const int height, const int depth, const int nChannels, const T* image1, const T* image2, 
		const T* vel_image, const T weight1, T* out)
	{
		memset(out,0,sizeof(T)*width*height*depth*nChannels);
		T* result = new T[nChannels];
		for(int k = 0;k < depth;k++)
		{
			for(int j = 0;j < height;j++)
			{
				for(int i = 0;i < width;i++)
				{
					int offset = k*height*width+j*width+i;
					T coord_x = i+vel_image[offset*6+0];
					T coord_y = j+vel_image[offset*6+1];
					T coord_z = k+vel_image[offset*6+2];
					ZQ_ImageProcessing3D::TrilinearInterpolate(image1,width,height,depth,nChannels,coord_x,coord_y,coord_z,result,false);
					for(int c = 0;c < nChannels;c++)
						out[offset*nChannels+c] += weight1*result[c];
					coord_x = i+vel_image[offset*6+3];
					coord_y = j+vel_image[offset*6+4];
					coord_z = k+vel_image[offset*6+5];
					ZQ_ImageProcessing3D::TrilinearInterpolate(image2,width,height,depth,nChannels,coord_x,coord_y,coord_z,result,false);
					for(int c = 0; c < nChannels;c++)
						out[offset*nChannels+c] += (1-weight1)*result[c];
				}
			}
		}
		delete []result;
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_compute_boundingbox(const int num, const T* coord_x, const T* coord_y, const T* coord_z, T boxmin[3], T boxmax[3])
	{
		if(num < 1)
			return;
		boxmin[0] = coord_x[0];
		boxmin[1] = coord_y[0];
		boxmin[2] = coord_z[0];
		boxmax[0] = coord_x[0];
		boxmax[1] = coord_y[0];
		boxmax[2] = coord_z[0];

		for(int i = 1;i < num;i++)
		{
			if(boxmin[0] > coord_x[i])
				boxmin[0] = coord_x[i];
			if(boxmin[1] > coord_y[i])
				boxmin[1] = coord_y[i];
			if(boxmin[2] > coord_z[i])
				boxmin[2] = coord_z[i];
			if(boxmax[0] < coord_x[i])
				boxmax[0] = coord_x[i];
			if(boxmax[1] < coord_y[i])
				boxmax[1] = coord_y[i];
			if(boxmax[2] < coord_z[i])
				boxmax[2] = coord_z[i];
		}
	}

	template<class T>
	void ZQ_BlendTwoImages3D::_scattered_interpolation(const int num, const T* coord_x, const T* coord_y, const T* coord_z, const int nChannels, const T* values, const T radius, 
		const int iterations,const int out_width, const int out_height, const int out_depth, T* out_images)
	{

		T time1 = 0, time2 = 0, time3 = 0;

		clock_t start1 = clock();

		T boxmin[3],boxmax[3];
		_compute_boundingbox(num,coord_x,coord_y,coord_z,boxmin,boxmax);


		int bucket_width = (boxmax[0] - boxmin[0])/radius + 1;
		int bucket_height = (boxmax[1] - boxmin[1])/radius + 1;
		int bucket_depth = (boxmax[2] - boxmin[2])/radius + 1;
		int bucket_num = bucket_width*bucket_height*bucket_depth;
		int* bucket_stored_num = new int[bucket_num];
		int* bucket_stored_offset = new int[bucket_num];
		int* bucket_stored_index = new int[num];
		int* coord_in_which_bucket = new int[num];
		memset(bucket_stored_num,0,sizeof(int)*bucket_num);
		memset(bucket_stored_offset,0,sizeof(int)*bucket_num);
		memset(bucket_stored_index,0,sizeof(int)*num);
		memset(coord_in_which_bucket,0,sizeof(int)*num);

		_distribute_bucket(num,coord_x,coord_y,coord_z,boxmin[0],boxmin[1],boxmin[2],radius,bucket_width,bucket_height,bucket_depth,
			bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket);

		/*FILE* out_sto = fopen("cpu_sto.txt","w");
		for(int i = 0;i < bucket_num;i++)
		{
		fprintf(out_sto,"%d\n",bucket_stored_num[i]);
		}
		fclose(out_sto);*/

		/*FILE* out_which = fopen("cpu_which.txt","w");
		for(int i = 0;i < num;i++)
		{
		fprintf(out_which,"%d\n",coord_in_which_bucket[i]);
		}
		fclose(out_which);*/


		int* neighbor_num = new int[num];
		int** neighbor_index = new int*[num];
		T** neighbor_weight = new T*[num];
		memset(neighbor_num,0,sizeof(int)*num);
		memset(neighbor_index,0,sizeof(int*)*num);
		memset(neighbor_weight,0,sizeof(T*)*num);
		_compute_neightbors(num,coord_x,coord_y,coord_z,radius,bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,coord_in_which_bucket,
			neighbor_num,neighbor_index,neighbor_weight);


		clock_t stop1 = clock();
		time1 = (stop1-start1);


		clock_t start2 = clock();

		T* coeffs = new T[num*nChannels];
		memset(coeffs,0,sizeof(T)*num*nChannels);

		for(int it = 0;it < iterations;it++)
		{
			_solve_coeffs(num,nChannels,values,bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
				neighbor_num,(const int**)neighbor_index,(const T**)neighbor_weight,coeffs);
		}

		delete []neighbor_num;
		for(int i = 0;i < num;i++)
		{
			if(neighbor_index[i])
				delete []neighbor_index[i];
			if(neighbor_weight[i])
				delete []neighbor_weight[i];
		}
		delete []neighbor_index;
		delete []neighbor_weight;

		clock_t stop2 = clock();
		time2 = (stop2-start2);

		clock_t start3 = clock();

		_splat_data(bucket_width,bucket_height,bucket_depth,bucket_stored_num,bucket_stored_offset,bucket_stored_index,
			coord_x,coord_y,coord_z,radius,nChannels,coeffs,out_width,out_height,out_depth,out_images);

		delete []coeffs;
		delete []bucket_stored_num;
		delete []bucket_stored_offset;
		delete []bucket_stored_index;
		delete []coord_in_which_bucket;

		clock_t stop3 = clock();
		time3 = (stop3-start3);

		//printf("prepare=%f,solve=%f,splat=%f\n",time1*0.001,time2*0.001,time3*0.001);
	}


	template<class T>
	void ZQ_BlendTwoImages3D::BlendTwoImages(const int width, const int height, const int depth, const int nChannels, const T* image1, const T* image2, 
		const T* u, const T* v, const T* w, const T weight1, const int skip, const T radius, const int iterations, T* out_image)
	{

		int seed_width = width/skip;
		int seed_height = height/skip;
		int seed_depth = depth/skip;
		int num = seed_width*seed_height*seed_depth;
		T* coord_x = new T[num];
		T* coord_y = new T[num];
		T* coord_z = new T[num];
		T* values = new T[num*6];
		memset(coord_x,0,sizeof(T)*num);
		memset(coord_y,0,sizeof(T)*num);
		memset(coord_z,0,sizeof(T)*num);
		memset(values,0,sizeof(T)*num*6);

		clock_t start1 = clock();
		_compute_coords_and_values(seed_width,seed_height,seed_depth,width,height,depth,skip,u,v,w,weight1,coord_x,coord_y,coord_z,values);
		clock_t end1 = clock();
		//printf("compute coords:%f\n",0.001*(end1-start1));

		T* vel_image = new T[width*height*depth*6];
		memset(vel_image,0,sizeof(T)*width*height*depth*6);

		_scattered_interpolation(num, coord_x, coord_y, coord_z, 6, values, radius, iterations, width, height, depth, vel_image);

		clock_t start2 = clock();
		_warp_and_blend(width, height, depth, nChannels, image1, image2, vel_image, weight1, out_image);
		clock_t end2 = clock();
		//printf("warp_and_blend:%f\n",0.001*(end2-start2));

		delete []coord_x;
		delete []coord_y;
		delete []coord_z;
		delete []values;
		delete []vel_image;
	}
}

#endif