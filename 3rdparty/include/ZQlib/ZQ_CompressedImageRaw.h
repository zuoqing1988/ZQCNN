#ifndef _ZQ_COMPRESSED_IMAGE_RAW_H_
#define _ZQ_COMPRESSED_IMAGE_RAW_H_

#include "ZQ_WaveletRawHead.h"
#include "ZQ_Wavelet.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_DoubleImage3D.h"
#include "ZQ_MergeSort.h"
#include <time.h>

namespace ZQ
{
	namespace ZQ_CompressedImageRaw
	{
		template<class T>
		bool CompressImage(const ZQ_DImage<T> &im, ZQ_WaveletRawHead& raw, const double ratio = 50, const double min_quality = 0.999);
		
		template<class T>
		bool DecompressImage(const ZQ_WaveletRawHead& raw, ZQ_DImage<T>& im);

		template<class T>
		bool CompressImage(const ZQ_DImage3D<T> &im, ZQ_WaveletRawHead3D& raw, const double ratio = 50, const double min_quality = 0.999);

		template<class T>
		bool DecompressImage(const ZQ_WaveletRawHead3D& raw, ZQ_DImage3D<T>& im);

		/********************** definitions **************************/


		template<class T>
		bool ZQ_CompressedImageRaw::CompressImage(const ZQ_DImage<T> &image, ZQ_WaveletRawHead& raw, const double ratio /* = 50 */, const double min_quality /* = 0.999 */)
		{
			raw.clear();
			if(image.nchannels() != 1)
			{
				printf("only support 1 channel image\n");
				return false;
			}

			clock_t t1 = clock();

			double mQuality = __min(0.9999,__max(0.80,min_quality));
			double mRatio = __max(1,ratio);

			int image_width = image.width();
			int image_height = image.height();

			const T*& image_ptr = image.data();

			int min_resolution = __min(image_width,image_height);

			char wavename[16] = "db1";

			const int max_levels = 8;

			int levels = 1;
			int wave_resolution = 4; // is related to the wave filter length
			while(wave_resolution < min_resolution && levels < max_levels)
			{
				wave_resolution *= 2;
				levels++;
			}

			int padding_width = (image_width+wave_resolution-1)/wave_resolution * wave_resolution;
			int padding_height = (image_height+wave_resolution-1)/wave_resolution * wave_resolution;

			int scale = pow(2.0,levels);
			int min_width = padding_width / scale;
			int min_height = padding_height / scale;


			raw.width = image_width;
			raw.height = image_height;
			raw.nLevels = levels;

			ZQ_Wavelet<T>::PaddingMode pad_mod = ZQ_Wavelet<T>::PADDING_ZPD;
			ZQ_Wavelet<T> m_wave;

			ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
			T*& image_each_channel_ptr = image_each_channel.data();
			for(int h = 0;h < image_height;h++)
			{
				for(int w = 0;w < image_width;w++)
				{
					image_each_channel_ptr[h*padding_width+w] = image_ptr[h*image_width+w];
				}
			}
			if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
			{
				raw.clear();
				return false;
			}
			ZQ_DImage<T> output;
			m_wave.GetWaveletImage(output);
			int output_nelements = output.nelements();
			T*& output_ptr = output.data();

			clock_t t2 = clock();

			T* coeffs = new T[output_nelements];
			double total_energy = 0;
			for(int iii = 0; iii < output_nelements;iii++)
			{
				coeffs[iii] = fabs(output_ptr[iii]);
				total_energy += coeffs[iii]*coeffs[iii];
			}

			ZQ_MergeSort::MergeSort(coeffs,output_nelements,false);


			double sum = 0;
			double sum_threshold = total_energy*mQuality; 
			double threshold = 0;
			for(int iii = 0;iii < output_nelements;iii++)
			{
				threshold = coeffs[iii];
				sum += threshold*threshold;
				if(sum >= sum_threshold && iii * ratio >= output_nelements)
					break;
			}
			delete []coeffs;

			clock_t t3 = clock();

			int output_width = output.width();
			int output_height = output.height();
			std::vector<int> indices;
			std::vector<float> values;

			for(int hh = 0;hh < output_height;hh++)
			{
				for(int ww = 0;ww < output_width;ww++)
				{
					if(fabs(output_ptr[hh*output_width+ww]) >= threshold)
					{
						indices.push_back(hh*output_width+ww);
						values.push_back(output_ptr[hh*output_width+ww]);
					}
				}
			}
			int tmp_num = indices.size();
			int* tmp_idx = 0;
			float* tmp_values = 0;
			if(tmp_num > 0)
			{
				tmp_idx = new int[tmp_num];
				tmp_values = new float[tmp_num];
				memcpy(tmp_idx,&indices[0],sizeof(int)*tmp_num);
				memcpy(tmp_values,&values[0],sizeof(float)*tmp_num);
			}
			raw.coeff_num = tmp_num;
			raw.coeff_idx = tmp_idx;
			raw.coeff_vals = tmp_values;

			clock_t t4 = clock();

			//printf("dwt = %f, sort = %f, select = %f\n",0.001*(t2-t1),0.001*(t3-t2),0.001*(t4-t3));

			return true;
		}

		template<class T>
		bool ZQ_CompressedImageRaw::DecompressImage(const ZQ_WaveletRawHead& raw, ZQ_DImage<T>& image)
		{
			const char* wavename = "db1";

			int width = raw.width;
			int height = raw.height;
			ZQ_Wavelet<T>::PaddingMode pad_mode = ZQ_Wavelet<T>::PADDING_ZPD;
			int levels = raw.nLevels;

			int wave_resolution = 4;

			int scale = pow(2.0,levels);
			wave_resolution *= scale/2;

			int padding_width = (width+wave_resolution-1)/wave_resolution * wave_resolution;
			int padding_height = (height+wave_resolution-1)/wave_resolution * wave_resolution;



			//ZQ_DImage<T> image_each_channel;

			int wave_image_width = padding_width;
			int wave_image_height = padding_height;
			int min_width = wave_image_width/scale;
			int min_height = wave_image_height/scale;


			T* wave_image = new T[wave_image_width*wave_image_height];
			memset(wave_image,0,sizeof(T)*wave_image_width*wave_image_height);

			for(int i = 0;i < raw.coeff_num;i++)
			{
				int cur_idx = raw.coeff_idx[i];
				float val = raw.coeff_vals[i];
				int h = cur_idx/wave_image_width;
				int w = cur_idx%wave_image_width;
				wave_image[h*wave_image_width+w] = val;
			}



			int cur_width = min_width;
			int cur_height = min_height;
			int cur_level = 0;
			ZQ_DImage<T> ca(cur_width,cur_height);
			T*& ca_ptr = ca.data();
			for(int hh = 0;hh < cur_height;hh++)
			{
				for(int ww = 0;ww < cur_width;ww++)
				{
					ca_ptr[hh*cur_width+ww] = wave_image[hh*wave_image_width+ww];
				}
			}

			while(cur_level < levels)
			{
				ZQ_DImage<T> ch(cur_width,cur_height);
				ZQ_DImage<T> cv(cur_width,cur_height);
				ZQ_DImage<T> cd(cur_width,cur_height);
				ZQ_DImage<T> tmp_out;
				T*& ch_ptr = ch.data();
				T*& cv_ptr = cv.data();
				T*& cd_ptr = cd.data();
				for(int hh = 0;hh < cur_height;hh++)
				{
					for(int ww = 0;ww < cur_width;ww++)
					{
						ch_ptr[hh*cur_width+ww] = wave_image[hh*wave_image_width+ww+cur_width];
						cv_ptr[hh*cur_width+ww] = wave_image[(hh+cur_height)*wave_image_width+ww];
						cd_ptr[hh*cur_width+ww] = wave_image[(hh+cur_height)*wave_image_width+ww+cur_width];
					}
				}
				if(!ZQ_Wavelet<T>::IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
				{
					delete []wave_image;
					return false;
				}
				ca.copyData(tmp_out);
				cur_width *= 2;
				cur_height *= 2;
				cur_level ++;
			}

			delete []wave_image;

			image.allocate(width,height,1);
			T*& image_ptr = image.data();

			
			for(int h = 0;h < height && h < wave_image_height; h++)
			{
				for(int w = 0;w < width && w < wave_image_width; w++)
				{
					image_ptr[h*width+w] = ca_ptr[h*wave_image_width+w];
				}
			}

			return true;
		}

		template<class T>
		bool ZQ_CompressedImageRaw::CompressImage(const ZQ_DImage3D<T> &image, ZQ_WaveletRawHead3D& raw, const double ratio /* = 50 */, const double min_quality /* = 0.999 */)
		{
			raw.clear();

			double mQuality = __min(0.9999,__max(0.80,min_quality));
			double mRatio = __max(1,ratio);

			clock_t t1 = clock();

			int image_width = image.width();
			int image_height = image.height();
			int image_depth = image.depth();

			if(image.nchannels() != 1)
			{
				printf("only support 1 channel image\n");
				return false;
			}

			const T*& image_ptr = image.data();

			int min_resolution = __min(image_width,image_height);

			char wavename[16] = "db1";

			const int max_levels = 8;

			int levels = 1;
			int wave_resolution = 4; // is related to the wave filter length
			while(wave_resolution < min_resolution && levels < max_levels)
			{
				wave_resolution *= 2;
				levels++;
			}

			int padding_width = (image_width+wave_resolution-1)/wave_resolution * wave_resolution;
			int padding_height = (image_height+wave_resolution-1)/wave_resolution * wave_resolution;

			int scale = pow(2.0,levels);
			int min_width = padding_width / scale;
			int min_height = padding_height / scale;


			raw.width = image_width;
			raw.height = image_height;
			raw.depth = image_depth;
			raw.nLevels = levels;

			ZQ_Wavelet<T>::PaddingMode pad_mod = ZQ_Wavelet<T>::PADDING_ZPD;

			T* coeffs = new T[padding_width*padding_height*image_depth];
			int slice_size = padding_width*padding_height;
			int total_size = slice_size*image_depth;

			ZQ_Wavelet<T> m_wave;

			std::vector<ZQ_DImage<T>> output(image_depth);
			for(int kk = 0;kk < image_depth;kk++)
			{
				ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
				T*& image_each_channel_ptr = image_each_channel.data();
				for(int h = 0;h < image_height;h++)
				{
					for(int w = 0;w < image_width;w++)
					{
						image_each_channel_ptr[h*padding_width+w] = image_ptr[kk*image_height*image_width+h*image_width+w];
					}
				}
				if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
				{
					raw.clear();
					delete []coeffs;
					return false;
				}

				m_wave.GetWaveletImage(output[kk]);

				T*& output_ptr = output[kk].data();




				int cur_slice_offset = slice_size*kk;
				for(int iii = 0; iii < slice_size;iii++)
				{
					coeffs[iii+cur_slice_offset] = fabs(output_ptr[iii]);
				}
			}

			clock_t t2 = clock();

			double total_energy = 0;
			for(int iii = 0;iii < total_size;iii++)
				total_energy += coeffs[iii]*coeffs[iii];

			//printf("total_energy=%f\n",total_energy);

			ZQ_MergeSort::MergeSort(coeffs,total_size,false);

			double sum = 0;
			double sum_threshold = total_energy*mQuality; 
			double threshold = 0;
			for(int iii = 0;iii < total_size;iii++)
			{
				threshold = coeffs[iii];
				sum += threshold*threshold;
				if(sum >= sum_threshold && iii * ratio >= total_size)
					break;
			}
			delete []coeffs;

			clock_t t3 = clock();


			std::vector<int> indices;
			std::vector<float> values;

			for(int kk = 0;kk < image_depth;kk++)
			{
				T*& output_ptr = output[kk].data();
				for(int hh = 0;hh < padding_height;hh++)
				{
					for(int ww = 0;ww < padding_width;ww++)
					{
						if(fabs(output_ptr[hh*padding_width+ww]) >= threshold)
						{
							indices.push_back(kk*slice_size+hh*padding_width+ww);
							values.push_back(output_ptr[hh*padding_width+ww]);
						}
					}
				}
			}

			int tmp_num = indices.size();
			int* tmp_idx = 0;
			float* tmp_values = 0;
			if(tmp_num > 0)
			{
				tmp_idx = new int[tmp_num];
				tmp_values = new float[tmp_num];
				memcpy(tmp_idx,&indices[0],sizeof(int)*tmp_num);
				memcpy(tmp_values,&values[0],sizeof(float)*tmp_num);
			}
			raw.coeff_num = tmp_num;
			raw.coeff_idx = tmp_idx;
			raw.coeff_vals = tmp_values;

			clock_t t4 = clock();

			//printf("dwt = %f, sort = %f, select = %f\n",0.001*(t2-t1),0.001*(t3-t2),0.001*(t4-t3));


			return true;
		}

		template<class T>
		bool ZQ_CompressedImageRaw::DecompressImage(const ZQ_WaveletRawHead3D& raw, ZQ_DImage3D<T>& image)
		{
			const char* wavename = "db1";

			int width = raw.width;
			int height = raw.height;
			int depth = raw.depth;
			ZQ_Wavelet<T>::PaddingMode pad_mode = ZQ_Wavelet<T>::PADDING_ZPD;
			int levels = raw.nLevels;

			int wave_resolution = 4;

			int scale = pow(2.0,levels);
			wave_resolution *= scale/2;

			int padding_width = (width+wave_resolution-1)/wave_resolution * wave_resolution;
			int padding_height = (height+wave_resolution-1)/wave_resolution * wave_resolution;



			std::vector<ZQ_DImage<T>> image_each_channel;

			int wave_image_width = padding_width;
			int wave_image_height = padding_height;
			int min_width = wave_image_width/scale;
			int min_height = wave_image_height/scale;

			T* wave_image = new T[wave_image_width*wave_image_height*depth];
			int slice_size = wave_image_width*wave_image_height;
			int total_size = slice_size*depth;
			memset(wave_image,0,sizeof(T)*total_size);

			for(int i = 0;i < raw.coeff_num;i++)
			{
				int cur_idx = raw.coeff_idx[i];
				float val = raw.coeff_vals[i];
				int d = cur_idx/slice_size;
				int rest = cur_idx%slice_size;
				int h = rest/wave_image_width;
				int w = rest%wave_image_width;
				wave_image[d*slice_size+h*wave_image_width+w] = val;
			}

			for(int kk = 0;kk < depth;kk++)
			{	
				int cur_offset = kk*slice_size;
				int cur_width = min_width;
				int cur_height = min_height;
				int cur_level = 0;
				ZQ_DImage<T> ca(cur_width,cur_height);
				T*& ca_ptr = ca.data();
				for(int hh = 0;hh < cur_height;hh++)
				{
					for(int ww = 0;ww < cur_width;ww++)
					{
						ca_ptr[hh*cur_width+ww] = wave_image[cur_offset+hh*wave_image_width+ww];
					}
				}

				while(cur_level < levels)
				{
					ZQ_DImage<T> ch(cur_width,cur_height);
					ZQ_DImage<T> cv(cur_width,cur_height);
					ZQ_DImage<T> cd(cur_width,cur_height);
					ZQ_DImage<T> tmp_out;
					T*& ch_ptr = ch.data();
					T*& cv_ptr = cv.data();
					T*& cd_ptr = cd.data();
					for(int hh = 0;hh < cur_height;hh++)
					{
						for(int ww = 0;ww < cur_width;ww++)
						{
							ch_ptr[hh*cur_width+ww] = wave_image[cur_offset+hh*wave_image_width+ww+cur_width];
							cv_ptr[hh*cur_width+ww] = wave_image[cur_offset+(hh+cur_height)*wave_image_width+ww];
							cd_ptr[hh*cur_width+ww] = wave_image[cur_offset+(hh+cur_height)*wave_image_width+ww+cur_width];
						}
					}
					if(!ZQ_Wavelet<T>::IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
					{
						delete []wave_image;
						return false;
					}
					ca.copyData(tmp_out);
					cur_width *= 2;
					cur_height *= 2;
					cur_level ++;
				}
				image_each_channel.push_back(ca);
			}

			delete []wave_image;

			image.allocate(width,height,depth,1);
			T*& image_ptr = image.data();

			for(int kk = 0;kk < depth;kk++)
			{
				T*& image_each_channel_ptr = image_each_channel[kk].data();
				for(int h = 0;h < height && h < wave_image_height; h++)
				{
					for(int w = 0;w < width && w < wave_image_width; w++)
					{
						image_ptr[kk*height*width+h*width+w] = image_each_channel_ptr[h*wave_image_width+w];
					}
				}
			}

			return true;
		}
	}
}

#endif
