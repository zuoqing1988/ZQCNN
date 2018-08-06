#ifndef _ZQ_WAVELET_RAW_HEAD_H_
#define _ZQ_WAVELET_RAW_HEAD_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "ZQ_DoubleImage.h"
#include "ZQ_DoubleImage3D.h"

namespace ZQ
{
	/*wavename must be 'db1'*/
	class ZQ_WaveletRawHead
	{
	public:
		ZQ_WaveletRawHead()
		{
			width=height=nLevels=coeff_num= 0;
			coeff_idx = 0;
			coeff_vals = 0;
		}

		ZQ_WaveletRawHead(const ZQ_WaveletRawHead& other)
		{
			width = other.width;
			height = other.height;
			nLevels = other.nLevels;
			coeff_num = other.coeff_num;
			if(coeff_num > 0)
			{
				coeff_idx = new int[coeff_num];
				coeff_vals = new float[coeff_num];
				memcpy(coeff_idx,other.coeff_idx,sizeof(int)*coeff_num);
				memcpy(coeff_vals,other.coeff_vals,sizeof(float)*coeff_num);
			}
		}

		~ZQ_WaveletRawHead()
		{
			clear();
		}

		void clear()
		{
			if(coeff_idx)
				delete []coeff_idx;
			coeff_idx = 0;
			if(coeff_vals)
				delete []coeff_vals;
			coeff_vals = 0;
			width = height = nLevels = coeff_num = 0;
		}

		int nLevels;
		int width;
		int height;

		int coeff_num;
		int* coeff_idx;
		float* coeff_vals;

	public:
		bool LoadFromFile(const char* file)
		{
			FILE* in = fopen(file,"rb");
			if(in == 0)
				return false;

			int w,h,nlvls,num;
			fread(&w,sizeof(int),1,in);
			fread(&h,sizeof(int),1,in);
			fread(&nlvls,sizeof(int),1,in);
			fread(&num,sizeof(int),1,in);
			int* idx = new int[num];
			if(idx == 0)
			{
				fclose(in);
				return false;
			}
			if(fread(idx,sizeof(int),num,in) != num)
			{
				delete []idx;
				fclose(in);
				return false;
			}
			float* vals = new float[num];
			if(num == 0)
			{
				delete []idx;
				fclose(in);
				return false;
			}
			if(fread(vals,sizeof(float),num,in) != num)
			{
				delete []vals;
				delete []idx;
				fclose(in);
				return false;
			}
			clear();
			width = w;
			height = h;
			nLevels = nlvls;
			coeff_num = num;
			coeff_idx = idx;
			coeff_vals = vals;
			return true;
		}

		bool WriteToFile(const char* file) const
		{
			FILE* out = fopen(file,"wb");
			if(out == 0)
				return false;

			fwrite(&width,sizeof(int),1,out);
			fwrite(&height,sizeof(int),1,out);
			fwrite(&nLevels,sizeof(int),1,out);
			fwrite(&coeff_num,sizeof(int),1,out);
			fwrite(coeff_idx,sizeof(int),coeff_num,out);
			fwrite(coeff_vals,sizeof(float),coeff_num,out);
			fclose(out);
			return true;
		}
	};

	/*wavename must be 'db1'*/
	class ZQ_WaveletRawHead3D
	{
	public:
		ZQ_WaveletRawHead3D()
		{
			width = height = depth = nLevels = 0;
			coeff_num = 0;
			coeff_idx = 0;
			coeff_vals = 0;
		}
		
		ZQ_WaveletRawHead3D(const ZQ_WaveletRawHead3D& other)
		{
			width = other.width;
			height = other.height;
			depth = other.depth;
			nLevels = other.nLevels;
			coeff_num = other.coeff_num;
			if(coeff_num > 0)
			{
				coeff_idx = new int[coeff_num];
				coeff_vals = new float[coeff_num];
				memcpy(coeff_idx,other.coeff_idx,sizeof(int)*coeff_num);
				memcpy(coeff_vals,other.coeff_vals,sizeof(float)*coeff_num);
			}
		}

		~ZQ_WaveletRawHead3D(){clear();}

		int width, height, depth;
		int nLevels;
		int coeff_num;
		int* coeff_idx;
		float* coeff_vals;

		void clear()
		{
			width = height = depth = nLevels = 0;

			if(coeff_idx)
				delete []coeff_idx;

			coeff_idx = 0;
			if(coeff_vals)
				delete []coeff_vals;

			coeff_vals = 0;
		}

	public:
		bool LoadFromFile(const char* file)
		{
			clear();
			FILE* in = fopen(file,"rb");
			if(in == 0)
			{
				return false;
			}

			fread(&width,sizeof(int),1,in);
			fread(&height,sizeof(int),1,in);
			fread(&depth,sizeof(int),1,in);
			fread(&nLevels,sizeof(int),1,in);


			int num;
			fread(&num,sizeof(in),1,in);
			int* idx = new int[num];
			if(idx == 0)
			{
				clear();
				fclose(in);
				return false;
			}
			if(fread(idx,sizeof(int),num,in) != num)
			{
				delete []idx;
				clear();
				fclose(in);
				return false;
			}
			float* vals = new float[num];
			if(vals == 0)
			{
				clear();
				delete []idx;
				fclose(in);
				return false;
			}
			if(fread(vals,sizeof(float),num,in) != num)
			{
				clear();
				delete []idx;
				delete []vals;
				fclose(in);
				return false;
			}
			coeff_num = num;
			coeff_idx = idx;
			coeff_vals = vals;


			return true;
		}

		bool WriteToFile(const char* file) const
		{
			FILE* out = fopen(file,"wb");
			if(out == 0)
				return false;

			fwrite(&width,sizeof(int),1,out);
			fwrite(&height,sizeof(int),1,out);
			fwrite(&depth,sizeof(int),1,out);
			fwrite(&nLevels,sizeof(int),1,out);

			fwrite(&coeff_num,sizeof(int),1,out);
			fwrite(coeff_idx,sizeof(int),coeff_num,out);
			fwrite(coeff_vals,sizeof(int),coeff_num,out);


			fclose(out);
			return true;
		}
	};

}


#endif