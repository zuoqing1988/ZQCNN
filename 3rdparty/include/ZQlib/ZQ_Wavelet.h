#ifndef _ZQ_WAVELET_H_
#define _ZQ_WAVELET_H_
#pragma  once

#include "ZQ_DoubleImage.h"
#include "ZQ_DoubleImage3D.h"
#include <vector>

namespace ZQ
{
	
	template<class T>
	class ZQ_Wavelet
	{
	public:
		enum PaddingMode{PADDING_ZPD,PADDING_SYM};
		enum ConvolutionMode{CONV_VAILD,CONV_FULL};
	public:
		ZQ_Wavelet();
		~ZQ_Wavelet();

	private:
		int nLevels;
		std::vector<ZQ_DImage<T>> ca;
		std::vector<ZQ_DImage<T>> ch;
		std::vector<ZQ_DImage<T>> cv;
		std::vector<ZQ_DImage<T>> cd;

	public:
		bool DiscreteWaveletImageNLevels(const ZQ_DImage<T>& input, const char* wavename, const int nLevels = 1, const PaddingMode mode = PADDING_ZPD);

		void GetWaveletImage(ZQ_DImage<T>& output);

		int GetLevelsNum() const {return nLevels;}

		const ZQ_DImage<T>& GetCA(const int level)const {return ca[level];}
		const ZQ_DImage<T>& GetCH(const int level)const {return ch[level];}
		const ZQ_DImage<T>& GetCV(const int level)const {return cv[level];}
		const ZQ_DImage<T>& GetCD(const int level)const {return cd[level];}

	private:
		void _clear(){nLevels = 0; ca.clear(); ch.clear(); cv.clear(); cd.clear();}


	public:

		static bool DWT(const ZQ_DImage<T>& input, ZQ_DImage<T>& ca, ZQ_DImage<T>& cd, const char* wavename,const PaddingMode mode);
		static bool IDWT(const ZQ_DImage<T>& ca, const ZQ_DImage<T>& cd, ZQ_DImage<T>& output, const char* wavename, const PaddingMode mode);

		static bool DWT2(const ZQ_DImage<T>& input, ZQ_DImage<T>& ca, ZQ_DImage<T>& ch, ZQ_DImage<T>& cv, ZQ_DImage<T>& cd, const char* wavename, const PaddingMode mode);
		static bool IDWT2(const ZQ_DImage<T>& ca, const ZQ_DImage<T>& ch, const ZQ_DImage<T>& cv, const ZQ_DImage<T>& cd, ZQ_DImage<T>& output, const char* wavename, const PaddingMode mode);

	private:
		/***************************  WARNING:  ******************/
		/* CONV_VALID: outputlen = inputlen - filterlen + 1
		/* CONV_FULL:  outputlen = inputlen + filterlen - 1
		/**********************************************************/
		static void _convfiltering(const T* input, const int inputlen, const T* filter, const int filterlen, T*& output, int& output_len, const ConvolutionMode mode);

		//the memory of lowfilter and highfilter will be allocated in this function
		// wavename = {"db1","haar", "db2", "db3", "db4", "db5"};
		static bool _selectfilter(const char* wavename,  int &filterlen, T*& lowfilter, T*& highfilter, bool inverseflag);

		static void _padding(const T* input, const int input_len, const int filter_len, T*& padding_img, int& paddingimg_len, const PaddingMode mode);

		static void _mergesort(T* vals,int start, int end);
	public:
		static bool LoadWaveletFile(const char* file, ZQ_DImage<T>& image);

		static bool LoadWaveletFile(const char* file, ZQ_DImage3D<T>& image);

		static bool LoadWaveletFromBytes(const int input_len, const unsigned char* input_bytes, ZQ_DImage<T>& image);

		static bool LoadWaveletFromBytes(const int input_len, const unsigned char* input_bytes, ZQ_DImage3D<T>& image);

		/*	first we will compute the energy for first [num_coeffs/ratio] coefficients, if the quality is smaller than [min_quality], it will use more coefficients, 
		/*	thus the real ratio will be less than [ratio] 
		/*	quality suggest: 0.9999*/
		static bool SaveWaveletFile(const char* file, const ZQ_DImage<T>& image, const double ratio = 50, const double min_quality = 0.9999);

		static bool SaveWaveletFile(const char* file, const ZQ_DImage3D<T>& image, const double ratio = 50, const double min_quality = 0.9999);

		static bool SaveWaveletToBytes(int& output_len, unsigned char*& output_bytes, const ZQ_DImage<T>& image, const double ratio = 50, const double min_quality = 0.999);

		static bool SaveWaveletToBytes(int& output_len, unsigned char*& output_bytes, const ZQ_DImage3D<T>& image, const double ratio = 50, const double min_quality = 0.999);
	
	

	
	};

	/********************************* definitions ***************************************/
	
	template<class T>
	ZQ_Wavelet<T>::ZQ_Wavelet()
	{

		nLevels = 0;
	}

	template<class T>
	ZQ_Wavelet<T>::~ZQ_Wavelet()
	{
		_clear();
	}


	template<class T>
	bool ZQ_Wavelet<T>::DiscreteWaveletImageNLevels(const ZQ_DImage<T>& input, const char* wavename, const int nLevels /* = 1 */, const PaddingMode mode /* = PADDING_ZPD*/)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		if(nChannels != 1)
			return false;

		if(nLevels < 1)
			return false;

		int ext_len = 2 << nLevels;
		if(width % ext_len != 0 || height % ext_len != 0)
			return false;

		_clear();

		this->nLevels = nLevels;
		for(int i = 0;i < nLevels;i++)
		{
			ZQ_DImage<T> tmp_ca,tmp_ch,tmp_cv,tmp_cd;
			if(i == 0)
			{
				if(!DWT2(input,tmp_ca,tmp_ch,tmp_cv,tmp_cd,wavename,mode))
				{
					_clear();
					return false;
				}
				ca.push_back(tmp_ca);
				ch.push_back(tmp_ch);
				cv.push_back(tmp_cv);
				cd.push_back(tmp_cd);
			}
			else
			{
				if(!DWT2(ca[i-1],tmp_ca,tmp_ch,tmp_cv,tmp_cd,wavename,mode))
				{
					_clear();
					return false;
				}
				ca.push_back(tmp_ca);
				ch.push_back(tmp_ch);
				cv.push_back(tmp_cv);
				cd.push_back(tmp_cd);
			}
		}

		return true;
	}

	template<class T>
	void ZQ_Wavelet<T>::GetWaveletImage(ZQ_DImage<T>& output)
	{
		output.clear();
		if(nLevels == 0)
			return ;

		int out_width = ca[0].width()*2;
		int out_height = ca[0].height()*2;

		output.allocate(out_width,out_height,1);

		T*& output_ptr = output.data();

		for(int i = 0;i < nLevels;i++)
		{
			int tmp_width = ca[i].width();
			int tmp_height = ca[i].height();

			T*& ca_ptr = ca[i].data();
			T*& ch_ptr = ch[i].data();
			T*& cv_ptr = cv[i].data();
			T*& cd_ptr = cd[i].data();

			for(int h = 0;h < tmp_height;h++)
			{
				for(int w = 0;w < tmp_width;w++)
				{
					output_ptr[h*out_width+w+tmp_width] = ch_ptr[h*tmp_width+w];
					output_ptr[(h+tmp_height)*out_width+w] = cv_ptr[h*tmp_width+w];
					output_ptr[(h+tmp_height)*out_width+w+tmp_width] = cd_ptr[h*tmp_width+w];
				}
			}

			if(i == nLevels-1)
			{
				for(int h = 0;h < tmp_height;h++)
				{
					for(int w = 0;w < tmp_width;w++)
					{
						output_ptr[h*out_width+w] = ca_ptr[h*tmp_width+w];
					}
				}
			}
		}
	}

	template<class T>
	bool ZQ_Wavelet<T>::DWT(const ZQ_DImage<T>& input, ZQ_DImage<T>& ca, ZQ_DImage<T>& cd, const char* wavename, const PaddingMode mode)
	{
		if(input.nchannels() != 1)
			return false;

		if(input.width() != 1 && input.height() != 1)
			return false;

		int input_len = input.width()*input.height();
		if(input_len < 1)
			return false;

		T* lowfilter = 0;
		T* highfilter = 0;
		int filter_len = 0;
		if(!_selectfilter(wavename,filter_len,lowfilter,highfilter,false))
			return false;

		if(input_len < filter_len) 
		{
			delete []lowfilter;
			delete []highfilter;
			return false;
		}


		T* padding_img = 0;
		int paddingimg_len = 0;
		_padding(input.data(),input_len,filter_len,padding_img,paddingimg_len,mode);


		int output_len = (filter_len+input_len-2-1)/2+1;
		if(input.width() == 1)
		{
			ca.allocate(1,output_len,1);
			cd.allocate(1,output_len,1);
		}
		else
		{
			ca.allocate(output_len,1,1);
			cd.allocate(output_len,1,1);
		}

		T*& ca_ptr = ca.data();
		T*& cd_ptr = cd.data();

		T* z_img = 0;
		int z_img_len;
		_convfiltering(padding_img,paddingimg_len,lowfilter,filter_len,z_img,z_img_len,CONV_VAILD);
		for(int i = 0;i < output_len;i++)
		{
			ca_ptr[i] = z_img[i*2+1];
		}

		delete []z_img;

		_convfiltering(padding_img,paddingimg_len,highfilter,filter_len,z_img,z_img_len,CONV_VAILD);
		for(int i = 0;i < output_len;i++)
		{
			cd_ptr[i] = z_img[i*2+1];
		}

		delete []z_img;

		delete []padding_img;
		delete []lowfilter;
		delete []highfilter;
		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::IDWT(const ZQ_DImage<T>& ca, const ZQ_DImage<T>& cd, ZQ_DImage<T>& output, const char* wavename, const PaddingMode mode)
	{
		int ca_width = ca.width();
		int ca_height = ca.height();
		int ca_nchannels = ca.nchannels();
		if(ca_nchannels != 1)
			return false;

		if(!cd.matchDimension(ca_width,ca_height,ca_nchannels))
			return false;

		int filterlen = 0;
		T* lowfilter = 0;
		T* highfilter = 0;
		if(!_selectfilter(wavename,filterlen,lowfilter,highfilter,true))
		{
			return false;
		}

		int input_len = ca_height*ca_width;
		int upsample_len = input_len*2;
		int output_len = upsample_len-filterlen+2;
		if(ca_width == 1)
		{
			output.allocate(1,output_len,1);
		}
		else
		{
			output.allocate(output_len,1,1);
		}

		const T*& ca_ptr = ca.data();
		const T*& cd_ptr = cd.data();
		T*& output_ptr = output.data();

		T* upsample_ca = new T[upsample_len];
		T* upsample_cd = new T[upsample_len];



		for(int i = 0;i < input_len;i++)
		{
			upsample_ca[2*i+0] = ca_ptr[i];
			upsample_ca[2*i+1] = 0;
			upsample_cd[2*i+0] = cd_ptr[i];
			upsample_cd[2*i+1] = 0;
		}

		T* z_img = 0;
		int z_img_len;
		_convfiltering(upsample_ca,upsample_len-1,lowfilter,filterlen,z_img,z_img_len,CONV_FULL);
		int offset_z_img = (z_img_len - output_len)/2;
		for(int i = 0;i < output_len;i++)
			output_ptr[i] += z_img[i+offset_z_img];

		delete []z_img;
		_convfiltering(upsample_cd,upsample_len-1,highfilter,filterlen,z_img,z_img_len,CONV_FULL);
		for(int i = 0;i < output_len;i++)
			output_ptr[i] += z_img[i+offset_z_img];

		delete []z_img;

		delete []upsample_ca;
		delete []upsample_cd;
		delete []lowfilter;
		delete []highfilter;

		return true;
	}


	template<class T>
	bool ZQ_Wavelet<T>::DWT2(const ZQ_DImage<T>& input, ZQ_DImage<T>& ca, ZQ_DImage<T>& ch, ZQ_DImage<T>& cv, ZQ_DImage<T>& cd, const char* wavename, const PaddingMode mode)
	{
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		if(nChannels != 1)
			return false;

		int filter_len = 0;
		T* lowfilter = 0;
		T* highfilter = 0;
		if(!_selectfilter(wavename,filter_len,lowfilter,highfilter,false))
		{
			return false;
		}

		if(width < filter_len || height < filter_len)
		{
			delete []lowfilter;
			delete []highfilter;
			return false;
		}

		int last_height = (height + filter_len - 2);
		int last_width = (width + filter_len - 2);
		int row_len = (last_width-1)/2+1;
		int col_len = (last_height-1)/2+1;

		ca.allocate(col_len,row_len);
		ch.allocate(col_len,row_len);
		cv.allocate(col_len,row_len);
		cd.allocate(col_len,row_len);

		const T*& input_ptr = input.data();
		T*& ca_ptr = ca.data();
		T*& ch_ptr = ch.data();
		T*& cv_ptr = cv.data();
		T*& cd_ptr = cd.data();


		// first pass horizontal low part
		T** low_part_rows = new T*[height];
		int low_part_width;
		for(int i = 0;i < height;i++)
		{
			T* padding_row = 0;
			int padding_len = 0;
			T* z_img_row = 0;
			_padding(input_ptr+i*width,width,filter_len,padding_row,padding_len,mode);
			_convfiltering(padding_row,padding_len,lowfilter,filter_len,z_img_row,low_part_width,CONV_VAILD);
			low_part_rows[i] = z_img_row;
			delete []padding_row;
		}

		// ca : second pass : vertical low part of the horizontal low part
		// ch : second pass : vertical high part of the horizontal low part

		int column_height = 0;
		for(int i = 0;i < col_len;i++)
		{
			T* tmp_col = new T[height];
			int real_col_id = 2*i+1;
			T* padding_col = 0;
			int padding_len = 0;
			T* z_img_col = 0;
			for(int j = 0;j < height;j++)
				tmp_col[j] = low_part_rows[j][real_col_id];
			_padding(tmp_col,height,filter_len,padding_col,padding_len,mode);
			delete []tmp_col;


			_convfiltering(padding_col,padding_len,lowfilter,filter_len,z_img_col,column_height,CONV_VAILD);

			for(int j = 0;j < row_len;j++)
				ca_ptr[j*col_len+i] = z_img_col[2*j+1];
			delete []z_img_col;

			_convfiltering(padding_col,padding_len,highfilter,filter_len,z_img_col,column_height,CONV_VAILD);

			for(int j = 0;j < row_len;j++)
				ch_ptr[j*col_len+i] = z_img_col[2*j+1];
			delete []z_img_col;


			delete []padding_col;
		}

		for(int i = 0;i < height;i++)
			delete []low_part_rows[i];
		delete []low_part_rows;

		// first pass horizontal high part

		T** high_part_rows = new T*[height];
		int high_part_width;
		for(int i = 0;i < height;i++)
		{
			T* padding_row = 0;
			int padding_len = 0;
			T* z_img_row = 0;
			_padding(input_ptr+i*width,width,filter_len,padding_row,padding_len,mode);
			_convfiltering(padding_row,padding_len,highfilter,filter_len,z_img_row,high_part_width,CONV_VAILD);
			high_part_rows[i] = z_img_row;
			delete []padding_row;
		}


		// cv : second pass : vertical low part of the horizontal high part
		// cd : second pass : vertical high part of the horizontal high part

		for(int i = 0;i < col_len;i++)
		{
			T* tmp_col = new T[height];
			int real_col_id = 2*i+1;
			T* padding_col = 0;
			int padding_len = 0;
			T* z_img_col = 0;
			for(int j = 0;j < height;j++)
				tmp_col[j] = high_part_rows[j][real_col_id];
			_padding(tmp_col,height,filter_len,padding_col,padding_len,mode);
			delete []tmp_col;


			_convfiltering(padding_col,padding_len,lowfilter,filter_len,z_img_col,column_height,CONV_VAILD);

			for(int j = 0;j < row_len;j++)
				cv_ptr[j*col_len+i] = z_img_col[2*j+1];
			delete []z_img_col;

			_convfiltering(padding_col,padding_len,highfilter,filter_len,z_img_col,column_height,CONV_VAILD);

			for(int j = 0;j < row_len;j++)
				cd_ptr[j*col_len+i] = z_img_col[2*j+1];

			delete []z_img_col;
			delete []padding_col;
		}

		for(int i = 0;i < height;i++)
			delete []high_part_rows[i];
		delete []high_part_rows;

		delete []lowfilter;
		delete []highfilter;

		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::IDWT2(const ZQ_DImage<T>& ca, const ZQ_DImage<T>& ch, const ZQ_DImage<T>& cv, const ZQ_DImage<T>& cd, ZQ_DImage<T>& output, const char* wavename, const PaddingMode mode)
	{
		int ca_width = ca.width();
		int ca_height = ca.height();
		int ca_nchannels = ca.nchannels();

		if(ca_nchannels != 1)
			return false;

		if(!ch.matchDimension(ca_width,ca_height,ca_nchannels)
			|| !cv.matchDimension(ca_width,ca_height,ca_nchannels)
			|| !cd.matchDimension(ca_width,ca_height,ca_nchannels))
			return false;

		int filterlen = 0;
		T* lowfilter = 0;
		T* highfilter = 0;
		if(!_selectfilter(wavename,filterlen,lowfilter,highfilter,true))
		{
			return false;
		}
		int upsample_width = ca_width*2;
		int upsample_height = ca_height*2;
		int output_width = upsample_width - filterlen + 2;
		int output_height = upsample_height - filterlen + 2;
		output.allocate(output_width,output_height);

		const T*& ca_ptr = ca.data();
		const T*& ch_ptr = ch.data();
		const T*& cv_ptr = cv.data();
		const T*& cd_ptr = cd.data();
		T*& output_ptr = output.data();


		int z_img_height = 2*ca_height+filterlen-2;
		int z_img_width = 2*ca_width+filterlen-2;
		int offset_z_img_x = (z_img_width-output_width)/2;
		int offset_z_img_y = (z_img_height-output_height)/2;

		T* buffer = new T[z_img_height*z_img_width];
		memset(buffer,0,sizeof(T)*z_img_height*z_img_width);

		T** low_part_cols = new T*[ca_width];
		// ca recover: first pass, vertical low filter, second pass, horizontal low filter
		for(int i = 0;i < ca_width;i++)
		{
			T* tmp_upsample = new T[upsample_height];
			for(int j = 0;j < ca_height;j++)
			{
				tmp_upsample[2*j+0] = ca_ptr[j*ca_width+i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_col = 0;
			int z_img_col_len;
			_convfiltering(tmp_upsample,upsample_height-1,lowfilter,filterlen,z_img_col,z_img_col_len,CONV_FULL);
			low_part_cols[i] = z_img_col;
			delete []tmp_upsample;
		}
		for(int i = 0;i < z_img_height;i++)
		{
			T* tmp_upsample = new T[upsample_width];
			for(int j = 0;j < ca_width;j++)
			{
				tmp_upsample[2*j+0] = low_part_cols[j][i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_row = 0;
			int z_img_row_len;
			_convfiltering(tmp_upsample,upsample_width-1,lowfilter,filterlen,z_img_row,z_img_row_len,CONV_FULL);
			for(int j = 0;j < z_img_width;j++)
			{
				buffer[i*z_img_width+j] += z_img_row[j];
			}
			delete []z_img_row;
			delete []tmp_upsample;
		}
		for(int i = 0;i < ca_width;i++)
			delete []low_part_cols[i];

		// ch recover: first pass, vertical high filter, second pass, horizontal low filter
		for(int i = 0;i < ca_width;i++)
		{
			T* tmp_upsample = new T[upsample_height];
			for(int j = 0;j < ca_height;j++)
			{
				tmp_upsample[2*j+0] = ch_ptr[j*ca_width+i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_col = 0;
			int z_img_col_len;
			_convfiltering(tmp_upsample,upsample_height-1,highfilter,filterlen,z_img_col,z_img_col_len,CONV_FULL);
			low_part_cols[i] = z_img_col;
			delete []tmp_upsample;
		}
		for(int i = 0;i < z_img_height;i++)
		{
			T* tmp_upsample = new T[upsample_width];
			for(int j = 0;j < ca_width;j++)
			{
				tmp_upsample[2*j+0] = low_part_cols[j][i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_row = 0;
			int z_img_row_len;
			_convfiltering(tmp_upsample,upsample_width-1,lowfilter,filterlen,z_img_row,z_img_row_len,CONV_FULL);
			for(int j = 0;j < z_img_width;j++)
			{
				buffer[i*z_img_width+j] += z_img_row[j];
			}
			delete []z_img_row;
			delete []tmp_upsample;
		}
		for(int i = 0;i < ca_width;i++)
			delete []low_part_cols[i];


		// cv recover: first pass, vertical low filter, second pass, horizontal high filter
		for(int i = 0;i < ca_width;i++)
		{
			T* tmp_upsample = new T[upsample_height];
			for(int j = 0;j < ca_height;j++)
			{
				tmp_upsample[2*j+0] = cv_ptr[j*ca_width+i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_col = 0;
			int z_img_col_len;
			_convfiltering(tmp_upsample,upsample_height-1,lowfilter,filterlen,z_img_col,z_img_col_len,CONV_FULL);
			low_part_cols[i] = z_img_col;
			delete []tmp_upsample;
		}
		for(int i = 0;i < z_img_height;i++)
		{
			T* tmp_upsample = new T[upsample_width];
			for(int j = 0;j < ca_width;j++)
			{
				tmp_upsample[2*j+0] = low_part_cols[j][i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_row = 0;
			int z_img_row_len;
			_convfiltering(tmp_upsample,upsample_width-1,highfilter,filterlen,z_img_row,z_img_row_len,CONV_FULL);
			for(int j = 0;j < z_img_width;j++)
			{
				buffer[i*z_img_width+j] += z_img_row[j];
			}
			delete []z_img_row;
			delete []tmp_upsample;
		}
		for(int i = 0;i < ca_width;i++)
			delete []low_part_cols[i];

		// cd recover: first pass, vertical high filter, second pass, horizontal high filter
		for(int i = 0;i < ca_width;i++)
		{
			T* tmp_upsample = new T[upsample_height];
			for(int j = 0;j < ca_height;j++)
			{
				tmp_upsample[2*j+0] = cd_ptr[j*ca_width+i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_col = 0;
			int z_img_col_len;
			_convfiltering(tmp_upsample,upsample_height-1,highfilter,filterlen,z_img_col,z_img_col_len,CONV_FULL);
			low_part_cols[i] = z_img_col;
			delete []tmp_upsample;
		}
		for(int i = 0;i < z_img_height;i++)
		{
			T* tmp_upsample = new T[upsample_width];
			for(int j = 0;j < ca_width;j++)
			{
				tmp_upsample[2*j+0] = low_part_cols[j][i];
				tmp_upsample[2*j+1] = 0;
			}
			T* z_img_row = 0;
			int z_img_row_len;
			_convfiltering(tmp_upsample,upsample_width-1,highfilter,filterlen,z_img_row,z_img_row_len,CONV_FULL);
			for(int j = 0;j < z_img_width;j++)
			{
				buffer[i*z_img_width+j] += z_img_row[j];
			}
			delete []z_img_row;
			delete []tmp_upsample;
		}
		for(int i = 0;i < ca_width;i++)
			delete []low_part_cols[i];

		delete []low_part_cols;

		for(int i = 0;i < output_height;i++)
		{
			int real_h = offset_z_img_y + i;
			for(int j = 0;j < output_width;j++)
			{
				int real_w = offset_z_img_x + j;
				output_ptr[i*output_width+j] = buffer[real_h*z_img_width+real_w];
			}
		}

		delete []buffer;
		delete []lowfilter;
		delete []highfilter;

		return true;
	}


	template<class T>
	void ZQ_Wavelet<T>::_convfiltering(const T* input, const int inputlen, const T* filter, const int filterlen, T *& output, int& outputlen, ConvolutionMode mode)
	{
		if(mode == CONV_VAILD)
		{
			outputlen = inputlen - filterlen + 1;
			output = new T[outputlen];
			for(int i = 0;i < outputlen;i++)
			{
				output[i] = 0;
				for(int j = 0;j < filterlen;j++)
				{
					output[i] += input[i+j] * filter[filterlen-1-j];
				}
			}
		}
		else if(mode == CONV_FULL)
		{
			outputlen = inputlen + filterlen - 1;
			output = new T[outputlen];
			for(int i = 0;i < outputlen;i++)
			{
				output[i] = 0;
				for(int j = 0;j < filterlen;j++)
				{
					T tmp = 0;
					if(i-filterlen+1+j >=  0 && i-filterlen+1+j < inputlen)
						tmp = input[i-filterlen+1+j] ;
					else
						tmp = 0;
					output[i] += filter[filterlen-1-j] * tmp;
				}
			}
		}
		else
		{
			printf("error:%s:(%d)\n",__FILE__,__LINE__);
		}
	}

	template<class T>
	bool ZQ_Wavelet<T>::_selectfilter(const char* wavename, int &filterlen, T*& lowfilter, T*& highfilter, bool inverseflag)
	{
		filterlen = 0;
		T* de_filter = 0;

		static T haar_data[2] = {0.70710678118654, 0.70710678118654};
		static T db2_data[4] = {-0.1294095225509, 0.2241438680419, 0.836516303737, 0.48296291314469028};
		static T db3_data[6] = {0.035226291882,-0.085441273882,-0.13501102001039,0.4598775021193312,0.8068915093133387, 0.332670552951};
		static T db4_data[8] = {-0.010597401785, 0.032883011667, 0.030841381836, -0.1870348117188811, -0.0279837694169835, 0.6308807679295903,  0.71484657055254152, 0.2303778133088552};
		static T db5_data[10] = {0.0033357252850015, -0.0125807519990155, -0.0062414902130117, 0.077571493840065, -0.03224486958502951, -0.2422948870661901,
			0.1384281459011034, 0.72430852843857438, 0.6038292697974729,  0.160102397974125023};
		//low pass decomposition filter
		if(_strcmpi(wavename, "haar") == 0 || _strcmpi(wavename,"db1") == 0)
		{
			filterlen = 2;		
			de_filter = new T[filterlen]; 
			for(int i = 0; i < filterlen; i++)
			{
				de_filter[i] = haar_data[i];
			}
		}	
		else if(_strcmpi(wavename, "db2") == 0)
		{
			filterlen = 4;
			de_filter = new T[filterlen]; 
			for(int i = 0; i < filterlen; i++)
			{
				de_filter[i] = db2_data[i];
			}
		}
		else if(_strcmpi(wavename, "db3") == 0)
		{
			filterlen = 6;
			de_filter = new T[filterlen]; 
			for(int i = 0; i < filterlen; i++)
			{
				de_filter[i] = db3_data[i];
			}
		}
		else if(_strcmpi(wavename, "db4") == 0)
		{
			filterlen = 8;
			de_filter = new T[filterlen]; 
			for(int i = 0; i < filterlen; i++)
			{
				de_filter[i] = db4_data[i];
			}
		}
		else if(_strcmpi(wavename, "db5") == 0)
		{
			filterlen = 10;
			de_filter = new T[filterlen]; 
			for(int i = 0; i < filterlen; i++)
			{
				de_filter[i] = db5_data[i];
			}
		}
		else
		{
			return false;
		}

		highfilter = new T[filterlen];

		if(!inverseflag)
		{
			for(int i = 0;i < filterlen;i++)
			{
				highfilter[i] = de_filter[filterlen - 1 - i] * (i%2 ? 1 : -1);
			}
			lowfilter = de_filter;

		}
		else
		{
			lowfilter = new T[filterlen];
			for(int i = 0;i < filterlen;i++)
			{
				lowfilter[i] = de_filter[filterlen - 1 - i];
				highfilter[i] = de_filter[i] * (i%2 ? -1 : 1);
			}
			delete []de_filter;
		}

		return true;
	}

	template<class T>
	void ZQ_Wavelet<T>::_padding(const T* input, const int input_len, const int filter_len, T*& padding_img, int& paddingimg_len, const PaddingMode mode)
	{
		paddingimg_len = input_len + 2*(filter_len-1);
		switch(mode)
		{
		case PADDING_SYM:
			{
				padding_img = new T[paddingimg_len];
				for(int i = 0;i < filter_len - 1;i++)
				{
					padding_img[filter_len - 2 - i] = input[i];
				}
				memcpy(padding_img+filter_len-1,input,sizeof(T)*input_len);
				for(int i = 0;i < filter_len - 1;i++)
				{
					padding_img[input_len + filter_len - 1 + i] = input[input_len-1-i];
				}
			}
			break;
		case PADDING_ZPD: default:
			padding_img = new T[paddingimg_len];
			memset(padding_img,0,sizeof(T)*paddingimg_len);
			memcpy(padding_img+filter_len-1,input,sizeof(T)*input_len);
			break;
		}
	}

	template<class T>
	void ZQ_Wavelet<T>::_mergesort(T* vals,int start, int end)
	{
		if(start >= end)
			return ;

		int mid = (start+end)/2;
		int left_len = mid-start+1;
		int right_len = end-mid;
		T* tmp_left = new T[left_len];
		T* tmp_right = new T[right_len];
		for(int i = 0;i < left_len;i++)
			tmp_left[i] = vals[start+i];
		for(int i = 0;i < right_len;i++)
			tmp_right[i] = vals[mid+1+i];

		_mergesort(tmp_left,0,left_len-1);
		_mergesort(tmp_right,0,right_len-1);

		int i_idx = 0;
		int j_idx = 0;
		int k_idx = 0;
		for(;i_idx < left_len && j_idx < right_len;)
		{
			if(tmp_left[i_idx] > tmp_right[j_idx])
			{
				vals[start+k_idx] = tmp_left[i_idx];
				i_idx++;
				k_idx++;
			}
			else
			{
				vals[start+k_idx] = tmp_right[j_idx];
				j_idx++;
				k_idx++;
			}
		}
		if(i_idx == left_len)
		{
			for(;j_idx < right_len;j_idx++,k_idx++)
				vals[start+k_idx] = tmp_right[j_idx];
		}
		else
		{
			for(; i_idx < left_len;i_idx++,k_idx++)
				vals[start+k_idx] = tmp_left[i_idx];
		}
		delete []tmp_left;
		delete []tmp_right;
		return ;
	}

	template<class T>
	bool ZQ_Wavelet<T>::LoadWaveletFile(const char* file, ZQ_DImage<T>& image)
	{
		int width,height,nchannels;
		PaddingMode pad_mode;
		char wavename[16];
		int levels;
		int min_width,min_height;

		const short max_short_val = 32767;

		FILE* in = fopen(file,"rb");
		fread(&width,sizeof(int),1,in);
		fread(&height,sizeof(int),1,in);
		int depth = 1;
		fread(&depth,sizeof(int),1,in);	//ignored for 2D image
		fread(&nchannels,sizeof(int),1,in);

		fread(&pad_mode,sizeof(PaddingMode),1,in);
		fread(wavename,sizeof(char),16,in);
		fread(&levels,sizeof(int),1,in);
		fread(&min_width,sizeof(int),1,in);
		fread(&min_height,sizeof(int),1,in);

		if(width <= 0 || height <= 0 || width > 32767 || height > 32767 || nchannels <= 0)
		{
			printf("invalid image resolution : width = %d, height = %d, nchannels = %d\n",width,height,nchannels);
			fclose(in);
			return false;
		}

		std::vector<ZQ_DImage<T>> image_each_channel;

		int wave_image_width = min_width;
		int wave_image_height = min_height;
		for(int i = 1;i <= levels;i++)
		{
			wave_image_width *= 2;
			wave_image_height *= 2;
		}

		T* wave_image = new T[wave_image_width*wave_image_height];

		//each channel is separately stored
		for(int c = 0;c < nchannels;c++)
		{
			int num;
			unsigned short w;
			short val_sh;
			float scale;
			float val;

			memset(wave_image,0,sizeof(T)*wave_image_width*wave_image_height);
			fread(&scale,sizeof(float),1,in);
			for(int h = 0;h < wave_image_height;h++)
			{
				fread(&num,sizeof(int),1,in);
				for(int nn = 0;nn < num;nn++)
				{
					fread(&w,sizeof(unsigned short),1,in);
					fread(&val_sh,sizeof(short),1,in);
					val = val_sh * scale / max_short_val;
					if(h < 0 || h >= wave_image_height || w < 0 || w >= wave_image_width)
					{
						printf("invalid wavelet image\n");
						delete []wave_image;
						fclose(in);
						return false;
					}
					wave_image[h*wave_image_width+w] = val;
				}
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
				if(!IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
				{
					printf("invalid wavelet image\n");
					delete []wave_image;
					fclose(in);
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

		image.allocate(width,height,nchannels);
		T*& image_ptr = image.data();
		for(int c = 0;c < nchannels;c++)
		{
			T*& image_each_channel_ptr = image_each_channel[c].data();
			for(int h = 0;h < height && h < wave_image_height; h++)
			{
				for(int w = 0;w < width && w < wave_image_width; w++)
				{
					image_ptr[(h*width+w)*nchannels+c] = image_each_channel_ptr[h*wave_image_width+w];
				}
			}
		}


		fclose(in);
		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::LoadWaveletFile(const char* file, ZQ_DImage3D<T>& image)
	{
		int width,height,depth,nchannels;
		PaddingMode pad_mode;
		char wavename[16];
		int levels;
		int min_width,min_height;

		const short max_short_val = 32767;

		FILE* in = fopen(file,"rb");
		fread(&width,sizeof(int),1,in);
		fread(&height,sizeof(int),1,in);
		fread(&depth,sizeof(int),1,in);	
		fread(&nchannels,sizeof(int),1,in);

		fread(&pad_mode,sizeof(PaddingMode),1,in);
		fread(wavename,sizeof(char),16,in);
		fread(&levels,sizeof(int),1,in);
		fread(&min_width,sizeof(int),1,in);
		fread(&min_height,sizeof(int),1,in);

		if(width <= 0 || height <= 0 || width > 32767 || height > 32767 || nchannels <= 0)
		{
			printf("invalid image resolution : width = %d, height = %d, nchannels = %d\n",width,height,nchannels);
			fclose(in);
			return false;
		}

		std::vector<ZQ_DImage<T>> image_each_channel;

		int wave_image_width = min_width;
		int wave_image_height = min_height;
		for(int i = 1;i <= levels;i++)
		{
			wave_image_width *= 2;
			wave_image_height *= 2;
		}

		T* wave_image = new T[wave_image_width*wave_image_height];

		//each channel is separately stored
		for(int kk = 0;kk < depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				int num;
				unsigned short w;
				short val_sh;
				float scale;
				float val;

				memset(wave_image,0,sizeof(T)*wave_image_width*wave_image_height);
				fread(&scale,sizeof(float),1,in);
				for(int h = 0;h < wave_image_height;h++)
				{
					fread(&num,sizeof(int),1,in);
					for(int nn = 0;nn < num;nn++)
					{
						fread(&w,sizeof(unsigned short),1,in);
						fread(&val_sh,sizeof(short),1,in);
						val = val_sh * scale / max_short_val;
						if(h < 0 || h >= wave_image_height || w < 0 || w >= wave_image_width)
						{
							printf("invalid wavelet image\n");
							delete []wave_image;
							fclose(in);
							return false;
						}
						wave_image[h*wave_image_width+w] = val;
					}
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
					if(!IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
					{
						printf("invalid wavelet image\n");
						fclose(in);
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
		}


		delete []wave_image;

		image.allocate(width,height,depth,nchannels);
		T*& image_ptr = image.data();
		for(int kk = 0;kk < depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				T*& image_each_channel_ptr = image_each_channel[kk*nchannels+c].data();
				for(int h = 0;h < height && h < wave_image_height; h++)
				{
					for(int w = 0;w < width && w < wave_image_width; w++)
					{
						image_ptr[(kk*height*width+h*width+w)*nchannels+c] = image_each_channel_ptr[h*wave_image_width+w];
					}
				}
			}
		}

		fclose(in);
		return true;
	}


	template<class T>
	bool ZQ_Wavelet<T>::LoadWaveletFromBytes(const int input_len, const unsigned char* input_bytes, ZQ_DImage<T>& image)
	{
		if(input_len <= 0 || input_bytes == 0)
			return false;

		int width,height,nchannels;
		PaddingMode pad_mode;
		char wavename[16];
		int levels;
		int min_width,min_height;

		const short max_short_val = 32767;

		const unsigned char* bytes_ptr = input_bytes;
		memcpy(&width,bytes_ptr,sizeof(int)); 	bytes_ptr += sizeof(int);
		memcpy(&height,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		int depth = 1;
		memcpy(&depth,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int); //ignored for 2D image
		memcpy(&nchannels,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&pad_mode,bytes_ptr,sizeof(PaddingMode));	bytes_ptr += sizeof(int);
		memcpy(wavename,bytes_ptr,sizeof(char)*16);		bytes_ptr += 16;
		memcpy(&levels,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&min_width,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&min_height,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);

		if(width <= 0 || height <= 0 || width > 32767 || height > 32767 || nchannels <= 0)
		{
			printf("invalid image resolution : width = %d, height = %d, nchannels = %d\n",width,height,nchannels);
			return false;
		}

		std::vector<ZQ_DImage<T>> image_each_channel;

		int wave_image_width = min_width;
		int wave_image_height = min_height;
		for(int i = 1;i <= levels;i++)
		{
			wave_image_width *= 2;
			wave_image_height *= 2;
		}

		T* wave_image = new T[wave_image_width*wave_image_height];

		//each channel is separately stored
		for(int c = 0;c < nchannels;c++)
		{
			int num;
			unsigned short w;
			short val_sh;
			float scale;
			float val;

			memset(wave_image,0,sizeof(T)*wave_image_width*wave_image_height);

			memcpy(&scale,bytes_ptr,sizeof(float));		bytes_ptr += sizeof(float);

			for(int h = 0;h < wave_image_height;h++)
			{
				memcpy(&num,bytes_ptr,sizeof(int));		bytes_ptr += sizeof(int);
				for(int nn = 0;nn < num;nn++)
				{
					memcpy(&w,bytes_ptr,sizeof(unsigned short));	bytes_ptr += sizeof(unsigned short);
					memcpy(&val_sh,bytes_ptr,sizeof(short));	bytes_ptr += sizeof(short);
					val = val_sh * scale / max_short_val;
					if(h < 0 || h >= wave_image_height || w < 0 || w >= wave_image_width)
					{
						printf("invalid wavelet image\n");
						delete []wave_image;
						return false;
					}

					wave_image[h*wave_image_width+w] = val;
				}
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
				if(!IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
				{
					printf("invalid wavelet image\n");
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

		image.allocate(width,height,nchannels);
		T*& image_ptr = image.data();
		for(int c = 0;c < nchannels;c++)
		{
			T*& image_each_channel_ptr = image_each_channel[c].data();
			for(int h = 0;h < height && h < wave_image_height; h++)
			{
				for(int w = 0;w < width && w < wave_image_width; w++)
				{
					image_ptr[(h*width+w)*nchannels+c] = image_each_channel_ptr[h*wave_image_width+w];
				}
			}
		}

		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::LoadWaveletFromBytes(const int input_len, const unsigned char* input_bytes, ZQ_DImage3D<T>& image)
	{
		if(input_len <= 0 || input_bytes == 0)
			return false;

		int width,height,depth,nchannels;
		PaddingMode pad_mode;
		char wavename[16];
		int levels;
		int min_width,min_height;

		const short max_short_val = 32767;

		const unsigned char* bytes_ptr = input_bytes;
		memcpy(&width,bytes_ptr,sizeof(int)); 	bytes_ptr += sizeof(int);
		memcpy(&height,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&depth,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int); 
		memcpy(&nchannels,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&pad_mode,bytes_ptr,sizeof(PaddingMode));	bytes_ptr += sizeof(int);
		memcpy(wavename,bytes_ptr,sizeof(char)*16);		bytes_ptr += 16;
		memcpy(&levels,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&min_width,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(&min_height,bytes_ptr,sizeof(int));	bytes_ptr += sizeof(int);

		if(width <= 0 || height <= 0 || width > 32767 || height > 32767 || nchannels <= 0)
		{
			printf("invalid image resolution : width = %d, height = %d, nchannels = %d\n",width,height,nchannels);
			return false;
		}

		std::vector<ZQ_DImage<T>> image_each_channel;

		int wave_image_width = min_width;
		int wave_image_height = min_height;
		for(int i = 1;i <= levels;i++)
		{
			wave_image_width *= 2;
			wave_image_height *= 2;
		}

		T* wave_image = new T[wave_image_width*wave_image_height];

		//each channel is separately stored
		for(int kk = 0;kk < depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				int num;
				unsigned short w;
				short val_sh;
				float scale;
				float val;

				memset(wave_image,0,sizeof(T)*wave_image_width*wave_image_height);

				memcpy(&scale,bytes_ptr,sizeof(float));		bytes_ptr += sizeof(float);

				for(int h = 0;h < wave_image_height;h++)
				{
					memcpy(&num,bytes_ptr,sizeof(int));		bytes_ptr += sizeof(int);
					for(int nn = 0;nn < num;nn++)
					{
						memcpy(&w,bytes_ptr,sizeof(unsigned short));	bytes_ptr += sizeof(unsigned short);
						memcpy(&val_sh,bytes_ptr,sizeof(short));	bytes_ptr += sizeof(short);
						val = val_sh * scale / max_short_val;
						if(h < 0 || h >= wave_image_height || w < 0 || w >= wave_image_width)
						{
							printf("invalid wavelet image\n");
							delete []wave_image;
							return false;
						}

						wave_image[h*wave_image_width+w] = val;
					}
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
					if(!IDWT2(ca,ch,cv,cd,tmp_out,wavename,pad_mode))
					{
						printf("invalid wavelet image\n");
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
		}


		delete []wave_image;

		image.allocate(width,height,depth,nchannels);
		T*& image_ptr = image.data();
		for(int kk = 0;kk < depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				T*& image_each_channel_ptr = image_each_channel[kk*nchannels+c].data();
				for(int h = 0;h < height && h < wave_image_height; h++)
				{
					for(int w = 0;w < width && w < wave_image_width; w++)
					{
						image_ptr[(kk*height*width+h*width+w)*nchannels+c] = image_each_channel_ptr[h*wave_image_width+w];
					}
				}
			}
		}


		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::SaveWaveletFile(const char* file, const ZQ_DImage<T>& image, const double ratio /*= 50*/, const double min_quality /* = 0.9999 */)
	{
		T mQuality = __min(0.9999,__max(0.80,min_quality));
		T mRatio = __max(1,ratio);

		int image_width = image.width();
		int image_height = image.height();
		int nchannels = image.nchannels();

		const T*& image_ptr = image.data();

		int min_resolution = __min(image_width,image_height);

		char wavename[16] = "db1";

		const int max_levels = 8;
		const short max_short_val = 32767;

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

		FILE* out = fopen(file,"wb");
		if(out == 0)
		{
			return false;
		}

		fwrite(&image_width,sizeof(int),1,out);
		fwrite(&image_height,sizeof(int),1,out);
		int image_depth = 1;
		fwrite(&image_depth,sizeof(int),1,out);//ignored for 2D image
		fwrite(&nchannels,sizeof(int),1,out);
		PaddingMode pad_mod = PADDING_ZPD;
		fwrite(&pad_mod,sizeof(PaddingMode),1,out);
		fwrite(wavename,sizeof(char),16,out);
		fwrite(&levels,sizeof(int),1,out);
		fwrite(&min_width,sizeof(int),1,out);
		fwrite(&min_height,sizeof(int),1,out);

		ZQ_Wavelet m_wave;
		for(int c = 0;c < nchannels;c++)
		{
			ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
			T*& image_each_channel_ptr = image_each_channel.data();
			for(int h = 0;h < image_height;h++)
			{
				for(int w = 0;w < image_width;w++)
				{
					image_each_channel_ptr[h*padding_width+w] = image_ptr[(h*image_width+w)*nchannels+c];
				}
			}
			if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
			{
				fclose(out);
				return false;
			}
			ZQ_DImage<T> output;
			m_wave.GetWaveletImage(output);
			int output_nelements = output.nelements();
			T*& output_ptr = output.data();

			T* coeffs = new T[output_nelements];
			T total_energy = 0;
			for(int iii = 0; iii < output_nelements;iii++)
			{
				coeffs[iii] = fabs(output_ptr[iii]);
				total_energy += coeffs[iii]*coeffs[iii];
			}

			_mergesort(coeffs,0,output_nelements-1);

			float scale = coeffs[0];

			T sum = 0;
			T sum_threshold = total_energy*mQuality; 
			T threshold = 0;
			for(int iii = 0;iii < output_nelements;iii++)
			{
				threshold = coeffs[iii];
				sum += threshold*threshold;
				if(sum >= sum_threshold && iii * ratio >= output_nelements)
					break;
			}

			delete []coeffs;

			for(int iii = 0; iii < output_nelements;iii++)
			{
				if(fabs(output_ptr[iii]) < threshold)
				{
					output_ptr[iii] = 0;
				}
				else
				{
					output_ptr[iii] = (short)(output_ptr[iii]/scale*max_short_val);
				}
			}

			fwrite(&scale,sizeof(float),1,out);

			int output_width = output.width();
			int output_height = output.height();
			for(int hh = 0;hh < output_height;hh++)
			{
				int cur_nnz = 0;
				for(int ww = 0;ww < output_width;ww++)
				{
					if(output_ptr[hh*output_width+ww] != 0)
						cur_nnz++;
				}
				fwrite(&cur_nnz,sizeof(int),1,out);
				for(int ww = 0;ww < output_width;ww++)
				{
					if(output_ptr[hh*output_width+ww] != 0)
					{
						unsigned short th = hh;
						unsigned short tw = ww;
						short val = output_ptr[hh*output_width+ww];
						fwrite(&tw,sizeof(unsigned short),1,out);
						fwrite(&val,sizeof(short),1,out);
					}
				}
			}
		}

		fclose(out);
		return true;
	}


	template<class T>
	bool ZQ_Wavelet<T>::SaveWaveletFile(const char* file, const ZQ_DImage3D<T>& image, const double ratio /*= 50*/, const double min_quality /* = 0.9999 */)
	{
		T mQuality = __min(0.9999,__max(0.80,min_quality));
		T mRatio = __max(1,ratio);

		int image_width = image.width();
		int image_height = image.height();
		int image_depth = image.depth();
		int nchannels = image.nchannels();

		const T*& image_ptr = image.data();

		int min_resolution = __min(image_width,image_height);

		char wavename[16] = "db1";

		const int max_levels = 8;
		const short max_short_val = 32767;

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

		FILE* out = fopen(file,"wb");
		if(out == 0)
		{
			return false;
		}

		fwrite(&image_width,sizeof(int),1,out);
		fwrite(&image_height,sizeof(int),1,out);
		fwrite(&image_depth,sizeof(int),1,out);
		fwrite(&nchannels,sizeof(int),1,out);
		PaddingMode pad_mod = PADDING_ZPD;
		fwrite(&pad_mod,sizeof(PaddingMode),1,out);
		fwrite(wavename,sizeof(char),16,out);
		fwrite(&levels,sizeof(int),1,out);
		fwrite(&min_width,sizeof(int),1,out);
		fwrite(&min_height,sizeof(int),1,out);

		ZQ_Wavelet m_wave;
		for(int kk = 0;kk < image_depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
				T*& image_each_channel_ptr = image_each_channel.data();
				for(int h = 0;h < image_height;h++)
				{
					for(int w = 0;w < image_width;w++)
					{
						image_each_channel_ptr[h*padding_width+w] = image_ptr[(kk*image_height*image_width+h*image_width+w)*nchannels+c];
					}
				}
				if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
				{
					fclose(out);
					return false;
				}
				ZQ_DImage<T> output;
				m_wave.GetWaveletImage(output);
				int output_nelements = output.nelements();
				T*& output_ptr = output.data();

				T* coeffs = new T[output_nelements];
				T total_energy = 0;
				for(int iii = 0; iii < output_nelements;iii++)
				{
					coeffs[iii] = fabs(output_ptr[iii]);
					total_energy += coeffs[iii]*coeffs[iii];
				}

				_mergesort(coeffs,0,output_nelements-1);

				float scale = coeffs[0];

				T sum = 0;
				T sum_threshold = total_energy*mQuality; 
				T threshold = 0;
				for(int iii = 0;iii < output_nelements;iii++)
				{
					threshold = coeffs[iii];
					sum += threshold*threshold;
					if(sum >= sum_threshold && iii * ratio >= output_nelements)
						break;
				}

				delete []coeffs;

				for(int iii = 0; iii < output_nelements;iii++)
				{
					if(fabs(output_ptr[iii]) < threshold)
					{
						output_ptr[iii] = 0;
					}
					else
					{
						output_ptr[iii] = (short)(output_ptr[iii]/scale*max_short_val);
					}
				}

				fwrite(&scale,sizeof(float),1,out);

				int output_width = output.width();
				int output_height = output.height();
				for(int hh = 0;hh < output_height;hh++)
				{
					int cur_nnz = 0;
					for(int ww = 0;ww < output_width;ww++)
					{
						if(output_ptr[hh*output_width+ww] != 0)
							cur_nnz++;
					}
					fwrite(&cur_nnz,sizeof(int),1,out);
					for(int ww = 0;ww < output_width;ww++)
					{
						if(output_ptr[hh*output_width+ww] != 0)
						{
							unsigned short th = hh;
							unsigned short tw = ww;
							short val = output_ptr[hh*output_width+ww];
							fwrite(&tw,sizeof(unsigned short),1,out);
							fwrite(&val,sizeof(short),1,out);
						}
					}
				}
			}
		}


		fclose(out);
		return true;
	}

	template<class T>
	bool ZQ_Wavelet<T>::SaveWaveletToBytes(int& output_len, unsigned char*& output_bytes, const ZQ_DImage<T>& image, const double ratio /* = 50 */, const double min_quality /* = 0.999 */)
	{
		T mQuality = __min(0.9999,__max(0.80,min_quality));
		T mRatio = __max(1,ratio);

		int image_width = image.width();
		int image_height = image.height();
		int nchannels = image.nchannels();

		const T*& image_ptr = image.data();

		int min_resolution = __min(image_width,image_height);

		char wavename[16] = "db1";

		const int max_levels = 8;
		const short max_short_val = 32767;

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

		// compute max_buffer_len
		int buffer_head_len = sizeof(int)*3 + sizeof(PaddingMode) + sizeof(char)*16 + sizeof(int)*3;
		int buffer_max_len = buffer_head_len + nchannels*(sizeof(int)+padding_width*padding_height*(2*sizeof(unsigned short)+sizeof(T)));
		unsigned char* buffer_output = new unsigned char[buffer_max_len];
		unsigned char* bytes_ptr = buffer_output;

		memcpy(bytes_ptr,&image_width,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&image_height,sizeof(int));	bytes_ptr += sizeof(int);
		int image_depth = 1;
		memcpy(bytes_ptr,&image_depth,sizeof(int));	bytes_ptr += sizeof(int);	//ignored for 2D image
		memcpy(bytes_ptr,&nchannels,sizeof(int));	bytes_ptr += sizeof(int);
		PaddingMode pad_mod = PADDING_ZPD;
		memcpy(bytes_ptr,&pad_mod,sizeof(PaddingMode));	bytes_ptr += sizeof(PaddingMode);
		memcpy(bytes_ptr,wavename,sizeof(char)*16);	bytes_ptr += 16;
		memcpy(bytes_ptr,&levels,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&min_width,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&min_height,sizeof(int));	bytes_ptr += sizeof(int);



		ZQ_Wavelet m_wave;
		for(int c = 0;c < nchannels;c++)
		{
			ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
			T*& image_each_channel_ptr = image_each_channel.data();
			for(int h = 0;h < image_height;h++)
			{
				for(int w = 0;w < image_width;w++)
				{
					image_each_channel_ptr[h*padding_width+w] = image_ptr[(h*image_width+w)*nchannels+c];
				}
			}
			if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
			{
				delete []buffer_output;
				return false;
			}
			ZQ_DImage<T> output;
			m_wave.GetWaveletImage(output);
			int output_nelements = output.nelements();
			T*& output_ptr = output.data();

			T* coeffs = new T[output_nelements];
			T total_energy = 0;
			for(int iii = 0; iii < output_nelements;iii++)
			{
				coeffs[iii] = fabs(output_ptr[iii]);
				total_energy += coeffs[iii]*coeffs[iii];
			}

			_mergesort(coeffs,0,output_nelements-1);

			float scale = fabs(coeffs[0]);

			T sum = 0;
			T sum_threshold = total_energy*mQuality; 
			T threshold = 0;
			for(int iii = 0;iii < output_nelements;iii++)
			{
				threshold = coeffs[iii];
				sum += threshold*threshold;
				if(sum >= sum_threshold && iii * ratio >= output_nelements)
					break;
			}
			delete []coeffs;

			for(int iii = 0; iii < output_nelements;iii++)
			{
				if(fabs(output_ptr[iii]) < threshold)
				{
					output_ptr[iii] = 0;
				}
				else
				{
					output_ptr[iii] = (short)(output_ptr[iii] / scale * max_short_val);
				}
			}

			memcpy(bytes_ptr,&scale,sizeof(float));	bytes_ptr += sizeof(float);

			int output_width = output.width();
			int output_height = output.height();

			for(int hh = 0;hh < output_height;hh++)
			{
				int cur_nnz = 0;
				for(int ww = 0;ww < output_width;ww++)
				{
					if(output_ptr[hh*output_width+ww] != 0)
						cur_nnz++;
				}
				memcpy(bytes_ptr,&cur_nnz,sizeof(int));	bytes_ptr += sizeof(int);

				for(int ww = 0;ww < output_width;ww++)
				{
					if(output_ptr[hh*output_width+ww] != 0)
					{
						unsigned short tw = ww;
						short val = output_ptr[hh*output_width+ww];
						memcpy(bytes_ptr,&tw,sizeof(unsigned short));	bytes_ptr += sizeof(unsigned short);
						memcpy(bytes_ptr,&val,sizeof(short));		bytes_ptr += sizeof(short);
					}
				}
			}
		}

		output_len = bytes_ptr - buffer_output;
		output_bytes = new unsigned char[output_len];
		memcpy(output_bytes,buffer_output,sizeof(unsigned char)*output_len);
		delete []buffer_output;
		return true;
	}


	template<class T>
	bool ZQ_Wavelet<T>::SaveWaveletToBytes(int& output_len, unsigned char*& output_bytes, const ZQ_DImage3D<T>& image, const double ratio /* = 50 */, const double min_quality /* = 0.999 */)
	{
		T mQuality = __min(0.9999,__max(0.80,min_quality));
		T mRatio = __max(1,ratio);

		int image_width = image.width();
		int image_height = image.height();
		int image_depth = image.depth();
		int nchannels = image.nchannels();

		const T*& image_ptr = image.data();

		int min_resolution = __min(image_width,image_height);

		char wavename[16] = "db1";

		const int max_levels = 8;
		const short max_short_val = 32767;

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

		// compute max_buffer_len
		int buffer_head_len = sizeof(int)*3 + sizeof(PaddingMode) + sizeof(char)*16 + sizeof(int)*3;
		int buffer_max_len = buffer_head_len + image_depth*nchannels*(sizeof(int)+padding_width*padding_height*(2*sizeof(unsigned short)+sizeof(T)));
		unsigned char* buffer_output = new unsigned char[buffer_max_len];
		unsigned char* bytes_ptr = buffer_output;

		memcpy(bytes_ptr,&image_width,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&image_height,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&image_depth,sizeof(int));	bytes_ptr += sizeof(int);	
		memcpy(bytes_ptr,&nchannels,sizeof(int));	bytes_ptr += sizeof(int);
		PaddingMode pad_mod = PADDING_ZPD;
		memcpy(bytes_ptr,&pad_mod,sizeof(PaddingMode));	bytes_ptr += sizeof(PaddingMode);
		memcpy(bytes_ptr,wavename,sizeof(char)*16);	bytes_ptr += 16;
		memcpy(bytes_ptr,&levels,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&min_width,sizeof(int));	bytes_ptr += sizeof(int);
		memcpy(bytes_ptr,&min_height,sizeof(int));	bytes_ptr += sizeof(int);



		ZQ_Wavelet m_wave;
		for(int kk = 0;kk < image_depth;kk++)
		{
			for(int c = 0;c < nchannels;c++)
			{
				ZQ_DImage<T> image_each_channel(padding_width,padding_height,1);
				T*& image_each_channel_ptr = image_each_channel.data();
				for(int h = 0;h < image_height;h++)
				{
					for(int w = 0;w < image_width;w++)
					{
						image_each_channel_ptr[h*padding_width+w] = image_ptr[(kk*image_height*image_width+h*image_width+w)*nchannels+c];
					}
				}
				if(!m_wave.DiscreteWaveletImageNLevels(image_each_channel,wavename,levels,pad_mod))
				{
					delete []buffer_output;
					return false;
				}
				ZQ_DImage<T> output;
				m_wave.GetWaveletImage(output);
				int output_nelements = output.nelements();
				T*& output_ptr = output.data();

				T* coeffs = new T[output_nelements];
				T total_energy = 0;
				for(int iii = 0; iii < output_nelements;iii++)
				{
					coeffs[iii] = fabs(output_ptr[iii]);
					total_energy += coeffs[iii]*coeffs[iii];
				}

				_mergesort(coeffs,0,output_nelements-1);

				float scale = fabs(coeffs[0]);

				T sum = 0;
				T sum_threshold = total_energy*mQuality; 
				T threshold = 0;
				for(int iii = 0;iii < output_nelements;iii++)
				{
					threshold = coeffs[iii];
					sum += threshold*threshold;
					if(sum >= sum_threshold && iii * ratio >= output_nelements)
						break;
				}
				delete []coeffs;

				for(int iii = 0; iii < output_nelements;iii++)
				{
					if(fabs(output_ptr[iii]) < threshold)
					{
						output_ptr[iii] = 0;
					}
					else
					{
						output_ptr[iii] = (short)(output_ptr[iii] / scale * max_short_val);
					}
				}

				memcpy(bytes_ptr,&scale,sizeof(float));	bytes_ptr += sizeof(float);

				int output_width = output.width();
				int output_height = output.height();

				for(int hh = 0;hh < output_height;hh++)
				{
					int cur_nnz = 0;
					for(int ww = 0;ww < output_width;ww++)
					{
						if(output_ptr[hh*output_width+ww] != 0)
							cur_nnz++;
					}
					memcpy(bytes_ptr,&cur_nnz,sizeof(int));	bytes_ptr += sizeof(int);

					for(int ww = 0;ww < output_width;ww++)
					{
						if(output_ptr[hh*output_width+ww] != 0)
						{
							unsigned short tw = ww;
							short val = output_ptr[hh*output_width+ww];
							memcpy(bytes_ptr,&tw,sizeof(unsigned short));	bytes_ptr += sizeof(unsigned short);
							memcpy(bytes_ptr,&val,sizeof(short));		bytes_ptr += sizeof(short);
						}
					}
				}
			}
		}

		output_len = bytes_ptr - buffer_output;
		output_bytes = new unsigned char[output_len];
		memcpy(output_bytes,buffer_output,sizeof(unsigned char)*output_len);
		delete []buffer_output;
		return true;
	}
}

#endif