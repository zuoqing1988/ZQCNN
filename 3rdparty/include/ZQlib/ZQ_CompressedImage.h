#ifndef _ZQ_COMPRESSED_IMAGE_H_
#define _ZQ_COMPRESSED_IMAGE_H_
#pragma once

#include "ZQ_Huffman.h"
#include "ZQ_DoubleImage.h"
#include "ZQ_DoubleImage3D.h"
#include "ZQ_Wavelet.h"

namespace ZQ
{
	class ZQ_CompressedImage
	{
	public:
		template<class T>
		static bool LoadCompressedImage(const char* file, ZQ::ZQ_DImage<T>& img);

		template<class T>
		static bool LoadCompressedImage(const char* file, ZQ::ZQ_DImage3D<T>& img);

		template<class T>
		static bool SaveCompressedImage(const char* file, const ZQ::ZQ_DImage<T>& img, const double ratio = 50, const double min_quality = 0.999);

		template<class T>
		static bool SaveCompressedImage(const char* file, const ZQ::ZQ_DImage3D<T>& img, const double ratio = 50, const double min_quality = 0.999);
	};

	/********************* definitions **********************/

	template<class T>
	bool ZQ_CompressedImage::LoadCompressedImage(const char* file, ZQ_DImage<T>& img)
	{
		FILE* in = fopen(file,"rb");
		if(in == 0)
			return false;
		fseek(in,0,SEEK_END);
		int file_len = ftell(in);
		fseek(in,0,SEEK_SET);
		unsigned char* bytes = new unsigned char[file_len];
		fread(bytes,sizeof(unsigned char),file_len,in);
		fclose(in);

		unsigned char* wave_bytes = 0;
		unsigned long wave_len = 0;
		if(!ZQ_HuffmanEndec::ZQ_HuffmanDecodeByteStream(bytes,file_len,&wave_bytes,&wave_len))
		{

			delete []bytes;
			return false;
		}

		bool flag = ZQ_Wavelet<T>::LoadWaveletFromBytes(wave_len,wave_bytes,img);
		delete []bytes;
		delete []wave_bytes;
		return flag;
	}



	template<class T>
	bool ZQ_CompressedImage::LoadCompressedImage(const char* file, ZQ_DImage3D<T>& img)
	{
		FILE* in = fopen(file,"rb");
		if(in == 0)
			return false;
		fseek(in,0,SEEK_END);
		int file_len = ftell(in);
		fseek(in,0,SEEK_SET);
		unsigned char* bytes = new unsigned char[file_len];
		fread(bytes,sizeof(unsigned char),file_len,in);
		fclose(in);

		unsigned char* wave_bytes = 0;
		unsigned long wave_len = 0;
		if(!ZQ_HuffmanEndec::ZQ_HuffmanDecodeByteStream(bytes,file_len,&wave_bytes,&wave_len))
		{

			delete []bytes;
			return false;
		}

		bool flag = ZQ_Wavelet<T>::LoadWaveletFromBytes(wave_len,wave_bytes,img);
		delete []bytes;
		delete []wave_bytes;
		return flag;
	}


	template<class T>
	bool ZQ_CompressedImage::SaveCompressedImage(const char* file, const ZQ_DImage<T>& img, const double ratio /*= 50*/, const double min_quality /*= 0.999*/)
	{
		int wave_len = 0;
		unsigned char* wave_bytes = 0;
		if(!ZQ_Wavelet<T>::SaveWaveletToBytes(wave_len,wave_bytes,img,ratio,min_quality))
			return false;

		unsigned char* output_bytes = 0;
		unsigned long output_len = 0;

		if(!ZQ_HuffmanEndec::ZQ_HuffmanEncodeByteStream(wave_bytes,wave_len,&output_bytes,&output_len))
		{
			delete []wave_bytes;
			return false;
		}
		delete []wave_bytes;

		FILE* out = fopen(file,"wb");
		if(out == 0)
		{
			delete []output_bytes;
			return false;
		}

		fwrite(output_bytes,sizeof(unsigned char),output_len,out);
		fclose(out);
		delete []output_bytes;
		return true;
	}


	template<class T>
	bool ZQ_CompressedImage::SaveCompressedImage(const char* file, const ZQ_DImage3D<T>& img, const double ratio /*= 50*/, const double min_quality /*= 0.999*/)
	{
		int wave_len = 0;
		unsigned char* wave_bytes = 0;
		if(!ZQ_Wavelet<T>::SaveWaveletToBytes(wave_len,wave_bytes,img,ratio,min_quality))
			return false;

		unsigned char* output_bytes = 0;
		unsigned long output_len = 0;

		if(!ZQ_HuffmanEndec::ZQ_HuffmanEncodeByteStream(wave_bytes,wave_len,&output_bytes,&output_len))
		{
			delete []wave_bytes;
			return false;
		}
		delete []wave_bytes;

		FILE* out = fopen(file,"wb");
		if(out == 0)
		{
			delete []output_bytes;
			return false;
		}

		fwrite(output_bytes,sizeof(unsigned char),output_len,out);
		fclose(out);
		delete []output_bytes;
		return true;
	}
}



#endif