#ifndef _ZQ_DOUBLE_IMAGE_H_
#define _ZQ_DOUBLE_IMAGE_H_
#pragma once

#include "ZQ_ImageProcessing.h"
#include <stdio.h>
#include <typeinfo>
#include <malloc.h>

namespace ZQ
{
	template<class T>
	class ZQ_DImage
	{
	public:
		T* pData;
	protected:
		int imWidth, imHeight, nChannels;
		int nPixels, nElements;
	public:
		ZQ_DImage(void);
		ZQ_DImage(int width, int height, int nchannels = 1);
		ZQ_DImage(const ZQ_DImage& other);
		~ZQ_DImage(void);

		void computeDimension(){nPixels = imWidth*imHeight ;nElements = nPixels*nChannels;}
		void allocate(int width, int height, int nchannels=1);
		void allocate(const ZQ_DImage &other);
		void clear();
		void reset();
		void copyData(const ZQ_DImage& other);
		ZQ_DImage& operator=(const ZQ_DImage& other);

		void swap(ZQ_DImage& other)
		{
			int tmp;
			tmp = imWidth;	imWidth = other.imWidth;	other.imWidth = tmp;
			tmp = imHeight;	imHeight = other.imHeight;	other.imHeight = tmp;
			tmp = nChannels;	nChannels = other.nChannels;	other.nChannels = tmp;
			tmp = nPixels;	nPixels = other.nPixels;	other.nPixels = tmp;
			tmp = nElements;	nElements = other.nElements;	other.nElements = tmp;
			T* tmp_ptr = pData;	pData = other.pData;	other.pData = tmp_ptr;
		}

		T immax() const
		{
			if(nElements == 0)
				return 0;

			return ZQ_ImageProcessing::ImageMaxValue(pData, imWidth, imHeight, nChannels, false);
		};

		T immin() const
		{
			if (nElements == 0)
				return 0;
			return ZQ_ImageProcessing::ImageMinValue(pData, imWidth, imHeight, nChannels, false);
		}

		void imclamp(T min_val, T max_val)
		{
			ZQ_ImageProcessing::ImageClamp(pData, min_val, max_val, imWidth, imHeight, nChannels, false);
		}

		// function to access the member variables
		T*& data(){return pData;}
		const T*& data() const{return (const T*&)pData;}
		int width() const {return imWidth;}
		int height() const {return imHeight;}
		int nchannels() const {return nChannels;}
		int npixels() const {return nPixels;}
		int nelements() const {return nElements;}
		bool IsEmpty() const {if(nElements == 0) return true;else return false;}
		bool matchDimension  (const ZQ_DImage& image) const;
		bool matchDimension (int width, int height, int nchannels) const;

		// function of basic image operations
		bool imresize(float ratio);
		bool imresize(int dstWidth, int dstHeight);
		void imresize(ZQ_DImage& result, float ratio) const;
		void imresize(ZQ_DImage& result, int dstWidth, int dstHeight) const;

		bool imresizeBicubic(float ratio);
		bool imresizeBicubic(int dstWidth, int dstHeight);
		void imresizeBicubic(ZQ_DImage& result, float ratio) const;
		void imresizeBicubic(ZQ_DImage& result, int dstWidth, int dstHeight) const;


		// drivatives
		void dx_3pt(ZQ_DImage<T>& result) const;
		void dx(ZQ_DImage& result, bool isAdvancedFilter = false) const;
		void dy_3pt(ZQ_DImage<T>& result) const;
		void dy(ZQ_DImage& result, bool isAdvancedFilter = false) const;

		// Gaussian smoothing
		void GaussianSmoothing(float sigma, int fsize);
		void GaussianSmoothing(ZQ_DImage& image, float sigma, int fsize) const;


		// funciton for filtering
		template<class FilterType>
		void imfilter_h(ZQ_DImage& image, const FilterType* filter, int fsize) const;
		template<class FilterType>
		void imfilter_v(ZQ_DImage& image, const FilterType* filter, int fsize) const;
		template<class FilterType>
		void imfilter_hv(ZQ_DImage& image, const FilterType* hfilter, int hfsize, const FilterType* vfilter, int vfsize) const;

		//collapse
		void collapse(ZQ_DImage& image) const;
		void collapse();


		// function to separate the channels of the image
		void separate(unsigned firstNChannels, ZQ_DImage& image1, ZQ_DImage& image2) const;

		void assemble(const ZQ_DImage& image1, const ZQ_DImage& image2);

		/*this = image1*image2*/
		void Multiply(const ZQ_DImage& image1, const ZQ_DImage& image2);

		/*this = image1*image2*image3*/
		void Multiply(const ZQ_DImage& image1, const ZQ_DImage& image2, const ZQ_DImage& image3);

		/*this *= image1*/
		void Multiplywith(const ZQ_DImage& image1);

		/*this *=value */
		void Multiplywith(T value);

		/* this = image1+image2*/
		void Add(const ZQ_DImage& image1, const ZQ_DImage& image2);

		/* this += image1*ratio*/
		void Addwith(const ZQ_DImage& image1, const T ratio);

		/* this += image1*/
		void Addwith(const ZQ_DImage& image1);

		/* this += value*/
		void Addwith(const T value);

		/*this = image1-image2*/
		void Subtract(const ZQ_DImage& image1, const ZQ_DImage& image2);

		/*this -= image1*/
		void Subtractwith(const ZQ_DImage& image1);

		void FlipX(ZQ_DImage& output) const;
		void FlipX();
		void FlipY(ZQ_DImage& output) const;
		void FlipY();

		// function for image warping
		void warpImage(ZQ_DImage& output, const ZQ_DImage& u, const ZQ_DImage& v) const;

		//median filter
		bool MedianFilter(ZQ_DImage& output, const int fsize) const;
		bool MedianFilter(const int fsize);
		bool MedianFilterWithMask(ZQ_DImage& output, const int fsize, const ZQ_DImage<bool>& keep_mask) const;
		bool MedianFilterWithMask(const int fsize, const ZQ_DImage<bool>& keep_mask);

		//autoadjust
		void AutoAdjust();

		// image IO's
		bool saveImage(const char* filename) const;
		bool loadImage(const char* filename);
	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/


	template<class T>
	ZQ_DImage<T>::ZQ_DImage()
	{
		pData = 0;
		imWidth = imHeight = nChannels = nPixels = nElements = 0;
	}

	template<class T>
	ZQ_DImage<T>::ZQ_DImage(int width, int height, int nchannels)
	{
		imWidth = width;
		imHeight = height;
		nChannels = nchannels;
		computeDimension();
		pData = 0;
		pData = new T[nElements];
		if(nElements > 0)
			memset(pData,0,sizeof(T)*nElements);
	}

	template<class T>
	ZQ_DImage<T>::ZQ_DImage(const ZQ_DImage<T>& other)
	{
		imWidth = imHeight = nChannels = nElements = 0;
		pData = 0;
		copyData(other);
	}


	template<class T>
	ZQ_DImage<T>::~ZQ_DImage()
	{
		if(pData != 0)
		{
			//printf("imWidth = %d, imHeight = %d\n", imWidth, imHeight);
			delete []pData;
			pData = 0;
		}
	}


	template<class T>
	void ZQ_DImage<T>::allocate(int width, int height, int nchannels)
	{
		clear();
		imWidth = width;
		imHeight = height;
		nChannels = nchannels;
		computeDimension();
		pData = 0;

		if(nElements > 0)
		{
			pData = new T[nElements];
			memset(pData,0,sizeof(T)*nElements);
		}
	}


	template<class T>
	void ZQ_DImage<T>::allocate(const ZQ_DImage<T> &other)
	{
		allocate(other.width(),other.height(),other.nchannels());
	}


	template<class T>
	void ZQ_DImage<T>::clear()
	{
		if(pData != 0)
			delete []pData;
		pData = 0;
		imWidth = imHeight = nChannels = nPixels = nElements = 0;
	}


	template<class T>
	void ZQ_DImage<T>::reset()
	{
		if(pData != 0)
			memset(pData,0,sizeof(T)*nElements);
	}



	template<class T>
	void ZQ_DImage<T>::copyData(const ZQ_DImage<T>& other)
	{
		imWidth = other.imWidth;
		imHeight = other.imHeight;
		nChannels = other.nChannels;
		nPixels = other.nPixels;

		if(nElements != other.nElements)
		{
			nElements = other.nElements;		
			if(pData != 0)
				delete []pData;
			pData = new T[nElements];
		}
		if(nElements > 0)
			memcpy(pData,other.pData,sizeof(T)*nElements);
	}



	template<class T>
	ZQ_DImage<T>& ZQ_DImage<T>::operator=(const ZQ_DImage<T>& other)
	{
		copyData(other);
		return *this;
	}


	template<class T>
	bool ZQ_DImage<T>::matchDimension(const ZQ_DImage<T>& image) const
	{
		if(imWidth == image.width() && imHeight == image.height() && nChannels == image.nchannels())
			return true;
		else
			return false;
	}



	template<class T>
	bool ZQ_DImage<T>::matchDimension(int width, int height, int nchannels) const
	{
		if(imWidth == width && imHeight == height && nChannels == nchannels)
			return true;
		else
			return false;
	}



	template<class T>
	bool ZQ_DImage<T>::imresize(float ratio)
	{
		if(pData == 0)
			return false;

		int dstWidth = imWidth*ratio + 0.5;
		int dstHeight = imHeight*ratio + 0.5;
		T* pDstData = new T[dstWidth*dstHeight*nChannels];

		ZQ_ImageProcessing::ResizeImage(pData, pDstData, imWidth, imHeight, nChannels, dstWidth, dstHeight, false);

		delete []pData;
		pData = pDstData;
		imWidth = dstWidth;
		imHeight = dstHeight;
		computeDimension();
		return true;
	}



	template<class T>
	bool ZQ_DImage<T>::imresize(int dstWidth, int dstHeight)
	{
		if(pData == 0)
			return false;

		ZQ_DImage<T> foo(dstWidth,dstHeight,nChannels);
		ZQ_ImageProcessing::ResizeImage(pData, foo.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
		copyData(foo);
		return true;
	}



	template<class T>
	void ZQ_DImage<T>::imresize(ZQ_DImage<T>& result, float ratio) const
	{
		int dstWidth = imWidth*ratio + 0.5;
		int dstHeight = imHeight*ratio + 0.5;
		if(result.width() != dstWidth || result.height() != dstHeight || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing::ResizeImage(pData, result.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
	}



	template<class T>
	void ZQ_DImage<T>::imresize(ZQ_DImage<T>& result, int dstWidth, int dstHeight) const
	{
		if(result.width() != dstWidth || result.height() != dstHeight || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing::ResizeImage(pData, result.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
	}



	template<class T>
	bool ZQ_DImage<T>::imresizeBicubic(float ratio)
	{
		if(pData == 0)
			return false;

		int dstWidth = imWidth*ratio + 0.5;
		int dstHeight = imHeight*ratio + 0.5;
		T* pDstData = new T[dstWidth*dstHeight*nChannels];

		ZQ_ImageProcessing::ResizeImageBicubic(pData, pDstData, imWidth, imHeight, nChannels, dstWidth, dstHeight, false);

		delete []pData;
		pData = pDstData;
		imWidth = dstWidth;
		imHeight = dstHeight;
		computeDimension();
		return true;
	}



	template<class T>
	bool ZQ_DImage<T>::imresizeBicubic(int dstWidth, int dstHeight)
	{
		if(pData == 0)
			return false;

		ZQ_DImage<T> foo(dstWidth,dstHeight,nChannels);
		ZQ_ImageProcessing::ResizeImageBicubic(pData, foo.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
		copyData(foo);
		return true;
	}



	template<class T>
	void ZQ_DImage<T>::imresizeBicubic(ZQ_DImage<T>& result, float ratio) const
	{
		int dstWidth = (double)imWidth*ratio + 0.5;
		int dstHeight = (double)imHeight*ratio + 0.5;
		if(result.width() != dstWidth || result.height() != dstHeight || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing::ResizeImageBicubic(pData, result.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
	}


	template<class T>
	void ZQ_DImage<T>::imresizeBicubic(ZQ_DImage<T>& result, int dstWidth, int dstHeight) const
	{
		if(result.width() != dstWidth || result.height() != dstHeight || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing::ResizeImageBicubic(pData, result.data(), imWidth, imHeight, nChannels, dstWidth, dstHeight, false);
	}


	template<class T>
	void ZQ_DImage<T>::dx_3pt(ZQ_DImage<T>& result) const
	{
		if (matchDimension(result) == false)
			result.allocate(imWidth, imHeight, nChannels);
		else
			result.reset();

		T*& data = result.pData;

		for (int i = 0; i < imHeight; i++)
		{
			for (int c = 0; c < nChannels; c++)
			{
				data[i*imWidth*nChannels + c] = pData[(i*imWidth + 1)*nChannels + c] - pData[i*imWidth*nChannels + c];
				data[(i*imWidth + imWidth - 1)*nChannels + c] = pData[(i*imWidth + imWidth - 1)*nChannels + c] - pData[(i*imWidth + imWidth - 2)*nChannels + c];
			}
			for (int j = 1; j < imWidth - 1; j++)
			{
				int offset = i*imWidth + j;
				for (int c = 0; c < nChannels; c++)
					data[offset*nChannels + c] = 0.5*(pData[(offset + 1)*nChannels + c] - pData[(offset - 1)*nChannels + c]);
			}
		}
	}

	template<class T>
	void ZQ_DImage<T>::dx(ZQ_DImage<T>& result, bool isAdvancedFilter) const
	{
		if(matchDimension(result) == false)
			result.allocate(imWidth,imHeight,nChannels);
		else
			result.reset();

		T*& data = result.pData;

		if(!isAdvancedFilter)
		{
			for (int i = 0; i < imHeight; i++)
			{
				for (int j = 0; j < imWidth - 1; j++)
				{
					int offset = i*imWidth + j;
					for (int c = 0; c < nChannels; c++)
						data[offset*nChannels + c] = pData[(offset + 1)*nChannels + c] - pData[offset*nChannels + c];
				}
			}
		}
		else
		{
			T xFilter[5] = {1,-8,0,8,-1};
			for(int i = 0; i < 5;i++)
				xFilter[i] /= 12;
			ZQ_ImageProcessing::Hfiltering(pData, data, imWidth, imHeight, nChannels, xFilter, 2, false);
		}
	}

	template<class T>
	void ZQ_DImage<T>::dy_3pt(ZQ_DImage<T>& result) const
	{
		if (matchDimension(result) == false)
			result.allocate(imWidth, imHeight, nChannels);
		else
			result.reset();

		T*& data = result.pData;

		for (int i = 0; i < imWidth; i++)
		{
			for (int c = 0; c < nChannels; c++)
			{
				data[i*nChannels + c] = pData[(imWidth + i)*nChannels + c] - pData[i*nChannels + c];
				data[((imHeight - 1)*imWidth + i)*nChannels + c] = pData[((imHeight - 1)*imWidth + i)*nChannels + c] - pData[((imHeight - 2)*imWidth + i)*nChannels + c];
			}
			for (int j = 1; j < imHeight - 1; j++)
			{
				int offset = j*imWidth + i;
				for (int c = 0; c < nChannels; c++)
					data[offset*nChannels + c] = 0.5*(pData[(offset + imWidth)*nChannels + c] - pData[(offset - imWidth)*nChannels + c]);
			}
		}
	}

	template<class T>
	void ZQ_DImage<T>::dy(ZQ_DImage<T>& result, bool isAdvancedFilter) const
	{
		if(matchDimension(result)==false)
			result.allocate(imWidth,imHeight,nChannels);
		else
			result.reset();

		T*& data = result.pData;

		if(!isAdvancedFilter)
		{
			for (int i = 0; i < imHeight - 1; i++)
			{
				for (int j = 0; j < imWidth; j++)
				{
					int offset = i*imWidth + j;
					for (int c = 0; c < nChannels; c++)
						data[offset*nChannels + c] = pData[(offset + imWidth)*nChannels + c] - pData[offset*nChannels + c];
				}
			}
		}
		else
		{
			T yFilter[5]={1,-8,0,8,-1};
			for(int i = 0;i < 5;i++)
				yFilter[i] /= 12;
			ZQ_ImageProcessing::Vfiltering(pData, data, imWidth, imHeight, nChannels, yFilter, 2, false);
		}
	}

	template<class T>
	void ZQ_DImage<T>::GaussianSmoothing(float sigma, int fsize)
	{
		ZQ_DImage<T> foo;
		GaussianSmoothing(foo,sigma,fsize);
		copyData(foo);
	}


	template<class T>
	void ZQ_DImage<T>::GaussianSmoothing(ZQ_DImage<T>& image, float sigma, int fsize) const
	{
		float* gFilter = new float[fsize*2+1];
		double sum = 0;
		double m_sigma = sigma*sigma*2;
		for(int i = -fsize;i <= fsize;i++)
		{
			gFilter[i+fsize] = exp(-(float)(i*i)/m_sigma);
			sum += gFilter[i+fsize];
		}
		for(int i = 0;i < 2*fsize+1;i++)
			gFilter[i] /= sum;

		imfilter_hv(image, gFilter, fsize, gFilter, fsize);

		delete []gFilter;
	}


	template<class T>template<class FilterType>
	void ZQ_DImage<T>::imfilter_h(ZQ_DImage<T>& image, const FilterType* filter, int fsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,nChannels);
		ZQ_ImageProcessing::Hfiltering(pData, image.data(), imWidth, imHeight, nChannels, filter, fsize, false);
	}


	template<class T>template<class FilterType>
	void ZQ_DImage<T>::imfilter_v(ZQ_DImage<T>& image, const FilterType* filter, int fsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,nChannels);
		ZQ_ImageProcessing::Vfiltering(pData, image.data(), imWidth, imHeight, nChannels, filter, fsize, false);
	}



	template<class T>template<class FilterType>
	void ZQ_DImage<T>::imfilter_hv(ZQ_DImage<T> &image, const FilterType *hfilter, int hfsize, const FilterType *vfilter, int vfsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,nChannels);
		float* pTempBuffer = new float[nElements];
		ZQ_ImageProcessing::Hfiltering(pData, pTempBuffer, imWidth, imHeight, nChannels, hfilter, hfsize, false);
		ZQ_ImageProcessing::Vfiltering(pTempBuffer, image.data(), imWidth, imHeight, nChannels, vfilter, vfsize, false);
		delete []pTempBuffer;
	}


	template<class T>
	void ZQ_DImage<T>::collapse(ZQ_DImage<T>& image) const
	{
		if(!(image.imWidth == imWidth && image.imHeight == imHeight && image.nChannels == 1))
			image.allocate(imWidth,imHeight,1);

		if(nChannels == 1)
		{
			image.copyData(*this);
			return;
		}
		T* data = image.data();

		for (int i = 0; i < nPixels; i++)
		{
			int offset = i*nChannels;
			T temp = 0;

			for (int c = 0; c < nChannels; c++)
				temp += pData[offset + c];
			data[i] = temp / nChannels;
		}
	}



	template<class T>
	void ZQ_DImage<T>::collapse()
	{
		if(nChannels == 1)
			return;
		ZQ_DImage<T> result;
		collapse(result);
		copyData(result);
	}


	template<class T>
	void ZQ_DImage<T>::separate(unsigned int firstNChannels, ZQ_DImage<T> &image1, ZQ_DImage<T> &image2) const
	{
		if(firstNChannels >= nChannels)
		{
			image1 = *this;
			image2.allocate(imWidth,imHeight,0);
			return;
		}
		else if(firstNChannels == 0)
		{
			image1.allocate(imWidth,imHeight,0);
			image2 = *this;
			return ;
		}
		else
		{
			int secondNChannels = nChannels-firstNChannels;
			if(image1.width() != imWidth || image1.height() != imHeight || image1.nchannels() != firstNChannels)
				image1.allocate(imWidth,imHeight,firstNChannels);
			if(image2.width() != imWidth || image2.height() != imHeight || image2.nchannels() != secondNChannels)
				image2.allocate(imWidth,imHeight,secondNChannels);

			for (int i = 0; i < nPixels; i++)
			{
				int offset = i;
				for (int c = 0; c < firstNChannels; c++)
					image1.pData[offset*firstNChannels + c] = pData[offset*nChannels + c];
				for (int c = firstNChannels; c < nChannels; c++)
					image2.pData[offset*secondNChannels + c - firstNChannels] = pData[offset*nChannels + c];
			}
		}
	}


	template<class T>
	void ZQ_DImage<T>::assemble(const ZQ_DImage<T> &u, const  ZQ_DImage<T> &v)
	{
		int width = u.width();
		int height = u.height();
		int nChannels1 = u.nchannels();
		int nChannels2 = v.nchannels();

		if (v.width() != width || v.height() != height)
			return;

		allocate(width, height, nChannels1 + nChannels2);

		for (int i = 0; i < nPixels; i++)
		{
			for (int c = 0; c < nChannels1; c++)
				pData[i*nChannels + c] = u.pData[i*nChannels1 + c];
			for (int c = 0; c < nChannels2; c++)
				pData[i*nChannels + c + nChannels1] = v.pData[i*nChannels2 + c];
		}
	}


	template<class T>
	void ZQ_DImage<T>::Multiply(const ZQ_DImage<T>& image1, const ZQ_DImage<T>& image2)
	{
		if(image1.matchDimension(image2) == false)
		{
			printf("Error in image dimensions--function Image::Multiply()!\n");
			return;
		}
		if(matchDimension(image1)==false)
			allocate(image1);

		const T*& pData1 = image1.data();
		const T*& pData2 = image2.data();

		for (int i = 0; i < nElements; i++)
			pData[i] = pData1[i] * pData2[i];

	}



	template<class T>
	void ZQ_DImage<T>::Multiply(const ZQ_DImage<T>& image1, const ZQ_DImage<T>& image2, const ZQ_DImage<T>& image3)
	{
		if(image1.matchDimension(image2) == false || image2.matchDimension(image3) == false)
		{
			printf("Error in image dimensions--function ZQ_DImage::Multiply()!\n");
			return;
		}
		if(matchDimension(image1) == false)
			allocate(image1);

		const T*& pData1 = image1.data();
		const T*& pData2 = image2.data();
		const T*& pData3 = image3.data();

		for (int i = 0; i < nElements; i++)
			pData[i] = pData1[i] * pData2[i] * pData3[i];

	}



	template<class T>
	void ZQ_DImage<T>::Multiplywith(const ZQ_DImage<T> &image1)
	{
		if(matchDimension(image1)==false)
		{
			printf("Error in image dimensions--function ZQ_DImage::Multiplywith()!\n");
			return;
		}
		const T*& pData1 = image1.data();

		for (int i = 0; i < nElements; i++)
			pData[i] *= pData1[i];
	}

	template<class T>
	void ZQ_DImage<T>::Multiplywith(T value)
	{
		for (int i = 0; i < nElements; i++)
			pData[i] *= value;
	}


	template<class T>
	void ZQ_DImage<T>::Add(const ZQ_DImage<T>& image1, const ZQ_DImage<T>& image2)
	{
		if(image1.matchDimension(image2) == false)
		{
			printf("Error in image dimensions--function ZQ_DImage::Add()!\n");
			return;
		}
		if(matchDimension(image1) == false)
			allocate(image1);

		const T*& pData1 = image1.data();
		const T*& pData2 = image2.data();

		for (int i = 0; i < nElements; i++)
			pData[i] = pData1[i] + pData2[i];
	}


	template<class T>
	void ZQ_DImage<T>::Addwith(const ZQ_DImage<T>& image1, const T ratio)
	{
		if (matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::Addwith()!\n");
			return;
		}
		const T*& pData1 = image1.data();

		for (int i = 0; i < nElements; i++)
			pData[i] += pData1[i] * ratio;
	}



	template<class T>
	void ZQ_DImage<T>::Addwith(const ZQ_DImage<T>& image1)
	{
		if (matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::Addwith()!\n");
			return;
		}
		const T*& pData1 = image1.data();

		for (int i = 0; i < nElements; i++)
			pData[i] += pData1[i];
	}


	template<class T>
	void ZQ_DImage<T>::Addwith(const T value)
	{
		for (int i = 0; i < nElements; i++)
			pData[i] += value;
	}



	template<class T>
	void ZQ_DImage<T>::Subtract(const ZQ_DImage<T>& image1, const ZQ_DImage<T>& image2)
	{
		if(image1.matchDimension(image2) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::Subtract()!\n");
			return;
		}
		if(matchDimension(image1) == false)
			allocate(image1);

		const T*& pData1 = image1.data();
		const T*& pData2 = image2.data();

		for (int i = 0; i < nElements; i++)
			pData[i] = pData1[i] - pData2[i];
	}


	template<class T>
	void ZQ_DImage<T>::Subtractwith(const ZQ_DImage<T>& image1)
	{
		if(matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::SubtractWith()!\n");
			return;
		}

		const T*& pData1 = image1.data();

		for (int i = 0; i < nElements; i++)
			pData[i] -= pData1[i];
	}

	template<class T>
	void ZQ_DImage<T>::FlipX(ZQ_DImage<T>& output) const
	{
		if(!output.matchDimension(*this))
			output.allocate(*this);

		for (int i = 0; i < imHeight; i++)
		{
			for (int j = 0; j < imWidth; j++)
			{
				memcpy(output.pData + (i*imWidth + imWidth - 1 - j)*nChannels, pData + (i*imWidth + j)*nChannels, sizeof(T)*nChannels);
			}
		}
	}

	template<class T>
	void ZQ_DImage<T>::FlipX()
	{
		ZQ_DImage<T> tmp(*this);
		tmp.FlipX(*this);
	}

	template<class T>
	void ZQ_DImage<T>::FlipY(ZQ_DImage<T>& output) const
	{
		if(!output.matchDimension(*this))
			output.allocate(*this);

		for (int i = 0; i < imHeight; i++)
		{
			memcpy(output.pData + (imHeight - 1 - i)*imWidth*nChannels, pData + i*imWidth*nChannels, sizeof(T)*imWidth*nChannels);
		}
	}

	template<class T>
	void ZQ_DImage<T>::FlipY()
	{
		ZQ_DImage<T> tmp(*this);
		tmp.FlipY(*this);
	}

	template<class T>
	void ZQ_DImage<T>::warpImage(ZQ_DImage<T> &output, const ZQ_DImage<T>& u, const ZQ_DImage<T>& v) const
	{
		if(!output.matchDimension(*this))
			output.allocate(*this);
		ZQ_ImageProcessing::WarpImage(output.data(), pData, u.data(), v.data(), imWidth, imHeight, nChannels, false);
	}

	//median filter
	template<class T>
	bool ZQ_DImage<T>::MedianFilter(ZQ_DImage<T>& output, const int fsize) const
	{
		output.copyData(*this);
		return ZQ_ImageProcessing::MedianFilter(pData, output.pData, imWidth, imHeight, nChannels, fsize, false);
	}

	template<class T>
	bool ZQ_DImage<T>::MedianFilter(const int fsize)
	{
		ZQ_DImage<T> tmp;
		if (!MedianFilter(tmp, fsize))
			return false;
		copyData(tmp);
		return true;
	}

	template<class T>
	bool ZQ_DImage<T>::MedianFilterWithMask(ZQ_DImage<T>& output, const int fsize, const ZQ_DImage<bool>& keep_mask) const
	{
		output.copyData(*this);
		return ZQ_ImageProcessing::MedianFilterWithMask(pData, output.pData, imWidth, imHeight, nChannels, fsize, keep_mask.data(), false);
	}

	template<class T>
	bool ZQ_DImage<T>::MedianFilterWithMask(const int fsize, const ZQ_DImage<bool>& keep_mask)
	{
		ZQ_DImage<T> tmp;
		if (!MedianFilterWithMask(tmp, fsize, keep_mask))
			return false;
		copyData(tmp);
		return true;
	}

	template<class T>
	void ZQ_DImage<T>::AutoAdjust()
	{
		if (nElements == 0)
			return;
		T min_v = immin();
		T max_v = immax();
		if (min_v == max_v || (min_v == 0 && max_v == 1))
			return;
		T scale = 1.0 / (max_v - min_v);

		for (int i = 0; i < nElements; i++)
		{
			pData[i] = (pData[i] - min_v)*scale;
		}
	}

	template<class T>
	bool ZQ_DImage<T>::saveImage(const char *filename) const
	{
		FILE* out = 0;
		if(0 != fopen_s(&out, filename, "wb"))
			return false;

		char type[16];
		sprintf_s(type,"%s",typeid(T).name());
		fwrite(type,sizeof(char),16,out);
		fwrite(&imWidth,sizeof(int),1,out);
		fwrite(&imHeight,sizeof(int),1,out);
		fwrite(&nChannels,sizeof(int),1,out);

		bool isDerivativeImage = false;//unused

		fwrite(&isDerivativeImage,sizeof(bool),1,out);
		fwrite(pData,sizeof(T),nElements,out);

		fclose(out);
		return true;
	}


	template<class T>
	bool ZQ_DImage<T>::loadImage(const char* filename)
	{
		FILE* in = 0;
		if(0 != fopen_s(&in, filename, "rb"))
			return false;

		char type[16];
		fread(type,sizeof(char),16,in);

		int width,height,nchannels;
		fread(&width,sizeof(int),1,in);
		fread(&height,sizeof(int),1,in);
		fread(&nchannels,sizeof(int),1,in);

		bool isDerivative;
		fread(&isDerivative,sizeof(bool),1,in);

		if(!matchDimension(width,height,nchannels))
			allocate(width,height,nchannels);

		if(_strcmpi(type,typeid(double).name()) == 0)
		{
			if(strcmp(typeid(T).name(),"double") == 0)
			{
				if(fread(pData,sizeof(double),nElements,in) == nElements)
				{
					fclose(in);
					return true;
				}
				else
				{
					clear();
					fclose(in);
					return false;
				}
			}
			else
			{
				double* tmp = new double[nElements];
				if(fread(tmp,sizeof(double),nElements,in) == nElements)
				{
					for(int i = 0;i < nElements;i++)
						pData[i] = tmp[i];
					delete []tmp;
					fclose(in);
					return true;
				}
				else
				{
					delete []tmp;
					clear();
					fclose(in);
					return false;
				}
			}
		}
		else if(_strcmpi(type,typeid(float).name()) == 0)
		{
			if(strcmp(typeid(T).name(), "float") == 0)
			{
				if(fread(pData,sizeof(float),nElements,in) == nElements)
				{
					fclose(in);
					return true;
				}
				else
				{
					clear();
					fclose(in);
					return false;
				}
			}
			else
			{
				float* tmp = new float[nElements];
				if(fread(tmp,sizeof(float),nElements,in) == nElements)
				{
					for(int i = 0;i < nElements;i++)
						pData[i] = tmp[i];
					delete []tmp;
					fclose(in);
					return true;
				}
				else
				{
					delete []tmp;
					clear();
					fclose(in);
					return false;
				}
			}
		}
		else
		{
			fclose(in);
			return false;
		}
	}
}



#endif