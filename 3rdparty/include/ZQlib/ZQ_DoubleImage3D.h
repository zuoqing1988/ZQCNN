#ifndef _ZQ_DOUBLE_IMAGE_3D_H_
#define _ZQ_DOUBLE_IMAGE_3D_H_
#pragma once

#include <stdio.h>
#include "ZQ_ImageProcessing3D.h"
#include <typeinfo>

namespace ZQ
{
	template<class T>
	class ZQ_DImage3D
	{
	public:
		T* pData;
	protected:
		int imWidth, imHeight, imDepth, nChannels;
		int nPixels, nElements;
	public:
		ZQ_DImage3D(void);
		ZQ_DImage3D(int width, int height, int depth, int nchannels = 1);
		ZQ_DImage3D(const ZQ_DImage3D& other);
		~ZQ_DImage3D(void);

		void computeDimension(){nPixels = imWidth*imHeight*imDepth ;nElements = nPixels*nChannels;}
		void allocate(int width, int height, int depth, int nchannels=1);
		void allocate(const ZQ_DImage3D &other);
		void clear();
		void reset();
		void copyData(const ZQ_DImage3D& other);
		ZQ_DImage3D& operator=(const ZQ_DImage3D& other);

		void swap(ZQ_DImage3D& other)
		{
			int tmp;
			tmp = imWidth; imWidth = other.imWidth; other.imWidth = tmp;
			tmp = imHeight;	imHeight = other.imHeight;	other.imHeight = tmp;
			tmp = imDepth;	imDepth = other.imDepth;	other.imDepth = tmp;
			tmp = nChannels;	nChannels = other.nChannels;	other.nChannels = tmp;
			tmp = nPixels;	nPixels = other.nPixels;	other.nPixels = tmp;
			tmp = nElements;	nElements = other.nElements;	other.nElements = tmp;
			T* tmp_ptr = pData;	pData = other.pData;	other.pData = tmp_ptr;
		}

		T immax() const
		{
			T Max = pData[0];
			for(int i = 1;i < nElements;i++)
				Max = __max(Max,pData[i]);
			return Max;
		};

		T immin() const
		{
			T Min = pData[0];
			for(int i = 1;i < nElements;i++)
				Min = __min(Min,pData[i]);
			return Min;
		}


		// function to access the member variables
		T*& data(){return pData;}
		const T*& data() const{return (const T*&)pData;}
		int width() const {return imWidth;}
		int height() const {return imHeight;}
		int depth() const {return imDepth;}
		int nchannels() const {return nChannels;}
		int npixels() const {return nPixels;}
		int nelements() const {return nElements;}
		bool IsEmpty() const {if(nElements == 0) return true;else return false;}
		bool matchDimension  (const ZQ_DImage3D& image) const;
		bool matchDimension (int width,int height, int depth, int nchannels) const;

		// function of basic image operations
		bool imresize(float ratio);
		bool imresize(int dstWidth, int dstHeight, int dstDepth);
		void imresize(ZQ_DImage3D& result, float ratio) const;
		void imresize(ZQ_DImage3D& result, int dstWidth, int dstHeight, int dstDepth) const;

		bool imresizeTricubic(float ratio);
		bool imresizeTricubic(int dstWidth, int dstHeight, int dstDepth);
		void imresizeTricubic(ZQ_DImage3D& result, float ratio) const;
		void imresizeTricubic(ZQ_DImage3D& result, int dstWidth, int dstHeight, int dstDepth) const;


		// drivatives
		void dx(ZQ_DImage3D& result, bool isAdvancedFilter = false) const;
		void dy(ZQ_DImage3D& result, bool isAdvancedFilter = false) const;
		void dz(ZQ_DImage3D& result, bool isAdvancedFilter = false) const;

		// Gaussian smoothing
		void GaussianSmoothing(float sigma, int fsize);
		void GaussianSmoothing(ZQ_DImage3D& image, float sigma, int fsize) const;


		// funciton for filtering
		void imfilter_h(ZQ_DImage3D& image, T* filter, int fsize) const;
		void imfilter_v(ZQ_DImage3D& image, T* filter, int fsize) const;
		void imfilter_d(ZQ_DImage3D& image, T* filter, int fsize) const;
		void imfilter_hvd(ZQ_DImage3D& image, const T* hfilter, int hfsize, const T* vfilter, int vfsize, const T* dfilter, int dfsize) const;

		//collapse
		void collapse(ZQ_DImage3D& image) const;
		void collapse();


		// function to separate the channels of the image
		void separate(unsigned firstNChannels, ZQ_DImage3D& image1, ZQ_DImage3D& image2) const;

		void assemble(const ZQ_DImage3D& u, const ZQ_DImage3D& v, const ZQ_DImage3D& w);

		/*this = image1*image2*/
		void Multiply(const ZQ_DImage3D& image1, const ZQ_DImage3D& image2);

		/*this = image1*image2*image3*/
		void Multiply(const ZQ_DImage3D& image1, const ZQ_DImage3D& image2, const ZQ_DImage3D& image3);

		/*this *= image1*/
		void Multiplywith(const ZQ_DImage3D& image1);

		/*this *=value */
		void Multiplywith(T value);

		/* this = image1+image2*/
		void Add(const ZQ_DImage3D& image1, const ZQ_DImage3D& image2);

		/* this += image1*ratio*/
		void Addwith(const ZQ_DImage3D& image1, const T ratio);

		/* this += image1*/
		void Addwith(const ZQ_DImage3D& image1);

		/* this += value*/
		void Addwith(const T value);

		/*this = image1-image2*/
		void Subtract(const ZQ_DImage3D& image1, const ZQ_DImage3D& image2);

		/*this -= image1*/
		void Subtractwith(const ZQ_DImage3D& image1);

		// function for image warping
		void warpImage(ZQ_DImage3D& output, const ZQ_DImage3D& u, const ZQ_DImage3D& v, const ZQ_DImage3D& w) const;

		// image IO's
		bool saveImage(const char* filename) const;
		bool loadImage(const char* filename);
	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/


	template<class T>
	ZQ_DImage3D<T>::ZQ_DImage3D()
	{
		pData = 0;
		imWidth = imHeight = imDepth = nChannels = nPixels = nElements = 0;
	}

	template<class T>
	ZQ_DImage3D<T>::ZQ_DImage3D(int width, int height, int depth, int nchannels)
	{
		imWidth = width;
		imHeight = height;
		imDepth = depth;
		nChannels = nchannels;
		computeDimension();
		pData = 0;
		pData = new T[nElements];
		if(nElements > 0)
			memset(pData,0,sizeof(T)*nElements);
	}

	template<class T>
	ZQ_DImage3D<T>::ZQ_DImage3D(const ZQ_DImage3D<T>& other)
	{
		imWidth = imHeight = imDepth = nChannels = nElements = 0;
		pData = 0;
		copyData(other);
	}


	template<class T>
	ZQ_DImage3D<T>::~ZQ_DImage3D()
	{
		if(pData != 0)
		{
			delete []pData;
			pData = 0;
		}
	}


	template<class T>
	void ZQ_DImage3D<T>::allocate(int width, int height, int depth, int nchannels)
	{
		clear();
		imWidth = width;
		imHeight = height;
		imDepth = depth;
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
	void ZQ_DImage3D<T>::allocate(const ZQ_DImage3D<T> &other)
	{
		allocate(other.width(), other.height(), other.depth() ,other.nchannels());
	}


	template<class T>
	void ZQ_DImage3D<T>::clear()
	{
		if(pData != 0)
			delete []pData;
		pData = 0;
		imWidth = imHeight = imDepth = nChannels = nPixels = nElements = 0;
	}


	template<class T>
	void ZQ_DImage3D<T>::reset()
	{
		if(pData != 0)
			memset(pData,0,sizeof(T)*nElements);
	}



	template<class T>
	void ZQ_DImage3D<T>::copyData(const ZQ_DImage3D<T>& other)
	{
		imWidth = other.imWidth;
		imHeight = other.imHeight;
		imDepth = other.imDepth;
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
	ZQ_DImage3D<T>& ZQ_DImage3D<T>::operator=(const ZQ_DImage3D<T>& other)
	{
		copyData(other);
		return *this;
	}


	template<class T>
	bool ZQ_DImage3D<T>::matchDimension(const ZQ_DImage3D<T>& image) const
	{
		if(imWidth == image.width() && imHeight == image.height() && imDepth == image.depth() && nChannels == image.nchannels())
			return true;
		else
			return false;
	}



	template<class T>
	bool ZQ_DImage3D<T>::matchDimension(int width, int height, int depth, int nchannels) const
	{
		if(imWidth == width && imHeight == height && imDepth == depth && nChannels == nchannels)
			return true;
		else
			return false;
	}



	template<class T>
	bool ZQ_DImage3D<T>::imresize(float ratio)
	{
		if(pData == 0)
			return false;

		int dstWidth = imWidth*ratio;
		int dstHeight = imHeight*ratio;
		int dstDepth = imDepth*ratio;
		T* pDstData = new T[dstWidth*dstHeight*dstDepth*nChannels];

		ZQ_ImageProcessing3D::ResizeImage(pData,pDstData,imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);

		delete []pData;
		pData = pDstData;
		imWidth = dstWidth;
		imHeight = dstHeight;
		imDepth = dstDepth;
		computeDimension();
		return true;
	}



	template<class T>
	bool ZQ_DImage3D<T>::imresize(int dstWidth, int dstHeight, int dstDepth)
	{
		if(pData == 0)
			return false;

		ZQ_DImage3D<T> foo(dstWidth,dstHeight,dstDepth,nChannels);
		ZQ_ImageProcessing3D::ResizeImage(pData,foo.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
		copyData(foo);
		return true;
	}



	template<class T>
	void ZQ_DImage3D<T>::imresize(ZQ_DImage3D<T>& result, float ratio) const
	{
		int dstWidth = imWidth*ratio;
		int dstHeight = imHeight*ratio;
		int dstDepth = imDepth*ratio;
		if(result.width() != dstWidth || result.height() != dstHeight || result.depth() != dstDepth || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,dstDepth,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing3D::ResizeImage(pData,result.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
	}



	template<class T>
	void ZQ_DImage3D<T>::imresize(ZQ_DImage3D<T>& result, int dstWidth, int dstHeight, int dstDepth) const
	{
		if(result.width() != dstWidth || result.height() != dstHeight || result.depth() != dstDepth || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,dstDepth,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing3D::ResizeImage(pData,result.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
	}



	template<class T>
	bool ZQ_DImage3D<T>::imresizeTricubic(float ratio)
	{
		if(pData == 0)
			return false;

		int dstWidth = imWidth*ratio;
		int dstHeight = imHeight*ratio;
		int dstDepth = imDepth*ratio;
		T* pDstData = new T[dstWidth*dstHeight*dstDepth*nChannels];

		ZQ_ImageProcessing3D::ResizeImageTricubic(pData,pDstData,imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);

		delete []pData;
		pData = pDstData;
		imWidth = dstWidth;
		imHeight = dstHeight;
		imDepth = dstDepth;
		computeDimension();
		return true;
	}



	template<class T>
	bool ZQ_DImage3D<T>::imresizeTricubic(int dstWidth, int dstHeight, int dstDepth)
	{
		if(pData == 0)
			return false;

		ZQ_DImage3D<T> foo(dstWidth,dstHeight,dstDepth,nChannels);
		ZQ_ImageProcessing3D::ResizeImageTricubic(pData,foo.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
		copyData(foo);
		return true;
	}



	template<class T>
	void ZQ_DImage3D<T>::imresizeTricubic(ZQ_DImage3D<T>& result, float ratio) const
	{
		int dstWidth = (double)imWidth*ratio;
		int dstHeight = (double)imHeight*ratio;
		int dstDepth = (double)imDepth*ratio;
		if(result.width() != dstWidth || result.height() != dstHeight || result.depth() != dstDepth || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,dstDepth,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing3D::ResizeImageTricubic(pData,result.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
	}


	template<class T>
	void ZQ_DImage3D<T>::imresizeTricubic(ZQ_DImage3D<T>& result, int dstWidth, int dstHeight, int dstDepth) const
	{
		if(result.width() != dstWidth || result.height() != dstHeight || result.depth() != dstDepth || result.nchannels() != nChannels)
			result.allocate(dstWidth,dstHeight,dstDepth,nChannels);
		else
			result.reset();
		ZQ_ImageProcessing3D::ResizeImageTricubic(pData,result.data(),imWidth,imHeight,imDepth,nChannels,dstWidth,dstHeight,dstDepth);
	}



	template<class T>
	void ZQ_DImage3D<T>::dx(ZQ_DImage3D<T>& result, bool isAdvancedFilter) const
	{
		if(matchDimension(result) == false)
			result.allocate(imWidth,imHeight,imDepth,nChannels);
		else
			result.reset();

		T*& data = result.pData;

		if(!isAdvancedFilter)
		{
			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth-1;i++)
					{
						int offset = k*imHeight*imWidth+j*imWidth+i;
						for(int c = 0;c < nChannels;c++)
							data[offset*nChannels+c] = pData[(offset+1)*nChannels+c]-pData[offset*nChannels+c];
					}
				}
			}
		}
		else
		{
			T xFilter[5] = {1,-8,0,8,-1};
			for(int i = 0; i < 5;i++)
				xFilter[i] /= 12;
			ZQ_ImageProcessing3D::Hfiltering(pData,data,imWidth,imHeight,imDepth,nChannels,xFilter,2);
		}
	}


	template<class T>
	void ZQ_DImage3D<T>::dy(ZQ_DImage3D<T>& result,bool isAdvancedFilter) const
	{
		if(matchDimension(result)==false)
			result.allocate(imWidth,imHeight,imDepth,nChannels);
		else
			result.reset();

		T*& data = result.pData;

		if(!isAdvancedFilter)
		{
			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight-1;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imHeight*imWidth+j*imWidth+i;
						for(int c = 0;c < nChannels;c++)
							data[offset*nChannels+c]=pData[(offset+imWidth)*nChannels+c]-pData[offset*nChannels+c];
					}
				}
			}
		}
		else
		{
			T yFilter[5]={1,-8,0,8,-1};
			for(int i = 0;i < 5;i++)
				yFilter[i] /= 12;
			ZQ_ImageProcessing3D::Vfiltering(pData,data,imWidth,imHeight,imDepth,nChannels,yFilter,2);
		}
	}

	template<class T>
	void ZQ_DImage3D<T>::dz(ZQ_DImage3D& result, bool isAdvancedFilter) const
	{
		if(matchDimension(result)==false)
			result.allocate(imWidth,imHeight,imDepth,nChannels);
		else
			result.reset();

		T*& data = result.pData;

		if(!isAdvancedFilter)
		{
			for(int k = 0;k < imDepth-1;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imHeight*imWidth+j*imWidth+i;
						for(int c = 0;c < nChannels;c++)
							data[offset*nChannels+c]=pData[(offset+imWidth*imHeight)*nChannels+c]-pData[offset*nChannels+c];
					}
				}
			}
		}
		else
		{
			T zFilter[5]={1,-8,0,8,-1};
			for(int i = 0;i < 5;i++)
				zFilter[i] /= 12;
			ZQ_ImageProcessing3D::Dfiltering(pData,data,imWidth,imHeight,imDepth,nChannels,zFilter,2);
		}
	}



	template<class T>
	void ZQ_DImage3D<T>::GaussianSmoothing(float sigma,int fsize) 
	{
		ZQ_DImage3D<T> foo;
		GaussianSmoothing(foo,sigma,fsize);
		copyData(foo);
	}


	template<class T>
	void ZQ_DImage3D<T>::GaussianSmoothing(ZQ_DImage3D<T>& image, float sigma,int fsize) const 
	{
		T* gFilter = new T[fsize*2+1];
		double sum = 0;
		double m_sigma = sigma*sigma*2;
		for(int i = -fsize;i <= fsize;i++)
		{
			gFilter[i+fsize] = exp(-(double)(i*i)/m_sigma);
			sum += gFilter[i+fsize];
		}
		for(int i = 0;i < 2*fsize+1;i++)
			gFilter[i] /= sum;

		imfilter_hvd(image,gFilter,fsize,gFilter,fsize,gFilter,fsize);

		delete []gFilter;
	}


	template<class T>
	void ZQ_DImage3D<T>::imfilter_h(ZQ_DImage3D<T>& image, T* filter,int fsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,imDepth,nChannels);
		ZQ_ImageProcessing3D::Hfiltering(pData,image.data(),imWidth,imHeight,imDepth,nChannels,filter,fsize);
	}


	template<class T>
	void ZQ_DImage3D<T>::imfilter_v(ZQ_DImage3D<T>& image, T* filter,int fsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,imDepth,nChannels);
		ZQ_ImageProcessing3D::Vfiltering(pData,image.data(),imWidth,imHeight,imDepth,nChannels,filter,fsize);
	}

	template<class T>
	void ZQ_DImage3D<T>::imfilter_d(ZQ_DImage3D<T>& image, T* filter, int fsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,imDepth,nChannels);
		ZQ_ImageProcessing3D::Dfiltering(pData,image.data(),imWidth,imHeight,imDepth,nChannels,filter,fsize);
	}



	template<class T>
	void ZQ_DImage3D<T>::imfilter_hvd(ZQ_DImage3D<T> &image, const T *hfilter, int hfsize, const T *vfilter, int vfsize, const T *dfilter, int dfsize) const
	{
		if(matchDimension(image) == false)
			image.allocate(imWidth,imHeight,imDepth,nChannels);
		T* pTempBuffer = new T[nElements];
		ZQ_ImageProcessing3D::Hfiltering(pData,image.data(),imWidth,imHeight,imDepth,nChannels,hfilter,hfsize);
		ZQ_ImageProcessing3D::Vfiltering(image.data(),pTempBuffer,imWidth,imHeight,imDepth,nChannels,vfilter,vfsize);
		ZQ_ImageProcessing3D::Dfiltering(pTempBuffer,image.data(),imWidth,imHeight,imDepth,nChannels,dfilter,dfsize);
		delete []pTempBuffer;
	}


	template<class T>
	void ZQ_DImage3D<T>::collapse(ZQ_DImage3D<T>& image) const
	{
		if(!(image.imWidth == imWidth && image.imHeight == imHeight && image.imDepth == imDepth && image.nChannels == 1))
			image.allocate(imWidth,imHeight,imDepth,1);

		if(nChannels == 1)
		{
			image.copyData(*this);
			return;
		}
		T* data = image.data();

		for(int i = 0;i < nPixels;i++)
		{
			int offset = i*nChannels;
			T temp = 0;

			for(int c = 0;c < nChannels;c++)
				temp += pData[offset+c];
			data[i] = temp / nChannels;
		}
	}



	template<class T>
	void ZQ_DImage3D<T>::collapse()
	{
		if(nChannels == 1)
			return;
		ZQ_DImage3D<T> result;
		collapse(result);
		copyData(result);
	}


	template<class T>
	void ZQ_DImage3D<T>::separate(unsigned int firstNChannels, ZQ_DImage3D<T> &image1, ZQ_DImage3D<T> &image2) const
	{
		if(firstNChannels >= nChannels)
		{
			image1 = *this;
			image2.allocate(imWidth,imHeight,imDepth,0);
			return;
		}
		else if(firstNChannels == 0)
		{
			image1.allocate(imWidth,imHeight,imDepth,0);
			image2 = *this;
			return ;
		}
		else
		{
			int secondNChannels = nChannels-firstNChannels;
			if(image1.width() != imWidth || image1.height() != imHeight || image1.depth() != imDepth || image1.nchannels() != firstNChannels)
				image1.allocate(imWidth,imHeight,imDepth,firstNChannels);
			if(image2.width() != imWidth || image2.height() != imHeight || image2.depth() != imDepth || image2.nchannels() != secondNChannels)
				image2.allocate(imWidth,imHeight,imDepth,secondNChannels);

			for(int k = 0;k < imDepth;k++)
			{
				for(int j = 0;j < imHeight;j++)
				{
					for(int i = 0;i < imWidth;i++)
					{
						int offset = k*imHeight*imWidth+j*imWidth+i;
						for(int c = 0;c < firstNChannels;c++)
							image1.pData[offset*firstNChannels+c] = pData[offset*nChannels+c];
						for(int c = firstNChannels;c < nChannels;c++)
							image2.pData[offset*secondNChannels+c-firstNChannels] = pData[offset*nChannels+c];
					}
				}
			}
		}
	}


	template<class T>
	void ZQ_DImage3D<T>::assemble(const ZQ_DImage3D<T> &u, const  ZQ_DImage3D<T> &v, const ZQ_DImage3D<T> &w)
	{
		int width = u.width();
		int height = u.height();
		int depth = u.depth();
		int nchannles = u.nchannels();

		if(nchannles != 1 || v.matchDimension(u) == false || w.matchDimension(u) == false)
			return ;

		allocate(width,height,depth,3);
		for(int i = 0;i < nPixels;i++)
		{
			pData[i*3+0] = u.pData[i];
			pData[i*3+1] = v.pData[i];
			pData[i*3+2] = w.pData[i];
		}
	}


	template<class T>
	void ZQ_DImage3D<T>::Multiply(const ZQ_DImage3D<T>& image1,const ZQ_DImage3D<T>& image2)
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

		for(int i = 0;i < nElements;i++)
			pData[i] = pData1[i]*pData2[i];
	}



	template<class T>
	void ZQ_DImage3D<T>::Multiply(const ZQ_DImage3D<T>& image1,const ZQ_DImage3D<T>& image2,const ZQ_DImage3D<T>& image3)
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

		for(int i = 0;i < nElements;i++)
			pData[i] = pData1[i]*pData2[i]*pData3[i];
	}



	template<class T>
	void ZQ_DImage3D<T>::Multiplywith(const ZQ_DImage3D<T> &image1)
	{
		if(matchDimension(image1)==false)
		{
			printf("Error in image dimensions--function ZQ_DImage::Multiplywith()!\n");
			return;
		}
		const T*& pData1 = image1.data();
		for(int i = 0;i < nElements;i++)
			pData[i] *= pData1[i];
	}

	template<class T>
	void ZQ_DImage3D<T>::Multiplywith(T value)
	{
		for(int i = 0;i < nElements;i++)
			pData[i] *= value;
	}


	template<class T>
	void ZQ_DImage3D<T>::Add(const ZQ_DImage3D<T>& image1,const ZQ_DImage3D<T>& image2)
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
		for(int i = 0;i < nElements;i++)
			pData[i] = pData1[i]+pData2[i];	
	}


	template<class T>
	void ZQ_DImage3D<T>::Addwith(const ZQ_DImage3D<T>& image1,const T ratio)
	{
		if(matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::Addwith()!\n");
			return;
		}
		const T*& pData1 = image1.data();
		for(int i = 0;i < nElements;i++)
			pData[i] += pData1[i]*ratio;	
	}



	template<class T>
	void ZQ_DImage3D<T>::Addwith(const ZQ_DImage3D<T>& image1)
	{
		if(matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::Addwith()!\n");
			return;
		}
		const T*& pData1 = image1.data();
		for(int i = 0;i < nElements;i++)
			pData[i] += pData1[i];	
	}


	template<class T>
	void ZQ_DImage3D<T>::Addwith(const T value)
	{
		for(int i = 0;i < nElements;i++)
			pData[i] += value;
	}



	template<class T>
	void ZQ_DImage3D<T>::Subtract(const ZQ_DImage3D<T>& image1, const ZQ_DImage3D<T>& image2)
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
		for(int i = 0;i < nElements;i++)
			pData[i] = pData1[i]-pData2[i];
	}


	template<class T>
	void ZQ_DImage3D<T>::Subtractwith(const ZQ_DImage3D<T>& image1)
	{
		if(matchDimension(image1) == false)
		{
			printf("Error in image dimensions--function ZQ_Image::SubtractWith()!\n");
			return;
		}

		const T*& pData1 = image1.data();
		for(int i = 0;i < nElements;i++)
			pData[i] -= pData1[i];
	}



	template<class T>
	void ZQ_DImage3D<T>::warpImage(ZQ_DImage3D<T> &output, const ZQ_DImage3D<T>& u, const ZQ_DImage3D<T>& v, const ZQ_DImage3D<T>& w) const
	{
		if(!output.matchDimension(*this))
			output.allocate(*this);
		ZQ_ImageProcessing3D::WarpImage(output.data(),pData,u.data(),v.data(),w.data(),imWidth,imHeight,imDepth,nChannels);
	}


	template<class T>
	bool ZQ_DImage3D<T>::saveImage(const char *filename) const
	{
		FILE* out = 0;
		if(0 != fopen_s(&out, filename, "wb"))
			return false;

		char type[16];
		sprintf_s(type,"%s",typeid(T).name());
		fwrite(type,sizeof(char),16,out);
		fwrite(&imWidth,sizeof(int),1,out);
		fwrite(&imHeight,sizeof(int),1,out);
		fwrite(&imDepth,sizeof(int),1,out);
		fwrite(&nChannels,sizeof(int),1,out);

		bool isDerivativeImage = false;//unused

		fwrite(&isDerivativeImage,sizeof(bool),1,out);
		fwrite(pData,sizeof(T),nElements,out);

		fclose(out);
		return true;
	}

	template<class T>
	bool ZQ_DImage3D<T>::loadImage(const char* filename)
	{
		FILE* in = 0;
		if(0 != fopen_s(&in,filename, "rb"))
			return false;

		char type[16];
		fread(type,sizeof(char),16,in);

		int width,height,depth,nchannels;
		fread(&width,sizeof(int),1,in);
		fread(&height,sizeof(int),1,in);
		fread(&depth,sizeof(int),1,in);
		fread(&nchannels,sizeof(int),1,in);

		bool isDerivative;
		fread(&isDerivative,sizeof(bool),1,in);

		if(!matchDimension(width,height,depth,nchannels))
			allocate(width,height,depth,nchannels);

		if(_strcmpi(type,typeid(double).name()) == 0)
		{
			if(strcmp(typeid(T).name(), "double") == 0)
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