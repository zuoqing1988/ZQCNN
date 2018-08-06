#ifndef _ZQ_STRUCTURE_FROM_TEXTURE_H_
#define _ZQ_STRUCTURE_FROM_TEXTURE_H_

#include "ZQ_DoubleImage.h"
#include "ZQ_SparseMatrix.h"
#include "ZQ_PCGSolver.h"
#include "ZQ_StructureFromTextureOptions.h"
#include <time.h>

namespace ZQ
{
	class ZQ_StructureFromTexture
	{
	public:
		ZQ_StructureFromTexture(){}
		~ZQ_StructureFromTexture(){}

	public:
		template<class T>
		static bool StructureFromTexture(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

	protected:
		template<class T>
		static bool StructureFromTextureTVL1(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

		template<class T>
		static bool StructureFromTextureTVL2(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

		template<class T>
		static bool StructureFromTextureRTVL1_OLD(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

		template<class T>
		static bool StructureFromTextureRTVL1(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

		template<class T>
		static bool StructureFromTextureRTVL2(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

		template<class T>
		static bool StructureFromTexturePenaltyGradientWeight(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt);

	};


	/*********************************************************************************/
	/********************************** definitions **********************************/
	/*********************************************************************************/

	/**************************  functions for users *****************************/

	

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTexture(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{
		switch (opt.type)
		{
		case ZQ_StructureFromTextureOptions::TYPE_TVL1:
			{
				return StructureFromTextureTVL1(input,output,opt);
			}
			break;
		case ZQ_StructureFromTextureOptions::TYPE_TVL2:
			{
				return StructureFromTextureTVL2(input,output,opt);
			}
			break;
		case ZQ_StructureFromTextureOptions::TYPE_RTVL1_OLD:
			{
				return StructureFromTextureRTVL1(input,output,opt);
			}
			break;
		case ZQ_StructureFromTextureOptions::TYPE_RTVL1:
			{
				return StructureFromTextureRTVL1(input,output,opt);
			}
			break;
		case ZQ_StructureFromTextureOptions::TYPE_RTVL2:
			{
				return StructureFromTextureRTVL2(input,output,opt);
			}
			break;
		case ZQ_StructureFromTextureOptions::TYPE_PENALTY_GRADIENT_WEIGHT:
			{
				return StructureFromTexturePenaltyGradientWeight(input,output,opt);
			}
			break;
		}
		return false;
	}

	
	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTextureTVL1(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{
		/*
		* Two models have the similar effect
		* (1) \int |S-I|_2^2 d\Omega + \lambda \int |S_x| + |S_y| d\Omega
		*     Let: |S_x| := \sqrt{S_x^2 + epsilon}
		*          |S_y| := \sqrt{S_y^2 + epsilon}
		*    Then Euler-Lagrange equation is :
		*        2(S-I) = \lambda \partial_x {\frac{S_x}{\sqrt{S_x^2+epsilon}}} + \lambda \partial_y {\frac{S_y}{\sqrt{S_y^2+epsilon}}}
		*    Use Fixed-Point Iteration on S
		*        2(S^{k+1}-I) = \lambda \partial_x {C^k S_x^{k+1}} + \lambda \partial_y {D^k S_y^{k+1}},
		*      where C^k := 1/sqrt{(S_x^k)^2+epsilon}, D^k = 1/sqrt{(S_y^k)^2+epislon},
		*     We can have a smoother value for S_x^k, S_y^k , or have a smoother value for C^k, D^k, that means we use a Gaussian filter to convolve it.
		*
		* (2) \int |S-I|_2^2 d\Omega + \lambda \int g*|S_x| + g*|S_y|d\Omega
		*     Here g* means using a Gaussian to convolve it,
		*     Let : |S_x| := 1/\sqrt{S_x^2 + epsilon} |S_x|^2
		*           |S_y| := 1/\sqrt{S_y^2 + epsilon} |S_y|^2
		*     \int g*|S_x| d\Omega = \int g* (C S_x^2) d\Omega, where C := 1/\sqrt{S_x^2 + epsilon}
		*                          = \int (g*C) S_x^2 d\Omega, with period boundary 
		*     Use Fixed-Point Iteration on S
		*        \int |S^{k+1}-I|_2^2 d\Omega + \lambda \int (g*C^k) (S_x^{k+1})^2 + \lambda \int (g*D^k) (S_y^{k+1})^2 d\Omega
		*     Then Euler-Lagrange equation is :
		*        2(S-I) = 2\lambda \partial_x {C^k S_x^{k+1}} + 2\lambda \partial_y {D^k S_y^{k+1}},
		*
		*/
		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		epsilon *= epsilon;
		float weight = opt.weight;

		int dim = width*height;

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();

		/* handle each channel of the image*/
		for(int c = 0;c < nChannels;c++)
		{
			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				ZQ_DImage<T> Sx,Sy;
				image.dx(Sx);
				image.dy(Sy);
				
				
				T*& Sx_data = Sx.data();
				T*& Sy_data = Sy.data();

				for(int pp = 0;pp < height*width;pp++)
				{
					Sx_data[pp] = 0.5/sqrt(Sx_data[pp]*Sx_data[pp]+epsilon);
					Sy_data[pp] = 0.5/sqrt(Sy_data[pp]*Sy_data[pp]+epsilon);
				}

				Sx.GaussianSmoothing(2,3);
				Sy.GaussianSmoothing(2,3);


				for(int it = 0;it < opt.nSolverIteration;it++)
				{
					for(int h = 0;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							float coeff = 0, sigma = 0;
							if(h < height-1)
							{
								coeff += weight*Sy_data[h*width+w];
								sigma += weight*Sy_data[h*width+w]*image_data[(h+1)*width+w];
							}
							if(h > 0)
							{
								coeff += weight*Sy_data[(h-1)*width+w];
								sigma += weight*Sy_data[(h-1)*width+w]*image_data[(h-1)*width+w];
							}
							if(w < width-1)
							{
								coeff += weight*Sx_data[h*width+w];
								sigma += weight*Sx_data[h*width+w]*image_data[h*width+w+1];
							}
							if(w > 0)
							{
								coeff += weight*Sx_data[h*width+w-1];
								sigma += weight*Sx_data[h*width+w-1]*image_data[h*width+w-1];
							}

							coeff += 1;
							sigma += input_data[(h*width+w)*nChannels+c];
							image_data[h*width+w] = sigma/coeff;
						}
					}
				}

			}

			for(int pp = 0;pp < dim;pp++)
			{
				output_data[pp*nChannels+c] = image_data[pp];
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTextureTVL2(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{
		/* (1) The simplest model is 
		*      \int |S-I|_2^2 d\Omega + \int S_x^2 + S_y^2 d\Omega
		*     Euler-Lagrange equation is 
		*       S-I = \lambda^2 \Delta S 
		*     However, we find this mode does not work well.
		*  
		*
		*/

		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		float weight = opt.weight;
		weight *= weight;

		int dim = width*height;

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();

		/* handle each channel of the image*/
		for(int c = 0;c < nChannels;c++)
		{
			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				for(int it = 0;it < opt.nSolverIteration;it++)
				{
					for(int h = 0;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							float coeff = 0, sigma = 0;
							if(h < height-1)
							{
								coeff += weight;
								sigma += weight*image_data[(h+1)*width+w];
							}
							if(h > 0)
							{
								coeff += weight;
								sigma += weight*image_data[(h-1)*width+w];
							}
							if(w < width-1)
							{
								coeff += weight;
								sigma += weight*image_data[h*width+w+1];
							}
							if(w > 0)
							{
								coeff += weight;
								sigma += weight*image_data[h*width+w-1];
							}

							coeff += 1;
							sigma += input_data[(h*width+w)*nChannels+c];
							image_data[h*width+w] = sigma/coeff;
						}
					}
				}

			}

			for(int pp = 0;pp < dim;pp++)
			{
				output_data[pp*nChannels+c] = image_data[pp];
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTextureRTVL1_OLD(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{
		/***********************/
		/* The RTV-L1 model is proposed by the paper "Structure Extraction from Texture via Relative Total Variation, 2012".
		/* The model is given as:
		/*   E(S) = \int (S-I)^2 + \lambda (\frac{D_x}{P_x+\epsilon}  + \frac{D_y}{P_y+\epsilon} )  d\Omega 
		/*  where, E is the image to estimate, I the input image,
		/*   D_x = g * |S_x|, D_y = g * |S_y|, 
		/*   L_x = |g * S_x|, L_y = |g * S_y|,
		/*  the operator "*" is a convolution, and g is implemented as a Gaussian filter 
		/*****/
		clock_t t1 = clock();

		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();
		
		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		float epsilon_s = opt.epsilon_for_s;

		int dim = width*height;

		ZQ_SparseMatrix Iden(dim,dim);
		for(int pp = 0;pp < dim;pp++)
			Iden.SetValue(pp,pp,1);

		taucs_ccs_matrix* Iden_mat = Iden.ExportCCS(TAUCS_DOUBLE);

		taucs_ccs_matrix* Cx_mat = 0;
		taucs_ccs_matrix* Cy_mat = 0;
		taucs_ccs_matrix* CxT_mat = 0;
		taucs_ccs_matrix* CyT_mat = 0;


		/*** calculate derivative operator Begin ****/
		ZQ_SparseMatrix Cx(dim,dim),Cy(dim,dim);
		for(int h = 0;h < height;h++)
		{
			for(int w = 0;w < width-1;w++)
			{
				Cx.SetValue(h*width+w,h*width+w,-1);
				Cx.SetValue(h*width+w,h*width+w+1,1);
			}
		}

		for(int h = 0;h < height-1;h++)
		{
			for(int w = 0;w < width;w++)
			{
				Cy.SetValue(h*width+w,h*width+w,-1);
				Cy.SetValue(h*width+w,(h+1)*width+w,1);
			}
		}

		Cx_mat = Cx.ExportCCS(TAUCS_DOUBLE);
		Cy_mat = Cy.ExportCCS(TAUCS_DOUBLE);
		CxT_mat = TaucsBase::ZQ_taucs_ccs_matrixTranspose(Cx_mat);
		CyT_mat = TaucsBase::ZQ_taucs_ccs_matrixTranspose(Cy_mat);


		/*** calculate derivative operator End ****/

		clock_t t2 = clock();
		//printf("pre:%f\n",0.001*(t2-t1));

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();
		
		/* handle each channel of the image*/
		for(int c = 0;c < nChannels;c++)
		{
			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			double* b = new double[dim];
			for(int pp = 0;pp < dim;pp++)
			{
				b[pp] = input_data[pp*nChannels+c];;
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				

				ZQ_DImage<T> Sx,Sy,ux(width,height),uy(width,height),wx(width,height),wy(width,height);
				image.dx(Sx);
				image.dy(Sy);
				Sx.GaussianSmoothing(ux,sigma,fsize);
				Sy.GaussianSmoothing(uy,sigma,fsize);

				T*& Sx_data = Sx.data();
				T*& Sy_data = Sy.data();
				T*& ux_data = ux.data();
				T*& uy_data = uy.data();
				T*& wx_data = wx.data();
				T*& wy_data = wy.data();
				for(int pp = 0;pp < height*width;pp++)
				{
					ux_data[pp] = 1.0/(fabs(ux_data[pp])+epsilon);
					uy_data[pp] = 1.0/(fabs(uy_data[pp])+epsilon);
					wx_data[pp] = 1.0/(fabs(Sx_data[pp])+epsilon_s);
					wy_data[pp] = 1.0/(fabs(Sy_data[pp])+epsilon_s);
				}

				ux.GaussianSmoothing(sigma,fsize);
				uy.GaussianSmoothing(sigma,fsize);


				//caculate UxWx, UyWy
				ZQ_SparseMatrix UxWx(dim,dim),UyWy(dim,dim);
				for(int pp = 0;pp < height*width;pp++)
				{
					UxWx.SetValue(pp,pp,ux_data[pp]*wx_data[pp]);
					UyWy.SetValue(pp,pp,uy_data[pp]*wy_data[pp]);
				}

				

				taucs_ccs_matrix* UxWx_mat = UxWx.ExportCCS(TAUCS_DOUBLE);
				taucs_ccs_matrix* UyWy_mat = UyWy.ExportCCS(TAUCS_DOUBLE);

				

				clock_t t3 = clock();
				
				taucs_ccs_matrix* CxtUxWx_mat = TaucsBase::ZQ_taucs_ccs_mul2NonSymmetricMatrices(CxT_mat,UxWx_mat);
				taucs_ccs_matrix* CxtUxWxCx_mat = TaucsBase::ZQ_taucs_ccs_mul2NonSymmetricMatrices(CxtUxWx_mat,Cx_mat);
				taucs_ccs_matrix* CytUyWy_mat = TaucsBase::ZQ_taucs_ccs_mul2NonSymmetricMatrices(CyT_mat,UyWy_mat);
				taucs_ccs_matrix* CytUyWyCy_mat = TaucsBase::ZQ_taucs_ccs_mul2NonSymmetricMatrices(CytUyWy_mat,Cy_mat);
				clock_t t4 = clock();
				clock_t t5 = clock();
				taucs_ccs_matrix* tmp_L_mat = TaucsBase::ZQ_taucs_ccs_add2NonSymmetricMatrices(CxtUxWxCx_mat,CytUyWyCy_mat);
				taucs_ccs_matrix* L_mat = TaucsBase::ZQ_taucs_ccs_scaleMatrix(tmp_L_mat,opt.weight);
				
				taucs_ccs_matrix* A_mat = TaucsBase::ZQ_taucs_ccs_add2NonSymmetricMatrices(Iden_mat,L_mat);

				clock_t t6 = clock();

				TaucsBase::ZQ_taucs_ccs_free(UxWx_mat);
				TaucsBase::ZQ_taucs_ccs_free(UyWy_mat);
				TaucsBase::ZQ_taucs_ccs_free(CxtUxWx_mat);
				TaucsBase::ZQ_taucs_ccs_free(CytUyWy_mat);
				TaucsBase::ZQ_taucs_ccs_free(CxtUxWxCx_mat);
				TaucsBase::ZQ_taucs_ccs_free(CytUyWyCy_mat);
				TaucsBase::ZQ_taucs_ccs_free(tmp_L_mat);
				TaucsBase::ZQ_taucs_ccs_free(L_mat);

				

				clock_t t7 = clock();

				ZQ_PCGSolver solver;
				double tol = 1e-9;
				int it = 0;
				int max_iter = opt.nSolverIteration;
				double* x0 = new double[dim];
				double* x = new double[dim];
				for(int pp = 0;pp < dim;pp++)
				{
					x0[pp] = image_data[pp];
					x[pp] = image_data[pp];
				}
				solver.PCG_sparse_unsquare(A_mat,b,x0,opt.nSolverIteration,tol,x,it,false);
				for(int pp = 0;pp < dim;pp++)
					image_data[pp] = x[pp];

				delete []x0;
				delete []x;

				TaucsBase::ZQ_taucs_ccs_free(A_mat);

				clock_t t8 = clock();

			//	printf("part1 = %f, part2 = %f, part3 = %f\n",0.001*(t4-t3),0.001*(t6-t5),0.001*(t8-t7));
				
			}

			for(int pp = 0;pp < dim;pp++)
			{
				output_data[pp*nChannels+c] = image_data[pp];
			}

			delete []b;
		}

		TaucsBase::ZQ_taucs_ccs_free(Iden_mat);
		TaucsBase::ZQ_taucs_ccs_free(Cx_mat);
		TaucsBase::ZQ_taucs_ccs_free(Cy_mat);
		TaucsBase::ZQ_taucs_ccs_free(CxT_mat);
		TaucsBase::ZQ_taucs_ccs_free(CyT_mat);

		return true;

	}

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTextureRTVL1(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{

		/***********************/
		/* The RTV-L1 model is proposed by the paper "Structure Extraction from Texture via Relative Total Variation, 2012".
		/* The model is given as:
		/*   E(S) = \int (S-I)^2 + \lambda (\frac{D_x}{P_x+\epsilon}  + \frac{D_y}{P_y+\epsilon} )  d\Omega 
		/*  where, E is the image to estimate, I the input image,
		/*   D_x = g * |S_x|, D_y = g * |S_y|, 
		/*   L_x = |g * S_x|, L_y = |g * S_y|,
		/*  the operator "*" is a convolution, and g is implemented as a Gaussian filter 
		/*****/

		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		float epsilon_s = opt.epsilon_for_s;
		float weight = opt.weight;

		int dim = width*height;

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();

		/* handle each channel of the image*/
		for(int c = 0;c < nChannels;c++)
		{
			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				ZQ_DImage<T> Sx,Sy,ux(width,height),uy(width,height),wx(width,height),wy(width,height);
				image.dx(Sx);
				image.dy(Sy);
				Sx.GaussianSmoothing(ux,sigma,fsize);
				Sy.GaussianSmoothing(uy,sigma,fsize);

				T*& Sx_data = Sx.data();
				T*& Sy_data = Sy.data();
				T*& ux_data = ux.data();
				T*& uy_data = uy.data();
				T*& wx_data = wx.data();
				T*& wy_data = wy.data();
				for(int pp = 0;pp < height*width;pp++)
				{
					ux_data[pp] = 1.0/(fabs(ux_data[pp])+epsilon);
					uy_data[pp] = 1.0/(fabs(uy_data[pp])+epsilon);
					wx_data[pp] = 1.0/(fabs(Sx_data[pp])+epsilon_s);
					wy_data[pp] = 1.0/(fabs(Sy_data[pp])+epsilon_s);
				}

				ux.GaussianSmoothing(sigma,fsize);
				uy.GaussianSmoothing(sigma,fsize);


				//caculate UxWx, UyWy
				ZQ_DImage<T> UxWx,UyWy;
				UxWx.Multiply(ux,wx);
				UyWy.Multiply(uy,wy);

				T*& UxWx_data = UxWx.data();
				T*& UyWy_data = UyWy.data();

				for(int it = 0;it < opt.nSolverIteration;it++)
				{
					for(int h = 0;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							float coeff = 0, sigma = 0;
							if(h < height-1)
							{
								coeff += weight*UyWy_data[h*width+w];
								sigma += weight*UyWy_data[h*width+w]*image_data[(h+1)*width+w];
							}
							if(h > 0)
							{
								coeff += weight*UyWy_data[(h-1)*width+w];
								sigma += weight*UyWy_data[(h-1)*width+w]*image_data[(h-1)*width+w];
							}
							if(w < width-1)
							{
								coeff += weight*UxWx_data[h*width+w];
								sigma += weight*UxWx_data[h*width+w]*image_data[h*width+w+1];
							}
							if(w > 0)
							{
								coeff += weight*UxWx_data[h*width+w-1];
								sigma += weight*UxWx_data[h*width+w-1]*image_data[h*width+w-1];
							}

							coeff += 1;
							sigma += input_data[(h*width+w)*nChannels+c];
							image_data[h*width+w] = sigma/coeff;
						}
					}
				}

			}

			for(int pp = 0;pp < dim;pp++)
			{
				output_data[pp*nChannels+c] = image_data[pp];
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTextureRTVL2(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{

		
		/***********************/
		/*  Here, we propose a RTV-L2 model as
		/*   E(S) = \int (S-I)^2 + \lambda (\frac{D_x}{P_x+\epsilon}  + \frac{D_y}{P_y+\epsilon} )  d\Omega 
		/*  where
		/*   D_x = g * ||S_x||^2, D_y = g * ||S_y||^2, 
		/*   L_x = ||g * S_x||^2, L_y = ||g * S_y||^2,
		/*
		/*****/


		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		epsilon *= epsilon;
		float weight = opt.weight;
		weight *= weight;

		int dim = width*height;

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();

		/* handle each channel of the image*/
		for(int c = 0;c < nChannels;c++)
		{
			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				ZQ_DImage<T> Sx,Sy,ux(width,height),uy(width,height);
				image.dx(Sx);
				image.dy(Sy);
				Sx.GaussianSmoothing(ux,sigma,fsize);
				Sy.GaussianSmoothing(uy,sigma,fsize);

				T*& Sx_data = Sx.data();
				T*& Sy_data = Sy.data();
				T*& ux_data = ux.data();
				T*& uy_data = uy.data();
				
				for(int pp = 0;pp < height*width;pp++)
				{
					ux_data[pp] = 1.0/(ux_data[pp]*ux_data[pp]+epsilon);
					uy_data[pp] = 1.0/(uy_data[pp]*uy_data[pp]+epsilon);
				}

				ux.GaussianSmoothing(sigma,fsize);
				uy.GaussianSmoothing(sigma,fsize);


				T*& UxWx_data = ux.data();
				T*& UyWy_data = uy.data();

				for(int it = 0;it < opt.nSolverIteration;it++)
				{
					for(int h = 0;h < height;h++)
					{
						for(int w = 0;w < width;w++)
						{
							float coeff = 0, sigma = 0;
							if(h < height-1)
							{
								coeff += weight*UyWy_data[h*width+w];
								sigma += weight*UyWy_data[h*width+w]*image_data[(h+1)*width+w];
							}
							if(h > 0)
							{
								coeff += weight*UyWy_data[(h-1)*width+w];
								sigma += weight*UyWy_data[(h-1)*width+w]*image_data[(h-1)*width+w];
							}
							if(w < width-1)
							{
								coeff += weight*UxWx_data[h*width+w];
								sigma += weight*UxWx_data[h*width+w]*image_data[h*width+w+1];
							}
							if(w > 0)
							{
								coeff += weight*UxWx_data[h*width+w-1];
								sigma += weight*UxWx_data[h*width+w-1]*image_data[h*width+w-1];
							}

							coeff += 1;
							sigma += input_data[(h*width+w)*nChannels+c];
							image_data[h*width+w] = sigma/coeff;
						}
					}
				}

			}

			for(int pp = 0;pp < dim;pp++)
			{
				output_data[pp*nChannels+c] = image_data[pp];
			}
		}
		return true;
	}

	template<class T>
	bool ZQ_StructureFromTexture::StructureFromTexturePenaltyGradientWeight(const ZQ_DImage<T>& input, ZQ_DImage<T>& output, const ZQ_StructureFromTextureOptions& opt)
	{
		/************************
		/* WLS model see the paper "Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation, 2008"
		/* RTV model see the paper "Structure Extraction from Texture via Relative Total Variation, 2012"
		/* this function have extended the models in these two papers. 
		/* one contribution is that the norm can be 1-norm, 2-norm or p-norm,
		/* the other contribution is the optimization process
		/********************/

		int width = input.width();
		int height = input.height();
		int nChannels = input.nchannels();

		float sigma = opt.sigma_for_filter;
		float fsize = opt.fsize_for_filter;
		float epsilon = opt.epsilon;
		float epsilon_d = opt.epsilon_for_d;
		float epsilon_s = opt.epsilon_for_s;
		float weight = opt.weight;
		float norm_for_rtv = opt.norm_for_rtv;
		float norm_for_wls = opt.norm_for_wls;
		float norm_for_dataterm = opt.norm_for_dataterm;

		
		int dim = width*height;

		const T*& input_data = input.data();
		output.allocate(width,height,nChannels);
		T*& output_data = output.data();

		for(int c = 0;c < nChannels;c++)
		{

			ZQ_DImage<T> image(width,height);
			T*& image_data = image.data();

			for(int pp = 0;pp < dim;pp++)
			{
				image_data[pp] = input_data[pp*nChannels+c];
			}

			ZQ_DImage<T> weightfiledx_ori(width,height),weightfiledy_ori(width,height);
			T*& weightfieldx_ori_data = weightfiledx_ori.data();
			T*& weightfieldy_ori_data = weightfiledy_ori.data();

			switch(opt.penaltyWeightType)
			{
			case ZQ_StructureFromTextureOptions::WEIGHT_RTV_MIX:
				{
					ZQ_DImage<T> Sx,Sy;
					image.dx(Sx);
					image.dy(Sy);

					T*& Sx_data = Sx.data();
					T*& Sy_data = Sy.data();

					ZQ_DImage<T> Dx(width,height),Dy(width,height),Lx(width,height),Ly(width,height);
					T*& Dx_data = Dx.data();
					T*& Dy_data = Dy.data();
					T*& Lx_data = Lx.data();
					T*& Ly_data = Ly.data();


					Sx.GaussianSmoothing(Lx,sigma,fsize);
					Sy.GaussianSmoothing(Ly,sigma,fsize);

					for(int pp = 0;pp < dim;pp++)
					{
						Lx_data[pp] = 1.0/(pow(fabs((double)Lx_data[pp]),(double)norm_for_rtv)+epsilon);
						Ly_data[pp] = 1.0/(pow(fabs((double)Ly_data[pp]),(double)norm_for_rtv)+epsilon);
					}

					Lx.GaussianSmoothing(sigma,fsize);
					Ly.GaussianSmoothing(sigma,fsize);

					for(int pp = 0;pp < dim;pp++)
					{
						Dx_data[pp] = 1.0/(pow(fabs((double)Sx_data[pp]),2-(double)norm_for_rtv)+epsilon_s);
						Dy_data[pp] = 1.0/(pow(fabs((double)Sy_data[pp]),2-(double)norm_for_rtv)+epsilon_s);
					}


					for(int pp = 0;pp < dim;pp++)
					{
						weightfieldx_ori_data[pp] = Dx_data[pp]*Lx_data[pp];
						weightfieldy_ori_data[pp] = Dy_data[pp]*Ly_data[pp];
					}
				}
				break;
			case ZQ_StructureFromTextureOptions::WEIGHT_WLS_MIX:
				{
					ZQ_DImage<T> logimg(width,height);
					T*& logimg_data = logimg.data();
					for(int pp = 0;pp < dim;pp++)
						logimg_data[pp] = log(image_data[pp]+1e-6);

					ZQ_DImage<T> Sx,Sy;
					logimg.dx(Sx);
					logimg.dy(Sy);
					T*& Sx_data = Sx.data();
					T*& Sy_data = Sy.data();
					for(int pp = 0;pp < dim;pp++)
					{
						weightfieldx_ori_data[pp] = 1.0/(pow(fabs((double)Sx_data[pp]),(double)norm_for_wls)+epsilon);
						weightfieldy_ori_data[pp] = 1.0/(pow(fabs((double)Sy_data[pp]),(double)norm_for_wls)+epsilon);
					}
				}
				break;
			}

			for(int out_it = 0;out_it < opt.nOuterIteration;out_it++)
			{
				ZQ_DImage<T> weightfiledx(width,height),weightfiledy(width,height);
				T*& weightfieldx_data = weightfiledx.data();
				T*& weightfieldy_data = weightfiledy.data();

				switch(opt.penaltyWeightType)
				{
				case ZQ_StructureFromTextureOptions::WEIGHT_RTV: 
				case ZQ_StructureFromTextureOptions::WEIGHT_RTV_MIX:
					{
						ZQ_DImage<T> Sx,Sy;
						image.dx(Sx);
						image.dy(Sy);

						T*& Sx_data = Sx.data();
						T*& Sy_data = Sy.data();

						ZQ_DImage<T> Dx(width,height),Dy(width,height),Lx(width,height),Ly(width,height);
						T*& Dx_data = Dx.data();
						T*& Dy_data = Dy.data();
						T*& Lx_data = Lx.data();
						T*& Ly_data = Ly.data();


						Sx.GaussianSmoothing(Lx,sigma,fsize);
						Sy.GaussianSmoothing(Ly,sigma,fsize);

						for(int pp = 0;pp < dim;pp++)
						{
							Lx_data[pp] = 1.0/(pow(fabs((double)Lx_data[pp]),(double)norm_for_rtv)+epsilon);
							Ly_data[pp] = 1.0/(pow(fabs((double)Ly_data[pp]),(double)norm_for_rtv)+epsilon);
						}

						Lx.GaussianSmoothing(sigma,fsize);
						Ly.GaussianSmoothing(sigma,fsize);

						for(int pp = 0;pp < dim;pp++)
						{
							Dx_data[pp] = 1.0/(pow(fabs((double)Sx_data[pp]),2-(double)norm_for_rtv)+epsilon_s);
							Dy_data[pp] = 1.0/(pow(fabs((double)Sy_data[pp]),2-(double)norm_for_rtv)+epsilon_s);
						}


						for(int pp = 0;pp < dim;pp++)
						{
							weightfieldx_data[pp] = Dx_data[pp]*Lx_data[pp];
							weightfieldy_data[pp] = Dy_data[pp]*Ly_data[pp];
						}
					}
					break;
				case ZQ_StructureFromTextureOptions::WEIGHT_WLS:
				case ZQ_StructureFromTextureOptions::WEIGHT_WLS_MIX:
					{
						ZQ_DImage<T> logimg(width,height);
						T*& logimg_data = logimg.data();
						for(int pp = 0;pp < dim;pp++)
							logimg_data[pp] = log(image_data[pp]+1e-6);

						ZQ_DImage<T> Sx,Sy;
						logimg.dx(Sx);
						logimg.dy(Sy);
						T*& Sx_data = Sx.data();
						T*& Sy_data = Sy.data();
						for(int pp = 0;pp < dim;pp++)
						{
							weightfieldx_data[pp] = 1.0/(pow(fabs((double)Sx_data[pp]),(double)norm_for_wls)+epsilon);
							weightfieldy_data[pp] = 1.0/(pow(fabs((double)Sy_data[pp]),(double)norm_for_wls)+epsilon);
						}
					}
					break;
				}


				switch(opt.penaltyWeightType)
				{
				case ZQ_StructureFromTextureOptions::WEIGHT_WLS_MIX:
				case ZQ_StructureFromTextureOptions::WEIGHT_RTV_MIX:
					{
						for(int pp = 0;pp < dim;pp++)
						{
							weightfieldx_data[pp] = 0.5*(weightfieldx_data[pp]+weightfieldx_ori_data[pp]);
							weightfieldy_data[pp] = 0.5*(weightfieldy_data[pp]+weightfieldy_ori_data[pp]);
						}
					}
					break;
				}

				
				if(norm_for_dataterm != 2)
				{
					ZQ_DImage<T> dataterm_weight(width,height);
					T*& dataterm_weight_data = dataterm_weight.data();

					for(int pp = 0;pp < dim;pp++)
						dataterm_weight_data[pp] = 1.0/(pow(fabs((double)image_data[pp]-input_data[pp*nChannels+c]),2-(double)norm_for_dataterm)+epsilon_d);

					for(int it = 0;it < opt.nSolverIteration;it++)
					{
						for(int h = 0;h < height;h++)
						{
							for(int w = 0;w < width;w++)
							{
								float coeff = 0, sigma = 0;
								if(h < height-1)
								{
									coeff += weight*weightfieldy_data[h*width+w];
									sigma += weight*weightfieldy_data[h*width+w]*image_data[(h+1)*width+w];
								}
								if(h > 0)
								{
									coeff += weight*weightfieldy_data[(h-1)*width+w];
									sigma += weight*weightfieldy_data[(h-1)*width+w]*image_data[(h-1)*width+w];
								}
								if(w < width-1)
								{
									coeff += weight*weightfieldx_data[h*width+w];
									sigma += weight*weightfieldx_data[h*width+w]*image_data[h*width+w+1];
								}
								if(w > 0)
								{
									coeff += weight*weightfieldx_data[h*width+w-1];
									sigma += weight*weightfieldx_data[h*width+w-1]*image_data[h*width+w-1];
								}

								coeff += dataterm_weight_data[h*width+w];
								sigma += dataterm_weight_data[h*width+w]*input_data[(h*width+w)*nChannels+c];
								image_data[h*width+w] = sigma/coeff;
							}
						}
					}

				}
				else
				{
					for(int it = 0;it < opt.nSolverIteration;it++)
					{
						for(int h = 0;h < height;h++)
						{
							for(int w = 0;w < width;w++)
							{
								float coeff = 0, sigma = 0;
								if(h < height-1)
								{
									coeff += weight*weightfieldy_data[h*width+w];
									sigma += weight*weightfieldy_data[h*width+w]*image_data[(h+1)*width+w];
								}
								if(h > 0)
								{
									coeff += weight*weightfieldy_data[(h-1)*width+w];
									sigma += weight*weightfieldy_data[(h-1)*width+w]*image_data[(h-1)*width+w];
								}
								if(w < width-1)
								{
									coeff += weight*weightfieldx_data[h*width+w];
									sigma += weight*weightfieldx_data[h*width+w]*image_data[h*width+w+1];
								}
								if(w > 0)
								{
									coeff += weight*weightfieldx_data[h*width+w-1];
									sigma += weight*weightfieldx_data[h*width+w-1]*image_data[h*width+w-1];
								}

								coeff += 1;
								sigma += input_data[(h*width+w)*nChannels+c];
								image_data[h*width+w] = sigma/coeff;
							}
						}
					}
				}

				
				for(int pp = 0;pp < dim;pp++)
				{
					output_data[pp*nChannels+c] = image_data[pp];
				}

			}
			

		}
		return true;
	}

}

#endif