#ifndef _ZQ_SHAPE_DEFORMATION_OPTIONS_H_
#define _ZQ_SHAPE_DEFORMATION_OPTIONS_H_
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace ZQ
{
	class ZQ_ShapeDeformationOptions
	{
	public:
		enum MethodType
		{
			METHOD_LAPLACIAN,
			METHOD_ARAP_VERT_AS_CENTER,
			METHOD_ARAP_TRIANGLE_AS_CENTER
		};
	public:
		ZQ_ShapeDeformationOptions()
		{
			Reset();
		}
		~ZQ_ShapeDeformationOptions()
		{

		}

		MethodType methodType;
		int FPIteration;
		int Iteration;

		void Reset()
		{
			methodType = METHOD_ARAP_VERT_AS_CENTER;
			FPIteration = 3;
			Iteration = 1000;
		}

		bool HandleArgs(const int argc, const char** argv)
		{
			for(int k = 0;k < argc;k++)
			{
				if(_strcmpi(argv[k],"methodtype") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of %s ?\n",argv[k-1]);
						return false;
					}

					if(_strcmpi(argv[k],"LAPLACIAN") == 0)
						methodType = METHOD_LAPLACIAN;
					else if(_strcmpi(argv[k],"ARAP_VERT") == 0)
						methodType = METHOD_ARAP_VERT_AS_CENTER;
					else if(_strcmpi(argv[k],"ARAP_TRIANGLE") == 0)
						methodType = METHOD_ARAP_TRIANGLE_AS_CENTER;
					else
					{
						printf("unknown para %s\n",argv[k]);
						return false;
					}
				}
				else if(_strcmpi(argv[k],"FPIteration") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of %s ?\n",argv[k-1]);
						return false;
					}
					FPIteration = atoi(argv[k]);
				}
				else if(_strcmpi(argv[k],"Iteration") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of %s ?\n",argv[k-1]);
						return false;
					}
					Iteration = atoi(argv[k]);
				}
				else
				{
					printf("unknown para %s \n",argv[k]);
					return false;
				}
			}
			return true;
		}
	};
}

#endif