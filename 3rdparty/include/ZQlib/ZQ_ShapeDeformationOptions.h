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
#if defined(_WIN32)
				if(_strcmpi(argv[k],"methodtype") == 0)
#else
				if (strcmp(argv[k], "methodtype") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of %s ?\n",argv[k-1]);
						return false;
					}
#if defined(_WIN32)
					if(_strcmpi(argv[k],"LAPLACIAN") == 0)
						methodType = METHOD_LAPLACIAN;
					else if(_strcmpi(argv[k],"ARAP_VERT") == 0)
						methodType = METHOD_ARAP_VERT_AS_CENTER;
					else if(_strcmpi(argv[k],"ARAP_TRIANGLE") == 0)
						methodType = METHOD_ARAP_TRIANGLE_AS_CENTER;
#else
					if (strcmp(argv[k], "LAPLACIAN") == 0)
						methodType = METHOD_LAPLACIAN;
					else if (strcmp(argv[k], "ARAP_VERT") == 0)
						methodType = METHOD_ARAP_VERT_AS_CENTER;
					else if (strcmp(argv[k], "ARAP_TRIANGLE") == 0)
						methodType = METHOD_ARAP_TRIANGLE_AS_CENTER;
#endif
					else
					{
						printf("unknown para %s\n",argv[k]);
						return false;
					}
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"FPIteration") == 0)
#else
				else if (strcmp(argv[k], "FPIteration") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of %s ?\n",argv[k-1]);
						return false;
					}
					FPIteration = atoi(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"Iteration") == 0)
#else
				else if (strcmp(argv[k], "Iteration") == 0)
#endif
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