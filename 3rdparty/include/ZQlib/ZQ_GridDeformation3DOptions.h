#ifndef _ZQ_GRID_DEFORMATION_3D_OPTIONS_H_
#define _ZQ_GRID_DEFORMATION_3D_OPTIONS_H_
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace ZQ
{
	class ZQ_GridDeformation3DOptions
	{
	public:
		enum MethodType{
			METHOD_LAPLACIAN,
			METHOD_LAPLACIAN_XLOOP,
			METHOD_ARAP_VERT_AS_CENTER,
			METHOD_ARAP_VERT_AS_CENTER_XLOOP
		};
		enum NeighborType{
			NEIGHBOR_6,
			NEIGHBOR_26
		};
	public:
		ZQ_GridDeformation3DOptions()
		{
			Reset();
		}
		~ZQ_GridDeformation3DOptions()
		{

		}

		MethodType methodType;
		NeighborType neighborType;
		float line_weight;
		float angle_weight;
		float distance_weight;
		float distance;
		float FPIteration;
		float iteration;

		void Reset()
		{
			methodType = METHOD_ARAP_VERT_AS_CENTER;
			line_weight = 1;
			angle_weight = 1;
			distance_weight = 1;
			distance = 1;
			FPIteration = 5;
			iteration = 1000;
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
						printf("the value of methodtype ?\n");
						return false;
					}
#if defined(_WIN32)
					if(_strcmpi(argv[k],"laplacian") == 0)
#else
					if (strcmp(argv[k], "laplacian") == 0)
#endif
					{
						methodType = METHOD_LAPLACIAN;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"laplacian_xloop") == 0)
#else
					else if(strcmp(argv[k], "laplacian_xloop") == 0)
#endif
					{
						methodType = METHOD_LAPLACIAN_XLOOP;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"arap_vert") == 0)
#else
					else if(strcmp(argv[k], "arap_vert") == 0)
#endif
					{
						methodType = METHOD_ARAP_VERT_AS_CENTER;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"arap_vert_xloop") == 0)
#else
					else if (strcmp(argv[k], "arap_vert_xloop") == 0)
#endif
					{
						methodType = METHOD_ARAP_VERT_AS_CENTER_XLOOP;
					}
					else
					{
						printf("unknown methodtype %s\n",argv[k]);
						return false;
					}
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"neighbortype") == 0)
#else
				else if (strcmp(argv[k], "neighbortype") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of neighbortype ?\n");
						return false;
					}
					if(strcmp(argv[k],"6") == 0)
					{
						neighborType = NEIGHBOR_6;
					}
					else if(strcmp(argv[k],"26") == 0)
					{
						neighborType = NEIGHBOR_26;
					}
					else
					{
						printf("unknown neighbortype %s\n",argv[k]);
						return false;
					}
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"line_weight") == 0)
#else
				else if (strcmp(argv[k], "line_weight") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of line_weight ?\n");
						return false;
					}
					line_weight = atof(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"angle_weight") == 0)
#else
				else if (strcmp(argv[k], "angle_weight") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of angle_weight ?\n");
						return false;
					}
					angle_weight = atof(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"distance_weight") == 0)
#else
				else if (strcmp(argv[k], "distance_weight") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of distance_weight ?\n");
						return false;
					}
					distance_weight = atof(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"distance") == 0)
#else
				else if (strcmp(argv[k], "distance") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of distance ?\n");
						return false;
					}
					distance = atof(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"fpiteration") == 0)
#else
				else if (strcmp(argv[k], "fpiteration") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of FPiteration ?\n");
						return false;
					}
					FPIteration = atoi(argv[k]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[k],"iteration") == 0)
#else
				else if (strcmp(argv[k], "iteration") == 0)
#endif
				{
					k++;
					if(k >= argc)
					{
						printf("the value of iteration ?\n");
						return false;
					}
					iteration = atoi(argv[k]);
				}
				else
				{
					printf("unknown parameters %s\n",argv[k]);
					return false;
				}
			}
			return true;
		}
	};
}

#endif