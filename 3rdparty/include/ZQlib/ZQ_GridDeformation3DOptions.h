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
				if(_strcmpi(argv[k],"methodtype") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of methodtype ?\n");
						return false;
					}
					if(_strcmpi(argv[k],"laplacian") == 0)
					{
						methodType = METHOD_LAPLACIAN;
					}
					else if(_strcmpi(argv[k],"laplacian_xloop") == 0)
					{
						methodType = METHOD_LAPLACIAN_XLOOP;
					}
					else if(_strcmpi(argv[k],"arap_vert") == 0)
					{
						methodType = METHOD_ARAP_VERT_AS_CENTER;
					}
					else if(_strcmpi(argv[k],"arap_vert_xloop") == 0)
					{
						methodType = METHOD_ARAP_VERT_AS_CENTER_XLOOP;
					}
					else
					{
						printf("unknown methodtype %s\n",argv[k]);
						return false;
					}
				}
				else if(_strcmpi(argv[k],"neighbortype") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of neighbortype ?\n");
						return false;
					}
					if(_strcmpi(argv[k],"6") == 0)
					{
						neighborType = NEIGHBOR_6;
					}
					else if(_strcmpi(argv[k],"26") == 0)
					{
						neighborType = NEIGHBOR_26;
					}
					else
					{
						printf("unknown neighbortype %s\n",argv[k]);
						return false;
					}
				}
				else if(_strcmpi(argv[k],"line_weight") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of line_weight ?\n");
						return false;
					}
					line_weight = atof(argv[k]);
				}
				else if(_strcmpi(argv[k],"angle_weight") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of angle_weight ?\n");
						return false;
					}
					angle_weight = atof(argv[k]);
				}
				else if(_strcmpi(argv[k],"distance_weight") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of distance_weight ?\n");
						return false;
					}
					distance_weight = atof(argv[k]);
				}
				else if(_strcmpi(argv[k],"distance") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of distance ?\n");
						return false;
					}
					distance = atof(argv[k]);
				}
				else if(_strcmpi(argv[k],"fpiteration") == 0)
				{
					k++;
					if(k >= argc)
					{
						printf("the value of FPiteration ?\n");
						return false;
					}
					FPIteration = atoi(argv[k]);
				}
				else if(_strcmpi(argv[k],"iteration") == 0)
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