#ifndef _ZQ_GRID_DEFORMATION_OPTIONS_H_
#define _ZQ_GRID_DEFORMATION_OPTIONS_H_
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace ZQ
{
	class ZQ_GridDeformationOptions
	{
	public:
		enum MethodType{
			METHOD_LINE_ANGLE_ENERGY,
			METHOD_LINE_ANGLE_ENERGY_SCALING,
			METHOD_LINE_ANGLE_DISTANCE_ENERGY,
			METHOD_LINE_ANGLE_ENERGY_XLOOP,
			METHOD_LINE_ANGLE_DISTANCE_ENERGY_XLOOP,
			METHOD_ARAP_VERT_AS_CENTER,
			METHOD_ARAP_VERT_AS_CENTER_XLOOP
		};
		enum NeighborType{
			NEIGHBOR_4,
			NEIGHBOR_8,
			NEIGHBOR_12
		};
	public:
		ZQ_GridDeformationOptions()
		{
			Reset();
		}
		~ZQ_GridDeformationOptions()
		{

		}

		MethodType methodType;
		NeighborType neighborType;
		float line_weight;
		float angle_weight;
		float distance_weight;
		float distance;
		float FPIteration;
		int iteration;

		void Reset()
		{
			methodType = METHOD_LINE_ANGLE_ENERGY;
			neighborType = NEIGHBOR_4;
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
					if(_strcmpi(argv[k],"line_angle") == 0)
#else
					if (strcmp(argv[k], "line_angle") == 0)
#endif
					{
						methodType = METHOD_LINE_ANGLE_ENERGY;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"line_angle_scaling") == 0)
#else
					else if (strcmp(argv[k], "line_angle_scaling") == 0)
#endif
					{
						methodType = METHOD_LINE_ANGLE_ENERGY_SCALING;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"line_angle_distance") == 0)
#else
					else if (strcmp(argv[k], "line_angle_distance") == 0)
#endif
					{
						methodType = METHOD_LINE_ANGLE_DISTANCE_ENERGY;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"line_angle_xloop") == 0)
#else
					else if (strcmp(argv[k], "line_angle_xloop") == 0)
#endif
					{
						methodType = METHOD_LINE_ANGLE_ENERGY_XLOOP;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"line_angle_distance_xloop") == 0)
#else
					else if (strcmp(argv[k], "line_angle_distance_xloop") == 0)
#endif
					{
						methodType = METHOD_LINE_ANGLE_DISTANCE_ENERGY_XLOOP;
					}
#if defined(_WIN32)
					else if(_strcmpi(argv[k],"arap_vert") == 0)
#else
					else if (strcmp(argv[k], "arap_vert") == 0)
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
					if(strcmp(argv[k],"4") == 0)
					{
						neighborType = NEIGHBOR_4;
					}
					else if(strcmp(argv[k],"8") == 0)
					{
						neighborType = NEIGHBOR_8;
					}
					else if(strcmp(argv[k],"12") == 0)
					{
						neighborType = NEIGHBOR_12;
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