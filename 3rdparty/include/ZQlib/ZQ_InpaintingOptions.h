#ifndef _ZQ_INPAINTING_OPTIONS_H_
#define _ZQ_INPAINTING_OPTIONS_H_
#pragma once 

#include <string.h>
#include <stdio.h>

namespace ZQ
{
	class ZQ_InpaintingOptions
	{
	public:
		enum MethodType{
			METHOD_PDE_THIRD_ORDER,
			METHOD_PYRAMID_TEXTURE_SYNTHESIS
		};
	public:
		ZQ_InpaintingOptions() {Reset();}
		~ZQ_InpaintingOptions() {}

		MethodType type;
				
		int nOuterIteration;//for PDE third order, and pyramid texture synthesis
		int nSORIteration;  //for PDE third order
		float lambda;		//for PDE third order

		float ratio;		//for pyramid texture synthesis
		int minWidth;		//for pyramid texture synthesis
		int winWidth;		//for pyramid texture synthesis
		int winHeight;		//for pyramid texture synthesis
		float gradWeight;	//for pyramid texture synthesis
		float probe;		//for pyramid texture synthesis

		bool display;
		

	public:
		void Reset()
		{
			type = METHOD_PYRAMID_TEXTURE_SYNTHESIS;
			nOuterIteration = 3; //3 for texture synthesis, 200 for PDE third order
			nSORIteration = 50;
			lambda = 0.1;

			ratio = 0.5;
			minWidth = 32;
			winWidth = 3;
			winHeight = 3;
			gradWeight = 0;
			probe = 1.0f;

			display = false;
		}

		bool HandleParas(int argc, const char** argv)
		{
			for(int i = 0;i < argc;i++)
			{
				if(_strcmpi(argv[i],"methodtype") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"PDE") == 0)
						type = METHOD_PDE_THIRD_ORDER;
					else if(_strcmpi(argv[i],"TextureSynthesis") == 0)
						type = METHOD_PYRAMID_TEXTURE_SYNTHESIS;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

				}
				else if(_strcmpi(argv[i],"nOuterIteration") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nOuterIteration = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"nSORIteration") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nSORIteration = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"lambda") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					lambda = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"ratio") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					ratio = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"minWidth") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					minWidth = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"winWidth") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					winWidth = atoi(argv[i]);
				}
				else if(_strcmpi(argv[i],"winHeight") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					winHeight = atoi(argv[i]);

				}
				else if(_strcmpi(argv[i],"gradWeight") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					gradWeight = atof(argv[i]);

				}
				else if(_strcmpi(argv[i],"probe") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					probe = atof(argv[i]);
				}
				else if(_strcmpi(argv[i],"display") == 0)
				{
					display = true;
				}
				else
				{
					printf("unknown para: %s\n",argv[i]);
					return false;
				}
			}
			return true;
		}

		void showArgs()
		{
			printf(
				"[\"methodtype\" \t \"PDE\", \"TextureSynthesis\"]\n"
				"[\"nOuterIteration\" \t nOuterIteration]\n"
				"[\"nSORIteration\" \t nSORIteration]\n"
				"[\"lambda\" \t lambda]\n"
				"[\"ratio\" \t ratio]\n"
				"[\"minWidth\" \t minWidth]\n"
				"[\"winWidth\" \t winWidth]\n"
				"[\"winHeight\" \t winHeight]\n"
				"[\"probe\" \t probe]\n"
				"[\"gradWeight\" \t gradWeight]\n"
				);
		}

	};
}

#endif
