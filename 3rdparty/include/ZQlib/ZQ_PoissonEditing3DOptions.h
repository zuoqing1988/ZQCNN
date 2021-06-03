#ifndef _ZQ_POISSON_EDITING_3D_OPTIONS_H_
#define _ZQ_POISSON_EDITING_3D_OPTIONS_H_

namespace ZQ
{
	class ZQ_PoissonEditing3DOptions
	{
	public:
		enum MethodType{
			METHOD_NAIVE,
		};
	public:
		ZQ_PoissonEditing3DOptions() {Reset();}
		~ZQ_PoissonEditing3DOptions() {}

		MethodType type;

		int nSORIteration;  
		float grad_scale;
		bool display;

	public:
		void Reset()
		{
			type = METHOD_NAIVE;
			nSORIteration = 500;
			grad_scale = 1.0f;
			display = false;
		}

		bool HandleParas(int argc, const char** argv)
		{
			for(int i = 0;i < argc;i++)
			{
#if defined(_WIN32)
				if(_strcmpi(argv[i],"methodtype") == 0)
#else
				if (strcmp(argv[i], "methodtype") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
#if defined(_WIN32)
					if(_strcmpi(argv[i],"naive") == 0)
#else
					if (strcmp(argv[i], "naive") == 0)
#endif
						type = METHOD_NAIVE;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

				}
#if defined(_WIN32)
				else if(_strcmpi(argv[i],"nSORIteration") == 0)
#else
				else if (strcmp(argv[i], "nSORIteration") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					nSORIteration = atoi(argv[i]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[i],"grad_scale") == 0)
#else
				else if (strcmp(argv[i], "grad_scale") == 0)
#endif
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					grad_scale = atof(argv[i]);
				}
#if defined(_WIN32)
				else if(_strcmpi(argv[i],"display") == 0)
#else
				else if (strcmp(argv[i], "display") == 0)
#endif
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

	};
}

#endif
