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
				if(_strcmpi(argv[i],"methodtype") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					if(_strcmpi(argv[i],"naive") == 0)
						type = METHOD_NAIVE;
					else
					{
						printf("unknown methodType: %s\n",argv[i]);
						return false;
					}

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
				else if(_strcmpi(argv[i],"grad_scale") == 0)
				{
					i++;
					if(i >= argc)
					{
						printf("the value of %s ?\n",argv[i-1]);
						return false;
					}
					grad_scale = atof(argv[i]);
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

	};
}

#endif
