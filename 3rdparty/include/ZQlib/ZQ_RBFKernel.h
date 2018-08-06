#ifndef _ZQ_RBF_KERNEL_H_
#define _ZQ_RBF_KERNEL_H_
#pragma once

namespace ZQ
{
	class ZQ_RBFKernel
	{
	public:
		enum RBF_TYPE
		{
			COMPACT_CPC0,		// (1-x)^2
			COMPACT_CPC2,		// (1-x)^4 (4x+1)
			COMPACT_CPC4,		// (1-x)^6 ((35/3)x^2 + 6x + 1)
			COMPACT_CPC6,		// (1-x)^8 (32x^3 + 25x^2 + 8x + 1)
			COMPACT_CTPS_C0,	// (1-x)^5
			COMPACT_CTPS_C1,	// 1 + (80/3)x^2 - 40x^3 + 15x^4 - (8/3)x^5 + 20x^2 log(x)
			COMPACT_CTPS_C2A,	// 1 - 30x^2 - 10x^3 + 45x^4 - 6x^5 - 60x^3log(x)
			COMPACT_CTPS_C2B,	// 1 - 20x^2 + 80x^3 - 45x^4 - 16x^5 + 60x^4log(x)

			GLOBAL_TPS,			// x^2log(x)
			GLOBAL_MQB,			// sqrt(a^2+x^2)
			GLOBAL_IMQB,		// sqrt(1/(a^2+x^2))
			GLOBAL_QB,			// 1+x^2
			GLOBAL_IQB,			// 1/(1+x^2)
			GLOBAL_GAUSS		// e^(-x^2)
		};

		static double _compact_kernel(bool &flag, const RBF_TYPE type, const double distance, const double radius)
		{
			
			double x = fabs(distance/radius);
			switch(type)
			{
			case COMPACT_CPC0:
				{
					flag = true;
					// (1-x)^2
					if(x > 1)
						return 0;
					else
						return (1-x)*(1-x);
				}
				break;
			case COMPACT_CPC2:
				{
					flag = true;
					// (1-x)^4 (4x+1)
					if(x > 1)
						return 0;
					else
						return (1-x)*(1-x)*(1-x)*(1-x)*(4*x+1);
				}
				break;
			case COMPACT_CPC4:
				{
					flag = true;
					// (1-x)^6 ((35/3)x^2 + 6x + 1)
					if(x > 1)
						return 0;
					else
					{
						double tmp = (1-x)*(1-x)*(1-x);
						return tmp*tmp*((35.0/3.0)*x*x+6*x+1);
					}
				}
				break;
			case COMPACT_CPC6:
				{
					flag = true;
					// (1-x)^8 (32x^3 + 25x^2 + 8x + 1)
					if(x > 1)
						return 0;
					else
					{
						double tmp = (1-x)*(1-x)*(1-x)*(1-x);
						return tmp*tmp*(32*x*x*x+25*x*x+8*x+1);
					}
				}
				break;
			case COMPACT_CTPS_C0:
				{
					flag = true;
					// (1-x)^5
					if(x > 1)
						return 0;
					else
						return (1-x)*(1-x)*(1-x)*(1-x)*(1-x);
				}
				break;
			case COMPACT_CTPS_C1:
				{
					flag = true;
					// 1 + (80/3)x^2 - 40x^3 + 15x^4 - (8/3)x^5 + 20x^2 log(x)
					if(x > 1)
						return 0;
					else if(x == 0)
						return 1;
					else
					{
						return 1+(80.0/3.0)*x*x - 40*x*x*x + 15*x*x*x*x - (8.0/3.0)*x*x*x*x*x + 20*x*x*log(x);
					}
				}
				break;
			case COMPACT_CTPS_C2A:
				{
					flag = true;
					// 1 - 30x^2 - 10x^3 + 45x^4 - 6x^5 - 60x^3log(x)
					if(x > 1)
						return 0;
					else if(x == 0)
						return 1;
					else
					{
						return 1-30*x*x - 10*x*x*x + 45*x*x*x*x - 6*x*x*x*x*x - 60*x*x*x*log(x);
					}
				}
				break;
			case COMPACT_CTPS_C2B:
				{
					flag = true;
					// 1 - 20x^2 + 80x^3 - 45x^4 - 16x^5 + 60x^4log(x)
					if(x > 1)
						return 0;
					else if(x == 0)
						return 1;
					else
					{
						return 1 - 20*x*x + 80*x*x*x - 45*x*x*x*x - 16*x*x*x*x*x + 60*x*x*x*x*log(x);
					}
				}
				break;
			default:
				flag = false;
				break;
			}

			return 0;
		}

		static double _global_kernel(bool &flag, RBF_TYPE type, double distance, double sigma, double eps = 0.001)
		{
			double x = fabs(distance/sigma);
			if(eps == 0)
				eps = 1e-5;

			switch(type)
			{
			case GLOBAL_TPS:
				{
					flag = true;
					// x^2log(x)
					if(x == 0)
						return 0;
					else
						return x*x*log(x);
				}
				break;
			case GLOBAL_MQB:
				{
					flag = true;
					// sqrt(a^2+x^2)
					return sqrt(eps*eps+x*x);
				}
				break;
			case GLOBAL_IMQB:
				{
					flag = true;
					// sqrt(1/(a^2+x^2))
					return sqrt(1.0/(eps*eps+x*x));
				}
				break;
			case GLOBAL_QB:
				{
					flag = true;
					// 1+x^2
					return 1+x*x;
				}
				break;
			case GLOBAL_IQB:
				{
					flag = true;
					// 1/(1+x^2)
					return 1.0/(1.0+x*x);
				}
				break;
			case GLOBAL_GAUSS:
				{
					flag = true;
					// e^(-x^2)
					return exp(-x*x);
				}
				break;
			default:
				flag = false;
			}
			return 0;
		}
	};
}

#endif