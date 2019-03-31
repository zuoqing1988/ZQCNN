#include "ZQ_CNN_Net_NCHWC.h"

namespace ZQ
{
	template<> class ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC1>;
#if __ARM_NEON
	template<> class ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4>;
#else

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
	template<> class ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC4>;
#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
	template<> class ZQ_CNN_Net_NCHWC<ZQ_CNN_Tensor4D_NCHWC8>;
#endif

#endif//__ARM_NEON
}