#include <stdlib.h>
#include "../ZQ_CNN_CompileConfig.h"
#if __ARM_NEON
#include <arm_neon.h>
#else
#if defined(__GNUC__)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <smmintrin.h>
#endif
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <x86intrin.h>
#endif
#elif defined(_WIN32)
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#include <mmintrin.h> //MMX  
#include <xmmintrin.h> //SSE(include mmintrin.h)  
#include <emmintrin.h> //SSE2(include xmmintrin.h)  
#include <pmmintrin.h> //SSE3(include emmintrin.h)  
#include <tmmintrin.h>//SSSE3(include pmmintrin.h)  
#include <smmintrin.h>//SSE4.1(include tmmintrin.h)  
#include <nmmintrin.h>//SSE4.2(include smmintrin.h)  
#endif 
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#include <wmmintrin.h>//AES(include nmmintrin.h)  
#include <immintrin.h>//AVX(include wmmintrin.h)  
#include <intrin.h>//(include immintrin.h)  
#endif
#endif
#endif //__ARM_NEON

#if defined(__cplusplus) || defined(c_plusplus) 
extern "C" {
#endif

#if __ARM_NEON
#define zq_cnn_eltwise_sum_nchwc zq_cnn_eltwise_sum_nchwc4
#define zq_cnn_eltwise_sum_with_weight_nchwc zq_cnn_eltwise_sum_with_weight_nchwc4
#define zq_cnn_eltwise_mul_nchwc zq_cnn_eltwise_mul_nchwc4
#define zq_cnn_eltwise_max_nchwc zq_cnn_eltwise_max_nchwc4
#define zq_mm_load_ps vld1q_f32
#define zq_mm_store_ps vst1q_f32
#define zq_mm_mul_ps vmulq_f32
#define zq_mm_add_ps vaddq_f32
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps(A, B, C) vfmaq_f32(C, A, B)
#else
#define zq_mm_fmadd_ps(A, B, C) vaddq_f32(vmulq_f32(A, B), C)
#endif
#define zq_mm_max_ps vmaxq_f32
#define zq_mm_set1_ps vdupq_n_f32
#define zq_mm_type float32x4_t
#define zq_base_type float
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#define zq_mm_align_size16 64
#define zq_mm_align_size32 128
#define zq_mm_align_size64 256

#include "zq_cnn_eltwise_nchwc_raw.h"


#undef zq_cnn_eltwise_sum_nchwc
#undef zq_cnn_eltwise_sum_with_weight_nchwc
#undef zq_cnn_eltwise_mul_nchwc
#undef zq_cnn_eltwise_max_nchwc
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_align_size32
#undef zq_mm_align_size64

#else
#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_SSE
#define zq_cnn_eltwise_sum_nchwc zq_cnn_eltwise_sum_nchwc4
#define zq_cnn_eltwise_sum_with_weight_nchwc zq_cnn_eltwise_sum_with_weight_nchwc4
#define zq_cnn_eltwise_mul_nchwc zq_cnn_eltwise_mul_nchwc4
#define zq_cnn_eltwise_max_nchwc zq_cnn_eltwise_max_nchwc4
#define zq_mm_load_ps _mm_load_ps
#define zq_mm_store_ps _mm_store_ps
#define zq_mm_mul_ps _mm_mul_ps
#define zq_mm_add_ps _mm_add_ps
#if ZQ_CNN_USE_FMADD128
#define zq_mm_fmadd_ps _mm_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm_add_ps(_mm_mul_ps(A, B), C)
#endif
#define zq_mm_max_ps _mm_max_ps
#define zq_mm_set1_ps _mm_set1_ps
#define zq_mm_type __m128
#define zq_base_type float
#define zq_mm_align_size 4
#define zq_mm_align_size2 8
#define zq_mm_align_size3 12
#define zq_mm_align_size4 16
#define zq_mm_align_size5 20
#define zq_mm_align_size6 24
#define zq_mm_align_size7 28
#define zq_mm_align_size8 32
#define zq_mm_align_size16 64
#define zq_mm_align_size32 128
#define zq_mm_align_size64 256

#include "zq_cnn_eltwise_nchwc_raw.h"


#undef zq_cnn_eltwise_sum_nchwc
#undef zq_cnn_eltwise_sum_with_weight_nchwc
#undef zq_cnn_eltwise_mul_nchwc
#undef zq_cnn_eltwise_max_nchwc
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_align_size32
#undef zq_mm_align_size64

#endif

#if ZQ_CNN_USE_SSETYPE >= ZQ_CNN_SSETYPE_AVX
#define zq_cnn_eltwise_sum_nchwc zq_cnn_eltwise_sum_nchwc8
#define zq_cnn_eltwise_sum_with_weight_nchwc zq_cnn_eltwise_sum_with_weight_nchwc8
#define zq_cnn_eltwise_mul_nchwc zq_cnn_eltwise_mul_nchwc8
#define zq_cnn_eltwise_max_nchwc zq_cnn_eltwise_max_nchwc8
#define zq_mm_load_ps _mm256_load_ps
#define zq_mm_store_ps _mm256_store_ps
#define zq_mm_mul_ps _mm256_mul_ps
#define zq_mm_add_ps _mm256_add_ps
#if ZQ_CNN_USE_FMADD256
#define zq_mm_fmadd_ps _mm256_fmadd_ps
#else
#define zq_mm_fmadd_ps(A, B, C) _mm256_add_ps(_mm256_mul_ps(A, B), C)
#endif
#define zq_mm_max_ps _mm256_max_ps
#define zq_mm_set1_ps _mm256_set1_ps
#define zq_mm_type __m256
#define zq_base_type float
#define zq_mm_align_size 8
#define zq_mm_align_size2 16
#define zq_mm_align_size3 24
#define zq_mm_align_size4 32
#define zq_mm_align_size5 40
#define zq_mm_align_size6 48
#define zq_mm_align_size7 56
#define zq_mm_align_size8 64
#define zq_mm_align_size16 128
#define zq_mm_align_size32 256
#define zq_mm_align_size64 512

#include "zq_cnn_eltwise_nchwc_raw.h"


#undef zq_cnn_eltwise_sum_nchwc
#undef zq_cnn_eltwise_sum_with_weight_nchwc
#undef zq_cnn_eltwise_mul_nchwc
#undef zq_cnn_eltwise_max_nchwc
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_align_size32
#undef zq_mm_align_size64

#endif
#endif //__ARM_NEON

static inline float my_mm_load_ps(const float* ptr) { return *ptr; }
static inline void my_mm_store_ps(float* ptr, float val) { *ptr = val; }
static inline float my_mm_add_ps(float a, float b) { return a + b; }
static inline float my_mm_sub_ps(float a, float b) { return a - b; }
static inline float my_mm_mul_ps(float a, float b) { return a * b; }
static inline float my_mm_fmadd_ps(float a, float b, float c) { return a*b + c; }
static inline float my_mm_max_ps(float a, float b) { return a > b ? a : b; }
static inline float my_mm_min_ps(float a, float b) { return a < b ? a : b; }
static inline float my_mm_setzero_ps() { return 0; }
static inline float my_mm_set1_ps(float v) { return v; }

#define zq_cnn_eltwise_sum_nchwc zq_cnn_eltwise_sum_nchwc1
#define zq_cnn_eltwise_sum_with_weight_nchwc zq_cnn_eltwise_sum_with_weight_nchwc1
#define zq_cnn_eltwise_mul_nchwc zq_cnn_eltwise_mul_nchwc1
#define zq_cnn_eltwise_max_nchwc zq_cnn_eltwise_max_nchwc1
#define zq_mm_load_ps my_mm_load_ps
#define zq_mm_store_ps my_mm_store_ps
#define zq_mm_mul_ps my_mm_mul_ps
#define zq_mm_add_ps my_mm_add_ps
#define zq_mm_fmadd_ps my_mm_fmadd_ps
#define zq_mm_max_ps my_mm_max_ps
#define zq_mm_set1_ps my_mm_set1_ps
#define zq_mm_type float
#define zq_base_type float
#define zq_mm_align_size 1
#define zq_mm_align_size2 2
#define zq_mm_align_size3 3
#define zq_mm_align_size4 4
#define zq_mm_align_size5 5
#define zq_mm_align_size6 6
#define zq_mm_align_size7 7
#define zq_mm_align_size8 8
#define zq_mm_align_size16 16
#define zq_mm_align_size32 32
#define zq_mm_align_size64 64

#include "zq_cnn_eltwise_nchwc_raw.h"


#undef zq_cnn_eltwise_sum_nchwc
#undef zq_cnn_eltwise_sum_with_weight_nchwc
#undef zq_cnn_eltwise_mul_nchwc
#undef zq_cnn_eltwise_max_nchwc
#undef zq_mm_load_ps
#undef zq_mm_store_ps
#undef zq_mm_mul_ps
#undef zq_mm_add_ps
#undef zq_mm_fmadd_ps
#undef zq_mm_max_ps
#undef zq_mm_set1_ps
#undef zq_mm_type
#undef zq_base_type
#undef zq_mm_align_size
#undef zq_mm_align_size2
#undef zq_mm_align_size3
#undef zq_mm_align_size4
#undef zq_mm_align_size5
#undef zq_mm_align_size6
#undef zq_mm_align_size7
#undef zq_mm_align_size8
#undef zq_mm_align_size16
#undef zq_mm_align_size32
#undef zq_mm_align_size64

#if defined(__cplusplus) || defined(c_plusplus) 
}
#endif