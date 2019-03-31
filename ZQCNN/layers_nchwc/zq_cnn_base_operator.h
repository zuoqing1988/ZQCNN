#ifndef _ZQ_CNN_BASE_OPERATOR_H_
#define _ZQ_CNN_BASE_OPERATOR_H_
inline float my_mm_load_ps(const float* ptr);
inline void my_mm_store_ps(float* ptr, float val);
inline float my_mm_add_ps(float a, float b);
inline float my_mm_sub_ps(float a, float b);
inline float my_mm_mul_ps(float a, float b);
inline float my_mm_fmadd_ps(float a, float b, float c);
inline float my_mm_max_ps(float a, float b);
inline float my_mm_min_ps(float a, float b);
inline float my_mm_setzero_ps();
inline float my_mm_set1_ps(float v);
#endif