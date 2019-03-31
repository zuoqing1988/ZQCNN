#ifndef _ZQ_CNN_BASE_OPERATOR_H_
#define _ZQ_CNN_BASE_OPERATOR_H_
float my_mm_load_ps(const float* ptr);
void my_mm_store_ps(float* ptr, float val);
float my_mm_add_ps(float a, float b);
float my_mm_sub_ps(float a, float b);
float my_mm_mul_ps(float a, float b);
float my_mm_fmadd_ps(float a, float b, float c);
float my_mm_max_ps(float a, float b);
float my_mm_min_ps(float a, float b);
float my_mm_setzero_ps();
float my_mm_set1_ps(float v);
#endif