float my_mm_load_ps(const float* ptr) { return *ptr; }
void my_mm_store_ps(float* ptr, float val) { *ptr = val; }
float my_mm_add_ps(float a, float b) { return a + b; }
float my_mm_sub_ps(float a, float b) { return a - b; }
float my_mm_mul_ps(float a, float b) { return a * b; }
float my_mm_fmadd_ps(float a, float b, float c) { return a*b + c; }
float my_mm_max_ps(float a, float b) { return a > b ? a : b; }
float my_mm_min_ps(float a, float b) { return a < b ? a : b; }
float my_mm_setzero_ps() { return 0; }
float my_mm_set1_ps(float v) { return v; }
