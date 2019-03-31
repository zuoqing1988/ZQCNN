inline float my_mm_load_ps(const float* ptr) { return *ptr; }
inline void my_mm_store_ps(float* ptr, float val) { *ptr = val; }
inline float my_mm_add_ps(float a, float b) { return a + b; }
inline float my_mm_sub_ps(float a, float b) { return a - b; }
inline float my_mm_mul_ps(float a, float b) { return a * b; }
inline float my_mm_fmadd_ps(float a, float b, float c) { return a*b + c; }
inline float my_mm_max_ps(float a, float b) { return a > b ? a : b; }
inline float my_mm_min_ps(float a, float b) { return a < b ? a : b; }
inline float my_mm_setzero_ps() { return 0;}
inline float my_mm_set1_ps(float v) { return v; }
