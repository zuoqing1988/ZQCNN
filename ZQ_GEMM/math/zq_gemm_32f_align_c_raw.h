
#define op_1x1 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_1x1_2 \
	op_1x1;\
	op_1x1

#define op_1x1_4 \
	op_1x1_2;\
	op_1x1_2

#define op_1x1_8 \
	op_1x1_4;\
	op_1x1_4

#define op_1x1_16 \
	op_1x1_8;\
	op_1x1_8

#define op_1x1_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_1x1_2_first \
	op_1x1_first;\
	op_1x1

#define op_1x1_4_first \
	op_1x1_2_first;\
	op_1x1_2

#define op_1x1_8_first \
	op_1x1_4_first;\
	op_1x1_4

#define op_1x1_16_first \
	op_1x1_8_first;\
	op_1x1_8

#define op_1x2 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_1x2_2 \
	op_1x2;\
	op_1x2

#define op_1x2_4 \
	op_1x2_2;\
	op_1x2_2

#define op_1x2_8 \
	op_1x2_4;\
	op_1x2_4

#define op_1x2_16 \
	op_1x2_8;\
	op_1x2_8

#define op_1x2_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_1x2_2_first \
	op_1x2_first;\
	op_1x2

#define op_1x2_4_first \
	op_1x2_2_first;\
	op_1x2_2

#define op_1x2_8_first \
	op_1x2_4_first;\
	op_1x2_4

#define op_1x2_16_first \
	op_1x2_8_first;\
	op_1x2_8


#define op_1x4 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op_1x4_2 \
	op_1x4;\
	op_1x4

#define op_1x4_4 \
	op_1x4_2;\
	op_1x4_2

#define op_1x4_8 \
	op_1x4_4;\
	op_1x4_4

#define op_1x4_16 \
	op_1x4_8;\
	op_1x4_8

#define op_1x4_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op_1x4_2_first \
	op_1x4_first;\
	op_1x4

#define op_1x4_4_first \
	op_1x4_2_first;\
	op_1x4_2

#define op_1x4_8_first \
	op_1x4_4_first;\
	op_1x4_4

#define op_1x4_16_first \
	op_1x4_8_first;\
	op_1x4_8

#define op_1x8 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec5, sum_vec15);\
	sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec6, sum_vec16);\
	sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec7, sum_vec17);\
	sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec8, sum_vec18);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_1x8_2 \
	op_1x8;\
	op_1x8

#define op_1x8_4 \
	op_1x8_2;\
	op_1x8_2

#define op_1x8_8 \
	op_1x8_4;\
	op_1x8_4

#define op_1x8_16 \
	op_1x8_8;\
	op_1x8_8

#define op_1x8_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec15 = zq_mm_mul_ps(a_vec1, b_vec5);\
	sum_vec16 = zq_mm_mul_ps(a_vec1, b_vec6);\
	sum_vec17 = zq_mm_mul_ps(a_vec1, b_vec7);\
	sum_vec18 = zq_mm_mul_ps(a_vec1, b_vec8);\
	A_c_ptr1 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_1x8_2_first \
	op_1x8_first;\
	op_1x8

#define op_1x8_4_first \
	op_1x8_2_first;\
	op_1x8_2

#define op_1x8_8_first \
	op_1x8_4_first;\
	op_1x8_4

#define op_1x8_16_first \
	op_1x8_8_first;\
	op_1x8_8

#define op_2x1 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_2x1_2 \
	op_2x1;\
	op_2x1

#define op_2x1_4 \
	op_2x1_2;\
	op_2x1_2

#define op_2x1_8 \
	op_2x1_4;\
	op_2x1_4

#define op_2x1_16 \
	op_2x1_8;\
	op_2x1_8

#define op_2x1_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_2x1_2_first \
	op_2x1_first;\
	op_2x1

#define op_2x1_4_first \
	op_2x1_2_first;\
	op_2x1_2

#define op_2x1_8_first \
	op_2x1_4_first;\
	op_2x1_4

#define op_2x1_16_first \
	op_2x1_8_first;\
	op_2x1_8

#define op_2x2 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_2x2_2 \
	op_2x2;\
	op_2x2

#define op_2x2_4 \
	op_2x2_2;\
	op_2x2_2

#define op_2x2_8 \
	op_2x2_4;\
	op_2x2_4

#define op_2x2_16 \
	op_2x2_8;\
	op_2x2_8

#define op_2x2_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_2x2_2_first \
	op_2x2_first;\
	op_2x2

#define op_2x2_4_first \
	op_2x2_2_first;\
	op_2x2_2

#define op_2x2_8_first \
	op_2x2_4_first;\
	op_2x2_4

#define op_2x2_16_first \
	op_2x2_8_first;\
	op_2x2_8

#define op_2x4 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec23);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec24);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op_2x4_2 \
	op_2x4;\
	op_2x4

#define op_2x4_4 \
	op_2x4_2;\
	op_2x4_2

#define op_2x4_8 \
	op_2x4_4;\
	op_2x4_4

#define op_2x4_16 \
	op_2x4_8;\
	op_2x4_8

#define op_2x4_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec23 = zq_mm_mul_ps(a_vec2, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec24 = zq_mm_mul_ps(a_vec2, b_vec4);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op_2x4_2_first \
	op_2x4_first;\
	op_2x4

#define op_2x4_4_first \
	op_2x4_2_first;\
	op_2x4_2

#define op_2x4_8_first \
	op_2x4_4_first;\
	op_2x4_4

#define op_2x4_16_first \
	op_2x4_8_first;\
	op_2x4_8

#define op_2x8 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec23);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec24);\
	sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec5, sum_vec15);\
	sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec5, sum_vec25);\
	sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec6, sum_vec16);\
	sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec6, sum_vec26);\
	sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec7, sum_vec17);\
	sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec7, sum_vec27);\
	sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec8, sum_vec18);\
	sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec8, sum_vec28);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_2x8_2 \
	op_2x8;\
	op_2x8

#define op_2x8_4 \
	op_2x8_2;\
	op_2x8_2

#define op_2x8_8 \
	op_2x8_4;\
	op_2x8_4

#define op_2x8_16 \
	op_2x8_8;\
	op_2x8_8

#define op_2x8_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec23 = zq_mm_mul_ps(a_vec2, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec24 = zq_mm_mul_ps(a_vec2, b_vec4);\
	sum_vec15 = zq_mm_mul_ps(a_vec1, b_vec5);\
	sum_vec25 = zq_mm_mul_ps(a_vec2, b_vec5);\
	sum_vec16 = zq_mm_mul_ps(a_vec1, b_vec6);\
	sum_vec26 = zq_mm_mul_ps(a_vec2, b_vec6);\
	sum_vec17 = zq_mm_mul_ps(a_vec1, b_vec7);\
	sum_vec27 = zq_mm_mul_ps(a_vec2, b_vec7);\
	sum_vec18 = zq_mm_mul_ps(a_vec1, b_vec8);\
	sum_vec28 = zq_mm_mul_ps(a_vec2, b_vec8);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_2x8_2_first \
	op_2x8_first;\
	op_2x8

#define op_2x8_4_first \
	op_2x8_2_first;\
	op_2x8_2

#define op_2x8_8_first \
	op_2x8_4_first;\
	op_2x8_4

#define op_2x8_16_first \
	op_2x8_8_first;\
	op_2x8_8

#define op_4x1 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_4x1_2 \
	op_4x1;\
	op_4x1

#define op_4x1_4 \
	op_4x1_2;\
	op_4x1_2

#define op_4x1_8 \
	op_4x1_4;\
	op_4x1_4

#define op_4x1_16 \
	op_4x1_8;\
	op_4x1_8

#define op_4x1_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_4x1_2_first \
	op_4x1_first;\
	op_4x1

#define op_4x1_4_first \
	op_4x1_2_first;\
	op_4x1_2

#define op_4x1_8_first \
	op_4x1_4_first;\
	op_4x1_4

#define op_4x1_16_first \
	op_4x1_8_first;\
	op_4x1_8

#define op_4x2 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec2, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec2, sum_vec42);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_4x2_2 \
	op_4x2;\
	op_4x2

#define op_4x2_4 \
	op_4x2_2;\
	op_4x2_2

#define op_4x2_8 \
	op_4x2_4;\
	op_4x2_4

#define op_4x2_16 \
	op_4x2_8;\
	op_4x2_8

#define op_4x2_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec32 = zq_mm_mul_ps(a_vec3, b_vec2);\
	sum_vec42 = zq_mm_mul_ps(a_vec4, b_vec2);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size


#define op_4x2_2_first \
	op_4x2_first;\
	op_4x2

#define op_4x2_4_first \
	op_4x2_2_first;\
	op_4x2_2

#define op_4x2_8_first \
	op_4x2_4_first;\
	op_4x2_4

#define op_4x2_16_first \
	op_4x2_8_first;\
	op_4x2_8


#define op_4x4 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec2, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec2, sum_vec42);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec23);\
	sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec3, sum_vec33);\
	sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec3, sum_vec43);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec24);\
	sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec4, sum_vec34);\
	sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec4, sum_vec44);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op2_4x4 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	a_vec1_ = zq_mm_load_ps(A_c_ptr1+zq_mm_align_size);\
	a_vec2_ = zq_mm_load_ps(A_c_ptr2 + zq_mm_align_size); \
	a_vec3_ = zq_mm_load_ps(A_c_ptr3 + zq_mm_align_size); \
	a_vec4_ = zq_mm_load_ps(A_c_ptr4 + zq_mm_align_size); \
	b_vec1_ = zq_mm_load_ps(B_c_ptr1+zq_mm_align_size);\
	b_vec2_ = zq_mm_load_ps(B_c_ptr2+zq_mm_align_size);\
	b_vec3_ = zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size); \
	b_vec4_ = zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size); \
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1_, b_vec1_, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2_, b_vec1_, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3_, b_vec1_, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4_, b_vec1_, sum_vec41);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1_, b_vec2_, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2_, b_vec2_, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec2, sum_vec32);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3_, b_vec2_, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec2, sum_vec42);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4_, b_vec2_, sum_vec42);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1_, b_vec3_, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec23);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2_, b_vec3_, sum_vec23);\
	sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec3, sum_vec33);\
	sum_vec33 = zq_mm_fmadd_ps(a_vec3_, b_vec3_, sum_vec33);\
	sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec3, sum_vec43);\
	sum_vec43 = zq_mm_fmadd_ps(a_vec4_, b_vec3_, sum_vec43);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1_, b_vec4_, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec24);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2_, b_vec4_, sum_vec24);\
	sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec4, sum_vec34);\
	sum_vec34 = zq_mm_fmadd_ps(a_vec3_, b_vec4_, sum_vec34);\
	sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec4, sum_vec44);\
	sum_vec44 = zq_mm_fmadd_ps(a_vec4_, b_vec4_, sum_vec44);\
	A_c_ptr1 += zq_mm_align_size2;\
	A_c_ptr2 += zq_mm_align_size2;\
	A_c_ptr3 += zq_mm_align_size2;\
	A_c_ptr4 += zq_mm_align_size2;\
	B_c_ptr1 += zq_mm_align_size2;\
	B_c_ptr2 += zq_mm_align_size2;\
	B_c_ptr3 += zq_mm_align_size2;\
	B_c_ptr4 += zq_mm_align_size2

#define op_4x4_2 \
	op_4x4;\
	op_4x4

#define op2_4x4_2 \
	op2_4x4;\
	op2_4x4

#define op_4x4_4 \
	op_4x4_2;\
	op_4x4_2

#define op2_4x4_4 \
	op2_4x4_2;\
	op2_4x4_2

#define op_4x4_8 \
	op_4x4_4;\
	op_4x4_4

#define op2_4x4_8 \
	op2_4x4_4;\
	op2_4x4_4

#define op_4x4_16 \
	op_4x4_8;\
	op_4x4_8

#define op2_4x4_16 \
	op2_4x4_8;\
	op2_4x4_8


#define op_4x4_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec32 = zq_mm_mul_ps(a_vec3, b_vec2);\
	sum_vec42 = zq_mm_mul_ps(a_vec4, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec23 = zq_mm_mul_ps(a_vec2, b_vec3);\
	sum_vec33 = zq_mm_mul_ps(a_vec3, b_vec3);\
	sum_vec43 = zq_mm_mul_ps(a_vec4, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec24 = zq_mm_mul_ps(a_vec2, b_vec4);\
	sum_vec34 = zq_mm_mul_ps(a_vec3, b_vec4);\
	sum_vec44 = zq_mm_mul_ps(a_vec4, b_vec4);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size

#define op2_4x4_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec1_ = zq_mm_load_ps(A_c_ptr1+zq_mm_align_size);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec2_ = zq_mm_load_ps(A_c_ptr2+zq_mm_align_size);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec3_ = zq_mm_load_ps(A_c_ptr3+zq_mm_align_size);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec4_ = zq_mm_load_ps(A_c_ptr4+zq_mm_align_size);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec1_ = zq_mm_load_ps(B_c_ptr1+zq_mm_align_size);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec2_ = zq_mm_load_ps(B_c_ptr2+zq_mm_align_size);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec3_ = zq_mm_load_ps(B_c_ptr3+zq_mm_align_size);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec4_ = zq_mm_load_ps(B_c_ptr4+zq_mm_align_size);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec32 = zq_mm_mul_ps(a_vec3, b_vec2);\
	sum_vec42 = zq_mm_mul_ps(a_vec4, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec23 = zq_mm_mul_ps(a_vec2, b_vec3);\
	sum_vec33 = zq_mm_mul_ps(a_vec3, b_vec3);\
	sum_vec43 = zq_mm_mul_ps(a_vec4, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec24 = zq_mm_mul_ps(a_vec2, b_vec4);\
	sum_vec34 = zq_mm_mul_ps(a_vec3, b_vec4);\
	sum_vec44 = zq_mm_mul_ps(a_vec4, b_vec4);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1_, b_vec1_, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2_, b_vec1_, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3_, b_vec1_, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4_, b_vec1_, sum_vec41);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1_, b_vec2_, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2_, b_vec2_, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3_, b_vec2_, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4_, b_vec2_, sum_vec42);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1_, b_vec3_, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2_, b_vec3_, sum_vec23);\
	sum_vec33 = zq_mm_fmadd_ps(a_vec3_, b_vec3_, sum_vec33);\
	sum_vec43 = zq_mm_fmadd_ps(a_vec4_, b_vec3_, sum_vec43);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1_, b_vec4_, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2_, b_vec4_, sum_vec24);\
	sum_vec34 = zq_mm_fmadd_ps(a_vec3_, b_vec4_, sum_vec34);\
	sum_vec44 = zq_mm_fmadd_ps(a_vec4_, b_vec4_, sum_vec44);\
	A_c_ptr1 += zq_mm_align_size2;\
	A_c_ptr2 += zq_mm_align_size2;\
	A_c_ptr3 += zq_mm_align_size2;\
	A_c_ptr4 += zq_mm_align_size2;\
	B_c_ptr1 += zq_mm_align_size2;\
	B_c_ptr2 += zq_mm_align_size2;\
	B_c_ptr3 += zq_mm_align_size2;\
	B_c_ptr4 += zq_mm_align_size2

#define op_4x4_2_first \
	op_4x4_first;\
	op_4x4

#define op2_4x4_2_first \
	op2_4x4_first;\
	op2_4x4

#define op_4x4_4_first \
	op_4x4_2_first;\
	op_4x4_2

#define op2_4x4_4_first \
	op2_4x4_2_first;\
	op2_4x4_2

#define op_4x4_8_first \
	op_4x4_4_first;\
	op_4x4_4

#define op2_4x4_8_first \
	op2_4x4_4_first;\
	op2_4x4_4

#define op_4x4_16_first \
	op_4x4_8_first;\
	op_4x4_8

#define op2_4x4_16_first \
	op2_4x4_8_first;\
	op2_4x4_8


#define op_4x8 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec41);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec42);\
	sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec3, sum_vec13);\
	sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec23);\
	sum_vec33 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec33);\
	sum_vec43 = zq_mm_fmadd_ps(a_vec2, b_vec3, sum_vec43);\
	sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec4, sum_vec14);\
	sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec24);\
	sum_vec34 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec34);\
	sum_vec44 = zq_mm_fmadd_ps(a_vec2, b_vec4, sum_vec44);\
	sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec5, sum_vec15);\
	sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec5, sum_vec25);\
	sum_vec35 = zq_mm_fmadd_ps(a_vec2, b_vec5, sum_vec35);\
	sum_vec45 = zq_mm_fmadd_ps(a_vec2, b_vec5, sum_vec45);\
	sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec6, sum_vec16);\
	sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec6, sum_vec26);\
	sum_vec36 = zq_mm_fmadd_ps(a_vec2, b_vec6, sum_vec36);\
	sum_vec46 = zq_mm_fmadd_ps(a_vec2, b_vec6, sum_vec46);\
	sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec7, sum_vec17);\
	sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec7, sum_vec27);\
	sum_vec37 = zq_mm_fmadd_ps(a_vec2, b_vec7, sum_vec37);\
	sum_vec47 = zq_mm_fmadd_ps(a_vec2, b_vec7, sum_vec47);\
	sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec8, sum_vec18);\
	sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec8, sum_vec28);\
	sum_vec38 = zq_mm_fmadd_ps(a_vec2, b_vec8, sum_vec38);\
	sum_vec48 = zq_mm_fmadd_ps(a_vec2, b_vec8, sum_vec48);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_4x8_2 \
	op_4x8;\
	op_4x8

#define op_4x8_4 \
	op_4x8_2;\
	op_4x8_2

#define op_4x8_8 \
	op_4x8_4;\
	op_4x8_4

#define op_4x8_16 \
	op_4x8_8;\
	op_4x8_8

#define op_4x8_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	b_vec3 = zq_mm_load_ps(B_c_ptr3);\
	b_vec4 = zq_mm_load_ps(B_c_ptr4);\
	b_vec5 = zq_mm_load_ps(B_c_ptr5);\
	b_vec6 = zq_mm_load_ps(B_c_ptr6);\
	b_vec7 = zq_mm_load_ps(B_c_ptr7);\
	b_vec8 = zq_mm_load_ps(B_c_ptr8);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec32 = zq_mm_mul_ps(a_vec3, b_vec2);\
	sum_vec42 = zq_mm_mul_ps(a_vec4, b_vec2);\
	sum_vec13 = zq_mm_mul_ps(a_vec1, b_vec3);\
	sum_vec23 = zq_mm_mul_ps(a_vec2, b_vec3);\
	sum_vec33 = zq_mm_mul_ps(a_vec3, b_vec3);\
	sum_vec43 = zq_mm_mul_ps(a_vec4, b_vec3);\
	sum_vec14 = zq_mm_mul_ps(a_vec1, b_vec4);\
	sum_vec24 = zq_mm_mul_ps(a_vec2, b_vec4);\
	sum_vec34 = zq_mm_mul_ps(a_vec3, b_vec4);\
	sum_vec44 = zq_mm_mul_ps(a_vec4, b_vec4);\
	sum_vec15 = zq_mm_mul_ps(a_vec1, b_vec5);\
	sum_vec25 = zq_mm_mul_ps(a_vec2, b_vec5);\
	sum_vec35 = zq_mm_mul_ps(a_vec3, b_vec5);\
	sum_vec45 = zq_mm_mul_ps(a_vec4, b_vec5);\
	sum_vec16 = zq_mm_mul_ps(a_vec1, b_vec6);\
	sum_vec26 = zq_mm_mul_ps(a_vec2, b_vec6);\
	sum_vec36 = zq_mm_mul_ps(a_vec3, b_vec6);\
	sum_vec46 = zq_mm_mul_ps(a_vec4, b_vec6);\
	sum_vec17 = zq_mm_mul_ps(a_vec1, b_vec7);\
	sum_vec27 = zq_mm_mul_ps(a_vec2, b_vec7);\
	sum_vec37 = zq_mm_mul_ps(a_vec3, b_vec7);\
	sum_vec47 = zq_mm_mul_ps(a_vec4, b_vec7);\
	sum_vec18 = zq_mm_mul_ps(a_vec1, b_vec8);\
	sum_vec28 = zq_mm_mul_ps(a_vec2, b_vec8);\
	sum_vec38 = zq_mm_mul_ps(a_vec3, b_vec8);\
	sum_vec48 = zq_mm_mul_ps(a_vec4, b_vec8);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size;\
	B_c_ptr3 += zq_mm_align_size;\
	B_c_ptr4 += zq_mm_align_size;\
	B_c_ptr5 += zq_mm_align_size;\
	B_c_ptr6 += zq_mm_align_size;\
	B_c_ptr7 += zq_mm_align_size;\
	B_c_ptr8 += zq_mm_align_size

#define op_4x8_2_first \
	op_4x8_first;\
	op_4x8

#define op_4x8_4_first \
	op_4x8_2_first;\
	op_4x8_2

#define op_4x8_8_first \
	op_4x8_4_first;\
	op_4x8_4

#define op_4x8_16_first \
	op_4x8_8_first;\
	op_4x8_8

#define op_8x1 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec51 = zq_mm_fmadd_ps(a_vec5, b_vec1, sum_vec51);\
	sum_vec61 = zq_mm_fmadd_ps(a_vec6, b_vec1, sum_vec61);\
	sum_vec71 = zq_mm_fmadd_ps(a_vec7, b_vec1, sum_vec71);\
	sum_vec81 = zq_mm_fmadd_ps(a_vec8, b_vec1, sum_vec81);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_8x1_2 \
	op_8x1;\
	op_8x1

#define op_8x1_4 \
	op_8x1_2;\
	op_8x1_2

#define op_8x1_8 \
	op_8x1_4;\
	op_8x1_4

#define op_8x1_16 \
	op_8x1_8;\
	op_8x1_8

#define op_8x1_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec51 = zq_mm_mul_ps(a_vec5, b_vec1);\
	sum_vec61 = zq_mm_mul_ps(a_vec6, b_vec1);\
	sum_vec71 = zq_mm_mul_ps(a_vec7, b_vec1);\
	sum_vec81 = zq_mm_mul_ps(a_vec8, b_vec1);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_8x1_2_first \
	op_8x1_first;\
	op_8x1

#define op_8x1_4_first \
	op_8x1_2_first;\
	op_8x1_2

#define op_8x1_8_first \
	op_8x1_4_first;\
	op_8x1_4

#define op_8x1_16_first \
	op_8x1_8_first;\
	op_8x1_8

#define op_8x2 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec51 = zq_mm_fmadd_ps(a_vec5, b_vec1, sum_vec51);\
	sum_vec61 = zq_mm_fmadd_ps(a_vec6, b_vec1, sum_vec61);\
	sum_vec71 = zq_mm_fmadd_ps(a_vec7, b_vec1, sum_vec71);\
	sum_vec81 = zq_mm_fmadd_ps(a_vec8, b_vec1, sum_vec81);\
	sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec2, sum_vec12);\
	sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec2, sum_vec22);\
	sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec2, sum_vec32);\
	sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec2, sum_vec42);\
	sum_vec52 = zq_mm_fmadd_ps(a_vec5, b_vec2, sum_vec52);\
	sum_vec62 = zq_mm_fmadd_ps(a_vec6, b_vec2, sum_vec62);\
	sum_vec72 = zq_mm_fmadd_ps(a_vec7, b_vec2, sum_vec72);\
	sum_vec82 = zq_mm_fmadd_ps(a_vec8, b_vec2, sum_vec82);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size

#define op_8x2_2 \
	op_8x2;\
	op_8x2

#define op_8x2_4 \
	op_8x2_2;\
	op_8x2_2

#define op_8x2_8 \
	op_8x2_4;\
	op_8x2_4

#define op_8x2_16 \
	op_8x2_8;\
	op_8x2_8

#define op_8x2_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	b_vec2 = zq_mm_load_ps(B_c_ptr2);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec51 = zq_mm_mul_ps(a_vec5, b_vec1);\
	sum_vec61 = zq_mm_mul_ps(a_vec6, b_vec1);\
	sum_vec71 = zq_mm_mul_ps(a_vec7, b_vec1);\
	sum_vec81 = zq_mm_mul_ps(a_vec8, b_vec1);\
	sum_vec12 = zq_mm_mul_ps(a_vec1, b_vec2);\
	sum_vec22 = zq_mm_mul_ps(a_vec2, b_vec2);\
	sum_vec32 = zq_mm_mul_ps(a_vec3, b_vec2);\
	sum_vec42 = zq_mm_mul_ps(a_vec4, b_vec2);\
	sum_vec52 = zq_mm_mul_ps(a_vec5, b_vec2);\
	sum_vec62 = zq_mm_mul_ps(a_vec6, b_vec2);\
	sum_vec72 = zq_mm_mul_ps(a_vec7, b_vec2);\
	sum_vec82 = zq_mm_mul_ps(a_vec8, b_vec2);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size;\
	B_c_ptr2 += zq_mm_align_size


#define op_8x2_2_first \
	op_8x2_first;\
	op_8x2

#define op_8x2_4_first \
	op_8x2_2_first;\
	op_8x2_2

#define op_8x2_8_first \
	op_8x2_4_first;\
	op_8x2_4

#define op_8x2_16_first \
	op_8x2_8_first;\
	op_8x2_8

#define op_16x1 \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	a_vec9 = zq_mm_load_ps(A_c_ptr9);\
	a_vecA = zq_mm_load_ps(A_c_ptrA);\
	a_vecB = zq_mm_load_ps(A_c_ptrB);\
	a_vecC = zq_mm_load_ps(A_c_ptrC);\
	a_vecD = zq_mm_load_ps(A_c_ptrD);\
	a_vecE = zq_mm_load_ps(A_c_ptrE);\
	a_vecF = zq_mm_load_ps(A_c_ptrF);\
	a_vecG = zq_mm_load_ps(A_c_ptrG);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec1, sum_vec11);\
	sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec1, sum_vec21);\
	sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec1, sum_vec31);\
	sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec1, sum_vec41);\
	sum_vec51 = zq_mm_fmadd_ps(a_vec5, b_vec1, sum_vec51);\
	sum_vec61 = zq_mm_fmadd_ps(a_vec6, b_vec1, sum_vec61);\
	sum_vec71 = zq_mm_fmadd_ps(a_vec7, b_vec1, sum_vec71);\
	sum_vec81 = zq_mm_fmadd_ps(a_vec8, b_vec1, sum_vec81);\
	sum_vec91 = zq_mm_fmadd_ps(a_vec9, b_vec1, sum_vec91);\
	sum_vecA1 = zq_mm_fmadd_ps(a_vecA, b_vec1, sum_vecA1);\
	sum_vecB1 = zq_mm_fmadd_ps(a_vecB, b_vec1, sum_vecB1);\
	sum_vecC1 = zq_mm_fmadd_ps(a_vecC, b_vec1, sum_vecC1);\
	sum_vecD1 = zq_mm_fmadd_ps(a_vecD, b_vec1, sum_vecD1);\
	sum_vecE1 = zq_mm_fmadd_ps(a_vecE, b_vec1, sum_vecE1);\
	sum_vecF1 = zq_mm_fmadd_ps(a_vecF, b_vec1, sum_vecF1);\
	sum_vecG1 = zq_mm_fmadd_ps(a_vecG, b_vec1, sum_vecG1);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	A_c_ptr9 += zq_mm_align_size;\
	A_c_ptrA += zq_mm_align_size;\
	A_c_ptrB += zq_mm_align_size;\
	A_c_ptrC += zq_mm_align_size;\
	A_c_ptrD += zq_mm_align_size;\
	A_c_ptrE += zq_mm_align_size;\
	A_c_ptrF += zq_mm_align_size;\
	A_c_ptrG += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_16x1_2 \
	op_16x1;\
	op_16x1

#define op_16x1_4 \
	op_16x1_2;\
	op_16x1_2

#define op_16x1_8 \
	op_16x1_4;\
	op_16x1_4

#define op_16x1_16 \
	op_16x1_8;\
	op_16x1_8

#define op_16x1_first \
	a_vec1 = zq_mm_load_ps(A_c_ptr1);\
	a_vec2 = zq_mm_load_ps(A_c_ptr2);\
	a_vec3 = zq_mm_load_ps(A_c_ptr3);\
	a_vec4 = zq_mm_load_ps(A_c_ptr4);\
	a_vec5 = zq_mm_load_ps(A_c_ptr5);\
	a_vec6 = zq_mm_load_ps(A_c_ptr6);\
	a_vec7 = zq_mm_load_ps(A_c_ptr7);\
	a_vec8 = zq_mm_load_ps(A_c_ptr8);\
	a_vec9 = zq_mm_load_ps(A_c_ptr9);\
	a_vecA = zq_mm_load_ps(A_c_ptrA);\
	a_vecB = zq_mm_load_ps(A_c_ptrB);\
	a_vecC = zq_mm_load_ps(A_c_ptrC);\
	a_vecD = zq_mm_load_ps(A_c_ptrD);\
	a_vecE = zq_mm_load_ps(A_c_ptrE);\
	a_vecF = zq_mm_load_ps(A_c_ptrF);\
	a_vecG = zq_mm_load_ps(A_c_ptrG);\
	b_vec1 = zq_mm_load_ps(B_c_ptr1);\
	sum_vec11 = zq_mm_mul_ps(a_vec1, b_vec1);\
	sum_vec21 = zq_mm_mul_ps(a_vec2, b_vec1);\
	sum_vec31 = zq_mm_mul_ps(a_vec3, b_vec1);\
	sum_vec41 = zq_mm_mul_ps(a_vec4, b_vec1);\
	sum_vec51 = zq_mm_mul_ps(a_vec5, b_vec1);\
	sum_vec61 = zq_mm_mul_ps(a_vec6, b_vec1);\
	sum_vec71 = zq_mm_mul_ps(a_vec7, b_vec1);\
	sum_vec81 = zq_mm_mul_ps(a_vec8, b_vec1);\
	sum_vec91 = zq_mm_mul_ps(a_vec9, b_vec1);\
	sum_vecA1 = zq_mm_mul_ps(a_vecA, b_vec1);\
	sum_vecB1 = zq_mm_mul_ps(a_vecB, b_vec1);\
	sum_vecC1 = zq_mm_mul_ps(a_vecC, b_vec1);\
	sum_vecD1 = zq_mm_mul_ps(a_vecD, b_vec1);\
	sum_vecE1 = zq_mm_mul_ps(a_vecE, b_vec1);\
	sum_vecF1 = zq_mm_mul_ps(a_vecF, b_vec1);\
	sum_vecG1 = zq_mm_mul_ps(a_vecG, b_vec1);\
	A_c_ptr1 += zq_mm_align_size;\
	A_c_ptr2 += zq_mm_align_size;\
	A_c_ptr3 += zq_mm_align_size;\
	A_c_ptr4 += zq_mm_align_size;\
	A_c_ptr5 += zq_mm_align_size;\
	A_c_ptr6 += zq_mm_align_size;\
	A_c_ptr7 += zq_mm_align_size;\
	A_c_ptr8 += zq_mm_align_size;\
	A_c_ptr9 += zq_mm_align_size;\
	A_c_ptrA += zq_mm_align_size;\
	A_c_ptrB += zq_mm_align_size;\
	A_c_ptrC += zq_mm_align_size;\
	A_c_ptrD += zq_mm_align_size;\
	A_c_ptrE += zq_mm_align_size;\
	A_c_ptrF += zq_mm_align_size;\
	A_c_ptrG += zq_mm_align_size;\
	B_c_ptr1 += zq_mm_align_size

#define op_16x1_2_first \
	op_16x1_first;\
	op_16x1

#define op_16x1_4_first \
	op_16x1_2_first;\
	op_16x1_2

#define op_16x1_8_first \
	op_16x1_4_first;\
	op_16x1_4

#define op_16x1_16_first \
	op_16x1_8_first;\
	op_16x1_8

#if CUR_IS_AVX
#define store_1x1 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x2 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x4 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x8 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec15);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec16);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec17);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec18);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_2x1 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x2 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x4 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x8 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec15);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec16);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec17);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec18);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec25);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec26);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec27);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec28);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_4x1 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_4x2 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_4x4 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec33);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec34);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec43);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec44);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_8x1 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr8++) = zq_final_sum_q0_4
	

#define store_8x2 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec52);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec62);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec72);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr8++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec82);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr8++) = zq_final_sum_q0_4

#define store_16x1 \
	zq_store_to_q(q.s, sum_vec11);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr8++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec91);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptr9++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecA1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrA++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecB1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrB++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecC1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrC++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecD1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrD++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecE1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrE++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecF1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrF++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecG1);\
	q.p[0] = _mm_add_ps(q.p[0],q.p[1]);\
	*(C_c_ptrG++) = zq_final_sum_q0_4

#elif __ARM_NEON && __ARM_NEON_ARMV8

#define store_1x1 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11)

#define store_1x2 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12)

#define store_1x4 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec13);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec14)

#define store_1x8 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec13);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec14);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec15);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec16);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec17);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec18)

#define store_2x1 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21)

#define store_2x2 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22)

#define store_2x4 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec13);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec14);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec23);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec24)

#define store_2x8 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec13);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec14);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec15);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec16);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec17);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec18);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec23);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec24);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec25);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec26);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec27);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec28)

#define store_4x1 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41)

#define store_4x2 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec32);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec42)

#define store_4x4 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec13);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec14);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec23);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec24);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec32);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec33);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec34);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec42);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec43);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec44)

#define store_8x1 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41);\
	*(C_c_ptr5++) = vaddvq_f32(sum_vec51);\
	*(C_c_ptr6++) = vaddvq_f32(sum_vec61);\
	*(C_c_ptr7++) = vaddvq_f32(sum_vec71);\
	*(C_c_ptr8++) = vaddvq_f32(sum_vec81)

#define store_8x2 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr1++) = vaddvq_f32(sum_vec12);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec22);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec32);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec42);\
	*(C_c_ptr5++) = vaddvq_f32(sum_vec51);\
	*(C_c_ptr5++) = vaddvq_f32(sum_vec52);\
	*(C_c_ptr6++) = vaddvq_f32(sum_vec61);\
	*(C_c_ptr6++) = vaddvq_f32(sum_vec62);\
	*(C_c_ptr7++) = vaddvq_f32(sum_vec71);\
	*(C_c_ptr7++) = vaddvq_f32(sum_vec72);\
	*(C_c_ptr8++) = vaddvq_f32(sum_vec81);\
	*(C_c_ptr8++) = vaddvq_f32(sum_vec82)

#define store_16x1 \
	*(C_c_ptr1++) = vaddvq_f32(sum_vec11);\
	*(C_c_ptr2++) = vaddvq_f32(sum_vec21);\
	*(C_c_ptr3++) = vaddvq_f32(sum_vec31);\
	*(C_c_ptr4++) = vaddvq_f32(sum_vec41);\
	*(C_c_ptr5++) = vaddvq_f32(sum_vec51);\
	*(C_c_ptr6++) = vaddvq_f32(sum_vec61);\
	*(C_c_ptr7++) = vaddvq_f32(sum_vec71);\
	*(C_c_ptr8++) = vaddvq_f32(sum_vec81);\
	*(C_c_ptr9++) = vaddvq_f32(sum_vec91);\
	*(C_c_ptrA++) = vaddvq_f32(sum_vecA1);\
	*(C_c_ptrB++) = vaddvq_f32(sum_vecB1);\
	*(C_c_ptrC++) = vaddvq_f32(sum_vecC1);\
	*(C_c_ptrD++) = vaddvq_f32(sum_vecD1);\
	*(C_c_ptrE++) = vaddvq_f32(sum_vecE1);\
	*(C_c_ptrF++) = vaddvq_f32(sum_vecF1);\
	*(C_c_ptrG++) = vaddvq_f32(sum_vecG1)

#else
#define store_1x1 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x2 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x4 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_1x8 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec15);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec16);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec17);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec18);\
	*(C_c_ptr1++) = zq_final_sum_q0_4

#define store_2x1 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x2 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x4 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_2x8 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec15);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec16);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec17);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec18);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec25);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec26);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec27);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec28);\
	*(C_c_ptr2++) = zq_final_sum_q0_4

#define store_4x1 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_4x2 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_4x4 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec13);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec14);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec23);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec24);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec33);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec34);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec43);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec44);\
	*(C_c_ptr4++) = zq_final_sum_q0_4

#define store_8x1 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	*(C_c_ptr8++) = zq_final_sum_q0_4

#define store_8x2 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec12);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec22);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec32);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec42);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec52);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec62);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec72);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	*(C_c_ptr8++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec82);\
	*(C_c_ptr8++) = zq_final_sum_q0_4

#define store_16x1 \
	zq_store_to_q(q.s, sum_vec11);\
	*(C_c_ptr1++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec21);\
	*(C_c_ptr2++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec31);\
	*(C_c_ptr3++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec41);\
	*(C_c_ptr4++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec51);\
	*(C_c_ptr5++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec61);\
	*(C_c_ptr6++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec71);\
	*(C_c_ptr7++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec81);\
	*(C_c_ptr8++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vec91);\
	*(C_c_ptr9++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecA1);\
	*(C_c_ptrA++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecB1);\
	*(C_c_ptrB++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecC1);\
	*(C_c_ptrC++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecD1);\
	*(C_c_ptrD++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecE1);\
	*(C_c_ptrE++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecF1);\
	*(C_c_ptrF++) = zq_final_sum_q0_4;\
	zq_store_to_q(q.s, sum_vecG1);\
	*(C_c_ptrG++) = zq_final_sum_q0_4
#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N1_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	zq_base_type a1;
	int m, n, k;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
#if __ARM_NEON && __ARM_NEON_ARMV8
			*C_c_ptr1 = vaddvq_f32(sum_vec11);
#else
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
#endif

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
			}

			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N2_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	zq_base_type a1;
	int m, n, k;
	int ldb2 = ldb << 1;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x2;
			}
#if __ARM_NEON && __ARM_NEON_ARMV8
			*(C_c_ptr1) = vaddvq_f32(sum_vec11);
			*(C_c_ptr1 + 1) = vaddvq_f32(sum_vec12);
#else
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
#endif
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
			}

			C_c_ptr1 += 2;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (;k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				q.s[0] += a1 * (*(B_c_ptr1++));
			}
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N4_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	zq_base_type a1;
	int m, n, k;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 3; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
#if !__ARM_NEON
			for (;k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x4_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x4_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x4_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x4_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x4;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1 * (*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1 * (*(B_c_ptr4++));
			}

			C_c_ptr1 += 4;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				q.s[0] += a1 * (*(B_c_ptr1++));
			}
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N8_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	zq_base_type a1;
	int m, n, k;
	int ldb8 = ldb << 3;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 7; n += 8, Bptr += ldb8)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			sum_vec15 = zq_mm_setzero_ps();
			sum_vec16 = zq_mm_setzero_ps();
			sum_vec17 = zq_mm_setzero_ps();
			sum_vec18 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			k = 0; A_c_ptr1 = Aptr1; 
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x8_16;
				op_1x8_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x8_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x8_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x8_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x8_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x8;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec15);
			*(C_c_ptr1 + 4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec16);
			*(C_c_ptr1 + 5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec17);
			*(C_c_ptr1 + 6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec18);
			*(C_c_ptr1 + 7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1 * (*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1 * (*(B_c_ptr4++));
				*(C_c_ptr1 + 4) += a1 * (*(B_c_ptr5++));
				*(C_c_ptr1 + 5) += a1 * (*(B_c_ptr6++));
				*(C_c_ptr1 + 6) += a1 * (*(B_c_ptr7++));
				*(C_c_ptr1 + 7) += a1 * (*(B_c_ptr8++));
			}

			C_c_ptr1 += 8;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64;	k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				q.s[0] += a1 * (*(B_c_ptr1++));
			}
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N1_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type* Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	zq_base_type a1, a2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr2) += a2 * (*(B_c_ptr1++));
			}

			C_c_ptr1++;
			C_c_ptr2++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N2_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type* Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	zq_base_type a1, a2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x2_16;
				op_2x2_16;
				op_2x2_16;
				op_2x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x2_16;
				op_2x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr2) += a2 * (*(B_c_ptr1++));
				*(C_c_ptr2 + 1) += a2 * (*(B_c_ptr2++));
			}

			C_c_ptr1 += 2;
			C_c_ptr2 += 2;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
			}
			C_c_ptr1 += 2;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N4_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type* Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	zq_base_type a1, a2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N - 3; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x4_16;
				op_2x4_16;
				op_2x4_16;
				op_2x4_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x4_16;
				op_2x4_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x4_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x4_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x4_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x4_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x4;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1 * (*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1 * (*B_c_ptr4);
				*(C_c_ptr2) += a2 * (*(B_c_ptr1++));
				*(C_c_ptr2 + 1) += a2 * (*(B_c_ptr2++));
				*(C_c_ptr2 + 2) += a2 * (*(B_c_ptr3++));
				*(C_c_ptr2 + 3) += a2 * (*(B_c_ptr4++));
			}

			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 3; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			k = 0; A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x4_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x4_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x4_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x4_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x4;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1 * (*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1 * (*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N8_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type* Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	zq_base_type a1, a2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb << 3;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N - 7; n += 8, Bptr += ldb8)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps();
			sum_vec15 = zq_mm_setzero_ps(); sum_vec25 = zq_mm_setzero_ps();
			sum_vec16 = zq_mm_setzero_ps(); sum_vec26 = zq_mm_setzero_ps();
			sum_vec17 = zq_mm_setzero_ps(); sum_vec27 = zq_mm_setzero_ps();
			sum_vec18 = zq_mm_setzero_ps(); sum_vec28 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x8_16;
				op_2x8_16;
				op_2x8_16;
				op_2x8_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x8_16;
				op_2x8_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x8_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x8_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x8_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x8_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x8;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec15);
			*(C_c_ptr1 + 4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec16);
			*(C_c_ptr1 + 5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec17);
			*(C_c_ptr1 + 6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec18);
			*(C_c_ptr1 + 7) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec25);
			*(C_c_ptr2 + 4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec26);
			*(C_c_ptr2 + 5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec27);
			*(C_c_ptr2 + 6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec28);
			*(C_c_ptr2 + 7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1 * (*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1 * (*B_c_ptr4);
				*(C_c_ptr1 + 4) += a1 * (*B_c_ptr5);
				*(C_c_ptr1 + 5) += a1 * (*B_c_ptr6);
				*(C_c_ptr1 + 6) += a1 * (*B_c_ptr7);
				*(C_c_ptr1 + 7) += a1 * (*B_c_ptr8);
				*(C_c_ptr2) += a2 * (*(B_c_ptr1++));
				*(C_c_ptr2 + 1) += a2 * (*(B_c_ptr2++));
				*(C_c_ptr2 + 2) += a2 * (*(B_c_ptr3++));
				*(C_c_ptr2 + 3) += a2 * (*(B_c_ptr4++));
				*(C_c_ptr2 + 4) += a2 * (*(B_c_ptr5++));
				*(C_c_ptr2 + 5) += a2 * (*(B_c_ptr6++));
				*(C_c_ptr2 + 6) += a2 * (*(B_c_ptr7++));
				*(C_c_ptr2 + 7) += a2 * (*(B_c_ptr8++));
			}

			C_c_ptr1 += 8;
			C_c_ptr2 += 8;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_2x1_16;
				op_2x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_2x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_2x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_2x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_2x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_2x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 7; n += 8, Bptr += ldb8)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			sum_vec15 = zq_mm_setzero_ps();
			sum_vec16 = zq_mm_setzero_ps();
			sum_vec17 = zq_mm_setzero_ps();
			sum_vec18 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			k = 0; A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x8_16;
				op_1x8_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x8_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x8_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x8_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x8_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x8;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec15);
			*(C_c_ptr1 + 4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec16);
			*(C_c_ptr1 + 5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec17);
			*(C_c_ptr1 + 6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec18);
			*(C_c_ptr1 + 7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1 * (*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1 * (*(B_c_ptr4++));
				*(C_c_ptr1 + 4) += a1 * (*(B_c_ptr5++));
				*(C_c_ptr1 + 5) += a1 * (*(B_c_ptr6++));
				*(C_c_ptr1 + 6) += a1 * (*(B_c_ptr7++));
				*(C_c_ptr1 + 7) += a1 * (*(B_c_ptr8++));
			}
			C_c_ptr1 += 8;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N1_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type a1, a2, a3, a4;
	int m, n, k;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_4x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_4x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_4x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_4x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_4x1;
			}

			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr4) += a4 * (*(B_c_ptr1++));
			}

			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N2_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type a1, a2, a3, a4;
	int m, n, k;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_4x2_16;
				op_4x2_16;
				op_4x2_16;
				op_4x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_4x2_16;
				op_4x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_4x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_4x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_4x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_4x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_4x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec32);
			*(C_c_ptr3 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec42);
			*(C_c_ptr4 + 1) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr2 + 1) += a2 * (*B_c_ptr2);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr3 + 1) += a3 * (*B_c_ptr2);
				*(C_c_ptr4) += a4 * (*(B_c_ptr1++));
				*(C_c_ptr4 + 1) += a4 * (*(B_c_ptr2++));
			}

			C_c_ptr1 += 2;
			C_c_ptr2 += 2;
			C_c_ptr3 += 2;
			C_c_ptr4 += 2;
		}

		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_4x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_4x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_4x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_4x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_4x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*C_c_ptr3 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*C_c_ptr4 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*B_c_ptr1);
				*C_c_ptr3 += a3 * (*B_c_ptr1);
				*C_c_ptr4 += a4 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
			}
			C_c_ptr1 += 2;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
				k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N4_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type a1, a2, a3, a4;
	int m, n, k;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N - 3; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps(); sum_vec33 = zq_mm_setzero_ps(); sum_vec43 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps(); sum_vec34 = zq_mm_setzero_ps(); sum_vec44 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_4x4_16;
				op_4x4_16;
				op_4x4_16;
				op_4x4_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_4x4_16;
				op_4x4_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_4x4_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_4x4_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_4x4_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_4x4_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_4x4;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec32);
			*(C_c_ptr3 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec33);
			*(C_c_ptr3 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec34);
			*(C_c_ptr3 + 3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec42);
			*(C_c_ptr4 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec43);
			*(C_c_ptr4 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec44);
			*(C_c_ptr4 + 3) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1 * (*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1 * (*B_c_ptr4);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr2 + 1) += a2 * (*B_c_ptr2);
				*(C_c_ptr2 + 2) += a2 * (*B_c_ptr3);
				*(C_c_ptr2 + 3) += a2 * (*B_c_ptr4);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr3 + 1) += a3 * (*B_c_ptr2);
				*(C_c_ptr3 + 2) += a3 * (*B_c_ptr3);
				*(C_c_ptr3 + 3) += a3 * (*B_c_ptr4);
				*(C_c_ptr4) += a4 * (*(B_c_ptr1++));
				*(C_c_ptr4 + 1) += a4 * (*(B_c_ptr2++));
				*(C_c_ptr4 + 2) += a4 * (*(B_c_ptr3++));
				*(C_c_ptr4 + 3) += a4 * (*(B_c_ptr4++));
			}

			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
			C_c_ptr3 += 4;
			C_c_ptr4 += 4;
		}

		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_4x1_16;
				op_4x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_4x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_4x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_4x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_4x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_4x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*C_c_ptr3 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*C_c_ptr4 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*B_c_ptr1);
				*C_c_ptr3 += a3 * (*B_c_ptr1);
				*C_c_ptr4 += a4 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 3; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			k = 0; A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x4_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x4_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x4_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x4_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x4;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1 * (*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1 * (*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_N1_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type a1, a2, a3, a4, a5, a6, a7, a8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, 
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec51 = zq_mm_setzero_ps(); sum_vec61 = zq_mm_setzero_ps(); sum_vec71 = zq_mm_setzero_ps(); sum_vec81 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_8x1_16;
				op_8x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_8x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_8x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_8x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_8x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_8x1;
			}

			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec51);
			*(C_c_ptr5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec61);
			*(C_c_ptr6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec71);
			*(C_c_ptr7) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec81);
			*(C_c_ptr8) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				a5 = *(A_c_ptr5++);
				a6 = *(A_c_ptr6++);
				a7 = *(A_c_ptr7++);
				a8 = *(A_c_ptr8++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr4) += a4 * (*B_c_ptr1);
				*(C_c_ptr5) += a5 * (*B_c_ptr1);
				*(C_c_ptr6) += a6 * (*B_c_ptr1);
				*(C_c_ptr7) += a7 * (*B_c_ptr1);
				*(C_c_ptr8) += a8 * (*(B_c_ptr1++));
			}

			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
			C_c_ptr5++;
			C_c_ptr6++;
			C_c_ptr7++;
			C_c_ptr8++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_N2_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type a1, a2, a3, a4, a5, a6, a7, a8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec51 = zq_mm_setzero_ps(); sum_vec61 = zq_mm_setzero_ps(); sum_vec71 = zq_mm_setzero_ps(); sum_vec81 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			sum_vec52 = zq_mm_setzero_ps(); sum_vec62 = zq_mm_setzero_ps(); sum_vec72 = zq_mm_setzero_ps(); sum_vec82 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_8x2_16;
				op_8x2_16;
				op_8x2_16;
				op_8x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_8x2_16;
				op_8x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_8x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_8x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_8x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_8x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_8x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec32);
			*(C_c_ptr3 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec42);
			*(C_c_ptr4 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec51);
			*(C_c_ptr5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec52);
			*(C_c_ptr5 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec61);
			*(C_c_ptr6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec62);
			*(C_c_ptr6 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec71);
			*(C_c_ptr7) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec72);
			*(C_c_ptr7 + 1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec81);
			*(C_c_ptr8) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec82);
			*(C_c_ptr8 + 1) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				a5 = *(A_c_ptr5++);
				a6 = *(A_c_ptr6++);
				a7 = *(A_c_ptr7++);
				a8 = *(A_c_ptr8++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1 * (*B_c_ptr2);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr2 + 1) += a2 * (*B_c_ptr2);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr3 + 1) += a3 * (*B_c_ptr2);
				*(C_c_ptr4) += a4 * (*B_c_ptr1);
				*(C_c_ptr4 + 1) += a4 * (*B_c_ptr2);
				*(C_c_ptr5) += a5 * (*B_c_ptr1);
				*(C_c_ptr5 + 1) += a5 * (*B_c_ptr2);
				*(C_c_ptr6) += a6 * (*B_c_ptr1);
				*(C_c_ptr6 + 1) += a6 * (*B_c_ptr2);
				*(C_c_ptr7) += a7 * (*B_c_ptr1);
				*(C_c_ptr7 + 1) += a7 * (*B_c_ptr2);
				*(C_c_ptr8) += a8 * (*(B_c_ptr1++));
				*(C_c_ptr8 + 1) += a8 * (*(B_c_ptr2++));
			}

			C_c_ptr1 += 2;
			C_c_ptr2 += 2;
			C_c_ptr3 += 2;
			C_c_ptr4 += 2;
			C_c_ptr5 += 2;
			C_c_ptr6 += 2;
			C_c_ptr7 += 2;
			C_c_ptr8 += 2;
		}

		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec51 = zq_mm_setzero_ps(); sum_vec61 = zq_mm_setzero_ps(); sum_vec71 = zq_mm_setzero_ps(); sum_vec81 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_8x1_16;
				op_8x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_8x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_8x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_8x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_8x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_8x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*C_c_ptr2 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*C_c_ptr3 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*C_c_ptr4 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec51);
			*C_c_ptr5 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec61);
			*C_c_ptr6 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec71);
			*C_c_ptr7 = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec81);
			*C_c_ptr8 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				a5 = *(A_c_ptr5++);
				a6 = *(A_c_ptr6++);
				a7 = *(A_c_ptr7++);
				a8 = *(A_c_ptr8++);
				*C_c_ptr1 += a1 * (*B_c_ptr1);
				*C_c_ptr2 += a2 * (*B_c_ptr1);
				*C_c_ptr3 += a3 * (*B_c_ptr1);
				*C_c_ptr4 += a4 * (*B_c_ptr1);
				*C_c_ptr5 += a5 * (*B_c_ptr1);
				*C_c_ptr6 += a6 * (*B_c_ptr1);
				*C_c_ptr7 += a7 * (*B_c_ptr1);
				*C_c_ptr8 += a8 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
			C_c_ptr5++;
			C_c_ptr6++;
			C_c_ptr7++;
			C_c_ptr8++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N - 1; n += 2, Bptr += ldb2)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			k = 0; A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x2_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x2_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x2_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x2_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x2;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1 * (*(B_c_ptr2++));
			}
			C_c_ptr1 += 2;
		}
		for (; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*C_c_ptr1 = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*C_c_ptr1 += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_N1_Kgeneral(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type* Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	zq_base_type a1, a2, a3, a4, a5, a6, a7, a8;
	zq_base_type a9, aA, aB, aC, aD, aE, aF, aG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size * zq_mm_align_size;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1, sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC, a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec51 = zq_mm_setzero_ps(); sum_vec61 = zq_mm_setzero_ps(); sum_vec71 = zq_mm_setzero_ps(); sum_vec81 = zq_mm_setzero_ps();
			sum_vec91 = zq_mm_setzero_ps(); sum_vecA1 = zq_mm_setzero_ps(); sum_vecB1 = zq_mm_setzero_ps(); sum_vecC1 = zq_mm_setzero_ps();
			sum_vecD1 = zq_mm_setzero_ps(); sum_vecE1 = zq_mm_setzero_ps(); sum_vecF1 = zq_mm_setzero_ps(); sum_vecG1 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_16x1_16;
				op_16x1_16;
				op_16x1_16;
				op_16x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_16x1_16;
				op_16x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_16x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_16x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_16x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_16x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_16x1;
			}

			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec51);
			*(C_c_ptr5) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec61);
			*(C_c_ptr6) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec71);
			*(C_c_ptr7) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec81);
			*(C_c_ptr8) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vec91);
			*(C_c_ptr9) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecA1);
			*(C_c_ptrA) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecB1);
			*(C_c_ptrB) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecC1);
			*(C_c_ptrC) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecD1);
			*(C_c_ptrD) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecE1);
			*(C_c_ptrE) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecF1);
			*(C_c_ptrF) = zq_final_sum_q;
			zq_store_to_q(q.s, sum_vecG1);
			*(C_c_ptrG) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				a5 = *(A_c_ptr5++);
				a6 = *(A_c_ptr6++);
				a7 = *(A_c_ptr7++);
				a8 = *(A_c_ptr8++);
				a9 = *(A_c_ptr9++);
				aA = *(A_c_ptrA++);
				aB = *(A_c_ptrB++);
				aC = *(A_c_ptrC++);
				aD = *(A_c_ptrD++);
				aE = *(A_c_ptrE++);
				aF = *(A_c_ptrF++);
				aG = *(A_c_ptrG++);
				*(C_c_ptr1) += a1 * (*B_c_ptr1);
				*(C_c_ptr2) += a2 * (*B_c_ptr1);
				*(C_c_ptr3) += a3 * (*B_c_ptr1);
				*(C_c_ptr4) += a4 * (*B_c_ptr1);
				*(C_c_ptr5) += a5 * (*B_c_ptr1);
				*(C_c_ptr6) += a6 * (*B_c_ptr1);
				*(C_c_ptr7) += a7 * (*B_c_ptr1);
				*(C_c_ptr8) += a8 * (*B_c_ptr1);
				*(C_c_ptr9) += a9 * (*B_c_ptr1);
				*(C_c_ptrA) += aA * (*B_c_ptr1);
				*(C_c_ptrB) += aB * (*B_c_ptr1);
				*(C_c_ptrC) += aC * (*B_c_ptr1);
				*(C_c_ptrD) += aD * (*B_c_ptr1);
				*(C_c_ptrE) += aE * (*B_c_ptr1);
				*(C_c_ptrF) += aF * (*B_c_ptr1);
				*(C_c_ptrG) += aG * (*(B_c_ptr1++));
			}

			C_c_ptr1++;
			C_c_ptr2++;
			C_c_ptr3++;
			C_c_ptr4++;
			C_c_ptr5++;
			C_c_ptr6++;
			C_c_ptr7++;
			C_c_ptr8++;
			C_c_ptr9++;
			C_c_ptrA++;
			C_c_ptrB++;
			C_c_ptrC++;
			C_c_ptrD++;
			C_c_ptrE++;
			C_c_ptrF++;
			C_c_ptrG++;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			sum_vec11 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			k = 0; A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
#if !__ARM_NEON
			for (; k < padK - zq_mm_align_size64; k += zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size32; k += zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			for (; k < padK - zq_mm_align_size16; k += zq_mm_align_size16)
			{
				op_1x1_16;
			}
#endif
			for (; k < padK - zq_mm_align_size8; k += zq_mm_align_size8)
			{
				op_1x1_8;
			}
			for (; k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				op_1x1_4;
			}
			for (; k < padK - zq_mm_align_size2; k += zq_mm_align_size2)
			{
				op_1x1_2;
			}
			for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
			{
				op_1x1;
			}
			zq_store_to_q(q.s, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1 * (*(B_c_ptr1++));
			}
			C_c_ptr1++;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x1_4;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x1_8;
			}
			store_1x1;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type a_vec1, b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n ++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			op_1x1_16;
			op_1x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x2_4;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x2_8;
			}
			store_1x2;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type a_vec1, b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			op_1x2_16;
			op_1x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

#endif 

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x4_4;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x4_8;
			}
			store_1x4;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type a_vec1, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			op_1x4_16;
			op_1x4_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_2_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_2;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_4_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_2_first;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_2;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x8_4;
			}
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x8_8;
			}
			store_1x8;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x8_16;
			}
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			op_1x8_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x8_16;
				op_1x8_16;
			}
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *A_c_ptr1, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *C_c_ptr1;
	int m, n, k;
	int ldb8 = ldb << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11;
	register zq_mm_type sum_vec12;
	register zq_mm_type sum_vec13;
	register zq_mm_type sum_vec14;
	register zq_mm_type sum_vec15;
	register zq_mm_type sum_vec16;
	register zq_mm_type sum_vec17;
	register zq_mm_type sum_vec18;
	register zq_mm_type a_vec1;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;

	Aptr1 = A;
	Cptr1 = C;
	for (m = 0; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			op_1x8_16;
			op_1x8_16;
			op_1x8_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
			}
			store_1x8;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_first;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_2_first;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_first;
			op_2x1_2;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_4_first;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_first;
			op_2x1_4;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_2_first;
			op_2x1_4;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_first;
			op_2x1_2;
			op_2x1_4;
			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_2x1_4;
			}

			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x1_4;
			}

			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_2x1_8;
			}

			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x1_8;
			}
			store_1x1;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_2x1_16;
			}

			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_16_first;
			op_2x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_2x1_16;
				op_2x1_16;
			}

			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type a_vec1, a_vec2, b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1;
			op_2x1_16_first;
			op_2x1_16;
			op_2x1_16;
			op_2x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
				op_2x1_16;
			}

			store_2x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			op_1x1_16;
			op_1x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_first;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_2_first;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_first;
			op_2x2_2;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_4_first;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_first;
			op_2x2_4;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_2_first;
			op_2x2_4;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_first;
			op_2x2_2;
			op_2x2_4;
			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_2x2_4;
			}

			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x2_4;
			}

			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_2x2_8;
			}

			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x2_8;
			}
			store_1x2;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_2x2_16;
			}

			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_16_first;
			op_2x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_2x2_16;
				op_2x2_16;
			}

			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_2x2_16_first;
			op_2x2_16;
			op_2x2_16;
			op_2x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_2x2_16;
				op_2x2_16;
				op_2x2_16;
				op_2x2_16;
			}

			store_2x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			op_1x2_16;
			op_1x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_first;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_2_first;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_first;
			op_2x4_2;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_4_first;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_first;
			op_2x4_4;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_2_first;
			op_2x4_4;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_first;
			op_2x4_2;
			op_2x4_4;
			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_2x4_4;
			}

			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x4_4;
			}

			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_2x4_8;
			}

			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x4_8;
			}
			store_1x4;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_2x4_16;
			}

			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_16_first;
			op_2x4_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_2x4_16;
				op_2x4_16;
			}

			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_2x4_16_first;
			op_2x4_16;
			op_2x4_16;
			op_2x4_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_2x4_16;
				op_2x4_16;
				op_2x4_16;
				op_2x4_16;
			}

			store_2x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			op_1x4_16;
			op_1x4_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_first;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_2_first;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_2_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_first;
			op_2x8_2;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_2;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_4_first;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_4_first;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_first;
			op_2x8_4;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_2_first;
			op_2x8_4;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_2_first;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_first;
			op_2x8_2;
			op_2x8_4;
			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_first;
			op_1x8_2;
			op_1x8_4;
			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_2x8_4;
			}

			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x8_4;
			}

			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_2x8_8;
			}

			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x8_8;
			}

			store_1x8;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_2x8_16;
			}

			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x8_16;
			}

			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_16_first;
			op_2x8_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_2x8_16;
				op_2x8_16;
			}

			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			op_1x8_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x8_16;
				op_1x8_16;
			}

			store_1x8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	zq_base_type* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda << 1;
	int ldc2 = ldc << 1;
	int ldb8 = ldb * 8;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4, b_vec5, b_vec6, b_vec7, b_vec8;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_2x8_16_first;
			op_2x8_16;
			op_2x8_16;
			op_2x8_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_2x8_16;
				op_2x8_16;
				op_2x8_16;
				op_2x8_16;
			}

			store_2x8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			Bptr5 = Bptr4 + ldb;
			Bptr6 = Bptr5 + ldb;
			Bptr7 = Bptr6 + ldb;
			Bptr8 = Bptr7 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			B_c_ptr5 = Bptr5; B_c_ptr6 = Bptr6; B_c_ptr7 = Bptr7; B_c_ptr8 = Bptr8;
			op_1x8_16_first;
			op_1x8_16;
			op_1x8_16;
			op_1x8_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
				op_1x8_16;
			}

			store_1x8;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_first;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_2_first;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_first;
			op_4x1_2;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_4_first;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_first;
			op_4x1_4;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_2_first;
			op_4x1_4;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_first;
			op_4x1_2;
			op_4x1_4;
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda4 = lda << 2;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_4x1_4;
			}
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x1_4;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_4x1_8;
			}
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x1_8;
			}
			store_1x1;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_4x1_16;
			}
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_16_first;
			op_4x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_4x1_16;
				op_4x1_16;
			}
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1;
			op_4x1_16_first;
			op_4x1_16;
			op_4x1_16;
			op_4x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
				op_4x1_16;
			}
			store_4x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			op_1x1_16;
			op_1x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_first;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_2_first;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_first;
			op_4x2_2;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_4_first;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_first;
			op_4x2_4;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_2_first;
			op_4x2_4;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_first;
			op_4x2_2;
			op_4x2_4;
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_4x2_4;
			}
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x2_4;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_4x2_8;
			}
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x2_8;
			}
			store_1x2;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_4x2_16;
			}
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_16_first;
			op_4x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_4x2_16;
				op_4x2_16;
			}
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_4x2_16_first;
			op_4x2_16;
			op_4x2_16;
			op_4x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_4x2_16;
				op_4x2_16;
				op_4x2_16;
				op_4x2_16;
			}
			store_4x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			op_1x2_16;
			op_1x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_first;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_2_first;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_first;
			op_4x4_2;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_4_first;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_first;
			op_4x4_4;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_2_first;
			op_4x4_4;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_2_first;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_first;
			op_4x4_2;
			op_4x4_4;
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_first;
			op_1x4_2;
			op_1x4_4;
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_4x4_4;
			}
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x4_4;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_4x4_8;
			}
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x4_8;
			}
			store_1x4;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_4x4_16;
			}
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_16_first;
			op_4x4_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_4x4_16;
				op_4x4_16;
			}
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const zq_base_type *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda << 1;
	int lda4 = lda << 2;
	int ldc2 = ldc << 1;
	int ldc4 = ldc << 2;
	int ldb4 = ldb << 2;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type b_vec1, b_vec2, b_vec3, b_vec4;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;
	//zq_base_type* packedB = (zq_base_type*)_aligned_malloc(N*ldb * sizeof(zq_base_type), 32);
	//int pack_size = N * ldb * sizeof(zq_base_type);
	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	for (m = 0; m < M - 3; m += 4, Aptr1 += lda4, Aptr2 += lda4, Aptr3 += lda4, Aptr4 += lda4,
		Cptr1 += ldc4, Cptr2 += ldc4, Cptr3 += ldc4, Cptr4 += ldc4)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			//memcpy(packedB, Bptr, pack_size);
			//Bptr1 = packedB;
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_4x4_16_first;
			op_4x4_16;
			op_4x4_16;
			op_4x4_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_4x4_16;
				op_4x4_16;
				op_4x4_16;
				op_4x4_16;
			}
			store_4x4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2; B_c_ptr3 = Bptr3; B_c_ptr4 = Bptr4;
			op_1x4_16_first;
			op_1x4_16;
			op_1x4_16;
			op_1x4_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
				op_1x4_16;
			}
			store_1x4;
		}
	}
	//_aligned_free(packedB);
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_first;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_2_first;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_first;
			op_8x1_2;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_4_first;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_first;
			op_8x1_4;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_2_first;
			op_8x1_4;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_first;
			op_8x1_2;
			op_8x1_4;
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_8x1_4;
			}
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x1_4;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_8x1_8;
			}
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x1_8;
			}
			store_1x1;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_8x1_16;
			}
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_16_first;
			op_8x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_8x1_16;
				op_8x1_16;
			}
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1;
			op_8x1_16_first;
			op_8x1_16;
			op_8x1_16;
			op_8x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
				op_8x1_16;
			}
			store_8x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			op_1x1_16;
			op_1x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_first;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_2_first;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_first;
			op_8x2_2;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_4_first;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_first;
			op_8x2_4;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_2_first;
			op_8x2_4;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_2_first;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_first;
			op_8x2_2;
			op_8x2_4;
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_first;
			op_1x2_2;
			op_1x2_4;
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8, 
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_8x2_4;
			}
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x2_4;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_8x2_8;
			}
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x2_8;
			}
			store_1x2;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_8x2_16;
			}
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_16_first;
			op_8x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_8x2_16;
				op_8x2_16;
			}
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type *Bptr, *Bptr1, *Bptr2;
	const zq_base_type *B_c_ptr1, *B_c_ptr2;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	int m, n, k;
	int lda8 = lda << 3;
	int ldc8 = ldc << 3;
	int ldb2 = ldb << 1;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41, sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42, sum_vec52, sum_vec62, sum_vec72, sum_vec82;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type b_vec1, b_vec2;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	for (m = 0; m < M - 7; m += 8,
		Aptr1 += lda8, Aptr2 += lda8, Aptr3 += lda8, Aptr4 += lda8,
		Aptr5 += lda8, Aptr6 += lda8, Aptr7 += lda8, Aptr8 += lda8,
		Cptr1 += ldc8, Cptr2 += ldc8, Cptr3 += ldc8, Cptr4 += ldc8,
		Cptr5 += ldc8, Cptr6 += ldc8, Cptr7 += ldc8, Cptr8 += ldc8)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_8x2_16_first;
			op_8x2_16;
			op_8x2_16;
			op_8x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_8x2_16;
				op_8x2_16;
				op_8x2_16;
				op_8x2_16;
			}
			store_8x2;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 2, Bptr += ldb2)
		{
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1; B_c_ptr2 = Bptr2;
			op_1x2_16_first;
			op_1x2_16;
			op_1x2_16;
			op_1x2_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
				op_1x2_16;
			}
			store_1x2;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_first;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_2_first;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign3(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_first;
			op_16x1_2;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_4_first;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign5(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_first;
			op_16x1_4;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign6(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_2_first;
			op_16x1_4;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_2_first;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign7(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_first;
			op_16x1_2;
			op_16x1_4;
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_first;
			op_1x1_2;
			op_1x1_4;
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16, 
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_16x1_4;
			}
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_4_first;
			for (k = K - zq_mm_align_size4; k; k -= zq_mm_align_size4)
			{
				op_1x1_4;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_16x1_8;
			}
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_8_first;
			for (k = K - zq_mm_align_size8; k; k -= zq_mm_align_size8)
			{
				op_1x1_8;
			}
			store_1x1;
		}
	}
}

#if !__ARM_NEON

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign16(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size16 = zq_mm_align_size << 4;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_16x1_16;
			}
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			for (k = K - zq_mm_align_size16; k; k -= zq_mm_align_size16)
			{
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign32(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size32 = zq_mm_align_size << 5;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_16_first;
			op_16x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_16x1_16;
				op_16x1_16;
			}
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			for (k = K - zq_mm_align_size32; k; k -= zq_mm_align_size32)
			{
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign64(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	const zq_base_type* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const zq_base_type* Aptr5, *Aptr6, *Aptr7, *Aptr8, *A_c_ptr5, *A_c_ptr6, *A_c_ptr7, *A_c_ptr8;
	const zq_base_type* Aptr9, *AptrA, *AptrB, *AptrC, *A_c_ptr9, *A_c_ptrA, *A_c_ptrB, *A_c_ptrC;
	const zq_base_type* AptrD, *AptrE, *AptrF, *AptrG, *A_c_ptrD, *A_c_ptrE, *A_c_ptrF, *A_c_ptrG;
	const zq_base_type *Bptr, *Bptr1;
	const zq_base_type *B_c_ptr1;
	zq_base_type* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	zq_base_type* Cptr5, *Cptr6, *Cptr7, *Cptr8, *C_c_ptr5, *C_c_ptr6, *C_c_ptr7, *C_c_ptr8;
	zq_base_type* Cptr9, *CptrA, *CptrB, *CptrC, *C_c_ptr9, *C_c_ptrA, *C_c_ptrB, *C_c_ptrC;
	zq_base_type* CptrD, *CptrE, *CptrF, *CptrG, *C_c_ptrD, *C_c_ptrE, *C_c_ptrF, *C_c_ptrG;
	int m, n, k;
	int lda16 = lda << 4;
	int ldc16 = ldc << 4;
	zq_q_type q;
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec51, sum_vec61, sum_vec71, sum_vec81;
	register zq_mm_type sum_vec91, sum_vecA1, sum_vecB1, sum_vecC1;
	register zq_mm_type sum_vecD1, sum_vecE1, sum_vecF1, sum_vecG1;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4;
	register zq_mm_type a_vec5, a_vec6, a_vec7, a_vec8;
	register zq_mm_type a_vec9, a_vecA, a_vecB, a_vecC;
	register zq_mm_type a_vecD, a_vecE, a_vecF, a_vecG;
	register zq_mm_type b_vec1;
	const int zq_mm_align_size64 = zq_mm_align_size << 6;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Aptr3 = Aptr2 + lda;
	Aptr4 = Aptr3 + lda;
	Aptr5 = Aptr4 + lda;
	Aptr6 = Aptr5 + lda;
	Aptr7 = Aptr6 + lda;
	Aptr8 = Aptr7 + lda;
	Aptr9 = Aptr8 + lda;
	AptrA = Aptr9 + lda;
	AptrB = AptrA + lda;
	AptrC = AptrB + lda;
	AptrD = AptrC + lda;
	AptrE = AptrD + lda;
	AptrF = AptrE + lda;
	AptrG = AptrF + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	Cptr3 = Cptr2 + ldc;
	Cptr4 = Cptr3 + ldc;
	Cptr5 = Cptr4 + ldc;
	Cptr6 = Cptr5 + ldc;
	Cptr7 = Cptr6 + ldc;
	Cptr8 = Cptr7 + ldc;
	Cptr9 = Cptr8 + ldc;
	CptrA = Cptr9 + ldc;
	CptrB = CptrA + ldc;
	CptrC = CptrB + ldc;
	CptrD = CptrC + ldc;
	CptrE = CptrD + ldc;
	CptrF = CptrE + ldc;
	CptrG = CptrF + ldc;
	for (m = 0; m < M - 15; m += 16,
		Aptr1 += lda16, Aptr2 += lda16, Aptr3 += lda16, Aptr4 += lda16,
		Aptr5 += lda16, Aptr6 += lda16, Aptr7 += lda16, Aptr8 += lda16,
		Aptr9 += lda16, AptrA += lda16, AptrB += lda16, AptrC += lda16,
		AptrD += lda16, AptrE += lda16, AptrF += lda16, AptrG += lda16,
		Cptr1 += ldc16, Cptr2 += ldc16, Cptr3 += ldc16, Cptr4 += ldc16,
		Cptr5 += ldc16, Cptr6 += ldc16, Cptr7 += ldc16, Cptr8 += ldc16,
		Cptr9 += ldc16, CptrA += ldc16, CptrB += ldc16, CptrC += ldc16,
		CptrD += ldc16, CptrE += ldc16, CptrF += ldc16, CptrG += ldc16)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		C_c_ptr3 = Cptr3;
		C_c_ptr4 = Cptr4;
		C_c_ptr5 = Cptr5;
		C_c_ptr6 = Cptr6;
		C_c_ptr7 = Cptr7;
		C_c_ptr8 = Cptr8;
		C_c_ptr9 = Cptr9;
		C_c_ptrA = CptrA;
		C_c_ptrB = CptrB;
		C_c_ptrC = CptrC;
		C_c_ptrD = CptrD;
		C_c_ptrE = CptrE;
		C_c_ptrF = CptrF;
		C_c_ptrG = CptrG;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1; A_c_ptr2 = Aptr2; A_c_ptr3 = Aptr3; A_c_ptr4 = Aptr4;
			A_c_ptr5 = Aptr5; A_c_ptr6 = Aptr6; A_c_ptr7 = Aptr7; A_c_ptr8 = Aptr8;
			A_c_ptr9 = Aptr9; A_c_ptrA = AptrA; A_c_ptrB = AptrB; A_c_ptrC = AptrC;
			A_c_ptrD = AptrD; A_c_ptrE = AptrE; A_c_ptrF = AptrF; A_c_ptrG = AptrG;
			B_c_ptr1 = Bptr1;
			op_16x1_16_first;
			op_16x1_16;
			op_16x1_16;
			op_16x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_16x1_16;
				op_16x1_16;
				op_16x1_16;
				op_16x1_16;
			}
			store_16x1;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n++, Bptr += ldb)
		{
			Bptr1 = Bptr;
			A_c_ptr1 = Aptr1;
			B_c_ptr1 = Bptr1;
			op_1x1_16_first;
			op_1x1_16;
			op_1x1_16;
			op_1x1_16;
			for (k = K - zq_mm_align_size64; k; k -= zq_mm_align_size64)
			{
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
				op_1x1_16;
			}
			store_1x1;
		}
	}
}

#endif

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign64(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 5) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign32(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 4) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign16(M, N, K, A, lda, Bt, ldb, C, ldc);
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size7)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign7(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size6)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign6(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size5)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign5(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size4)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size3)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign3(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size2)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign2(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KeqAlign1(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K %zq_mm_align_size4 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv1_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M1_N1_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 2;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv2_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if(restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_N2_Kgeneral(M, restN, K, A, lda, Bt+partN*ldb, ldb, C+partN, ldc);
	}
	else	
		zq_gemm_32f_align_AnoTrans_Btrans_M1_N2_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 4;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if(restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_N4_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M1_N4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_N8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 8;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_N8_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M1_N8_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign64(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 5) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign32(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 4) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign16(M, N, K, A, lda, Bt, ldb, C, ldc);
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size7)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign7(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size6)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign6(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size5)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign5(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size4)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size3)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign3(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size2)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign2(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KeqAlign1(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K %zq_mm_align_size4 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv1_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M2_N1_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 2;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv2_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_N2_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M2_N2_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 4;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_N4_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M2_N4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_N8(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 8;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_N8_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M2_N8_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign64(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 5) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign32(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 4) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign16(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
#endif
		if (K %zq_mm_align_size8 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size7)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign7(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size6)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign6(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size5)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign5(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size4)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size3)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign3(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size2)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign2(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KeqAlign1(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K %zq_mm_align_size4 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv1_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M4_N1_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 2;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv2_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M4_N2_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M4_N2_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_N4(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 4;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M4_N4_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M4_N4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_N1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign64(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 5) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign32(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 4) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign16(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
#endif
		if (K %zq_mm_align_size8 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size7)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign7(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size6)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign6(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size5)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign5(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size4)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size3)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign3(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size2)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign2(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KeqAlign1(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K %zq_mm_align_size4 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv1_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M8_N1_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M8_N2(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
	int restN = N % 2;
	int partN = N - restN;
	int handled = 0;
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign64(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 5) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign32(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K % (zq_mm_align_size << 4) == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign16(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else 
#endif
		if (K %zq_mm_align_size8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign8(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size7)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign7(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size6)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign6(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size5)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign5(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size4)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size3)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign3(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size2)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign2(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K == zq_mm_align_size)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KeqAlign1(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	else if (K %zq_mm_align_size4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M8_caseNdiv2_KdivAlign4(M, partN, K, A, lda, Bt, ldb, C, ldc);
		handled = 1;
	}
	

	if (handled)
	{
		if (restN != 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M8_N2_Kgeneral(M, restN, K, A, lda, Bt + partN * ldb, ldb, C + partN, ldc);
	}
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M8_N2_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}

void zq_gemm_32f_align_AnoTrans_Btrans_M16_N1(int M, int N, int K, const zq_base_type* A, int lda, const zq_base_type* Bt, int ldb, zq_base_type* C, int ldc)
{
#if !__ARM_NEON
	if (K % (zq_mm_align_size << 6) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign64(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 5) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign32(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K % (zq_mm_align_size << 4) == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign16(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
#endif
		if (K %zq_mm_align_size8 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size7)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign7(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size6)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign6(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size5)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign5(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size4)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size3)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign3(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size2)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign2(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K == zq_mm_align_size)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KeqAlign1(M, N, K, A, lda, Bt, ldb, C, ldc);
	else if (K %zq_mm_align_size4 == 0)
		zq_gemm_32f_align_AnoTrans_Btrans_M16_caseNdiv1_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
	else
		zq_gemm_32f_align_AnoTrans_Btrans_M16_N1_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
	return;
}


#undef op_1x1
#undef op_1x1_2
#undef op_1x1_4
#undef op_1x1_8
#undef op_1x1_16
#undef op_1x1_first
#undef op_1x1_2_first
#undef op_1x1_4_first
#undef op_1x1_8_first
#undef op_1x1_16_first
#undef op_1x2
#undef op_1x2_2
#undef op_1x2_4
#undef op_1x2_8
#undef op_1x2_16
#undef op_1x2_first
#undef op_1x2_2_first
#undef op_1x2_4_first
#undef op_1x2_8_first
#undef op_1x2_16_first
#undef op_1x4
#undef op_1x4_2
#undef op_1x4_4
#undef op_1x4_8
#undef op_1x4_16
#undef op_1x4_first
#undef op_1x4_2_first
#undef op_1x4_4_first
#undef op_1x4_8_first
#undef op_1x4_16_first
#undef op_1x8
#undef op_1x8_2
#undef op_1x8_4
#undef op_1x8_8
#undef op_1x8_16
#undef op_1x8_first
#undef op_1x8_2_first
#undef op_1x8_4_first
#undef op_1x8_8_first
#undef op_1x8_16_first
#undef op_2x1
#undef op_2x1_2
#undef op_2x1_4
#undef op_2x1_8
#undef op_2x1_16
#undef op_2x1_first
#undef op_2x1_2_first
#undef op_2x1_4_first
#undef op_2x1_8_first
#undef op_2x1_16_first
#undef op_2x2
#undef op_2x2_2
#undef op_2x2_4
#undef op_2x2_8
#undef op_2x2_16
#undef op_2x2_first
#undef op_2x2_2_first
#undef op_2x2_4_first
#undef op_2x2_8_first
#undef op_2x2_16_first
#undef op_2x4
#undef op_2x4_2
#undef op_2x4_4
#undef op_2x4_8
#undef op_2x4_16
#undef op_2x4_first
#undef op_2x4_2_first
#undef op_2x4_4_first
#undef op_2x4_8_first
#undef op_2x4_16_first
#undef op_2x8
#undef op_2x8_2
#undef op_2x8_4
#undef op_2x8_8
#undef op_2x8_16
#undef op_2x8_first
#undef op_2x8_2_first
#undef op_2x8_4_first
#undef op_2x8_8_first
#undef op_2x8_16_first
#undef op_4x1
#undef op_4x1_2
#undef op_4x1_4
#undef op_4x1_8
#undef op_4x1_16
#undef op_4x1_first
#undef op_4x1_2_first
#undef op_4x1_4_first
#undef op_4x1_8_first
#undef op_4x1_16_first
#undef op_4x2
#undef op_4x2_2
#undef op_4x2_4
#undef op_4x2_8
#undef op_4x2_16
#undef op_4x2_first
#undef op_4x2_2_first
#undef op_4x2_4_first
#undef op_4x2_8_first
#undef op_4x2_16_first
#undef op_4x4
#undef op_4x4_2
#undef op_4x4_4
#undef op_4x4_8
#undef op_4x4_16
#undef op_4x4_first
#undef op_4x4_2_first
#undef op_4x4_4_first
#undef op_4x4_8_first
#undef op_4x4_16_first
#undef op_8x1
#undef op_8x1_2
#undef op_8x1_4
#undef op_8x1_8
#undef op_8x1_16
#undef op_8x1_first
#undef op_8x1_2_first
#undef op_8x1_4_first
#undef op_8x1_8_first
#undef op_8x1_16_first
#undef op_8x2
#undef op_8x2_2
#undef op_8x2_4
#undef op_8x2_8
#undef op_8x2_16
#undef op_8x2_first
#undef op_8x2_2_first
#undef op_8x2_4_first
#undef op_8x2_8_first
#undef op_8x2_16_first
#undef op_16x1
#undef op_16x1_2
#undef op_16x1_4
#undef op_16x1_8
#undef op_16x1_16
#undef op_16x1_first
#undef op_16x1_2_first
#undef op_16x1_4_first
#undef op_16x1_8_first
#undef op_16x1_16_first
#undef store_1x1
#undef store_1x2
#undef store_1x4
#undef store_1x8
#undef store_2x1
#undef store_2x2
#undef store_2x4
#undef store_2x8
#undef store_4x1
#undef store_4x2
#undef store_4x4
#undef store_8x1
#undef store_8x2
#undef store_16x1