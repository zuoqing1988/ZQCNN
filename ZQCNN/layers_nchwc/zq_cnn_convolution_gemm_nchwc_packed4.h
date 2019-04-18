#define op6x4_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_mul_ps(a0, b0);\
c01 = zq_mm_mul_ps(a0, b1);\
c02 = zq_mm_mul_ps(a0, b2);\
c03 = zq_mm_mul_ps(a0, b3);\
c10 = zq_mm_mul_ps(a1, b0);\
c11 = zq_mm_mul_ps(a1, b1);\
c12 = zq_mm_mul_ps(a1, b2);\
c13 = zq_mm_mul_ps(a1, b3);\
c20 = zq_mm_mul_ps(a2, b0);\
c21 = zq_mm_mul_ps(a2, b1);\
c22 = zq_mm_mul_ps(a2, b2);\
c23 = zq_mm_mul_ps(a2, b3);\
c30 = zq_mm_mul_ps(a3, b0);\
c31 = zq_mm_mul_ps(a3, b1);\
c32 = zq_mm_mul_ps(a3, b2);\
c33 = zq_mm_mul_ps(a3, b3);\
c40 = zq_mm_mul_ps(a4, b0);\
c41 = zq_mm_mul_ps(a4, b1);\
c42 = zq_mm_mul_ps(a4, b2);\
c43 = zq_mm_mul_ps(a4, b3);\
c50 = zq_mm_mul_ps(a5, b0);\
c51 = zq_mm_mul_ps(a5, b1);\
c52 = zq_mm_mul_ps(a5, b2);\
c53 = zq_mm_mul_ps(a5, b3);\
src_ptr0 += zq_mm_align_size6;\
src_ptr1 += zq_mm_align_size4

#define op4x4_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_mul_ps(a0, b0);\
c01 = zq_mm_mul_ps(a0, b1);\
c02 = zq_mm_mul_ps(a0, b2);\
c03 = zq_mm_mul_ps(a0, b3);\
c10 = zq_mm_mul_ps(a1, b0);\
c11 = zq_mm_mul_ps(a1, b1);\
c12 = zq_mm_mul_ps(a1, b2);\
c13 = zq_mm_mul_ps(a1, b3);\
c20 = zq_mm_mul_ps(a2, b0);\
c21 = zq_mm_mul_ps(a2, b1);\
c22 = zq_mm_mul_ps(a2, b2);\
c23 = zq_mm_mul_ps(a2, b3);\
c30 = zq_mm_mul_ps(a3, b0);\
c31 = zq_mm_mul_ps(a3, b1);\
c32 = zq_mm_mul_ps(a3, b2);\
c33 = zq_mm_mul_ps(a3, b3);\
src_ptr0 += zq_mm_align_size4;\
src_ptr1 += zq_mm_align_size4


#define op1x4_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_mul_ps(a0, b0);\
c01 = zq_mm_mul_ps(a0, b1);\
c02 = zq_mm_mul_ps(a0, b2);\
c03 = zq_mm_mul_ps(a0, b3);\
src_ptr0 += zq_mm_align_size;\
src_ptr1 += zq_mm_align_size4

#define op6x4_1 \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_fmadd_ps(a0, b0, c00);\
c01 = zq_mm_fmadd_ps(a0, b1, c01);\
c02 = zq_mm_fmadd_ps(a0, b2, c02);\
c03 = zq_mm_fmadd_ps(a0, b3, c03);\
c10 = zq_mm_fmadd_ps(a1, b0, c10);\
c11 = zq_mm_fmadd_ps(a1, b1, c11);\
c12 = zq_mm_fmadd_ps(a1, b2, c12);\
c13 = zq_mm_fmadd_ps(a1, b3, c13);\
c20 = zq_mm_fmadd_ps(a2, b0, c20);\
c21 = zq_mm_fmadd_ps(a2, b1, c21);\
c22 = zq_mm_fmadd_ps(a2, b2, c22);\
c23 = zq_mm_fmadd_ps(a2, b3, c23);\
c30 = zq_mm_fmadd_ps(a3, b0, c30);\
c31 = zq_mm_fmadd_ps(a3, b1, c31);\
c32 = zq_mm_fmadd_ps(a3, b2, c32);\
c33 = zq_mm_fmadd_ps(a3, b3, c33);\
c40 = zq_mm_fmadd_ps(a4, b0, c40);\
c41 = zq_mm_fmadd_ps(a4, b1, c41);\
c42 = zq_mm_fmadd_ps(a4, b2, c42);\
c43 = zq_mm_fmadd_ps(a4, b3, c43);\
c50 = zq_mm_fmadd_ps(a5, b0, c50);\
c51 = zq_mm_fmadd_ps(a5, b1, c51);\
c52 = zq_mm_fmadd_ps(a5, b2, c52);\
c53 = zq_mm_fmadd_ps(a5, b3, c53);\
src_ptr0 += zq_mm_align_size6;\
src_ptr1 += zq_mm_align_size4

#define op4x4_1 \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_fmadd_ps(a0, b0, c00);\
c01 = zq_mm_fmadd_ps(a0, b1, c01);\
c02 = zq_mm_fmadd_ps(a0, b2, c02);\
c03 = zq_mm_fmadd_ps(a0, b3, c03);\
c10 = zq_mm_fmadd_ps(a1, b0, c10);\
c11 = zq_mm_fmadd_ps(a1, b1, c11);\
c12 = zq_mm_fmadd_ps(a1, b2, c12);\
c13 = zq_mm_fmadd_ps(a1, b3, c13);\
c20 = zq_mm_fmadd_ps(a2, b0, c20);\
c21 = zq_mm_fmadd_ps(a2, b1, c21);\
c22 = zq_mm_fmadd_ps(a2, b2, c22);\
c23 = zq_mm_fmadd_ps(a2, b3, c23);\
c30 = zq_mm_fmadd_ps(a3, b0, c30);\
c31 = zq_mm_fmadd_ps(a3, b1, c31);\
c32 = zq_mm_fmadd_ps(a3, b2, c32);\
c33 = zq_mm_fmadd_ps(a3, b3, c33);\
src_ptr0 += zq_mm_align_size4;\
src_ptr1 += zq_mm_align_size4

#define op1x4_1 \
a0 = zq_mm_load_ps(src_ptr0);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
b2 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size2);\
b3 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size3);\
c00 = zq_mm_fmadd_ps(a0, b0, c00);\
c01 = zq_mm_fmadd_ps(a0, b1, c01);\
c02 = zq_mm_fmadd_ps(a0, b2, c02);\
c03 = zq_mm_fmadd_ps(a0, b3, c03);\
src_ptr0 += zq_mm_align_size;\
src_ptr1 += zq_mm_align_size4

#define op6x4_2_first \
op6x4_1_first;\
op6x4_1

#define op6x4_2 \
op6x4_1;\
op6x4_1

#define op6x4_2 \
op6x4_1;\
op6x4_1

#define op6x4_4_first \
op6x4_2_first;\
op6x4_2

#define op6x4_4 \
op6x4_2;\
op6x4_2

#define op6x4_8_first \
op6x4_4_first;\
op6x4_4

#define op6x4_8 \
op6x4_4;\
op6x4_4

#define op6x4_16_first \
op6x4_8_first;\
op6x4_8

#define op6x4_16 \
op6x4_8;\
op6x4_8

#define op6x4_32_first \
op6x4_16_first;\
op6x4_16

#define op6x4_32 \
op6x4_16;\
op6x4_16

#define op6x4_64_first \
op6x4_32_first;\
op6x4_32

#define op6x4_64 \
op6x4_32;\
op6x4_32

#define op4x4_2_first \
op4x4_1_first;\
op4x4_1

#define op4x4_2 \
op4x4_1;\
op4x4_1

#define op4x4_2 \
op4x4_1;\
op4x4_1

#define op4x4_4_first \
op4x4_2_first;\
op4x4_2

#define op4x4_4 \
op4x4_2;\
op4x4_2

#define op4x4_8_first \
op4x4_4_first;\
op4x4_4

#define op4x4_8 \
op4x4_4;\
op4x4_4

#define op4x4_16_first \
op4x4_8_first;\
op4x4_8

#define op4x4_16 \
op4x4_8;\
op4x4_8

#define op4x4_32_first \
op4x4_16_first;\
op4x4_16

#define op4x4_32 \
op4x4_16;\
op4x4_16

#define op4x4_64_first \
op4x4_32_first;\
op4x4_32

#define op4x4_64 \
op4x4_32;\
op4x4_32

#define op1x4_2_first \
op1x4_1_first;\
op1x4_1

#define op1x4_2 \
op1x4_1;\
op1x4_1

#define op1x4_4_first \
op1x4_2_first;\
op1x4_2

#define op1x4_4 \
op1x4_2;\
op1x4_2

#define op1x4_8_first \
op1x4_4_first;\
op1x4_4

#define op1x4_8 \
op1x4_4;\
op1x4_4

#define op1x4_16_first \
op1x4_8_first;\
op1x4_8

#define op1x4_16 \
op1x4_8;\
op1x4_8

#define op1x4_32_first \
op1x4_16_first;\
op1x4_16

#define op1x4_32 \
op1x4_16;\
op1x4_16

#define op1x4_64_first \
op1x4_32_first;\
op1x4_32

#define op1x4_64 \
op1x4_32;\
op1x4_32

#if __ARM_NEON && __ARM_NEON_ARMV8

#if WITH_BIAS
#define op8x8_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c11 = vfmaq_laneq_f32(bias_v1, b1, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c21 = vfmaq_laneq_f32(bias_v1, b1, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c31 = vfmaq_laneq_f32(bias_v1, b1, a3, 0);\
c40 = vfmaq_laneq_f32(bias_v0, b0, a4, 0);\
c41 = vfmaq_laneq_f32(bias_v1, b1, a4, 0);\
c50 = vfmaq_laneq_f32(bias_v0, b0, a5, 0);\
c51 = vfmaq_laneq_f32(bias_v1, b1, a5, 0);\
c60 = vfmaq_laneq_f32(bias_v0, b0, a6, 0);\
c61 = vfmaq_laneq_f32(bias_v1, b1, a6, 0);\
c70 = vfmaq_laneq_f32(bias_v0, b0, a7, 0);\
c71 = vfmaq_laneq_f32(bias_v1, b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 3);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 3);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 3);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 3);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 3);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 3);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 3);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 3)

#define op8x8_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c11 = vfmaq_laneq_f32(bias_v1, b1, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c21 = vfmaq_laneq_f32(bias_v1, b1, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c31 = vfmaq_laneq_f32(bias_v1, b1, a3, 0);\
c40 = vfmaq_laneq_f32(bias_v0, b0, a4, 0);\
c41 = vfmaq_laneq_f32(bias_v1, b1, a4, 0);\
c50 = vfmaq_laneq_f32(bias_v0, b0, a5, 0);\
c51 = vfmaq_laneq_f32(bias_v1, b1, a5, 0);\
c60 = vfmaq_laneq_f32(bias_v0, b0, a6, 0);\
c61 = vfmaq_laneq_f32(bias_v1, b1, a6, 0);\
c70 = vfmaq_laneq_f32(bias_v0, b0, a7, 0);\
c71 = vfmaq_laneq_f32(bias_v1, b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2)

#define op8x4_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c40 = vfmaq_laneq_f32(bias_v0, b0, a4, 0);\
c50 = vfmaq_laneq_f32(bias_v0, b0, a5, 0);\
c60 = vfmaq_laneq_f32(bias_v0, b0, a6, 0);\
c70 = vfmaq_laneq_f32(bias_v0, b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 3);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 3);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 3);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 3)

#define op8x4_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c40 = vfmaq_laneq_f32(bias_v0, b0, a4, 0);\
c50 = vfmaq_laneq_f32(bias_v0, b0, a5, 0);\
c60 = vfmaq_laneq_f32(bias_v0, b0, a6, 0);\
c70 = vfmaq_laneq_f32(bias_v0, b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2)

#define op4x8_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c11 = vfmaq_laneq_f32(bias_v1, b1, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c21 = vfmaq_laneq_f32(bias_v1, b1, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c31 = vfmaq_laneq_f32(bias_v1, b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3)

#define op4x8_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c11 = vfmaq_laneq_f32(bias_v1, b1, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c21 = vfmaq_laneq_f32(bias_v1, b1, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
c31 = vfmaq_laneq_f32(bias_v1, b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2)

#define op4x4_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3)

#define op4x4_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
bias_v0 = zq_mm_load_ps(bias + j * 8);\
c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);\
c10 = vfmaq_laneq_f32(bias_v0, b0, a1, 0);\
c20 = vfmaq_laneq_f32(bias_v0, b0, a2, 0);\
c30 = vfmaq_laneq_f32(bias_v0, b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2)

#else

#define op8x8_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c01 = vmulq_laneq_f32(b1, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c11 = vmulq_laneq_f32(b1, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c21 = vmulq_laneq_f32(b1, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c31 = vmulq_laneq_f32(b1, a3, 0);\
c40 = vmulq_laneq_f32(b0, a4, 0);\
c41 = vmulq_laneq_f32(b1, a4, 0);\
c50 = vmulq_laneq_f32(b0, a5, 0);\
c51 = vmulq_laneq_f32(b1, a5, 0);\
c60 = vmulq_laneq_f32(b0, a6, 0);\
c61 = vmulq_laneq_f32(b1, a6, 0);\
c70 = vmulq_laneq_f32(b0, a7, 0);\
c71 = vmulq_laneq_f32(b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 3);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 3);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 3);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 3);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 3);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 3);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 3);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 3)

#define op8x8_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c01 = vmulq_laneq_f32(b1, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c11 = vmulq_laneq_f32(b1, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c21 = vmulq_laneq_f32(b1, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c31 = vmulq_laneq_f32(b1, a3, 0);\
c40 = vmulq_laneq_f32(b0, a4, 0);\
c41 = vmulq_laneq_f32(b1, a4, 0);\
c50 = vmulq_laneq_f32(b0, a5, 0);\
c51 = vmulq_laneq_f32(b1, a5, 0);\
c60 = vmulq_laneq_f32(b0, a6, 0);\
c61 = vmulq_laneq_f32(b1, a6, 0);\
c70 = vmulq_laneq_f32(b0, a7, 0);\
c71 = vmulq_laneq_f32(b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2)

#define op8x4_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c40 = vmulq_laneq_f32(b0, a4, 0);\
c50 = vmulq_laneq_f32(b0, a5, 0);\
c60 = vmulq_laneq_f32(b0, a6, 0);\
c70 = vmulq_laneq_f32(b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c30, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c30, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c30, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c30, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c30, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c30, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c30, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c30, b0, a7, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c40 = vfmaq_laneq_f32(c30, b0, a4, 3);\
c50 = vfmaq_laneq_f32(c30, b0, a5, 3);\
c60 = vfmaq_laneq_f32(c30, b0, a6, 3);\
c70 = vfmaq_laneq_f32(c30, b0, a7, 3)

#define op8x4_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c40 = vmulq_laneq_f32(b0, a4, 0);\
c50 = vmulq_laneq_f32(b0, a5, 0);\
c60 = vmulq_laneq_f32(b0, a6, 0);\
c70 = vmulq_laneq_f32(b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2)

#define op4x8_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c01 = vmulq_laneq_f32(b1, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c11 = vmulq_laneq_f32(b1, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c21 = vmulq_laneq_f32(b1, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c31 = vmulq_laneq_f32(b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3)

#define op4x8_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c01 = vmulq_laneq_f32(b1, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c11 = vmulq_laneq_f32(b1, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c21 = vmulq_laneq_f32(b1, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
c31 = vmulq_laneq_f32(b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2)

#define op4x4_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3)

#define op4x4_C3_other_1_first \
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vmulq_laneq_f32(b0, a0, 0);\
c10 = vmulq_laneq_f32(b0, a1, 0);\
c20 = vmulq_laneq_f32(b0, a2, 0);\
c30 = vmulq_laneq_f32(b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2)


#endif

#define op8x8_other_1 \
src_ptr0 += zq_mm_align_size8;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 0);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 0);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 0);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 0);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 0);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 0);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 0);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 0);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 3);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 3);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 3);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 3);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 3);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 3);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 3);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 3)

#define op8x8_C3_other_1 \
src_ptr0 += zq_mm_align_size8;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 0);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 0);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 0);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 0);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 0);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 0);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 0);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 0);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c41 = vfmaq_laneq_f32(c41, b1, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c51 = vfmaq_laneq_f32(c51, b1, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c61 = vfmaq_laneq_f32(c61, b1, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
c71 = vfmaq_laneq_f32(c71, b1, a7, 2)

#define op8x4_other_1 \
src_ptr0 += zq_mm_align_size8;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 0);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 0);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 0);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 3);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 3);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 3);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 3)

#define op8x4_C3_other_1 \
src_ptr0 += zq_mm_align_size8;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
a4 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size4);\
a5 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size5);\
a6 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size6);\
a7 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size7);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 0);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 0);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 0);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 1);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 1);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 1);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c40 = vfmaq_laneq_f32(c40, b0, a4, 2);\
c50 = vfmaq_laneq_f32(c50, b0, a5, 2);\
c60 = vfmaq_laneq_f32(c60, b0, a6, 2);\
c70 = vfmaq_laneq_f32(c70, b0, a7, 2)

#define op4x8_other_1 \
src_ptr0 += zq_mm_align_size4;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 3)

#define op4x8_C3_other_1 \
src_ptr0 += zq_mm_align_size4;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 0);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 1);\
src_ptr1 += zq_mm_align_size2;\
b0 = zq_mm_load_ps(src_ptr1);\
b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c01 = vfmaq_laneq_f32(c01, b1, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c11 = vfmaq_laneq_f32(c11, b1, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c21 = vfmaq_laneq_f32(c21, b1, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
c31 = vfmaq_laneq_f32(c31, b1, a3, 2)

#define op4x4_other_1 \
src_ptr0 += zq_mm_align_size4;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 3);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 3);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 3);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 3)

#define op4x4_C3_other_1 \
src_ptr0 += zq_mm_align_size4;\
a0 = zq_mm_load_ps(src_ptr0);\
a1 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size);\
a2 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size2);\
a3 = zq_mm_load_ps(src_ptr0 + zq_mm_align_size3);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 0);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 0);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 0);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 0);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 1);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 1);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 1);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 1);\
src_ptr1 += zq_mm_align_size;\
b0 = zq_mm_load_ps(src_ptr1);\
c00 = vfmaq_laneq_f32(c00, b0, a0, 2);\
c10 = vfmaq_laneq_f32(c10, b0, a1, 2);\
c20 = vfmaq_laneq_f32(c20, b0, a2, 2);\
c30 = vfmaq_laneq_f32(c30, b0, a3, 2)

#define op8x8_other_2_first \
op8x8_other_1_first;\
op8x8_other_1

#define op8x8_other_2 \
op8x8_other_1;\
op8x8_other_1

#define op8x8_other_4_first \
op8x8_other_2_first;\
op8x8_other_2

#define op8x8_other_4 \
op8x8_other_2;\
op8x8_other_2

#define op8x8_other_8_first \
op8x8_other_4_first;\
op8x8_other_4

#define op8x8_other_8 \
op8x8_other_4;\
op8x8_other_4

#define op8x8_other_16_first \
op8x8_other_8_first;\
op8x8_other_8

#define op8x8_other_16 \
op8x8_other_8;\
op8x8_other_8

#define op8x8_other_32_first \
op8x8_other_16_first;\
op8x8_other_16

#define op8x8_other_32 \
op8x8_other_16;\
op8x8_other_16

#define op8x8_other_64_first \
op8x8_other_32_first;\
op8x8_other_32

#define op8x8_other_64 \
op8x8_other_32;\
op8x8_other_32

#define op8x4_other_2_first \
op8x4_other_1_first;\
op8x4_other_1

#define op8x4_other_2 \
op8x4_other_1;\
op8x4_other_1

#define op8x4_other_4_first \
op8x4_other_2_first;\
op8x4_other_2

#define op8x4_other_4 \
op8x4_other_2;\
op8x4_other_2

#define op8x4_other_8_first \
op8x4_other_4_first;\
op8x4_other_4

#define op8x4_other_8 \
op8x4_other_4;\
op8x4_other_4

#define op8x4_other_16_first \
op8x4_other_8_first;\
op8x4_other_8

#define op8x4_other_16 \
op8x4_other_8;\
op8x4_other_8

#define op8x4_other_32_first \
op8x4_other_16_first;\
op8x4_other_16

#define op8x4_other_32 \
op8x4_other_16;\
op8x4_other_16

#define op8x4_other_64_first \
op8x4_other_32_first;\
op8x4_other_32

#define op8x4_other_64 \
op8x4_other_32;\
op8x4_other_32

#define op8x8_C3_other_2_first \
op8x8_C3_other_1_first;\
op8x8_C3_other_1

#define op8x8_C3_other_2 \
op8x8_C3_other_1;\
op8x8_C3_other_1

#define op8x8_C3_other_4_first \
op8x8_C3_other_2_first;\
op8x8_C3_other_2

#define op8x8_C3_other_4 \
op8x8_C3_other_2;\
op8x8_C3_other_2

#define op8x8_C3_other_8_first \
op8x8_C3_other_4_first;\
op8x8_C3_other_4

#define op8x8_C3_other_8 \
op8x8_C3_other_4;\
op8x8_C3_other_4

#define op8x8_C3_other_16_first \
op8x8_C3_other_8_first;\
op8x8_C3_other_8

#define op8x8_C3_other_16 \
op8x8_C3_other_8;\
op8x8_C3_other_8

#define op8x8_C3_other_32_first \
op8x8_C3_other_16_first;\
op8x8_C3_other_16

#define op8x8_C3_other_32 \
op8x8_C3_other_16;\
op8x8_C3_other_16

#define op8x8_C3_other_64_first \
op8x8_C3_other_32_first;\
op8x8_C3_other_32

#define op8x8_C3_other_64 \
op8x8_C3_other_32;\
op8x8_C3_other_32

#define op8x4_C3_other_2_first \
op8x4_C3_other_1_first;\
op8x4_C3_other_1

#define op8x4_C3_other_2 \
op8x4_C3_other_1;\
op8x4_C3_other_1

#define op8x4_C3_other_4_first \
op8x4_C3_other_2_first;\
op8x4_C3_other_2

#define op8x4_C3_other_4 \
op8x4_C3_other_2;\
op8x4_C3_other_2

#define op8x4_C3_other_8_first \
op8x4_C3_other_4_first;\
op8x4_C3_other_4

#define op8x4_C3_other_8 \
op8x4_C3_other_4;\
op8x4_C3_other_4

#define op8x4_C3_other_16_first \
op8x4_C3_other_8_first;\
op8x4_C3_other_8

#define op8x4_C3_other_16 \
op8x4_C3_other_8;\
op8x4_C3_other_8

#define op8x4_C3_other_32_first \
op8x4_C3_other_16_first;\
op8x4_C3_other_16

#define op8x4_C3_other_32 \
op8x4_C3_other_16;\
op8x4_C3_other_16

#define op8x4_C3_other_64_first \
op8x4_C3_other_32_first;\
op8x4_C3_other_32

#define op8x4_C3_other_64 \
op8x4_C3_other_32;\
op8x4_C3_other_32

#define op4x8_other_2_first \
op4x8_other_1_first;\
op4x8_other_1

#define op4x8_other_2 \
op4x8_other_1;\
op4x8_other_1

#define op4x8_other_4_first \
op4x8_other_2_first;\
op4x8_other_2

#define op4x8_other_4 \
op4x8_other_2;\
op4x8_other_2

#define op4x8_other_8_first \
op4x8_other_4_first;\
op4x8_other_4

#define op4x8_other_8 \
op4x8_other_4;\
op4x8_other_4

#define op4x8_other_16_first \
op4x8_other_8_first;\
op4x8_other_8

#define op4x8_other_16 \
op4x8_other_8;\
op4x8_other_8

#define op4x8_other_32_first \
op4x8_other_16_first;\
op4x8_other_16

#define op4x8_other_32 \
op4x8_other_16;\
op4x8_other_16

#define op4x8_other_64_first \
op4x8_other_32_first;\
op4x8_other_32

#define op4x8_other_64 \
op4x8_other_32;\
op4x8_other_32

#define op4x4_other_2_first \
op4x4_other_1_first;\
op4x4_other_1

#define op4x4_other_2 \
op4x4_other_1;\
op4x4_other_1

#define op4x4_other_4_first \
op4x4_other_2_first;\
op4x4_other_2

#define op4x4_other_4 \
op4x4_other_2;\
op4x4_other_2

#define op4x4_other_8_first \
op4x4_other_4_first;\
op4x4_other_4

#define op4x4_other_8 \
op4x4_other_4;\
op4x4_other_4

#define op4x4_other_16_first \
op4x4_other_8_first;\
op4x4_other_8

#define op4x4_other_16 \
op4x4_other_8;\
op4x4_other_8

#define op4x4_other_32_first \
op4x4_other_16_first;\
op4x4_other_16

#define op4x4_other_32 \
op4x4_other_16;\
op4x4_other_16

#define op4x4_other_64_first \
op4x4_other_32_first;\
op4x4_other_32

#define op4x4_other_64 \
op4x4_other_32;\
op4x4_other_32

#define op4x8_C3_other_2_first \
op4x8_C3_other_1_first;\
op4x8_C3_other_1

#define op4x8_C3_other_2 \
op4x8_C3_other_1;\
op4x8_C3_other_1

#define op4x8_C3_other_4_first \
op4x8_C3_other_2_first;\
op4x8_C3_other_2

#define op4x8_C3_other_4 \
op4x8_C3_other_2;\
op4x8_C3_other_2

#define op4x8_C3_other_8_first \
op4x8_C3_other_4_first;\
op4x8_C3_other_4

#define op4x8_C3_other_8 \
op4x8_C3_other_4;\
op4x8_C3_other_4

#define op4x8_C3_other_16_first \
op4x8_C3_other_8_first;\
op4x8_C3_other_8

#define op4x8_C3_other_16 \
op4x8_C3_other_8;\
op4x8_C3_other_8

#define op4x8_C3_other_32_first \
op4x8_C3_other_16_first;\
op4x8_C3_other_16

#define op4x8_C3_other_32 \
op4x8_C3_other_16;\
op4x8_C3_other_16

#define op4x8_C3_other_64_first \
op4x8_C3_other_32_first;\
op4x8_C3_other_32

#define op4x8_C3_other_64 \
op4x8_C3_other_32;\
op4x8_C3_other_32

#define op4x4_C3_other_2_first \
op4x4_C3_other_1_first;\
op4x4_C3_other_1

#define op4x4_C3_other_2 \
op4x4_C3_other_1;\
op4x4_C3_other_1

#define op4x4_C3_other_4_first \
op4x4_C3_other_2_first;\
op4x4_C3_other_2

#define op4x4_C3_other_4 \
op4x4_C3_other_2;\
op4x4_C3_other_2

#define op4x4_C3_other_8_first \
op4x4_C3_other_4_first;\
op4x4_C3_other_4

#define op4x4_C3_other_8 \
op4x4_C3_other_4;\
op4x4_C3_other_4

#define op4x4_C3_other_16_first \
op4x4_C3_other_8_first;\
op4x4_C3_other_8

#define op4x4_C3_other_16 \
op4x4_C3_other_8;\
op4x4_C3_other_8

#define op4x4_C3_other_32_first \
op4x4_C3_other_16_first;\
op4x4_C3_other_16

#define op4x4_C3_other_32 \
op4x4_C3_other_16;\
op4x4_C3_other_16

#define op4x4_C3_other_64_first \
op4x4_C3_other_32_first;\
op4x4_C3_other_32

#define op4x4_C3_other_64 \
op4x4_C3_other_32;\
op4x4_C3_other_32

#define store6x4 \
dst_ptr0[0] = vaddvq_f32(c00);\
dst_ptr0[1] = vaddvq_f32(c01);\
dst_ptr0[2] = vaddvq_f32(c02);\
dst_ptr0[3] = vaddvq_f32(c03);\
dst_ptr1[0] = vaddvq_f32(c10);\
dst_ptr1[1] = vaddvq_f32(c11);\
dst_ptr1[2] = vaddvq_f32(c12);\
dst_ptr1[3] = vaddvq_f32(c13);\
dst_ptr2[0] = vaddvq_f32(c20);\
dst_ptr2[1] = vaddvq_f32(c21);\
dst_ptr2[2] = vaddvq_f32(c22);\
dst_ptr2[3] = vaddvq_f32(c23);\
dst_ptr3[0] = vaddvq_f32(c30);\
dst_ptr3[1] = vaddvq_f32(c31);\
dst_ptr3[2] = vaddvq_f32(c32);\
dst_ptr3[3] = vaddvq_f32(c33);\
dst_ptr4[0] = vaddvq_f32(c40);\
dst_ptr4[1] = vaddvq_f32(c41);\
dst_ptr4[2] = vaddvq_f32(c42);\
dst_ptr4[3] = vaddvq_f32(c43);\
dst_ptr5[0] = vaddvq_f32(c50);\
dst_ptr5[1] = vaddvq_f32(c51);\
dst_ptr5[2] = vaddvq_f32(c52);\
dst_ptr5[3] = vaddvq_f32(c53)

#define store4x4 \
dst_ptr0[0] = vaddvq_f32(c00);\
dst_ptr0[1] = vaddvq_f32(c01);\
dst_ptr0[2] = vaddvq_f32(c02);\
dst_ptr0[3] = vaddvq_f32(c03);\
dst_ptr1[0] = vaddvq_f32(c10);\
dst_ptr1[1] = vaddvq_f32(c11);\
dst_ptr1[2] = vaddvq_f32(c12);\
dst_ptr1[3] = vaddvq_f32(c13);\
dst_ptr2[0] = vaddvq_f32(c20);\
dst_ptr2[1] = vaddvq_f32(c21);\
dst_ptr2[2] = vaddvq_f32(c22);\
dst_ptr2[3] = vaddvq_f32(c23);\
dst_ptr3[0] = vaddvq_f32(c30);\
dst_ptr3[1] = vaddvq_f32(c31);\
dst_ptr3[2] = vaddvq_f32(c32);\
dst_ptr3[3] = vaddvq_f32(c33)

#define store1x4 \
dst_ptr0[0] = vaddvq_f32(c00);\
dst_ptr0[1] = vaddvq_f32(c01);\
dst_ptr0[2] = vaddvq_f32(c02);\
dst_ptr0[3] = vaddvq_f32(c03)

#else 

#define store6x4 \
zq_mm_store_ps(q, c00);\
dst_ptr0[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c01);\
dst_ptr0[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c02);\
dst_ptr0[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c03);\
dst_ptr0[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c10);\
dst_ptr1[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c11);\
dst_ptr1[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c12);\
dst_ptr1[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c13);\
dst_ptr1[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c20);\
dst_ptr2[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c21);\
dst_ptr2[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c22);\
dst_ptr2[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c23);\
dst_ptr2[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c30);\
dst_ptr3[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c31);\
dst_ptr3[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c32);\
dst_ptr3[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c33);\
dst_ptr3[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c40);\
dst_ptr4[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c41);\
dst_ptr4[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c42);\
dst_ptr4[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c43);\
dst_ptr4[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c50);\
dst_ptr5[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c51);\
dst_ptr5[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c52);\
dst_ptr5[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c53);\
dst_ptr5[3] = zq_final_sum_q

#define store4x4 \
zq_mm_store_ps(q, c00);\
dst_ptr0[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c01);\
dst_ptr0[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c02);\
dst_ptr0[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c03);\
dst_ptr0[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c10);\
dst_ptr1[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c11);\
dst_ptr1[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c12);\
dst_ptr1[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c13);\
dst_ptr1[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c20);\
dst_ptr2[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c21);\
dst_ptr2[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c22);\
dst_ptr2[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c23);\
dst_ptr2[3] = zq_final_sum_q;\
zq_mm_store_ps(q, c30);\
dst_ptr3[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c31);\
dst_ptr3[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c32);\
dst_ptr3[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c33);\
dst_ptr3[3] = zq_final_sum_q

#define store1x4 \
zq_mm_store_ps(q, c00);\
dst_ptr0[0] = zq_final_sum_q;\
zq_mm_store_ps(q, c01);\
dst_ptr0[1] = zq_final_sum_q;\
zq_mm_store_ps(q, c02);\
dst_ptr0[2] = zq_final_sum_q;\
zq_mm_store_ps(q, c03);\
dst_ptr0[3] = zq_final_sum_q

#endif



/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM6N4_kernel1x1(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = in_H*in_W;
	int NHW = in_N*HW;
	int div6_size = NHW /6;
	int div4_size = (NHW - (div6_size * 6)) >> 2;
	int paddedC = (in_C + 3) >> 2 << 2;
	int packed_A_num = div6_size + div4_size + (NHW - (div6_size *6) - (div4_size << 2));
	int packed_A_step = paddedC * 6;
	int packed_B_step = paddedC * 4;
	int packed_B_num = (out_C + 3) >> 2;
	int i,ii,n,h,w,c,out_c;
	zq_base_type* A_buffer,*dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5;
	register zq_mm_type a0, a1, a2, a3, a4, a5;
	register zq_mm_type b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03, c04, c05;
	register zq_mm_type c10, c11, c12, c13, c14, c15;
	register zq_mm_type c20, c21, c22, c23;
	register zq_mm_type c30, c31, c32, c33;
	register zq_mm_type c40, c41, c42, c43;
	register zq_mm_type c50, c51, c52, c53;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[16];
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num*sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if(*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size,32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i=0; i < div6_size; i ++)
	{
		ii = i * 6;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr4 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr5 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size4, zq_mm_load_ps(src_ptr4));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size5, zq_mm_load_ps(src_ptr5));
			dst_ptr += zq_mm_align_size6;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
			src_ptr4 += in_sliceStep;
			src_ptr5 += in_sliceStep;
		}
	}

	for (i = 0;i < div4_size;i++)
	{
		ii = div6_size * 6 + i*4;
		dst_ptr = A_buffer + packed_A_step*(i+div6_size);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
		}
	}

	for (i=0; i < NHW-div6_size*6-div4_size*4; i++)
	{
		ii = div6_size*6+div4_size * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i+div6_size+div4_size);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			dst_ptr += zq_mm_align_size;
			src_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < div6_size; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*i;
		ii = i * 6;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr4 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr5 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		/*if (paddedC % zq_mm_align_size64 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);

				op6x4_64_first;
				c = zq_mm_align_size64;
				for (; c < paddedC; c += zq_mm_align_size64)
				{
					op6x4_64;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size32 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);

				op6x4_32_first;
				c = zq_mm_align_size32;
				for (; c < paddedC; c += zq_mm_align_size32)
				{
					op6x4_32;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else*/ if (paddedC % zq_mm_align_size16 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				
				op6x4_16_first;
				c = zq_mm_align_size16;
				for (; c < paddedC; c += zq_mm_align_size16)
				{
					op6x4_16;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (out_c = 0;out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op6x4_8_first;
				c = zq_mm_align_size8;
				for (; c < paddedC; c += zq_mm_align_size8)
				{
					op6x4_8;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op6x4_4_first;
				c = zq_mm_align_size4;
				for (; c < paddedC; c += zq_mm_align_size4)
				{
					op6x4_4;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (out_c = 0;out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op6x4_2_first;
				c = zq_mm_align_size2;
				for (; c < paddedC; c += zq_mm_align_size2)
				{
					op6x4_2;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			for (out_c = 0;out_c < out_C; out_c += zq_mm_align_size)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
				op6x4_1_first;
				c = zq_mm_align_size;
				for (; c < paddedC; c += zq_mm_align_size)
				{
					op6x4_1;
				}
				store6x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_6x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
			}
		}
	}

	for (i = 0; i < div4_size; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i+div6_size);
		ii = (i << 2) + (div6_size *6);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		/*if (paddedC % zq_mm_align_size64 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_64_first;
				c = zq_mm_align_size64;
				for (; c < paddedC; c += zq_mm_align_size64)
				{
					op4x4_64;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size32 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_32_first;
				c = zq_mm_align_size32;
				for (; c < paddedC; c += zq_mm_align_size32)
				{
					op4x4_32;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else*/ if (paddedC % zq_mm_align_size16 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_16_first;
				c = zq_mm_align_size16;
				for (; c < paddedC; c += zq_mm_align_size16)
				{
					op4x4_16;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_8_first;
				c = zq_mm_align_size8;
				for (; c < paddedC; c += zq_mm_align_size8)
				{
					op4x4_8;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_4_first;
				c = zq_mm_align_size4;
				for (; c < paddedC; c += zq_mm_align_size4)
				{
					op4x4_4;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_2_first;
				c = zq_mm_align_size2;
				for (; c < paddedC; c += zq_mm_align_size2)
				{
					op4x4_2;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_1_first;
				c = zq_mm_align_size;
				for (; c < paddedC; c += zq_mm_align_size)
				{
					op4x4_1;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
	}

	//rest 
	for (i = 0; i < NHW-(div6_size*6)-(div4_size<<2); i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + div4_size + div6_size);
		ii = (div6_size * 6) + (div4_size << 2) + i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		/*if (paddedC % zq_mm_align_size64 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_64_first;
				c = zq_mm_align_size64;
				for (; c < paddedC; c += zq_mm_align_size64)
				{
					op1x4_64;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size32 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_32_first;
				c = zq_mm_align_size32;
				for (; c < paddedC; c += zq_mm_align_size32)
				{
					op1x4_32;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else*/ if (paddedC % zq_mm_align_size16 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_16_first;
				c = zq_mm_align_size16;
				for (; c < paddedC; c += zq_mm_align_size16)
				{
					op1x4_16;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_8_first;
				c = zq_mm_align_size8;
				for (; c < paddedC; c += zq_mm_align_size8)
				{
					op1x4_8;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_4_first;
				c = zq_mm_align_size4;
				for (; c < paddedC; c += zq_mm_align_size4)
				{
					op1x4_4;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_2_first;
				c = zq_mm_align_size2;
				for (; c < paddedC; c += zq_mm_align_size2)
				{
					op1x4_2;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_1_first;
				c = zq_mm_align_size;
				for (; c < paddedC; c += zq_mm_align_size)
				{
					op1x4_1;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
	}
}

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM4N4_kernel1x1(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = in_H*in_W;
	int NHW = in_N*HW;
	int div4_size = NHW >> 2;
	int paddedC = (in_C + 3) >> 2 << 2;
	int packed_A_num = div4_size + (NHW - (div4_size << 2));
	int packed_A_step = paddedC * 4;
	int packed_B_step = paddedC * 4;
	int packed_B_num = (out_C + 3) >> 2;
	int i, ii, n, h, w, c, out_c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03;
	register zq_mm_type c10, c11, c12, c13;
	register zq_mm_type c20, c21, c22, c23;
	register zq_mm_type c30, c31, c32, c33;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[16];
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < div4_size; i++)
	{
		ii = i * 4;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
		}
	}

	for (i = 0; i < NHW - div4_size * 4; i++)
	{
		ii = div4_size * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + div4_size);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			dst_ptr += zq_mm_align_size;
			src_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < div4_size; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*i;
		ii = i * 4;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		/*if (paddedC % zq_mm_align_size64 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_64_first;
				c = zq_mm_align_size64;
				for (; c < paddedC; c += zq_mm_align_size64)
				{
					op4x4_64;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size32 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_32_first;
				c = zq_mm_align_size32;
				for (; c < paddedC; c += zq_mm_align_size32)
				{
					op4x4_32;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else*/ if (paddedC % zq_mm_align_size16 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;

				op4x4_16_first;
				c = zq_mm_align_size16;
				for (; c < paddedC; c += zq_mm_align_size16)
				{
					op4x4_16;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_8_first;
				c = zq_mm_align_size8;
				for (; c < paddedC; c += zq_mm_align_size8)
				{
					op4x4_8;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_4_first;
				c = zq_mm_align_size4;
				for (; c < paddedC; c += zq_mm_align_size4)
				{
					op4x4_4;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_2_first;
				c = zq_mm_align_size2;
				for (; c < paddedC; c += zq_mm_align_size2)
				{
					op4x4_2;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op4x4_1_first;
				c = zq_mm_align_size;
				for (; c < paddedC; c += zq_mm_align_size)
				{
					op4x4_1;
				}
				store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
	}

	//rest 
	for (i = 0; i < NHW - div4_size * 4; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + div4_size);
		ii = (div4_size << 2) + i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		/*if (paddedC % zq_mm_align_size64 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_64_first;
				c = zq_mm_align_size64;
				for (; c < paddedC; c += zq_mm_align_size64)
				{
					op1x4_64;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size32 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_32_first;
				c = zq_mm_align_size32;
				for (; c < paddedC; c += zq_mm_align_size32)
				{
					op1x4_32;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else*/ if (paddedC % zq_mm_align_size16 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_16_first;
				c = zq_mm_align_size16;
				for (; c < paddedC; c += zq_mm_align_size16)
				{
					op1x4_16;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_8_first;
				c = zq_mm_align_size8;
				for (; c < paddedC; c += zq_mm_align_size8)
				{
					op1x4_8;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_4_first;
				c = zq_mm_align_size4;
				for (; c < paddedC; c += zq_mm_align_size4)
				{
					op1x4_4;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_2_first;
				c = zq_mm_align_size2;
				for (; c < paddedC; c += zq_mm_align_size2)
				{
					op1x4_2;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
		else //if (paddedC % zq_mm_align_size == 0)
		{
			src_ptr3 = packed_filter;
			for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size, src_ptr3 += packed_B_step)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = src_ptr3;
				op1x4_1_first;
				c = zq_mm_align_size;
				for (; c < paddedC; c += zq_mm_align_size)
				{
					op1x4_1;
				}
				store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
				dst_ptr0 += out_sliceStep;
			}
		}
	}
}

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM4N8_other_kernel1x1(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = in_H*in_W;
	int NHW = in_N*HW;
	int A_div4_num = NHW >> 2;
	int paddedC = (in_C + 3) >> 2 << 2;
	int packed_A_num = A_div4_num + (NHW - (A_div4_num << 2));
	int packed_A_step = paddedC * 4;
	int packed_B_step = paddedC * 8;
	int out_alignC = (out_C + 3) >> 2 << 2;
	int B_div8_num = out_alignC >> 3;
	int B_div4_num = (out_alignC - (B_div8_num << 3)) >> 2;
	int B_pack_num = B_div8_num + B_div4_num;
	int i, j, ii, n, h, w, c, out_c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	register zq_mm_type a0, a1, a2, a3, b0, b1;
	register zq_mm_type c00, c01, c10, c11, c20, c21, c30, c31;
#if WITH_BIAS
	register zq_mm_type bias_v0, bias_v1;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v0, slope_v1;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
		}
	}

	for (i = 0; i < NHW - A_div4_num * 4; i++)
	{
		ii = A_div4_num * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + A_div4_num);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			dst_ptr += zq_mm_align_size;
			src_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < A_div4_num; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*i;
		ii = i * 4;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		if (paddedC % zq_mm_align_size16 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op4x8_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op4x4_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_8_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op4x8_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op4x4_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_4_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op4x8_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op4x4_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op4x8_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op4x4_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op4x8_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = A_buffer + packed_A_step*i;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op4x4_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
	}
	
	//rest 
	for (i = 0; i < NHW - A_div4_num * 4; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + A_div4_num);
		ii = (A_div4_num << 2) + i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(A_div4_num + i);
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			dst_ptr0 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(A_div4_num + i);
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
		}
	}
}

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM8N8_other_kernel1x1(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = in_H*in_W;
	int NHW = in_N*HW;
	int A_div8_num = NHW >> 3;
	int A_div4_num = (NHW - (A_div8_num << 3)) >> 2;
	int A_rest_num = NHW - (A_div8_num << 3) - (A_div4_num << 2);
	int paddedC = (in_C + 3) >> 2 << 2;
	int packed_A_num = A_div8_num + A_div4_num + A_rest_num;
	int packed_A_step = paddedC * 8;
	int packed_B_step = paddedC * 8;
	int out_alignC = (out_C + 3) >> 2 << 2;
	int B_div8_num = out_alignC >> 3;
	int B_div4_num = (out_alignC - (B_div8_num << 3)) >> 2;
	int B_pack_num = B_div8_num + B_div4_num;
	int i, j, ii, n, h, w, c, out_c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2, *src_ptr3, *src_ptr4, *src_ptr5, *src_ptr6, *src_ptr7;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5, *dst_ptr6, *dst_ptr7;
	register zq_mm_type a0, a1, a2, a3, a4, a5, a6, a7, b0, b1;
	register zq_mm_type c00, c01, c10, c11, c20, c21, c30, c31;
	register zq_mm_type c40, c41, c50, c51, c60, c61, c70, c71;
#if WITH_BIAS
	register zq_mm_type bias_v0, bias_v1;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v0, slope_v1;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < A_div8_num; i++)
	{
		ii = i * 8;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr4 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr5 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr6 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr7 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size4, zq_mm_load_ps(src_ptr4));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size5, zq_mm_load_ps(src_ptr5));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size6, zq_mm_load_ps(src_ptr6));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size7, zq_mm_load_ps(src_ptr7));
			dst_ptr += zq_mm_align_size8;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
			src_ptr4 += in_sliceStep;
			src_ptr5 += in_sliceStep;
			src_ptr6 += in_sliceStep;
			src_ptr7 += in_sliceStep;
		}
	}

	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4 + A_div8_num * 8;
		dst_ptr = A_buffer + packed_A_step*(i+A_div8_num);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr1 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr2 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr3 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(src_ptr1));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(src_ptr2));
			zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(src_ptr3));
			dst_ptr += zq_mm_align_size4;
			src_ptr0 += in_sliceStep;
			src_ptr1 += in_sliceStep;
			src_ptr2 += in_sliceStep;
			src_ptr3 += in_sliceStep;
		}
	}

	for (i = 0; i < NHW - A_div8_num * 8 - A_div4_num * 4; i++)
	{
		ii = A_div8_num * 8 + A_div4_num * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + A_div8_num + A_div4_num);
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		src_ptr0 = in_data + n*in_imStep + h*in_widthStep + w*zq_mm_align_size;
		for (c = 0; c < in_C; c += zq_mm_align_size)
		{
			zq_mm_store_ps(dst_ptr, zq_mm_load_ps(src_ptr0));
			dst_ptr += zq_mm_align_size;
			src_ptr0 += in_sliceStep;
		}
	}

	/* gemm */
	for (i = 0; i < A_div8_num; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*i;
		ii = i * 8;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr4 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr5 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr6 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr7 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		if (paddedC % zq_mm_align_size16 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op8x8_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op8x8_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
				c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				zq_mm_store_ps(dst_ptr4, c41);
				zq_mm_store_ps(dst_ptr5, c51);
				zq_mm_store_ps(dst_ptr6, c61);
				zq_mm_store_ps(dst_ptr7, c71);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op8x4_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op8x4_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op8x8_other_8_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op8x8_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
				c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				zq_mm_store_ps(dst_ptr4, c41);
				zq_mm_store_ps(dst_ptr5, c51);
				zq_mm_store_ps(dst_ptr6, c61);
				zq_mm_store_ps(dst_ptr7, c71);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op8x4_other_8_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op8x4_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op8x8_other_4_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op8x8_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
				c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				zq_mm_store_ps(dst_ptr4, c41);
				zq_mm_store_ps(dst_ptr5, c51);
				zq_mm_store_ps(dst_ptr6, c61);
				zq_mm_store_ps(dst_ptr7, c71);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op8x4_other_4_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op8x4_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op8x8_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op8x8_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
				c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				zq_mm_store_ps(dst_ptr4, c41);
				zq_mm_store_ps(dst_ptr5, c51);
				zq_mm_store_ps(dst_ptr6, c61);
				zq_mm_store_ps(dst_ptr7, c71);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op8x4_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op8x4_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}
		}
		else
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op8x8_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op8x8_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
				c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				zq_mm_store_ps(dst_ptr4, c41);
				zq_mm_store_ps(dst_ptr5, c51);
				zq_mm_store_ps(dst_ptr6, c61);
				zq_mm_store_ps(dst_ptr7, c71);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op8x4_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op8x4_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
				c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
				c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
				c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				zq_mm_store_ps(dst_ptr4, c40);
				zq_mm_store_ps(dst_ptr5, c50);
				zq_mm_store_ps(dst_ptr6, c60);
				zq_mm_store_ps(dst_ptr7, c70);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				dst_ptr4 += out_sliceStep;
				dst_ptr5 += out_sliceStep;
				dst_ptr6 += out_sliceStep;
				dst_ptr7 += out_sliceStep;
			}
		}
	}

	for (i = 0; i < A_div4_num; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i+A_div8_num);
		ii = i * 4 + A_div8_num * 8;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		if (paddedC % zq_mm_align_size16 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op4x8_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_16_first;
				for (c = zq_mm_align_size16; c < paddedC; c += zq_mm_align_size16)
				{
					op4x4_other_16;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size8 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_8_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op4x8_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size8; c < paddedC; c += zq_mm_align_size8)
				{
					op4x4_other_8;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size4 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_4_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op4x8_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size4; c < paddedC; c += zq_mm_align_size4)
				{
					op4x4_other_4;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else if (paddedC % zq_mm_align_size2 == 0)
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op4x8_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_2_first;
				for (c = zq_mm_align_size2; c < paddedC; c += zq_mm_align_size2)
				{
					op4x4_other_2;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
		else
		{
			for (j = 0; j < B_div8_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*j;
				op4x8_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op4x8_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + j * 8);
				slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
				c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
				zq_mm_store_ps(dst_ptr0, c01);
				zq_mm_store_ps(dst_ptr1, c11);
				zq_mm_store_ps(dst_ptr2, c21);
				zq_mm_store_ps(dst_ptr3, c31);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}

			for (j = 0; j < B_div4_num; j++)
			{
				src_ptr0 = src_ptr2;
				src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
				op4x4_other_1_first;
				for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
				{
					op4x4_other_1;
				}
#if WITH_PRELU	
				slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
				c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
				c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
				c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
				c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
				zq_mm_store_ps(dst_ptr0, c00);
				zq_mm_store_ps(dst_ptr1, c10);
				zq_mm_store_ps(dst_ptr2, c20);
				zq_mm_store_ps(dst_ptr3, c30);
				dst_ptr0 += out_sliceStep;
				dst_ptr1 += out_sliceStep;
				dst_ptr2 += out_sliceStep;
				dst_ptr3 += out_sliceStep;
			}
		}
	}

	//rest 
	for (i = 0; i < NHW - A_div8_num * 8 - A_div4_num * 4; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + A_div8_num + A_div4_num);
		ii = (A_div8_num << 3) + (A_div4_num << 2) + i;
		n = ii / HW;
		h = (ii%HW) / in_W;
		w = (ii%HW) % in_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			dst_ptr0 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			for (c = zq_mm_align_size; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 3);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
		}
	}
}

#endif

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packed4_kernel3x3_C3C4(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	int dilate_H,
	int dilate_W,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = out_H*out_W;
	int NHW = out_N*HW;
	int div4_size = NHW >> 2;
	int paddedC = 36;
	int packed_A_num = div4_size + (NHW - (div4_size << 2));
	int packed_A_step = paddedC * 4;
	int packed_B_step = paddedC * 4;
	int packed_B_num = (out_C + 3) >> 2;
	int stride_H_mul_widthStep = stride_H*in_widthStep;
	int stride_W_mul_pixStep = stride_W*zq_mm_align_size;
	int dilate_H_mul_widthStep = dilate_H*in_widthStep;
	int dilate_W_mul_pixStep = dilate_W*zq_mm_align_size;
	int i, ii, n, h, w, out_c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* im_ptr0, *im_ptr1, *im_ptr2, *im_ptr3;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	const zq_base_type* src_ptr0, *src_ptr1;
	register zq_mm_type a0, a1, a2, a3, b0, b1, b2, b3;
	register zq_mm_type c00, c01, c02, c03;
	register zq_mm_type c10, c11, c12, c13;
	register zq_mm_type c20, c21, c22, c23;
	register zq_mm_type c30, c31, c32, c33;
	ZQ_DECLSPEC_ALIGN32 zq_base_type q[16];
#if WITH_BIAS
	register zq_mm_type bias_v;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < div4_size; i++)
	{
		ii = i * 4;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr1 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr2 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr3 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		row_ptr1 = im_ptr1;
		row_ptr2 = im_ptr2;
		row_ptr3 = im_ptr3;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			pix_ptr1 = row_ptr1;
			pix_ptr2 = row_ptr2;
			pix_ptr3 = row_ptr3;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
				dst_ptr += zq_mm_align_size4;
				pix_ptr0 += dilate_W_mul_pixStep;
				pix_ptr1 += dilate_W_mul_pixStep;
				pix_ptr2 += dilate_W_mul_pixStep;
				pix_ptr3 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
			row_ptr1 += dilate_H_mul_widthStep;
			row_ptr2 += dilate_H_mul_widthStep;
			row_ptr3 += dilate_H_mul_widthStep;
		}
	}

	for (i = 0; i < NHW - div4_size * 4; i++)
	{
		ii = div4_size * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + div4_size);
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				dst_ptr += zq_mm_align_size;
				pix_ptr0 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
		}
	}

	/* gemm */
	for (i = 0; i < div4_size; i++)
	{
		ii = i * 4;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
		{
			src_ptr0 = A_buffer + packed_A_step*i;
			src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
			op4x4_1_first;
			op4x4_8;
			store4x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_4x4.h"
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}
	}

	//rest 
	for (i = 0; i < NHW - div4_size * 4; i++)
	{
		ii = (div4_size << 2) + i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (out_c = 0; out_c < out_C; out_c += zq_mm_align_size)
		{
			src_ptr0 = A_buffer + packed_A_step*(i + div4_size);
			src_ptr1 = packed_filter + packed_B_step*(out_c / zq_mm_align_size);
			op1x4_1_first;
			op1x4_8;
			store1x4;
#include "zq_cnn_convolution_gemm_nchwc_packed4_handle_bias_prelu_1x4.h"
			dst_ptr0 += out_sliceStep;
		}
	}
}

#if __ARM_NEON && __ARM_NEON_ARMV8

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM4N8_other_kernel3x3_C3(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	int dilate_H,
	int dilate_W,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = out_H*out_W;
	int NHW = out_N*HW;
	int A_div4_num = NHW >> 2;
	int packed_A_num = A_div4_num + (NHW - (A_div4_num << 2));
	int out_alignC = (out_C + 3) >> 2 << 2;
	int B_div8_num = out_alignC >> 3;
	int B_div4_num = (out_alignC - (B_div8_num << 3)) >> 2;
	int paddedC = 36;
	int packed_B_step = paddedC * 8;
	int packed_A_step = paddedC * 4;
	int stride_H_mul_widthStep = stride_H*in_widthStep;
	int stride_W_mul_pixStep = stride_W*zq_mm_align_size;
	int dilate_H_mul_widthStep = dilate_H*in_widthStep;
	int dilate_W_mul_pixStep = dilate_W*zq_mm_align_size;
	int i, j, ii, n, h, w, c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* im_ptr0, *im_ptr1, *im_ptr2, *im_ptr3;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3;
	const zq_base_type* src_ptr0, *src_ptr1;
	register zq_mm_type a0, a1, a2, a3, b0, b1;
	register zq_mm_type c00, c01, c10, c11, c20, c21, c30, c31;
#if WITH_BIAS
	register zq_mm_type bias_v0, bias_v1;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v0, slope_v1;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr1 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr2 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr3 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		row_ptr1 = im_ptr1;
		row_ptr2 = im_ptr2;
		row_ptr3 = im_ptr3;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			pix_ptr1 = row_ptr1;
			pix_ptr2 = row_ptr2;
			pix_ptr3 = row_ptr3;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
				dst_ptr += zq_mm_align_size4;
				pix_ptr0 += dilate_W_mul_pixStep;
				pix_ptr1 += dilate_W_mul_pixStep;
				pix_ptr2 += dilate_W_mul_pixStep;
				pix_ptr3 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
			row_ptr1 += dilate_H_mul_widthStep;
			row_ptr2 += dilate_H_mul_widthStep;
			row_ptr3 += dilate_H_mul_widthStep;
		}
	}

	for (i = 0; i < NHW - A_div4_num * 4; i++)
	{
		ii = A_div4_num * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + A_div4_num);
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				dst_ptr += zq_mm_align_size;
				pix_ptr0 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
		}
	}

	/* gemm */
	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*i;
			src_ptr1 = packed_filter + packed_B_step*j;
			op4x8_C3_other_1_first;
			op4x8_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
			c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			zq_mm_store_ps(dst_ptr1, c11);
			zq_mm_store_ps(dst_ptr2, c21);
			zq_mm_store_ps(dst_ptr3, c31);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*i;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			op4x4_C3_other_1_first;
			op4x4_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}
	}

	//rest 
	for (i = 0; i < NHW - A_div4_num * 4; i++)
	{
		ii = (A_div4_num << 2) + i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(i+A_div4_num);
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			for (c = 4; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			dst_ptr0 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = A_buffer + packed_A_step*(i+A_div4_num);
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			for (c = 4; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
		}
	}
}

/*zq_mm_align_size must be 4*/
void zq_cnn_convolution_gemm_nchwc_packedM8N8_other_kernel3x3_C3(
	const zq_base_type* in_data,
	int in_N,
	int in_H,
	int in_W,
	int in_C,
	int in_widthStep,
	int in_sliceStep,
	int in_imStep,
	const zq_base_type* packed_filter,
	int kernel_H,
	int kernel_W,
	int stride_H,
	int stride_W,
	int dilate_H,
	int dilate_W,
	zq_base_type* out_data,
	int out_N,
	int out_H,
	int out_W,
	int out_C,
	int out_widthStep,
	int out_sliceStep,
	int out_imStep,
#if WITH_BIAS
	const zq_base_type* bias,
#endif
#if WITH_PRELU
	const zq_base_type* slope,
#endif
	void** buffer,
	__int64* buffer_len
)
{
	int HW = out_H*out_W;
	int NHW = out_N*HW;
	int A_div8_num = NHW >> 3;
	int A_div4_num = (NHW - (A_div8_num << 3)) >> 2;
	int A_rest_num = NHW - (A_div8_num << 3) - (A_div4_num << 2);
	int packed_A_num = A_div8_num + A_div4_num + A_rest_num;
	int out_alignC = (out_C + 3) >> 2 << 2;
	int B_div8_num = out_alignC >> 3;
	int B_div4_num = (out_alignC - (B_div8_num << 3)) >> 2;
	int paddedC = 36;
	int packed_B_step = paddedC * 8;
	int packed_A_step = paddedC * 8;
	int stride_H_mul_widthStep = stride_H*in_widthStep;
	int stride_W_mul_pixStep = stride_W*zq_mm_align_size;
	int dilate_H_mul_widthStep = dilate_H*in_widthStep;
	int dilate_W_mul_pixStep = dilate_W*zq_mm_align_size;
	int i, j, ii, n, h, w, c;
	zq_base_type* A_buffer, *dst_ptr;
	const zq_base_type* im_ptr0, *im_ptr1, *im_ptr2, *im_ptr3, *im_ptr4, *im_ptr5, *im_ptr6, *im_ptr7;
	const zq_base_type* row_ptr0, *row_ptr1, *row_ptr2, *row_ptr3, *row_ptr4, *row_ptr5, *row_ptr6, *row_ptr7;
	const zq_base_type* pix_ptr0, *pix_ptr1, *pix_ptr2, *pix_ptr3, *pix_ptr4, *pix_ptr5, *pix_ptr6, *pix_ptr7;
	zq_base_type* dst_ptr0, *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5, *dst_ptr6, *dst_ptr7;
	const zq_base_type* src_ptr0, *src_ptr1, *src_ptr2;
	register zq_mm_type a0, a1, a2, a3, a4, a5, a6, a7, b0, b1;
	register zq_mm_type c00, c01, c10, c11, c20, c21, c30, c31;
	register zq_mm_type c40, c41, c50, c51, c60, c61, c70, c71;
#if WITH_BIAS
	register zq_mm_type bias_v0, bias_v1;
#endif
#if WITH_PRELU
	register zq_mm_type slope_v0, slope_v1;
	register zq_mm_type zero_v = zq_mm_setzero_ps();
#endif

	__int64 need_buffer_size = (__int64)packed_A_step*packed_A_num * sizeof(zq_base_type);
	if (*buffer_len < need_buffer_size)
	{
		if (*buffer != 0)
			_aligned_free(*buffer);
		*buffer = _aligned_malloc(need_buffer_size, 32);
		*buffer_len = need_buffer_size;
	}
	A_buffer = (zq_base_type*)(*buffer);
	/* pack in_data */
	for (i = 0; i < A_div8_num; i++)
	{
		ii = i * 8;
		dst_ptr = A_buffer + packed_A_step*i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr1 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr2 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr3 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr4 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr5 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr6 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr7 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		row_ptr1 = im_ptr1;
		row_ptr2 = im_ptr2;
		row_ptr3 = im_ptr3;
		row_ptr4 = im_ptr4;
		row_ptr5 = im_ptr5;
		row_ptr6 = im_ptr6;
		row_ptr7 = im_ptr7;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			pix_ptr1 = row_ptr1;
			pix_ptr2 = row_ptr2;
			pix_ptr3 = row_ptr3;
			pix_ptr4 = row_ptr4;
			pix_ptr5 = row_ptr5;
			pix_ptr6 = row_ptr6;
			pix_ptr7 = row_ptr7;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size4, zq_mm_load_ps(pix_ptr4));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size5, zq_mm_load_ps(pix_ptr5));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size6, zq_mm_load_ps(pix_ptr6));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size7, zq_mm_load_ps(pix_ptr7));
				dst_ptr += zq_mm_align_size8;
				pix_ptr0 += dilate_W_mul_pixStep;
				pix_ptr1 += dilate_W_mul_pixStep;
				pix_ptr2 += dilate_W_mul_pixStep;
				pix_ptr3 += dilate_W_mul_pixStep;
				pix_ptr4 += dilate_W_mul_pixStep;
				pix_ptr5 += dilate_W_mul_pixStep;
				pix_ptr6 += dilate_W_mul_pixStep;
				pix_ptr7 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
			row_ptr1 += dilate_H_mul_widthStep;
			row_ptr2 += dilate_H_mul_widthStep;
			row_ptr3 += dilate_H_mul_widthStep;
			row_ptr4 += dilate_H_mul_widthStep;
			row_ptr5 += dilate_H_mul_widthStep;
			row_ptr6 += dilate_H_mul_widthStep;
			row_ptr7 += dilate_H_mul_widthStep;
		}
	}

	for (i = 0; i < A_div4_num; i++)
	{
		ii = i * 4 + A_div8_num * 8;
		dst_ptr = A_buffer + packed_A_step*(i+A_div8_num);
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr1 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr2 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr3 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		row_ptr1 = im_ptr1;
		row_ptr2 = im_ptr2;
		row_ptr3 = im_ptr3;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			pix_ptr1 = row_ptr1;
			pix_ptr2 = row_ptr2;
			pix_ptr3 = row_ptr3;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size, zq_mm_load_ps(pix_ptr1));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size2, zq_mm_load_ps(pix_ptr2));
				zq_mm_store_ps(dst_ptr + zq_mm_align_size3, zq_mm_load_ps(pix_ptr3));
				dst_ptr += zq_mm_align_size4;
				pix_ptr0 += dilate_W_mul_pixStep;
				pix_ptr1 += dilate_W_mul_pixStep;
				pix_ptr2 += dilate_W_mul_pixStep;
				pix_ptr3 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
			row_ptr1 += dilate_H_mul_widthStep;
			row_ptr2 += dilate_H_mul_widthStep;
			row_ptr3 += dilate_H_mul_widthStep;
		}
	}

	for (i = 0; i < NHW - A_div8_num * 8 - A_div4_num * 4; i++)
	{
		ii = A_div8_num * 8 + A_div4_num * 4 + i;
		dst_ptr = A_buffer + packed_A_step*(i + A_div8_num + A_div4_num);
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		im_ptr0 = in_data + n*in_imStep + h*stride_H_mul_widthStep + w*stride_W_mul_pixStep;
		row_ptr0 = im_ptr0;
		for (h = 0; h < 3; h++)
		{
			pix_ptr0 = row_ptr0;
			for (w = 0; w < 3; w++)
			{
				zq_mm_store_ps(dst_ptr, zq_mm_load_ps(pix_ptr0));
				dst_ptr += zq_mm_align_size;
				pix_ptr0 += dilate_W_mul_pixStep;
			}
			row_ptr0 += dilate_H_mul_widthStep;
		}
	}

	/* gemm */
	for (i = 0; i < A_div8_num; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*i;
		ii = i * 8;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr4 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr5 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr6 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr7 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*j;
			op8x8_C3_other_1_first;
			op8x8_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
			c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
			c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
			c41 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c41, zero_v), zq_mm_max_ps(c41, zero_v));
			c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
			c51 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c51, zero_v), zq_mm_max_ps(c51, zero_v));
			c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
			c61 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c61, zero_v), zq_mm_max_ps(c61, zero_v));
			c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
			c71 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c71, zero_v), zq_mm_max_ps(c71, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			zq_mm_store_ps(dst_ptr4, c40);
			zq_mm_store_ps(dst_ptr5, c50);
			zq_mm_store_ps(dst_ptr6, c60);
			zq_mm_store_ps(dst_ptr7, c70);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			dst_ptr4 += out_sliceStep;
			dst_ptr5 += out_sliceStep;
			dst_ptr6 += out_sliceStep;
			dst_ptr7 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			zq_mm_store_ps(dst_ptr1, c11);
			zq_mm_store_ps(dst_ptr2, c21);
			zq_mm_store_ps(dst_ptr3, c31);
			zq_mm_store_ps(dst_ptr4, c41);
			zq_mm_store_ps(dst_ptr5, c51);
			zq_mm_store_ps(dst_ptr6, c61);
			zq_mm_store_ps(dst_ptr7, c71);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			dst_ptr4 += out_sliceStep;
			dst_ptr5 += out_sliceStep;
			dst_ptr6 += out_sliceStep;
			dst_ptr7 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			op8x4_C3_other_1_first;
			op8x4_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
			c40 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c40, zero_v), zq_mm_max_ps(c40, zero_v));
			c50 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c50, zero_v), zq_mm_max_ps(c50, zero_v));
			c60 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c60, zero_v), zq_mm_max_ps(c60, zero_v));
			c70 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c70, zero_v), zq_mm_max_ps(c70, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			zq_mm_store_ps(dst_ptr4, c40);
			zq_mm_store_ps(dst_ptr5, c50);
			zq_mm_store_ps(dst_ptr6, c60);
			zq_mm_store_ps(dst_ptr7, c70);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			dst_ptr4 += out_sliceStep;
			dst_ptr5 += out_sliceStep;
			dst_ptr6 += out_sliceStep;
			dst_ptr7 += out_sliceStep;
		}
	}

	for (i = 0; i < A_div4_num; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + A_div8_num);
		ii = i * 4 + A_div8_num * 8;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr1 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr2 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;
		ii++;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr3 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*j;
			op4x8_C3_other_1_first;
			op4x8_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c11 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c11, zero_v), zq_mm_max_ps(c11, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c21 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c21, zero_v), zq_mm_max_ps(c21, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
			c31 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c31, zero_v), zq_mm_max_ps(c31, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			zq_mm_store_ps(dst_ptr1, c11);
			zq_mm_store_ps(dst_ptr2, c21);
			zq_mm_store_ps(dst_ptr3, c31);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			op4x4_C3_other_1_first;
			op4x4_C3_other_8;
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c10 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c10, zero_v), zq_mm_max_ps(c10, zero_v));
			c20 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c20, zero_v), zq_mm_max_ps(c20, zero_v));
			c30 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c30, zero_v), zq_mm_max_ps(c30, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			zq_mm_store_ps(dst_ptr1, c10);
			zq_mm_store_ps(dst_ptr2, c20);
			zq_mm_store_ps(dst_ptr3, c30);
			dst_ptr0 += out_sliceStep;
			dst_ptr1 += out_sliceStep;
			dst_ptr2 += out_sliceStep;
			dst_ptr3 += out_sliceStep;
		}
	}

	//rest 
	for (i = 0; i < NHW - A_div8_num * 8 - A_div4_num * 4; i++)
	{
		src_ptr2 = A_buffer + packed_A_step*(i + A_div8_num + A_div4_num);
		ii = (A_div8_num << 3) + (A_div4_num << 2) + i;
		n = ii / HW;
		h = (ii%HW) / out_W;
		w = (ii%HW) % out_W;
		dst_ptr0 = out_data + n*out_imStep + h*out_widthStep + w*zq_mm_align_size;

		for (j = 0; j < B_div8_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*j;
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + j * 8);
			bias_v1 = zq_mm_load_ps(bias + j * 8 + 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
			c01 = vfmaq_laneq_f32(bias_v1, b1, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
			c01 = vmulq_laneq_f32(b1, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
			src_ptr1 += zq_mm_align_size2;
			b0 = zq_mm_load_ps(src_ptr1);
			b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			for (c = 4; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 1);
				src_ptr1 += zq_mm_align_size2;
				b0 = zq_mm_load_ps(src_ptr1);
				b1 = zq_mm_load_ps(src_ptr1 + zq_mm_align_size);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
				c01 = vfmaq_laneq_f32(c01, b1, a0, 2);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + j * 8);
			slope_v1 = zq_mm_load_ps(slope + j * 8 + 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
			c01 = zq_mm_fmadd_ps(slope_v1, zq_mm_min_ps(c01, zero_v), zq_mm_max_ps(c01, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
			zq_mm_store_ps(dst_ptr0, c01);
			dst_ptr0 += out_sliceStep;
		}

		for (j = 0; j < B_div4_num; j++)
		{
			src_ptr0 = src_ptr2;
			src_ptr1 = packed_filter + packed_B_step*(B_div8_num + j);
			a0 = zq_mm_load_ps(src_ptr0);
			b0 = zq_mm_load_ps(src_ptr1);
#if WITH_BIAS
			bias_v0 = zq_mm_load_ps(bias + B_div8_num * 8 + j * 4);
			c00 = vfmaq_laneq_f32(bias_v0, b0, a0, 0);
#else
			c00 = vmulq_laneq_f32(b0, a0, 0);
#endif
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
			src_ptr1 += zq_mm_align_size;
			b0 = zq_mm_load_ps(src_ptr1);
			c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			for (c = 4; c < paddedC; c += zq_mm_align_size)
			{
				src_ptr0 += zq_mm_align_size;
				a0 = zq_mm_load_ps(src_ptr0);

				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 1);
				src_ptr1 += zq_mm_align_size;
				b0 = zq_mm_load_ps(src_ptr1);
				c00 = vfmaq_laneq_f32(c00, b0, a0, 2);
			}
#if WITH_PRELU	
			slope_v0 = zq_mm_load_ps(slope + B_div8_num * 8 + j * 4);
			c00 = zq_mm_fmadd_ps(slope_v0, zq_mm_min_ps(c00, zero_v), zq_mm_max_ps(c00, zero_v));
#endif
			zq_mm_store_ps(dst_ptr0, c00);
			dst_ptr0 += out_sliceStep;
		}
	}
}

#endif

#undef op8x4_1_first
#undef op8x4_1
#undef op8x4_2_first
#undef op8x4_2
#undef op8x4_4_first
#undef op8x4_4
#undef op8x4_8_first
#undef op8x4_8
#undef op8x4_16_first
#undef op8x4_16
#undef store8x4

#undef op4x4_1_first
#undef op4x4_1
#undef op4x4_2_first
#undef op4x4_2
#undef op4x4_4_first
#undef op4x4_4
#undef op4x4_8_first
#undef op4x4_8
#undef op4x4_16_first
#undef op4x4_16
#undef store4x4

#undef op1x4_1_first
#undef op1x4_1
#undef op1x4_2_first
#undef op1x4_2
#undef op1x4_4_first
#undef op1x4_4
#undef op1x4_8_first
#undef op1x4_8
#undef op1x4_16_first
#undef op1x4_16
#undef store1x4

#if __ARM_NEON && __ARM_NEON_ARMV8
#undef op8x8_other_1_first
#undef op8x8_other_1
#undef op8x8_other_2_first
#undef op8x8_other_2
#undef op8x8_other_4_first
#undef op8x8_other_4
#undef op8x8_other_8_first
#undef op8x8_other_8
#undef op8x8_other_16_first
#undef op8x8_other_16
#undef op8x8_other_32_first
#undef op8x8_other_32
#undef op8x8_other_64_first
#undef op8x8_other_64
#undef op8x4_other_1_first
#undef op8x4_other_1
#undef op8x4_other_2_first
#undef op8x4_other_2
#undef op8x4_other_4_first
#undef op8x4_other_4
#undef op8x4_other_8_first
#undef op8x4_other_8
#undef op8x4_other_16_first
#undef op8x4_other_16
#undef op8x4_other_32_first
#undef op8x4_other_32
#undef op8x4_other_64_first
#undef op8x4_other_64
#undef op8x8_C3_other_1_first
#undef op8x8_C3_other_1
#undef op8x8_C3_other_2_first
#undef op8x8_C3_other_2
#undef op8x8_C3_other_4_first
#undef op8x8_C3_other_4
#undef op8x8_C3_other_8_first
#undef op8x8_C3_other_8
#undef op8x8_C3_other_16_first
#undef op8x8_C3_other_16
#undef op8x8_C3_other_32_first
#undef op8x8_C3_other_32
#undef op8x8_C3_other_64_first
#undef op8x8_C3_other_64
#undef op8x4_C3_other_1_first
#undef op8x4_C3_other_1
#undef op8x4_C3_other_2_first
#undef op8x4_C3_other_2
#undef op8x4_C3_other_4_first
#undef op8x4_C3_other_4
#undef op8x4_C3_other_8_first
#undef op8x4_C3_other_8
#undef op8x4_C3_other_16_first
#undef op8x4_C3_other_16
#undef op8x4_C3_other_32_first
#undef op8x4_C3_other_32
#undef op8x4_C3_other_64_first
#undef op8x4_C3_other_64
#undef op4x8_other_1_first
#undef op4x8_other_1
#undef op4x8_other_2_first
#undef op4x8_other_2
#undef op4x8_other_4_first
#undef op4x8_other_4
#undef op4x8_other_8_first
#undef op4x8_other_8
#undef op4x8_other_16_first
#undef op4x8_other_16
#undef op4x8_other_32_first
#undef op4x8_other_32
#undef op4x8_other_64_first
#undef op4x8_other_64
#undef op4x4_other_1_first
#undef op4x4_other_1
#undef op4x4_other_2_first
#undef op4x4_other_2
#undef op4x4_other_4_first
#undef op4x4_other_4
#undef op4x4_other_8_first
#undef op4x4_other_8
#undef op4x4_other_16_first
#undef op4x4_other_16
#undef op4x4_other_32_first
#undef op4x4_other_32
#undef op4x4_other_64_first
#undef op4x4_other_64
#undef op4x8_C3_other_1_first
#undef op4x8_C3_other_1
#undef op4x8_C3_other_2_first
#undef op4x8_C3_other_2
#undef op4x8_C3_other_4_first
#undef op4x8_C3_other_4
#undef op4x8_C3_other_8_first
#undef op4x8_C3_other_8
#undef op4x8_C3_other_16_first
#undef op4x8_C3_other_16
#undef op4x8_C3_other_32_first
#undef op4x8_C3_other_32
#undef op4x8_C3_other_64_first
#undef op4x8_C3_other_64
#undef op4x4_C3_other_1_first
#undef op4x4_C3_other_1
#undef op4x4_C3_other_2_first
#undef op4x4_C3_other_2
#undef op4x4_C3_other_4_first
#undef op4x4_C3_other_4
#undef op4x4_C3_other_8_first
#undef op4x4_C3_other_8
#undef op4x4_C3_other_16_first
#undef op4x4_C3_other_16
#undef op4x4_C3_other_32_first
#undef op4x4_C3_other_32
#undef op4x4_C3_other_64_first
#undef op4x4_C3_other_64
#endif