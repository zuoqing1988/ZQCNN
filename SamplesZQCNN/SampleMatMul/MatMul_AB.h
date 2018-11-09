#ifndef _ZQ_MAT_MUL_AB_H_
#define _ZQ_MAT_MUL_AB_H_

void MatMul0_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul1_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul2_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul3_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul4_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul5_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

void MatMul6_AB(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc);

#endif