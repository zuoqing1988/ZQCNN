

void zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	float a_val;
	int m, n, k;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q1[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q2[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q3[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q4[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type a_vec;

	if (K % zq_mm_align_size8 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda,Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N - 3; n += 4, Bptr += ldb*4)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K;
					k += zq_mm_align_size8, A_c_ptr += zq_mm_align_size8,
					B_c_ptr1 += zq_mm_align_size8, B_c_ptr2 += zq_mm_align_size8,
					B_c_ptr3 += zq_mm_align_size8, B_c_ptr4 += zq_mm_align_size8)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size4);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size4), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size4), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size4), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size5);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size5), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size5), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size5), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size5), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size6);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size6), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size6), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size6), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size6), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size7);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size7), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size7), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size7), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size7), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec1 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
					k < K;
					k += zq_mm_align_size8, A_c_ptr += zq_mm_align_size8, B_c_ptr1 += zq_mm_align_size8)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size4);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size4), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size5);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size5), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size6);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size6), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size7);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size7), sum_vec1);
				}
				zq_mm_store_ps(q1, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q1;
			}
		}

	}
	else if (K % zq_mm_align_size4 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda,Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N - 3; n += 4, Bptr += ldb*4)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec1 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
					k < K;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4, B_c_ptr1 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
				}
				zq_mm_store_ps(q1, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q1;
			}
		}
	}
	else
	{
		Aptr = A;
		Cptr = C;
		//printf("hello\n");
		for (m = 0; m < M; m++, Aptr += lda,Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N - 3; n += 4, Bptr += ldb*4)
			{
				//printf("n=%d\n", n);
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK - zq_mm_align_size4;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
				}
				for (; k < padK-zq_mm_align_size;
					k += zq_mm_align_size, A_c_ptr += zq_mm_align_size,
					B_c_ptr1 += zq_mm_align_size, B_c_ptr2 += zq_mm_align_size,
					B_c_ptr3 += zq_mm_align_size, B_c_ptr4 += zq_mm_align_size)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				for (; k < K; k ++)
				{
					a_val = *(A_c_ptr++);
					q1[0] += a_val*(*(B_c_ptr1++));
					q2[0] += a_val*(*(B_c_ptr2++));
					q3[0] += a_val*(*(B_c_ptr3++));
					q4[0] += a_val*(*(B_c_ptr4++));
				}
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec1 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1;
					k < padK - zq_mm_align_size4;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4, B_c_ptr1 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
				}
				for (; k < padK-zq_mm_align_size;
					k += zq_mm_align_size, A_c_ptr += zq_mm_align_size, B_c_ptr1 += zq_mm_align_size)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
				}
				zq_mm_store_ps(q1, sum_vec1);
				for (; k < K; k++)
				{
					a_val = *(A_c_ptr++);
					q1[0] += a_val*(*(B_c_ptr1++));
				}
				*(C_c_ptr++) = zq_final_sum_q1;
			}
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	float a_val;
	int m, n, k;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q1[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q2[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q3[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q4[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type a_vec;

	if (K % zq_mm_align_size8 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			B_c_ptr1;
			B_c_ptr2;
			B_c_ptr3;
			B_c_ptr4;
			for (n = 0; n < N; n += 4, Bptr += ldb*4)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + K;
				Bptr3 = Bptr2 + K;
				Bptr4 = Bptr3 + K;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K;
					k += zq_mm_align_size8, A_c_ptr += zq_mm_align_size8,
					B_c_ptr1 += zq_mm_align_size8, B_c_ptr2 += zq_mm_align_size8,
					B_c_ptr3 += zq_mm_align_size8, B_c_ptr4 += zq_mm_align_size8)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size4);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size4), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size4), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size4), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size5);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size5), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size5), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size5), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size5), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size6);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size6), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size6), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size6), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size6), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size7);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size7), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size7), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size7), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size7), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
		}
	}
	else if (K % zq_mm_align_size4 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N; n += 4, Bptr += ldb*4)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
		}
	}
	else
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N; n += 4, Bptr += ldb*4)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK - zq_mm_align_size4;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
				}
				for (; k < padK-zq_mm_align_size;
					k += zq_mm_align_size, A_c_ptr += zq_mm_align_size,
					B_c_ptr1 += zq_mm_align_size, B_c_ptr2 += zq_mm_align_size,
					B_c_ptr3 += zq_mm_align_size, B_c_ptr4 += zq_mm_align_size)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				for (; k < K; k ++)
				{
					a_val = *(A_c_ptr++);
					q1[0] += a_val*(*(B_c_ptr1++));
					q2[0] += a_val*(*(B_c_ptr2++));
					q3[0] += a_val*(*(B_c_ptr3++));
					q4[0] += a_val*(*(B_c_ptr4++));
				}
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
			}
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	float a_val;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q1[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q2[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q3[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q4[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q5[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q6[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q7[zq_mm_align_size];
	__declspec(align(zq_mm_align_size4)) float q8[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type sum_vec5;
	register zq_mm_type sum_vec6;
	register zq_mm_type sum_vec7;
	register zq_mm_type sum_vec8;
	register zq_mm_type a_vec;

	if (K % zq_mm_align_size8 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N; n += 8, Bptr += ldb*8)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				sum_vec5 = zq_mm_setzero_ps();
				sum_vec6 = zq_mm_setzero_ps();
				sum_vec7 = zq_mm_setzero_ps();
				sum_vec8 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				Bptr5 = Bptr4 + ldb;
				Bptr6 = Bptr5 + ldb;
				Bptr7 = Bptr6 + ldb;
				Bptr8 = Bptr7 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
					B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
					k < K;
					k += zq_mm_align_size8, A_c_ptr += zq_mm_align_size8,
					B_c_ptr1 += zq_mm_align_size8, B_c_ptr2 += zq_mm_align_size8,
					B_c_ptr3 += zq_mm_align_size8, B_c_ptr4 += zq_mm_align_size8,
					B_c_ptr5 += zq_mm_align_size8, B_c_ptr6 += zq_mm_align_size8,
					B_c_ptr7 += zq_mm_align_size8, B_c_ptr8 += zq_mm_align_size8)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size2), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size2), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size2), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size2), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size3), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size3), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size3), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size3), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size4);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size4), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size4), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size4), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size4), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size4), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size4), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size4), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size4), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size5);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size5), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size5), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size5), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size5), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size5), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size5), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size5), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size5), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size6);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size6), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size6), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size6), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size6), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size6), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size6), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size6), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size6), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size7);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size7), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size7), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size7), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size7), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size7), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size7), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size7), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size7), sum_vec8);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				zq_mm_store_ps(q5, sum_vec5);
				zq_mm_store_ps(q6, sum_vec6);
				zq_mm_store_ps(q7, sum_vec7);
				zq_mm_store_ps(q8, sum_vec8);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
				*(C_c_ptr++) = zq_final_sum_q5;
				*(C_c_ptr++) = zq_final_sum_q6;
				*(C_c_ptr++) = zq_final_sum_q7;
				*(C_c_ptr++) = zq_final_sum_q8;
			}
		}
	}
	else if (K % zq_mm_align_size4 == 0)
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N; n += 8, Bptr += ldb*8)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				sum_vec5 = zq_mm_setzero_ps();
				sum_vec6 = zq_mm_setzero_ps();
				sum_vec7 = zq_mm_setzero_ps();
				sum_vec8 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				Bptr5 = Bptr4 + ldb;
				Bptr6 = Bptr5 + ldb;
				Bptr7 = Bptr6 + ldb;
				Bptr8 = Bptr7 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
					B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
					k < K;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4,
					B_c_ptr5 += zq_mm_align_size4, B_c_ptr6 += zq_mm_align_size4,
					B_c_ptr7 += zq_mm_align_size4, B_c_ptr8 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size2), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size2), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size2), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size2), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size3), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size3), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size3), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size3), sum_vec8);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				zq_mm_store_ps(q5, sum_vec5);
				zq_mm_store_ps(q6, sum_vec6);
				zq_mm_store_ps(q7, sum_vec7);
				zq_mm_store_ps(q8, sum_vec8);
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
				*(C_c_ptr++) = zq_final_sum_q5;
				*(C_c_ptr++) = zq_final_sum_q6;
				*(C_c_ptr++) = zq_final_sum_q7;
				*(C_c_ptr++) = zq_final_sum_q8;
			}
		}
	}
	else
	{
		Aptr = A;
		Cptr = C;
		for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
		{
			Bptr = Bt;
			C_c_ptr = Cptr;
			for (n = 0; n < N; n += 8, Bptr += ldb*8)
			{
				sum_vec1 = zq_mm_setzero_ps();
				sum_vec2 = zq_mm_setzero_ps();
				sum_vec3 = zq_mm_setzero_ps();
				sum_vec4 = zq_mm_setzero_ps();
				sum_vec5 = zq_mm_setzero_ps();
				sum_vec6 = zq_mm_setzero_ps();
				sum_vec7 = zq_mm_setzero_ps();
				sum_vec8 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				Bptr2 = Bptr1 + ldb;
				Bptr3 = Bptr2 + ldb;
				Bptr4 = Bptr3 + ldb;
				Bptr5 = Bptr4 + ldb;
				Bptr6 = Bptr5 + ldb;
				Bptr7 = Bptr6 + ldb;
				Bptr8 = Bptr7 + ldb;
				for (k = 0, A_c_ptr = Aptr, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
					B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
					k < padK - zq_mm_align_size4;
					k += zq_mm_align_size4, A_c_ptr += zq_mm_align_size4,
					B_c_ptr1 += zq_mm_align_size4, B_c_ptr2 += zq_mm_align_size4,
					B_c_ptr3 += zq_mm_align_size4, B_c_ptr4 += zq_mm_align_size4,
					B_c_ptr5 += zq_mm_align_size4, B_c_ptr6 += zq_mm_align_size4,
					B_c_ptr7 += zq_mm_align_size4, B_c_ptr8 += zq_mm_align_size4)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size2);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size2), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size2), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size2), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size2), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size2), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size2), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size2), sum_vec8);
					a_vec = zq_mm_load_ps(A_c_ptr + zq_mm_align_size3);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1 + zq_mm_align_size3), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2 + zq_mm_align_size3), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3 + zq_mm_align_size3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4 + zq_mm_align_size3), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5 + zq_mm_align_size3), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6 + zq_mm_align_size3), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7 + zq_mm_align_size3), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8 + zq_mm_align_size3), sum_vec8);
				}
				for (; k < padK-zq_mm_align_size;
					k += zq_mm_align_size, A_c_ptr += zq_mm_align_size,
					B_c_ptr1 += zq_mm_align_size, B_c_ptr2 += zq_mm_align_size,
					B_c_ptr3 += zq_mm_align_size, B_c_ptr4 += zq_mm_align_size,
					B_c_ptr5 += zq_mm_align_size, B_c_ptr6 += zq_mm_align_size,
					B_c_ptr7 += zq_mm_align_size, B_c_ptr8 += zq_mm_align_size)
				{
					a_vec = zq_mm_load_ps(A_c_ptr);
					sum_vec1 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr1), sum_vec1);
					sum_vec2 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr2), sum_vec2);
					sum_vec3 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr3), sum_vec3);
					sum_vec4 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr4), sum_vec4);
					sum_vec5 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr5), sum_vec5);
					sum_vec6 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr6), sum_vec6);
					sum_vec7 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr7), sum_vec7);
					sum_vec8 = zq_mm_fmadd_ps(a_vec, zq_mm_load_ps(B_c_ptr8), sum_vec8);
				}
				zq_mm_store_ps(q1, sum_vec1);
				zq_mm_store_ps(q2, sum_vec2);
				zq_mm_store_ps(q3, sum_vec3);
				zq_mm_store_ps(q4, sum_vec4);
				zq_mm_store_ps(q5, sum_vec5);
				zq_mm_store_ps(q6, sum_vec6);
				zq_mm_store_ps(q7, sum_vec7);
				zq_mm_store_ps(q8, sum_vec8);
				for (; k < K;k ++)
				{
					a_val = *(A_c_ptr++);
					q1[0] += a_val*(*(B_c_ptr1++));
					q2[0] += a_val*(*(B_c_ptr2++));
					q3[0] += a_val*(*(B_c_ptr3++));
					q4[0] += a_val*(*(B_c_ptr4++));
					q5[0] += a_val*(*(B_c_ptr5++));
					q6[0] += a_val*(*(B_c_ptr6++));
					q7[0] += a_val*(*(B_c_ptr7++));
					q8[0] += a_val*(*(B_c_ptr8++));
				}
				*(C_c_ptr++) = zq_final_sum_q1;
				*(C_c_ptr++) = zq_final_sum_q2;
				*(C_c_ptr++) = zq_final_sum_q3;
				*(C_c_ptr++) = zq_final_sum_q4;
				*(C_c_ptr++) = zq_final_sum_q5;
				*(C_c_ptr++) = zq_final_sum_q6;
				*(C_c_ptr++) = zq_final_sum_q7;
				*(C_c_ptr++) = zq_final_sum_q8;
			}
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	if (N % 8 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv8(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else if (N % 4 == 0)
	{
		zq_gemm_32f_align_AnoTrans_Btrans_caseNdiv4(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else
	{
		zq_gemm_32f_align_AnoTrans_Btrans_caseNgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
}