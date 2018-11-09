

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	float a_val;
	int m, n, k;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
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
				zq_mm_store_ps(q, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec2);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec3);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q;
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
				zq_mm_store_ps(q, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q;
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
				zq_mm_store_ps(q, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec2);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec3);
				*(C_c_ptr++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec4);
				*(C_c_ptr++) = zq_final_sum_q;
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
				zq_mm_store_ps(q, sum_vec1);
				*(C_c_ptr++) = zq_final_sum_q;
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
				zq_mm_store_ps(q, sum_vec1);
				*(C_c_ptr) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec2);
				*(C_c_ptr+1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec3);
				*(C_c_ptr+2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec4);
				*(C_c_ptr+3) = zq_final_sum_q;
				for (; k < K; k ++)
				{
					a_val = *(A_c_ptr++);
					*(C_c_ptr) += a_val*(*(B_c_ptr1++));
					*(C_c_ptr+1) += a_val*(*(B_c_ptr2++));
					*(C_c_ptr+2) += a_val*(*(B_c_ptr3++));
					*(C_c_ptr+3) += a_val*(*(B_c_ptr4++));
				}

				C_c_ptr += 4;
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
				zq_mm_store_ps(q, sum_vec1);
				for (; k < K; k++)
				{
					a_val = *(A_c_ptr++);
					q[0] += a_val*(*(B_c_ptr1++));
				}
				*(C_c_ptr++) = zq_final_sum_q;
			}
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float* Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float a1,a2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb4 = ldb * 4;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1,a_vec2,b_vec;

	if (K % zq_mm_align_size8 == 0)
	{
		Aptr1 = A;
		Aptr2 = Aptr1 + lda;
		Cptr1 = C;
		Cptr2 = Cptr1 + ldc;
		for (m = 0; m < M-1; m+=2, Aptr1 += lda2, Aptr2+=lda2, Cptr1 += ldc2,Cptr2+=ldc2)
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, 
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size8)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				sum_vec21 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
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
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size8)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
		}
	}
	else if (K % zq_mm_align_size4 == 0)
	{
		Aptr1 = A;
		Aptr2 = Aptr1 + lda;
		Cptr1 = C;
		Cptr2 = Cptr1 + ldc;
		for (m = 0; m < M-1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				sum_vec21 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
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
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
		}
	}
	else
	{
		Aptr1 = A;
		Aptr2 = Aptr1 + lda;
		Cptr1 = C;
		Cptr2 = Cptr1 + ldc;
		for (m = 0; m < M-1; m+=2, Aptr1 += lda2,Aptr2+=lda2, Cptr1 += ldc2,Cptr2+=ldc2)
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1 + 1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1 + 2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1 + 3) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2 + 1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2 + 2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2 + 3) = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					a2 = *(A_c_ptr2++);
					*(C_c_ptr1) += a1*(*B_c_ptr1);
					*(C_c_ptr1 + 1) += a1*(*B_c_ptr2);
					*(C_c_ptr1 + 2) += a1*(*B_c_ptr3);
					*(C_c_ptr1 + 3) += a1*(*B_c_ptr4);
					*(C_c_ptr2) += a2*(*(B_c_ptr1++));
					*(C_c_ptr2 + 1) += a2*(*(B_c_ptr2++));
					*(C_c_ptr2 + 2) += a2*(*(B_c_ptr3++));
					*(C_c_ptr2 + 3) += a2*(*(B_c_ptr4++));
				}

				C_c_ptr1 += 4;
				C_c_ptr2 += 4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, B_c_ptr1 = Bptr1;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*C_c_ptr1 = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*C_c_ptr2 = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					a2 = *(A_c_ptr2++);
					*C_c_ptr1 += a1*(*B_c_ptr1);
					*C_c_ptr2 += a2*(*(B_c_ptr1++));
				}
				C_c_ptr1++;
				C_c_ptr2++;
			}
		}

		for (; m < M; m ++, Aptr1 += lda, Cptr1 += ldc)
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
				for (k = 0, A_c_ptr1 = Aptr1, 
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1 + 1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1 + 2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1 + 3) = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					*(C_c_ptr1) += a1*(*(B_c_ptr1++));
					*(C_c_ptr1 + 1) += a1*(*(B_c_ptr2++));
					*(C_c_ptr1 + 2) += a1*(*(B_c_ptr3++));
					*(C_c_ptr1 + 3) += a1*(*(B_c_ptr4++));
				}
				C_c_ptr1 += 4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*C_c_ptr1 = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					*C_c_ptr1 += a1*(*(B_c_ptr1++));
				}
				C_c_ptr1++;
			}
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const float* Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	float a1, a2, a3, a4;
	int m, n, k;
	int lda2 = lda + lda;
	int lda4 = lda2 + lda2;
	int ldc2 = ldc + ldc;
	int ldc4 = ldc2 + ldc2;
	int ldb4 = ldb * 4;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, b_vec;

	if (K % zq_mm_align_size8 == 0)
	{
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size8)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec32);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec33);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec34);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec42);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec43);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec44);
				*(C_c_ptr4++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				sum_vec21 = zq_mm_setzero_ps();
				sum_vec31 = zq_mm_setzero_ps();
				sum_vec41 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
					A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*(C_c_ptr4++) = zq_final_sum_q;
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
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size8)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
			}

			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
		}
	}
	else if (K % zq_mm_align_size4 == 0)
	{
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec32);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec33);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec34);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec42);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec43);
				*(C_c_ptr4++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec44);
				*(C_c_ptr4++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				sum_vec21 = zq_mm_setzero_ps();
				sum_vec31 = zq_mm_setzero_ps();
				sum_vec41 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
					A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*(C_c_ptr3++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*(C_c_ptr4++) = zq_final_sum_q;
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
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < K; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1++) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < K;	k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1++) = zq_final_sum_q;
			}
		}
	}
	else
	{
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK-zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}

				for (;	k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
					sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
					sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
					sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
					sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
					sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
					sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}

				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1+1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1+2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1+3) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*(C_c_ptr2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec22);
				*(C_c_ptr2+1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec23);
				*(C_c_ptr2+2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec24);
				*(C_c_ptr2+3) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*(C_c_ptr3) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec32);
				*(C_c_ptr3+1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec33);
				*(C_c_ptr3+2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec34);
				*(C_c_ptr3+3) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*(C_c_ptr4) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec42);
				*(C_c_ptr4+1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec43);
				*(C_c_ptr4+2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec44);
				*(C_c_ptr4+3) = zq_final_sum_q;

				for (; k < K; k ++)
				{
					a1 = *(A_c_ptr1++);
					a2 = *(A_c_ptr2++);
					a3 = *(A_c_ptr3++);
					a4 = *(A_c_ptr4++);
					*(C_c_ptr1) += a1*(*B_c_ptr1);
					*(C_c_ptr1+1) += a1*(*B_c_ptr2);
					*(C_c_ptr1+2) += a1*(*B_c_ptr3);
					*(C_c_ptr1+3) += a1*(*B_c_ptr4);
					*(C_c_ptr2) += a2*(*B_c_ptr1);
					*(C_c_ptr2 + 1) += a2*(*B_c_ptr2);
					*(C_c_ptr2 + 2) += a2*(*B_c_ptr3);
					*(C_c_ptr2 + 3) += a2*(*B_c_ptr4);
					*(C_c_ptr3) += a3*(*B_c_ptr1);
					*(C_c_ptr3 + 1) += a3*(*B_c_ptr2);
					*(C_c_ptr3 + 2) += a3*(*B_c_ptr3);
					*(C_c_ptr3 + 3) += a3*(*B_c_ptr4);
					*(C_c_ptr4) += a4*(*(B_c_ptr1++));
					*(C_c_ptr4 + 1) += a4*(*(B_c_ptr2++));
					*(C_c_ptr4 + 2) += a4*(*(B_c_ptr3++));
					*(C_c_ptr4 + 3) += a4*(*(B_c_ptr4++));
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
				for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
					B_c_ptr1 = Bptr1;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}

				for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					a_vec2 = zq_mm_load_ps(A_c_ptr2);
					a_vec3 = zq_mm_load_ps(A_c_ptr3);
					a_vec4 = zq_mm_load_ps(A_c_ptr4);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
					sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
					sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
					A_c_ptr1 += zq_mm_align_size;
					A_c_ptr2 += zq_mm_align_size;
					A_c_ptr3 += zq_mm_align_size;
					A_c_ptr4 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}

				zq_mm_store_ps(q, sum_vec11);
				*C_c_ptr1 = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec21);
				*C_c_ptr2 = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec31);
				*C_c_ptr3 = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec41);
				*C_c_ptr4 = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					a2 = *(A_c_ptr2++);
					a3 = *(A_c_ptr3++);
					a4 = *(A_c_ptr4++);
					*C_c_ptr1 += a1*(*B_c_ptr1);
					*C_c_ptr2 += a2*(*B_c_ptr1);
					*C_c_ptr3 += a3*(*B_c_ptr1);
					*C_c_ptr4 += a4*(*(B_c_ptr1++));
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
				for (k = 0, A_c_ptr1 = Aptr1,
					B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					b_vec = zq_mm_load_ps(B_c_ptr2);
					sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
					b_vec = zq_mm_load_ps(B_c_ptr3);
					sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
					b_vec = zq_mm_load_ps(B_c_ptr4);
					sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
					B_c_ptr2 += zq_mm_align_size;
					B_c_ptr3 += zq_mm_align_size;
					B_c_ptr4 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*(C_c_ptr1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec12);
				*(C_c_ptr1 + 1) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec13);
				*(C_c_ptr1 + 2) = zq_final_sum_q;
				zq_mm_store_ps(q, sum_vec14);
				*(C_c_ptr1 + 3) = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					*(C_c_ptr1) += a1*(*(B_c_ptr1++));
					*(C_c_ptr1 + 1) += a1*(*(B_c_ptr2++));
					*(C_c_ptr1 + 2) += a1*(*(B_c_ptr3++));
					*(C_c_ptr1 + 3) += a1*(*(B_c_ptr4++));
				}
				C_c_ptr1 += 4;
			}
			for (; n < N; n++, Bptr += ldb)
			{
				sum_vec11 = zq_mm_setzero_ps();
				Bptr1 = Bptr;
				for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1;
					k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;

					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				for (; k < padK - zq_mm_align_size; k += zq_mm_align_size)
				{
					a_vec1 = zq_mm_load_ps(A_c_ptr1);
					b_vec = zq_mm_load_ps(B_c_ptr1);
					sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
					A_c_ptr1 += zq_mm_align_size;
					B_c_ptr1 += zq_mm_align_size;
				}
				zq_mm_store_ps(q, sum_vec11);
				*C_c_ptr1 = zq_final_sum_q;
				for (; k < K; k++)
				{
					a1 = *(A_c_ptr1++);
					*C_c_ptr1 += a1*(*(B_c_ptr1++));
				}
				C_c_ptr1++;
			}
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type a_vec,b_vec;

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
		for (n = 0; n < N; n += 4, Bptr += ldb4)
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
				k < K;	k += zq_mm_align_size8)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M-1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}

			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}

			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda + lda;
	int lda4 = lda2 + lda2;
	int ldc2 = ldc + ldc;
	int ldc4 = ldc2 + ldc2;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, b_vec;

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
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps(); sum_vec33 = zq_mm_setzero_ps(); sum_vec43 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps(); sum_vec34 = zq_mm_setzero_ps(); sum_vec44 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec31);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec32);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec33);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec34);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec41);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec42);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec43);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec44);
			*(C_c_ptr4++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}



void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type a_vec, b_vec;

	Aptr = A;
	Cptr = C;
	for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
	{
		Bptr = Bt;
		C_c_ptr = Cptr;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
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
				k < K;	k += zq_mm_align_size4)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M-1; m += 2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda + lda;
	int lda4 = lda2 + lda2;
	int ldc2 = ldc + ldc;
	int ldc4 = ldc2 + ldc2;
	int ldb4 = ldb << 2;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, b_vec;

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
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps(); sum_vec33 = zq_mm_setzero_ps(); sum_vec43 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps(); sum_vec34 = zq_mm_setzero_ps(); sum_vec44 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec31);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec32);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec33);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec34);
			*(C_c_ptr3++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec41);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec42);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec43);
			*(C_c_ptr4++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec44);
			*(C_c_ptr4++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < K; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr, *C_c_ptr;
	float a_val;
	int m, n, k;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type a_vec, b_vec;

	Aptr = A;
	Cptr = C;
	for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
	{
		Bptr = Bt;
		C_c_ptr = Cptr;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
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
				k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a_val = *(A_c_ptr++);
				*(C_c_ptr) += a_val*(*(B_c_ptr1++));
				*(C_c_ptr + 1) += a_val*(*(B_c_ptr2++));
				*(C_c_ptr + 2) += a_val*(*(B_c_ptr3++));
				*(C_c_ptr + 3) += a_val*(*(B_c_ptr4++));
			}
			C_c_ptr += 4;
		}
	}

}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	float a1,a2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M-1; m+=2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 4, Bptr += ldb4)
		{
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, 
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1*(*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1*(*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1*(*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1*(*B_c_ptr4);
				*(C_c_ptr2) += a2*(*(B_c_ptr1++));
				*(C_c_ptr2 + 1) += a2*(*(B_c_ptr2++));
				*(C_c_ptr2 + 2) += a2*(*(B_c_ptr3++));
				*(C_c_ptr2 + 3) += a2*(*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1*(*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1*(*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1*(*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1*(*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *Aptr3, *Aptr4, *A_c_ptr1, *A_c_ptr2, *A_c_ptr3, *A_c_ptr4;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4;
	float* Cptr1, *Cptr2, *Cptr3, *Cptr4, *C_c_ptr1, *C_c_ptr2, *C_c_ptr3, *C_c_ptr4;
	int m, n, k;
	int lda2 = lda + lda;
	int lda4 = lda2 + lda2;
	int ldc2 = ldc + ldc;
	int ldc4 = ldc2 + ldc2;
	int ldb4 = ldb << 2;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21, sum_vec31, sum_vec41;
	register zq_mm_type sum_vec12, sum_vec22, sum_vec32, sum_vec42;
	register zq_mm_type sum_vec13, sum_vec23, sum_vec33, sum_vec43;
	register zq_mm_type sum_vec14, sum_vec24, sum_vec34, sum_vec44;
	register zq_mm_type a_vec1, a_vec2, a_vec3, a_vec4, b_vec;
	float a1, a2, a3, a4;

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
			sum_vec11 = zq_mm_setzero_ps(); sum_vec21 = zq_mm_setzero_ps(); sum_vec31 = zq_mm_setzero_ps(); sum_vec41 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps(); sum_vec22 = zq_mm_setzero_ps(); sum_vec32 = zq_mm_setzero_ps(); sum_vec42 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps(); sum_vec23 = zq_mm_setzero_ps(); sum_vec33 = zq_mm_setzero_ps(); sum_vec43 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps(); sum_vec24 = zq_mm_setzero_ps(); sum_vec34 = zq_mm_setzero_ps(); sum_vec44 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, A_c_ptr3 = Aptr3, A_c_ptr4 = Aptr4,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				a_vec3 = zq_mm_load_ps(A_c_ptr3);
				a_vec4 = zq_mm_load_ps(A_c_ptr4);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				sum_vec31 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec31);
				sum_vec41 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec41);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				sum_vec32 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec32);
				sum_vec42 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec42);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				sum_vec33 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec33);
				sum_vec43 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec43);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				sum_vec34 = zq_mm_fmadd_ps(a_vec3, b_vec, sum_vec34);
				sum_vec44 = zq_mm_fmadd_ps(a_vec4, b_vec, sum_vec44);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				A_c_ptr3 += zq_mm_align_size;
				A_c_ptr4 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec31);
			*(C_c_ptr3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec32);
			*(C_c_ptr3 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec33);
			*(C_c_ptr3 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec34);
			*(C_c_ptr3 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec41);
			*(C_c_ptr4) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec42);
			*(C_c_ptr4 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec43);
			*(C_c_ptr4 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec44);
			*(C_c_ptr4 + 3) = zq_final_sum_q;

			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				a3 = *(A_c_ptr3++);
				a4 = *(A_c_ptr4++);
				*(C_c_ptr1) += a1*(*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1*(*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1*(*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1*(*B_c_ptr4);
				*(C_c_ptr2) += a2*(*B_c_ptr1);
				*(C_c_ptr2 + 1) += a2*(*B_c_ptr2);
				*(C_c_ptr2 + 2) += a2*(*B_c_ptr3);
				*(C_c_ptr2 + 3) += a2*(*B_c_ptr4);
				*(C_c_ptr3) += a3*(*B_c_ptr1);
				*(C_c_ptr3 + 1) += a3*(*B_c_ptr2);
				*(C_c_ptr3 + 2) += a3*(*B_c_ptr3);
				*(C_c_ptr3 + 3) += a3*(*B_c_ptr4);
				*(C_c_ptr4) += a4*(*(B_c_ptr1++));
				*(C_c_ptr4 + 1) += a4*(*(B_c_ptr2++));
				*(C_c_ptr4 + 2) += a4*(*(B_c_ptr3++));
				*(C_c_ptr4 + 3) += a4*(*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
			C_c_ptr2 += 4;
			C_c_ptr3 += 4;
			C_c_ptr4 += 4;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 4, Bptr += ldb * 4)
		{
			sum_vec11 = zq_mm_setzero_ps();
			sum_vec12 = zq_mm_setzero_ps();
			sum_vec13 = zq_mm_setzero_ps();
			sum_vec14 = zq_mm_setzero_ps();
			Bptr1 = Bptr;
			Bptr2 = Bptr1 + ldb;
			Bptr3 = Bptr2 + ldb;
			Bptr4 = Bptr3 + ldb;
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4;
				k < padK - zq_mm_align_size4; k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1*(*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1*(*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1*(*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1*(*(B_c_ptr4++));
			}
			C_c_ptr1 += 4;
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type sum_vec5;
	register zq_mm_type sum_vec6;
	register zq_mm_type sum_vec7;
	register zq_mm_type sum_vec8;
	register zq_mm_type a_vec, b_vec;

	Aptr = A;
	Cptr = C;
	for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
	{
		Bptr = Bt;
		C_c_ptr = Cptr;
		for (n = 0; n < N; n += 8, Bptr += ldb * 8)
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
				k < K;	k += zq_mm_align_size8)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec5);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec6);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec7);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec8);
			*(C_c_ptr++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb8 = ldb * 8;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m+=2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
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
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < K;	k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec25);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec26);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec27);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec28);
			*(C_c_ptr2++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
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
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < K;	k += zq_mm_align_size8)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
		
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type sum_vec5;
	register zq_mm_type sum_vec6;
	register zq_mm_type sum_vec7;
	register zq_mm_type sum_vec8;
	register zq_mm_type a_vec, b_vec;


	Aptr = A;
	Cptr = C;
	for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
	{
		Bptr = Bt;
		C_c_ptr = Cptr;
		for (n = 0; n < N; n += 8, Bptr += ldb * 8)
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
				k < K; k += zq_mm_align_size4)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec5);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec6);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec7);
			*(C_c_ptr++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec8);
			*(C_c_ptr++) = zq_final_sum_q;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int ldb8 = ldb * 8;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M - 1; m+=2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
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
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2,
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < K;	k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec25);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec26);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec27);
			*(C_c_ptr2++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec28);
			*(C_c_ptr2++) = zq_final_sum_q;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb8)
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
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < K;	k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1++) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1++) = zq_final_sum_q;
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr, *A_c_ptr, *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr, *C_c_ptr;
	int m, n, k;
	float a_val;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec1;
	register zq_mm_type sum_vec2;
	register zq_mm_type sum_vec3;
	register zq_mm_type sum_vec4;
	register zq_mm_type sum_vec5;
	register zq_mm_type sum_vec6;
	register zq_mm_type sum_vec7;
	register zq_mm_type sum_vec8;
	register zq_mm_type a_vec, b_vec;

	Aptr = A;
	Cptr = C;
	for (m = 0; m < M; m++, Aptr += lda, Cptr += ldc)
	{
		Bptr = Bt;
		C_c_ptr = Cptr;
		for (n = 0; n < N; n += 8, Bptr += ldb * 8)
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
				k < padK - zq_mm_align_size4;	k += zq_mm_align_size4)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec = zq_mm_load_ps(A_c_ptr);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec1 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec1);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec2 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec2);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec3 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec3);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec4 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec4);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec5 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec5);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec6 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec6);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec7 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec7);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec8 = zq_mm_fmadd_ps(a_vec, b_vec, sum_vec8);
				A_c_ptr += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec1);
			*(C_c_ptr) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec2);
			*(C_c_ptr+1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec3);
			*(C_c_ptr+2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec4);
			*(C_c_ptr+3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec5);
			*(C_c_ptr+4) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec6);
			*(C_c_ptr+5) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec7);
			*(C_c_ptr+6) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec8);
			*(C_c_ptr+7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a_val = *(A_c_ptr++);
				*(C_c_ptr) += a_val*(*(B_c_ptr1++));
				*(C_c_ptr+1) += a_val*(*(B_c_ptr2++));
				*(C_c_ptr+2) += a_val*(*(B_c_ptr3++));
				*(C_c_ptr+3) += a_val*(*(B_c_ptr4++));
				*(C_c_ptr+4) += a_val*(*(B_c_ptr5++));
				*(C_c_ptr+5) += a_val*(*(B_c_ptr6++));
				*(C_c_ptr+6) += a_val*(*(B_c_ptr7++));
				*(C_c_ptr+7) += a_val*(*(B_c_ptr8++));
			}
			C_c_ptr += 8;
		}
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	const float* Aptr1, *Aptr2, *A_c_ptr1, *A_c_ptr2;
	const float *Bptr, *Bptr1, *Bptr2, *Bptr3, *Bptr4, *Bptr5, *Bptr6, *Bptr7, *Bptr8;
	const float *B_c_ptr1, *B_c_ptr2, *B_c_ptr3, *B_c_ptr4, *B_c_ptr5, *B_c_ptr6, *B_c_ptr7, *B_c_ptr8;
	float* Cptr1, *Cptr2, *C_c_ptr1, *C_c_ptr2;
	int m, n, k;
	float a1,a2;
	int lda2 = lda + lda;
	int ldc2 = ldc + ldc;
	int padK = (K + zq_mm_align_size - 1) / zq_mm_align_size*zq_mm_align_size;
	__declspec(align(zq_mm_align_size4)) float q[zq_mm_align_size];
	register zq_mm_type sum_vec11, sum_vec21;
	register zq_mm_type sum_vec12, sum_vec22;
	register zq_mm_type sum_vec13, sum_vec23;
	register zq_mm_type sum_vec14, sum_vec24;
	register zq_mm_type sum_vec15, sum_vec25;
	register zq_mm_type sum_vec16, sum_vec26;
	register zq_mm_type sum_vec17, sum_vec27;
	register zq_mm_type sum_vec18, sum_vec28;
	register zq_mm_type a_vec1, a_vec2, b_vec;

	Aptr1 = A;
	Aptr2 = Aptr1 + lda;
	Cptr1 = C;
	Cptr2 = Cptr1 + ldc;
	for (m = 0; m < M-1; m+=2, Aptr1 += lda2, Aptr2 += lda2, Cptr1 += ldc2, Cptr2 += ldc2)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		C_c_ptr2 = Cptr2;
		for (n = 0; n < N; n += 8, Bptr += ldb * 8)
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
			for (k = 0, A_c_ptr1 = Aptr1, A_c_ptr2 = Aptr2, 
				B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < padK - zq_mm_align_size4;	k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				a_vec2 = zq_mm_load_ps(A_c_ptr2);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				sum_vec21 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec21);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				sum_vec22 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec22);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				sum_vec23 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec23);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				sum_vec24 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec24);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				sum_vec25 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec25);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				sum_vec26 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec26);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				sum_vec27 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec27);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				sum_vec28 = zq_mm_fmadd_ps(a_vec2, b_vec, sum_vec28);
				A_c_ptr1 += zq_mm_align_size;
				A_c_ptr2 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1 + 4) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1 + 5) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1 + 6) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1 + 7) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec21);
			*(C_c_ptr2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec22);
			*(C_c_ptr2 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec23);
			*(C_c_ptr2 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec24);
			*(C_c_ptr2 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec25);
			*(C_c_ptr2 + 4) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec26);
			*(C_c_ptr2 + 5) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec27);
			*(C_c_ptr2 + 6) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec28);
			*(C_c_ptr2 + 7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				a2 = *(A_c_ptr2++);
				*(C_c_ptr1) += a1*(*B_c_ptr1);
				*(C_c_ptr1 + 1) += a1*(*B_c_ptr2);
				*(C_c_ptr1 + 2) += a1*(*B_c_ptr3);
				*(C_c_ptr1 + 3) += a1*(*B_c_ptr4);
				*(C_c_ptr1 + 4) += a1*(*B_c_ptr5);
				*(C_c_ptr1 + 5) += a1*(*B_c_ptr6);
				*(C_c_ptr1 + 6) += a1*(*B_c_ptr7);
				*(C_c_ptr1 + 7) += a1*(*B_c_ptr8);
				*(C_c_ptr2) += a2*(*(B_c_ptr1++));
				*(C_c_ptr2 + 1) += a2*(*(B_c_ptr2++));
				*(C_c_ptr2 + 2) += a2*(*(B_c_ptr3++));
				*(C_c_ptr2 + 3) += a2*(*(B_c_ptr4++));
				*(C_c_ptr2 + 4) += a2*(*(B_c_ptr5++));
				*(C_c_ptr2 + 5) += a2*(*(B_c_ptr6++));
				*(C_c_ptr2 + 6) += a2*(*(B_c_ptr7++));
				*(C_c_ptr2 + 7) += a2*(*(B_c_ptr8++));
			}
			C_c_ptr1 += 8;
			C_c_ptr2 += 8;
		}
	}

	for (; m < M; m++, Aptr1 += lda, Cptr1 += ldc)
	{
		Bptr = Bt;
		C_c_ptr1 = Cptr1;
		for (n = 0; n < N; n += 8, Bptr += ldb * 8)
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
			for (k = 0, A_c_ptr1 = Aptr1, B_c_ptr1 = Bptr1, B_c_ptr2 = Bptr2, B_c_ptr3 = Bptr3, B_c_ptr4 = Bptr4,
				B_c_ptr5 = Bptr5, B_c_ptr6 = Bptr6, B_c_ptr7 = Bptr7, B_c_ptr8 = Bptr8;
				k < padK - zq_mm_align_size4;	k += zq_mm_align_size4)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;

				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			for (; k < padK - zq_mm_align_size;
				k += zq_mm_align_size)
			{
				a_vec1 = zq_mm_load_ps(A_c_ptr1);
				b_vec = zq_mm_load_ps(B_c_ptr1);
				sum_vec11 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec11);
				b_vec = zq_mm_load_ps(B_c_ptr2);
				sum_vec12 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec12);
				b_vec = zq_mm_load_ps(B_c_ptr3);
				sum_vec13 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec13);
				b_vec = zq_mm_load_ps(B_c_ptr4);
				sum_vec14 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec14);
				b_vec = zq_mm_load_ps(B_c_ptr5);
				sum_vec15 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec15);
				b_vec = zq_mm_load_ps(B_c_ptr6);
				sum_vec16 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec16);
				b_vec = zq_mm_load_ps(B_c_ptr7);
				sum_vec17 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec17);
				b_vec = zq_mm_load_ps(B_c_ptr8);
				sum_vec18 = zq_mm_fmadd_ps(a_vec1, b_vec, sum_vec18);
				A_c_ptr1 += zq_mm_align_size;
				B_c_ptr1 += zq_mm_align_size;
				B_c_ptr2 += zq_mm_align_size;
				B_c_ptr3 += zq_mm_align_size;
				B_c_ptr4 += zq_mm_align_size;
				B_c_ptr5 += zq_mm_align_size;
				B_c_ptr6 += zq_mm_align_size;
				B_c_ptr7 += zq_mm_align_size;
				B_c_ptr8 += zq_mm_align_size;
			}
			zq_mm_store_ps(q, sum_vec11);
			*(C_c_ptr1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec12);
			*(C_c_ptr1 + 1) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec13);
			*(C_c_ptr1 + 2) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec14);
			*(C_c_ptr1 + 3) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec15);
			*(C_c_ptr1 + 4) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec16);
			*(C_c_ptr1 + 5) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec17);
			*(C_c_ptr1 + 6) = zq_final_sum_q;
			zq_mm_store_ps(q, sum_vec18);
			*(C_c_ptr1 + 7) = zq_final_sum_q;
			for (; k < K; k++)
			{
				a1 = *(A_c_ptr1++);
				*(C_c_ptr1) += a1*(*(B_c_ptr1++));
				*(C_c_ptr1 + 1) += a1*(*(B_c_ptr2++));
				*(C_c_ptr1 + 2) += a1*(*(B_c_ptr3++));
				*(C_c_ptr1 + 3) += a1*(*(B_c_ptr4++));
				*(C_c_ptr1 + 4) += a1*(*(B_c_ptr5++));
				*(C_c_ptr1 + 5) += a1*(*(B_c_ptr6++));
				*(C_c_ptr1 + 6) += a1*(*(B_c_ptr7++));
				*(C_c_ptr1 + 7) += a1*(*(B_c_ptr8++));
			}
			C_c_ptr1 += 8;
		}
	}
}


void zq_gemm_32f_align_AnoTrans_Btrans_M1(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	if (N % 8 == 0)
	{
		if (K %zq_mm_align_size8 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
		else if(K %zq_mm_align_size4 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
		else
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv8_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else if (N % 4 == 0)
	{
		if (K %zq_mm_align_size8 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
		else if (K %zq_mm_align_size4 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
		else
			zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNdiv4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M1_caseNgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M2(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	if (N % 8 == 0)
	{
		if (K %zq_mm_align_size8 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
		else if (K %zq_mm_align_size4 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
		else
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv8_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else if (N % 4 == 0)
	{
		if (K %zq_mm_align_size8 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
		else if (K %zq_mm_align_size4 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
		else
			zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNdiv4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M2_caseNgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
}

void zq_gemm_32f_align_AnoTrans_Btrans_M4(int M, int N, int K, const float* A, int lda, const float* Bt, int ldb, float* C, int ldc)
{
	if (N % 4 == 0)
	{
		if (K %zq_mm_align_size8 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign8(M, N, K, A, lda, Bt, ldb, C, ldc);
		else if (K %zq_mm_align_size4 == 0)
			zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_KdivAlign4(M, N, K, A, lda, Bt, ldb, C, ldc);
		else
			zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNdiv4_Kgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
	else
	{
		zq_gemm_32f_align_AnoTrans_Btrans_M4_caseNgeneral(M, N, K, A, lda, Bt, ldb, C, ldc);
		return;
	}
}