/*******************************************************************************
* Copyright (c) 1999-2018, Intel Corporation
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*     * Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Intel Corporation nor the names of its contributors
*       may be used to endorse or promote products derived from this software
*       without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*******************************************************************************/

/*
!  Content:
!     Intel(R) Math Kernel Library (Intel(R) MKL) interface for service routines
!******************************************************************************/

#ifndef _MKL_SERVICE_H_
#define _MKL_SERVICE_H_

#include <stdlib.h>
#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if !defined(MKL_CALL_CONV)
#   if defined(__MIC__) || defined(__TARGET_ARCH_MIC)
#       define MKL_CALL_CONV
#   else
#       if defined(MKL_STDCALL)
#           define MKL_CALL_CONV __stdcall
#       else
#           define MKL_CALL_CONV __cdecl
#       endif
#   endif
#endif

#if !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)    extern rtype MKL_CALL_CONV name arg;
#endif

_Mkl_Api(void,MKL_Get_Version,(MKLVersion *ver)) /* Returns information about the version of the Intel(R) MKL software */
#define mkl_get_version             MKL_Get_Version

_Mkl_Api(void,MKL_Get_Version_String,(char * buffer, int len)) /* Returns a string that contains Intel(R) MKL version information */
#define mkl_get_version_string      MKL_Get_Version_String

_Mkl_Api(void,MKL_Free_Buffers,(void)) /* Frees the memory allocated by the Intel(R) MKL Memory Manager */
#define mkl_free_buffers            MKL_Free_Buffers

_Mkl_Api(void,MKL_Thread_Free_Buffers,(void)) /* Frees the memory allocated by the Intel(R) MKL Memory Manager in the current thread only */
#define mkl_thread_free_buffers     MKL_Thread_Free_Buffers

_Mkl_Api(MKL_INT64,MKL_Mem_Stat,(int* nbuffers)) /* Intel(R) MKL Memory Manager statistical information. */
                                                 /* Returns an amount of memory, allocated by the Intel(R) MKL Memory Manager */
                                                 /* in <nbuffers> buffers. */
#define mkl_mem_stat                MKL_Mem_Stat

#define  MKL_PEAK_MEM_DISABLE       0
#define  MKL_PEAK_MEM_ENABLE        1
#define  MKL_PEAK_MEM_RESET        -1
#define  MKL_PEAK_MEM               2
_Mkl_Api(MKL_INT64,MKL_Peak_Mem_Usage,(int reset))    /* Returns the peak amount of memory, allocated by the Intel(R) MKL Memory Manager */
#define mkl_peak_mem_usage          MKL_Peak_Mem_Usage

_Mkl_Api(void*,MKL_malloc,(size_t size, int align)) /* Allocates the aligned buffer */
#define mkl_malloc                  MKL_malloc

_Mkl_Api(void*,MKL_calloc,(size_t num, size_t size, int align)) /* Allocates the aligned num*size - bytes memory buffer initialized by zeros */
#define mkl_calloc                  MKL_calloc

_Mkl_Api(void*,MKL_realloc,(void *ptr, size_t size)) /* Changes the size of memory buffer allocated by MKL_malloc/MKL_calloc */
#define mkl_realloc                  MKL_realloc

_Mkl_Api(void,MKL_free,(void *ptr))                 /* Frees the memory allocated by MKL_malloc() */
#define mkl_free                    MKL_free

_Mkl_Api(int,MKL_Disable_Fast_MM,(void))            /* Turns off the Intel(R) MKL Memory Manager */
#define  mkl_disable_fast_mm        MKL_Disable_Fast_MM

_Mkl_Api(void,MKL_Get_Cpu_Clocks,(unsigned MKL_INT64 *)) /* Gets CPU clocks */
#define mkl_get_cpu_clocks          MKL_Get_Cpu_Clocks

_Mkl_Api(double,MKL_Get_Cpu_Frequency,(void)) /* Gets CPU frequency in GHz */
#define mkl_get_cpu_frequency       MKL_Get_Cpu_Frequency

_Mkl_Api(double,MKL_Get_Max_Cpu_Frequency,(void)) /* Gets max CPU frequency in GHz */
#define mkl_get_max_cpu_frequency   MKL_Get_Max_Cpu_Frequency

_Mkl_Api(double,MKL_Get_Clocks_Frequency,(void)) /* Gets clocks frequency in GHz */
#define mkl_get_clocks_frequency    MKL_Get_Clocks_Frequency

_Mkl_Api(int,MKL_Set_Num_Threads_Local,(int nth))
#define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local
_Mkl_Api(void,MKL_Set_Num_Threads,(int nth))
#define mkl_set_num_threads         MKL_Set_Num_Threads
_Mkl_Api(int,MKL_Get_Max_Threads,(void))
#define mkl_get_max_threads         MKL_Get_Max_Threads
_Mkl_Api(void,MKL_Set_Num_Stripes,(int nstripes))
#define mkl_set_num_stripes         MKL_Set_Num_Stripes
_Mkl_Api(int,MKL_Get_Num_Stripes,(void))
#define mkl_get_num_stripes         MKL_Get_Num_Stripes
_Mkl_Api(int,MKL_Domain_Set_Num_Threads,(int nth, int MKL_DOMAIN))
#define mkl_domain_set_num_threads  MKL_Domain_Set_Num_Threads
_Mkl_Api(int,MKL_Domain_Get_Max_Threads,(int MKL_DOMAIN))
#define mkl_domain_get_max_threads  MKL_Domain_Get_Max_Threads
_Mkl_Api(void,MKL_Set_Dynamic,(int bool_MKL_DYNAMIC))
#define mkl_set_dynamic             MKL_Set_Dynamic
_Mkl_Api(int,MKL_Get_Dynamic,(void))
#define mkl_get_dynamic             MKL_Get_Dynamic

/* Intel(R) MKL Progress routine */
#ifndef _MKL_PROGRESS_H_
#define _MKL_PROGRESS_H_
_Mkl_Api(int,MKL_PROGRESS, ( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,MKL_PROGRESS_,( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,mkl_progress, ( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,mkl_progress_,( int* thread, int* step, char* stage, int lstage ))
#endif /* _MKL_PROGRESS_H_ */

_Mkl_Api(int,MKL_Enable_Instructions,(int))
#define  mkl_enable_instructions    MKL_Enable_Instructions
#define  MKL_ENABLE_SSE4_2          0
#define  MKL_ENABLE_AVX             1
#define  MKL_ENABLE_AVX2            2
#define  MKL_ENABLE_AVX512_MIC      3
#define  MKL_ENABLE_AVX512          4
#define  MKL_ENABLE_AVX512_MIC_E1   5
#define  MKL_ENABLE_AVX512_E1       6
#define  MKL_SINGLE_PATH_ENABLE     0x0600

/* Single Dynamic library interface */
#define MKL_INTERFACE_LP64          0x0
#define MKL_INTERFACE_ILP64         0x1
#define MKL_INTERFACE_GNU           0x2
_Mkl_Api(int,MKL_Set_Interface_Layer,(int code))
#define mkl_set_interface_layer     MKL_Set_Interface_Layer

/* Single Dynamic library threading */
#define MKL_THREADING_INTEL         0
#define MKL_THREADING_SEQUENTIAL    1
#define MKL_THREADING_PGI           2
#define MKL_THREADING_GNU           3
#define MKL_THREADING_TBB           4
_Mkl_Api(int,MKL_Set_Threading_Layer,(int code))
#define mkl_set_threading_layer     MKL_Set_Threading_Layer

typedef void (* XerblaEntry) (const char * Name, const int * Num, const int Len);
_Mkl_Api(XerblaEntry,mkl_set_xerbla,(XerblaEntry xerbla))

typedef int (* ProgressEntry) (int* thread, int* step, char* stage, int stage_len);
_Mkl_Api(ProgressEntry,mkl_set_progress,(ProgressEntry progress))

/* Intel(R) MKL CBWR */
_Mkl_Api(int,MKL_CBWR_Get,(int))
#define mkl_cbwr_get                MKL_CBWR_Get
_Mkl_Api(int,MKL_CBWR_Set,(int))
#define mkl_cbwr_set                MKL_CBWR_Set
_Mkl_Api(int,MKL_CBWR_Get_Auto_Branch,(void))
#define mkl_cbwr_get_auto_branch    MKL_CBWR_Get_Auto_Branch

_Mkl_Api(int,MKL_Set_Env_Mode,(int))
#define mkl_set_env_mode            MKL_Set_Env_Mode

_Mkl_Api(int,MKL_Verbose,(int))
#define mkl_verbose                MKL_Verbose

#if defined(MKL_STDCALL)
    _Mkl_Api(int,MKL_Verbose_Output_File,(char *, int))
#else
    _Mkl_Api(int,MKL_Verbose_Output_File,(char *))
#endif
#define mkl_verbose_output_file    MKL_Verbose_Output_File

#define MKL_EXIT_UNSUPPORTED_CPU    1
#define MKL_EXIT_CORRUPTED_INSTALL  2
#define MKL_EXIT_NO_MEMORY          3

typedef void (* MKLExitHandler)(int why);
_Mkl_Api(void,MKL_Set_Exit_Handler,(MKLExitHandler h));
#define mkl_set_exit_handler       MKL_Set_Exit_Handler

#define MKL_MEM_MCDRAM 1

_Mkl_Api(int,MKL_Set_Memory_Limit,(int mem_type,size_t limit));
#define mkl_set_memory_limit MKL_Set_Memory_Limit

enum {
    MKL_BLACS_CUSTOM = 0,
    MKL_BLACS_MSMPI = 1,
    MKL_BLACS_INTELMPI = 2,
    MKL_BLACS_MPICH2 = 3,
    MKL_BLACS_LASTMPI = 4
};
int MKL_Set_mpi(int vendor, const char *custom_library_name);
#define mkl_set_mpi MKL_Set_mpi

_Mkl_Api(void,MKL_Finalize,(void));
#define mkl_finalize MKL_Finalize

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_SERVICE_H_ */
