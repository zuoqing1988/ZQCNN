/*******************************************************************************
* Copyright (c) 2006-2018, Intel Corporation
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

/**
 * This header file describes how memory allocation can be replaced
 * in the Intel(R) Math Kernel Library (Intel(R) MKL). It contains
 * all declarations required by an application developer to replace
 * the memory allocation.
 *
 * Intel(R) MKL supporting this feature only use the following
 * functions to allocate or free memory:
 * - malloc
 * - calloc
 * - realloc
 * - free
 * and call those functions via globally visible function pointers:
 * - i_malloc
 * - i_calloc
 * - i_realloc
 * - i_free
 *
 * C++ new and delete operators are changed to also use these function
 * pointers, if they occur in a library at all. No library supporting memory 
 * allocation replacement will allocate memory before it is invoked explicitly
 * by the application for the first time.
 *
 * Therefore an application can safely set these function pointers
 * at the very beginning of its execution to some other replacement
 * functions. The function pointers must remain valid while
 * Intel(R) MKL is in use.
 *
 * Setting these pointers is optional because the copies contained in
 * Intel(R) MKL point to the standard C library functions by default.
 *
 * On Windows(R) data exported by a DLL and data contained in a static
 * library are accessed differently. To support mixing static
 * libraries and DLLs, the function pointers exist in two sets with
 * different names so that the application can override both sets in
 * the same source file without running into name conflicts.
 *
 * Here is an example:
\verbatim

   #include <i_malloc.h>
   #include <my_malloc.h>

   int main( int argc, int argv )
   {
       // override normal pointers
       i_malloc = my_malloc;
       i_calloc = my_calloc;
       i_realloc = my_realloc;
       i_free = my_free;

   #ifdef _WIN32
       // also override pointers used by DLLs
       i_malloc_dll = my_malloc;
       i_calloc_dll = my_calloc;
       i_realloc_dll = my_realloc;
       i_free_dll = my_free;
   #endif

   }
\endverbatim
 */

#ifndef _I_MALLOC_H_
#define _I_MALLOC_H_

#include <stdlib.h>     /* for size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* typedefs for all four function pointers */
typedef void * (* i_malloc_t)(size_t size);
typedef void * (* i_calloc_t)(size_t nmemb, size_t size);
typedef void * (* i_realloc_t)(void *ptr, size_t size);
typedef void (* i_free_t)(void *ptr);

#ifdef _WIN32

# ifdef INTEL_DLL_EXPORTS
#  define INTEL_API_DEF __declspec(dllexport)
# else
#  define INTEL_API_DEF __declspec(dllimport)
#endif

/* function pointers as used and exported by a DLL */
extern INTEL_API_DEF i_malloc_t i_malloc_dll;
extern INTEL_API_DEF i_calloc_t i_calloc_dll;
extern INTEL_API_DEF i_realloc_t i_realloc_dll;
extern INTEL_API_DEF i_free_t i_free_dll;

#else /* _WIN32 */

# define INTEL_API_DEF

#endif /* _WIN32 */

/* normal function pointers for static libraries */
extern i_malloc_t i_malloc;
extern i_calloc_t i_calloc;
extern i_realloc_t i_realloc;
extern i_free_t i_free;

#ifdef __cplusplus
}
#endif

#endif /* _I_MALLOC_H_ */
