/* file: mkl_vml_types.h */
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

/*
//++
//  User-level type definitions.
//--
*/

#ifndef __MKL_VML_TYPES_H__
#define __MKL_VML_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "mkl_types.h"

/*
//++
//  TYPEDEFS
//--
*/

/*
//  ERROR CALLBACK CONTEXT.
//  Error callback context structure is used in a user's error callback
//  function with the following interface:
//
//      int USER_CALLBACK_FUNC_NAME( DefVmlErrorContext par )
//
//  Error callback context fields:
//  iCode        - error status
//  iIndex       - index of bad argument
//  dbA1         - 1-st argument value, at which error occured
//  dbA2         - 2-nd argument value, at which error occured
//                 (2-argument functions only)
//  dbR1         - 1-st resulting value
//  dbR2         - 2-nd resulting value (2-result functions only)
//  cFuncName    - function name, for which error occured
//  iFuncNameLen - length of function name
*/
typedef struct _DefVmlErrorContext
{
    int     iCode;
    int     iIndex;
    double  dbA1;
    double  dbA2;
    double  dbR1;
    double  dbR2;
    char    cFuncName[64];
    int     iFuncNameLen;
    double  dbA1Im;
    double  dbA2Im;
    double  dbR1Im;
    double  dbR2Im;
} DefVmlErrorContext;

/*
// User error callback handler function type
*/
typedef int (*VMLErrorCallBack) (DefVmlErrorContext* pdefVmlErrorContext);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __MKL_VML_TYPES_H__ */
