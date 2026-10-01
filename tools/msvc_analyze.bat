@echo off
rem MSVC /analyze (Code Analysis) over the main project.
rem
rem Why (audit_k3_20261001.md appendix AW): the gcc side of the audit has been
rem driven by -Wall -Wextra (appendix AT/AU). /analyze is the Windows-side
rem equivalent and covers classes gcc has **no** warning for at all:
rem   C4701 / C6001  potentially uninitialized local variable used
rem   C6385/C6386   buffer overrun (wrong size or index)
rem   C26495        uninitialized member (ConC++ only, needs /analyze:concurrency)
rem   C6001         dereferencing NULL
rem It is interprocedural within a TU, so it sees things a single -W pass cannot.
rem
rem /external:W0 silences warnings from headers outside the project. Without it
rem the Windows SDK alone buries the real findings.
rem
rem Keep this file ASCII-only: cmd reads .bat in the OEM codepage and non-ASCII
rem comments break the line parser (2026-10-02 hit this twice).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0.."

if "%~1"=="" set FILES=ZQCNN\ZQ_CNN_Forward_SSEUtils.cpp ZQCNN\ZQ_CNN_Forward_SSEUtils_NCHWC.cpp ZQCNN\ZQ_CNN_LoadConfigUtils.cpp ZQCNN\ZQ_CNN_Net_NCHWC.cpp ZQCNN\ZQ_CNN_SSDDetectorPytorch.cpp ZQCNN\ZQ_CNN_Tensor4D.cpp ZQCNN\ZQ_CNN_Tensor4D_NCHWC.cpp ZQCNN\math\zq_avx_mathfun.c ZQCNN\math\zq_libm_compat.c ZQCNN\layers_c\zq_cnn_softmax_32f_align_c.c ZQCNN\layers_c\zq_cnn_lrn_32f_align_c.c ZQCNN\layers_c\zq_cnn_addbias_32f_align_c.c ZQCNN\layers_c\zq_cnn_dropout_32f_align_c.c ZQCNN\layers_c\zq_cnn_sqrt_32f_align_c.c
if not "%~1"=="" set FILES=%*

set FAILED=0
set TOTAL=0
for %%F in (%FILES%) do (
  set /a TOTAL+=1
  if /i "%%~xF"==".c" (set STD=/TC) else (set STD=/std:c++14)
  cl /nologo /c /EHsc %STD% /O2 /utf-8 /analyze /external:W0 /wd4996 ^
     /I ZQCNN /I ZQ_GEMM /I 3rdparty\include ^
     "%%F" /Fo:%TEMP%\zan_%%~nF.obj /Fd:%TEMP%\zan.pdb > "%TEMP%\zan_%%~nF.log" 2>&1
  set /a RC=!ERRORLEVEL!
  findstr /C:"warning C" "%TEMP%\zan_%%~nF.log" > "%TEMP%\zan_%%~nF.warn" 2>nul
  if exist "%TEMP%\zan_%%~nF.warn" (
    for /f %%L in ('type "%TEMP%\zan_%%~nF.warn"') do echo %%L
  ) else (
    if "!RC!"=="0" (echo %%~nxF  CLEAN) else (echo %%~nxF  BUILD FAIL ^(rc=!RC!^) & findstr /C:"error C" "%TEMP%\zan_%%~nF.log")
  )
)
exit /b 0
