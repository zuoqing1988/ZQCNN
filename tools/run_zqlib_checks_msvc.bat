@echo off
rem Compile and run every tools/zq_*_check.cpp with MSVC + /fsanitize=address.
rem
rem Why: tools/run_zqlib_checks.py runs the same tests under gcc inside WSL. Five of
rem the ZQlib headers these cover are LINKED into Windows sample binaries
rem (see tools/zqlib_reachability.py), and those samples cannot be exercised on
rem this machine (they need face databases / model files that are not in the repo).
rem So until now the only Windows-side check for those headers was "does it compile"
rem (tools/probe_zqlib_headers_msvc.py). This closes the runtime gap.
rem
rem /utf-8 is required: without it MSVC reads the sources as codepage 936 (GBK) and
rem the Chinese comments produce C2001/C2143. The main CMakeLists passes /utf-8 too.
rem
rem NOTE: keep this file ASCII-only -- cmd reads .bat in the OEM codepage, so
rem non-ASCII comments here break the line parser (2026-10-02 hit this).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
rem %~dp0 is <root>\tools\ ; go up one level instead of hardcoding D:\ZQCNN so the
rem repo can be cloned anywhere. cd /d first because %~dp0 is an absolute path.
cd /d "%~dp0.."
set ROOT=%CD%

set TESTS=zq_batch4 zq_batch6 zq_bitonicsort zq_imageprocessing zq_kmeans zq_matrix zq_mergesort zq_quaternion zq_quicksort
set FAILED=0
set TOTAL=0

for %%T in (%TESTS%) do (
  set /a TOTAL+=1
  cl /nologo /EHsc /std:c++14 /O1 /Zi /utf-8 /fsanitize=address /I3rdparty\include\ZQlib /Itools ^
     tools\%%T_check.cpp /Fe:%TEMP%\%%T.exe /Fo:%TEMP%\ /Fd:%TEMP%\zq_asan.pdb > %TEMP%\%%T.build.txt 2>&1
  if errorlevel 1 (
    echo %%T  BUILD FAIL
    type %TEMP%\%%T.build.txt
    set /a FAILED+=1
  ) else (
    %TEMP%\%%T.exe > %TEMP%\%%T.out.txt 2>&1
    if errorlevel 1 (
      echo %%T  RUN FAIL ^(rc=%errorlevel%^)
      type %TEMP%\%%T.out.txt
      set /a FAILED+=1
    ) else (
      echo %%T  PASS
    )
  )
)

echo ----------------------------------------
echo %TOTAL% tests, %FAILED% failed
exit /b %FAILED%
