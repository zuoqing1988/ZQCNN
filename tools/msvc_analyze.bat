@echo off
rem MSVC /analyze (Code Analysis) over the main project.
rem
rem Usage:  msvc_analyze.bat <file1> [file2] ...
rem          (list the TUs yourself -- see tools/run_msvc_analyze.py)
rem
rem Why (audit_k3_20261001.md appendix AW/AZ): the gcc side of the audit is
rem driven by -Wall -Wextra (appendix AT/AU). /analyze is the Windows-side
rem equivalent and covers classes gcc has no warning for at all:
rem   C4701 / C6001  potentially uninitialized local / NULL dereference
rem   C6385 / C6386  buffer overrun (wrong size or index)
rem   C6235          tautological condition
rem   C6246          variable shadowing
rem It is interprocedural within a TU, so it sees things a single -W pass cannot.
rem
rem /external:W0 silences warnings from headers outside the project. Without it
rem the Windows SDK alone buries the real findings.
rem
rem Keep this file ASCII-only: cmd reads .bat in the OEM codepage and non-ASCII
rem comments break the line parser (2026-10-02 hit this twice).
rem
rem NOTE: do NOT add goto/labels or a "scan everything" mode here. With labels
rem plus EnableDelayedExpansion, cmd started executing fragments of the rem
rem lines as commands (2026-10-02). Enumerating the TUs in Python and passing
rem them as arguments is boring but it works.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0.."

if "%~1"=="" (echo no input files & exit /b 1)

set TOTAL=0
for %%F in (%*) do (
  set /a TOTAL+=1
  if /i "%%~xF"==".c" (set STD=/TC) else (set STD=/std:c++14)
  cl /nologo /c /EHsc %STD% /O2 /utf-8 /analyze /external:W0 /wd4996 ^
     /I ZQCNN /I ZQ_GEMM /I 3rdparty\include ^
     "%%F" /Fo:%TEMP%\zan_%%~nF.obj /Fd:%TEMP%\zan.pdb > "%TEMP%\zan_%%~nF.log" 2>&1
  if errorlevel 1 (echo %%~nxF  BUILD FAIL & findstr /C:"error C" "%TEMP%\zan_%%~nF.log") else (echo %%~nxF  done)
)
echo %TOTAL% files
exit /b 0
