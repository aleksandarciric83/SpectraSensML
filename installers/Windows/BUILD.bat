@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  SpectraSensML v1.0 — Windows build script
REM  Run this script on a Windows 10/11 x64 machine to produce:
REM    packaging\Output\SpectraSensML_1.0_win64_setup.exe
REM
REM  Requirements:
REM    - Python 3.11 or 3.12 installed from python.org  (recommended 3.12)
REM    - Inno Setup 6 installed from https://jrsoftware.org/isinfo.php
REM    - Internet connection (to download pip packages ~4 GB)
REM
REM  Usage:  double-click BUILD.bat  OR  run from cmd.exe in this folder
REM ═══════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion

echo ================================================================
echo  SpectraSensML v1.0 - Windows Installer Builder
echo ================================================================

REM ── Locate Python ────────────────────────────────────────────────────
SET PYTHON=python
%PYTHON% --version >nul 2>&1
IF ERRORLEVEL 1 (
    SET PYTHON=python3
    %PYTHON% --version >nul 2>&1
    IF ERRORLEVEL 1 (
        echo ERROR: Python not found. Install Python 3.12 from https://python.org
        pause & exit /b 1
    )
)
echo Using: & %PYTHON% --version

REM Check version is 3.11 or 3.12
%PYTHON% -c "import sys; v=sys.version_info; assert (v.major==3 and v.minor in (11,12)), f'Need Python 3.11 or 3.12, got {v.major}.{v.minor}'" 2>nul
IF ERRORLEVEL 1 (
    echo WARNING: Python 3.11 or 3.12 is recommended. Continuing anyway...
)

REM ── Always run from the folder containing this script ────────────────
SET HERE=%~dp0
cd /d "%HERE%"

REM ── Create build venv ─────────────────────────────────────────────────
SET VENV=%HERE%.venv-build
IF NOT EXIST "%VENV%\Scripts\activate.bat" (
    echo Creating build virtual environment...
    %PYTHON% -m venv "%VENV%"
)
CALL "%VENV%\Scripts\activate.bat"
echo.

REM ── Install dependencies ──────────────────────────────────────────────
echo Installing dependencies (this may take 10-20 minutes)...
python -m pip install --quiet --upgrade pip wheel setuptools

REM Install everything except torch lines from requirements
python -c "lines=[l for l in open(r'%HERE%packaging\requirements-lock.txt') if l.strip() and not l.startswith('#') and 'torch' not in l]; open(r'%HERE%.tmp.txt','w').writelines(lines)"
python -m pip install --quiet -r "%HERE%.tmp.txt"
del "%HERE%.tmp.txt"

echo Installing PyTorch CPU (large download ~1 GB)...
python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu

REM ── Expose app source ─────────────────────────────────────────────────
SET PYTHONPATH=%HERE%;%PYTHONPATH%

REM ── Generate Help PDF ─────────────────────────────────────────────────
echo Generating Help PDF...
python -m lt2_gui.build_help_pdf 2>nul || echo (PDF generation skipped)

REM ── PyInstaller ───────────────────────────────────────────────────────
echo Running PyInstaller (this takes 5-15 minutes)...
python -m PyInstaller --noconfirm --clean packaging\LT2_Thermometry.spec

IF ERRORLEVEL 1 (
    echo ERROR: PyInstaller failed. See output above.
    pause & exit /b 1
)

REM ── Inno Setup ────────────────────────────────────────────────────────
SET ISCC="C:\Program Files (x86)\Inno Setup 6\iscc.exe"
IF NOT EXIST %ISCC% (
    SET ISCC="C:\Program Files\Inno Setup 6\iscc.exe"
)
IF NOT EXIST %ISCC% (
    echo.
    echo PyInstaller bundle is ready at:  dist\SpectraSensML\
    echo.
    echo To create the final .exe installer:
    echo   1. Install Inno Setup 6 from https://jrsoftware.org/isinfo.php
    echo   2. Run:  iscc.exe packaging\innosetup.iss
    pause & exit /b 0
)

echo Creating Windows installer with Inno Setup...
mkdir packaging\Output 2>nul
%ISCC% packaging\innosetup.iss

echo.
echo ================================================================
echo  DONE!
echo  Installer: packaging\Output\SpectraSensML_1.0_win64_setup.exe
echo ================================================================
pause
