@echo off
REM build_pyinstaller.bat — create the onedir PyInstaller bundle on Windows x64.
REM
REM Usage (run from repo root Yb\):
REM   packaging\build_pyinstaller.bat [path\to\python3.12.exe]
REM
REM The optional first argument lets you specify a custom Python 3.12 binary if
REM the default 'python' on PATH is not 3.12.
REM
REM Outputs:
REM   dist\LT2 Thermometry\          <-- self-contained app folder
REM   dist\LT2 Thermometry_setup.exe <-- installer (built by build_installer.bat)

setlocal enabledelayedexpansion

SET PYTHON=%~1
IF "%PYTHON%"=="" SET PYTHON=python

SET REPO_ROOT=%~dp0..
SET VENV_DIR=%REPO_ROOT%\.venv-build

echo =^> Repo root : %REPO_ROOT%
echo =^> Python    : %PYTHON%
%PYTHON% --version
echo =^> Venv      : %VENV_DIR%

REM ── 1. Create / reuse build venv ─────────────────────────────────────────────
IF NOT EXIST "%VENV_DIR%\Scripts\activate.bat" (
    echo =^> Creating build venv ...
    %PYTHON% -m venv "%VENV_DIR%"
)
CALL "%VENV_DIR%\Scripts\activate.bat"
echo =^> Active Python:
python --version

REM ── 2. Install dependencies ───────────────────────────────────────────────────
echo =^> Upgrading pip, wheel, setuptools ...
python -m pip install --quiet --upgrade pip wheel setuptools

echo =^> Installing runtime requirements (excluding torch) ...
REM Filter out comment lines, blank lines, and torch lines; install the rest.
REM We write a temp file without torch entries then install it.
python -c ^
  "import re; lines=[l for l in open(r'packaging/requirements-lock.txt') if l.strip() and not l.startswith('#') and not re.match(r'torch', l.strip())]; open('.tmp_reqs.txt','w').writelines(lines)"
python -m pip install --quiet -r .tmp_reqs.txt
del .tmp_reqs.txt

echo =^> Installing PyTorch CPU-only wheel ...
python -m pip install --quiet torch ^
    --index-url https://download.pytorch.org/whl/cpu

REM ── 3. Install the app packages ───────────────────────────────────────────────
echo =^> Installing lt2_core and lt2_gui ...
python -m pip install --quiet -e "%REPO_ROOT%" 2>nul || echo (no setup.py — using PYTHONPATH)
SET PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%

REM ── 4. Regenerate Help PDF ────────────────────────────────────────────────────
echo =^> Generating Help PDF ...
python -m lt2_gui.build_help_pdf || echo (Warning: build_help_pdf failed)

REM ── 5. Run PyInstaller ────────────────────────────────────────────────────────
echo =^> Running PyInstaller ...
cd "%REPO_ROOT%"
python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    packaging\LT2_Thermometry.spec

echo.
echo =^> Build complete.
echo     Bundle : %REPO_ROOT%\dist\LT2 Thermometry\
echo     Next   : compile packaging\innosetup.iss with Inno Setup
