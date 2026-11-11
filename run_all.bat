@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "MPLBACKEND=Agg"
set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"

if not exist "requirements.txt" (
    echo ERROR: requirements.txt is missing. Extract the complete MAI391_Code folder first.
    goto failed
)
if not exist "data\raw\Folds5x2_pp.xlsx" (
    echo ERROR: data\raw\Folds5x2_pp.xlsx is missing. Extract the complete MAI391_Code folder first.
    goto failed
)

if not exist "%VENV_PYTHON%" (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        if errorlevel 1 goto python_version
        py -3 -m venv ".venv"
        if errorlevel 1 goto failed
    ) else (
        where python >nul 2>nul
        if errorlevel 1 goto python_missing
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        if errorlevel 1 goto python_version
        python -m venv ".venv"
        if errorlevel 1 goto failed
    )
)

"%VENV_PYTHON%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 goto python_version

echo Installing project dependencies...
"%VENV_PYTHON%" -m pip install -r "requirements.txt"
if errorlevel 1 goto install_failed

echo.
echo Running questions 1 through 10...
for %%Q in (1 2 3 4 5 6 7 8 9 10) do (
    call :run_question %%Q
    if errorlevel 1 goto failed
)

echo.
echo All questions completed successfully.
echo Results: %CD%\results
echo Figures: %CD%\figures
echo.
pause
exit /b 0

:run_question
echo.
echo ===== Question %~1 =====
"%VENV_PYTHON%" "run_q%~1.py"
exit /b %errorlevel%

:python_missing
echo ERROR: Install Python 3.10 or later, then run this file again.
echo Download Python from https://www.python.org/downloads/windows/
goto failed

:python_version
echo ERROR: Python 3.10 or later is required.
goto failed

:install_failed
echo ERROR: Dependency installation failed. Check your Internet connection and try again.
goto failed

:failed
echo.
echo The project did not finish. Read the error above, fix it, and run run_all.bat again.
echo.
pause
exit /b 1
