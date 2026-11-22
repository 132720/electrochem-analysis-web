@echo off
setlocal
cd /d "%~dp0"
echo Installation of pyimpspec 5.3 requires NumPy 2.5+ and Pandas 3+.
echo Installing into your Anaconda base would break older Streamlit/Numba.
echo This installer now uses a dedicated environment to avoid that problem.
echo.
call setup_electrochem_pro_env.bat
exit /b %errorlevel%
