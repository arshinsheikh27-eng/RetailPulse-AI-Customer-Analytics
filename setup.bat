@echo off
setlocal
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create venv with Python 3.11. Install Python 3.11 and try again.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check the error above.
  pause
  exit /b 1
)
echo.
echo Setup complete. Run run_dashboard.bat to launch RetailPulse.
pause
