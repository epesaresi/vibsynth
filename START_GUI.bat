@echo off
rem vibsynth - start the browser interface (Windows)
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import numpy, scipy, matplotlib" 2>nul || (
  echo Installing numpy, scipy, matplotlib for the current user...
  %PY% -m pip install --user numpy scipy matplotlib openpyxl
)
%PY% -m vibsynth gui
pause
