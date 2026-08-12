@echo off
cd /d "%~dp0"
start /min cmd /c "python app.py"
timeout /t 3 /nobreak > nul
start brave --new-window "http://localhost:5000"
start brave --new-window "http://localhost:5000/outsiders"
