@echo off
set "MRI_VIEWER_DIR=%~dp0mri_viewer"
start "MRI viewer server" /min py -m http.server 8765 --directory "%MRI_VIEWER_DIR%"
timeout /t 1 /nobreak >nul
start "" http://127.0.0.1:8765/
