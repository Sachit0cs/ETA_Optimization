@echo off
rem Launch the Delhivery Network Intelligence LITE console (fast build).
rem Port 8502 so it can run side by side with the full console (app.py).
cd /d "%~dp0"
python -m streamlit run app_lite.py --server.port 8502
pause
