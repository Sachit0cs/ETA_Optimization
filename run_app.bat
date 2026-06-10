@echo off
rem Launch the Delhivery Network Intelligence console.
rem Uses "python -m streamlit" because pip user-installs often leave the
rem bare "streamlit" command off PATH.
cd /d "%~dp0"
python -m streamlit run app.py
pause
