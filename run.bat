@echo off
rem Starts the portal on this PC at http://localhost:8100 (no password needed on this computer).
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)
start "" http://localhost:8100
.venv\Scripts\python.exe cli.py serve
