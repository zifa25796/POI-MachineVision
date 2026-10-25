@echo off
cd /d "%~dp0"
if exist "..\.gitmodules" (
    uv run --project .. python machine_vision.py
) else (
    uv run python machine_vision.py
)
if errorlevel 1 pause
