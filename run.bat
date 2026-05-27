@echo off
cd /d "%~dp0.."
uv run python CameraView/machine_vision.py
pause
