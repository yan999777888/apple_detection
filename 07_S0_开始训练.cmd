@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_s0_f_drive.ps1" -Action train -Background
