@echo off
rem Double-click to start the SRT GUI (you can also drop an .srt file on this icon)
cd /d "%~dp0"
where pyw >nul 2>nul
if %errorlevel%==0 (
  start "" pyw -3 srt_gui.py %*
) else (
  start "" pythonw srt_gui.py %*
)
