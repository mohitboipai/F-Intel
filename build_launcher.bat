@echo off
echo Building FIntelLauncher.exe...
call .venv\Scripts\activate
pyinstaller --onefile --noconsole --name FIntelLauncher --add-data "FIntelLogger.py;." FIntelLauncher.py
echo.
echo Build complete. EXE is in the dist\ folder.
echo Copy dist\FIntelLauncher.exe to the project root to use it.
pause
