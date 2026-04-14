@echo off
REM Quick launcher for documentation
REM Opens docs/index.html in the default browser

setlocal

set script_dir=%~dp0
set docs_path=%script_dir%docs\index.html

if not exist "%docs_path%" (
    echo Error: Documentation file not found at %docs_path%
    pause
    exit /b 1
)

echo Opening documentation...
echo URL: file:///%docs_path:\=/%

start "" "%docs_path%"

exit /b 0
