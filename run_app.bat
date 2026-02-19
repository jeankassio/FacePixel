@echo off
setlocal

if not exist requirements.txt (
    echo Run this script inside the FacePixel project folder.
    exit /b 1
)

if not exist .venv (
    echo Creating local virtual environment in .venv ...
    py -3 -m venv .venv >nul 2>nul
    if errorlevel 1 (
        python -m venv .venv
        if errorlevel 1 (
            echo Could not create the virtual environment.
            echo Install Python 3.10+ and try again.
            exit /b 1
        )
    )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Could not activate virtual environment .venv.
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py

endlocal
pause