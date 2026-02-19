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

echo.
echo Installing dependencies...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA 11.8 (for GPU acceleration)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo Installing other requirements...
python -m pip install -r requirements.txt

echo.
echo Downloading face detection models (this may take a few minutes)...
python download_models.py
if errorlevel 1 (
    echo Warning: Some models could not be downloaded.
    echo The app may download them on first run.
    echo.
)

echo.
echo Building executable...
pyinstaller FacePixel.spec

echo.
echo Build finished. EXE at dist\FacePixel.exe
endlocal
pause