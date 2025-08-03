@echo off
echo ========================================
echo    AI Agents Collection Setup
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from https://python.org
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
python setup.py

echo.
echo ========================================
echo Setup completed!
echo.
echo Next steps:
echo 1. Edit .env file and add your HuggingFace token
echo 2. Start ChromaDB: docker run -p 8000:8000 chromadb/chroma
echo 3. Run: streamlit run myagents/tailored_coverletter_agent_ui.py
echo ========================================
pause
