@echo off
REM Setup script for Financial Chatbot Backend

echo ========================================
echo Financial Chatbot - Backend Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    pause
    exit /b 1
)

echo Installing Python dependencies...
echo.

REM Install core dependencies
pip install flask flask-cors python-dotenv pyyaml

REM Install LangChain core
pip install langchain langchain-core

REM Install LangChain providers
echo.
echo Installing LangChain provider integrations...
pip install langchain-openai langchain-groq

REM Optional providers (user can uncomment if needed)
REM pip install langchain-anthropic
REM pip install langchain-cohere

REM Install ML dependencies
echo.
echo Installing ML dependencies (this may take a while)...
pip install torch transformers sentence-transformers

REM Install other dependencies
echo.
echo Installing additional dependencies...
pip install yfinance numpy

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Configure your .env file with API keys
echo 2. Run: python app.py
echo.
echo See LLM_PROVIDER_GUIDE.md for configuration details
echo.
pause
