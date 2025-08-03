#!/bin/bash

echo "========================================"
echo "    AI Agents Collection Setup"
echo "========================================"
echo

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

python3 --version

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install pip3"
    exit 1
fi

echo
echo "Installing dependencies..."
python3 setup.py

echo
echo "========================================"
echo "Setup completed!"
echo
echo "Next steps:"
echo "1. Edit .env file and add your HuggingFace token"
echo "2. Start ChromaDB: docker run -p 8000:8000 chromadb/chroma"
echo "3. Run: streamlit run myagents/tailored_coverletter_agent_ui.py"
echo "========================================"
