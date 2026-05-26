#!/bin/bash
set -e

# setup_env.sh - Production-Grade Virtual Environment Bootstrap Script
# This script sets up a virtualenv, upgrades core pip utilities, and installs dependencies.

echo "====================================================================="
echo "        Bootstrapping Gemma 4 ESCO Pipeline Environment"
echo "====================================================================="

# Check Python version
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python is not installed. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "[+] Detected Python version: $PYTHON_VERSION"

# Ensure we are in a modern Python ecosystem
$PYTHON_CMD -c "
import sys
if sys.version_info < (3, 8):
    print('[-] Error: Python 3.8 or higher is required.')
    sys.exit(1)
"

# Create Virtual Environment using virtualenv or fallback to venv
ENV_DIR=".venv"
if [ ! -d "$ENV_DIR" ]; then
    echo "[+] Creating virtual environment in '$ENV_DIR'..."
    if command -v virtualenv &>/dev/null; then
        virtualenv -p "$PYTHON_CMD" "$ENV_DIR"
    else
        echo "[!] virtualenv command not found, falling back to python -m venv..."
        $PYTHON_CMD -m venv "$ENV_DIR"
    fi
else
    echo "[+] Existing virtual environment detected in '$ENV_DIR'."
fi

# Activate Virtual Environment
echo "[+] Activating virtual environment..."
source "$ENV_DIR/bin/activate"

# Upgrade pip, setuptools, and wheel
echo "[+] Upgrading core pip utilities..."
pip install --upgrade pip setuptools wheel

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "[+] Installing pipeline dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[-] Warning: requirements.txt not found. Skipping library installation."
fi

echo "====================================================================="
echo "[+] Environment installation completed successfully!"
echo "[+] To activate the environment, run:"
echo "    source .venv/bin/activate"
echo "====================================================================="
