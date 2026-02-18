#!/bin/bash
set -e

echo "=========================================="
echo "   UHK Mission Control (Frontend)"
echo "=========================================="

# 1. Install Dependencies
echo "[*] Installing Python Dependencies (Flask)..."
pip install flask

# 2. Check SDK
if [ -d "uhk" ]; then
    echo "[*] UHK SDK detected."
else
    echo "[!] UHK SDK not found (Simulation Mode Only)."
    # Attempt build if needed? No, user manual build is better.
fi

# 3. Run Server
echo "[*] Launching Dashboard on http://127.0.0.1:5000"
python frontend/app.py
