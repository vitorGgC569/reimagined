#!/bin/bash
set -e

echo "[1] Checking Python (Ruff)..."
# Check OXN scripts and nsos python bindings if any
# Ignoring line length for now as legacy code might be verbose
ruff check OXN/scripts/ --ignore=E501

echo "[2] Checking C++ (CppCheck)..."
# Enable all checks but suppress missingInclude because we don't have full system headers in lint env
cppcheck --error-exitcode=1 --enable=warning,performance,portability --suppress=missingInclude OXN/nsos/src/

echo "[SUCCESS] Code is clean."
