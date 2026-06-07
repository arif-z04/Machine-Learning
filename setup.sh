#!/bin/bash

# ================================================
# Python Project Setup Script
# Author: Noir
# Description: Creates venv, installs dependencies,
#              and auto-generates requirements.txt if missing.
# ================================================

# Exit if any command fails
set -e

echo "🚀 Starting project setup..."

# Step 1: Check for Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install it first."
    exit 1
fi

# Step 2: Create virtual environment if it doesn't exist
if [ ! -d ".ml-venv" ]; then
    echo "⚙️  Creating virtual environment (.ml-venv)..."
    python3 -m venv .ml-venv
else
    echo "✅ Virtual environment already exists."
fi

# Step 3: Activate the virtual environment
echo "🔹 Activating virtual environment..."
source .ml-venv/bin/activate

# Step 4: Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Step 5: Check for requirements.txt
if [ -f "requirements.txt" ]; then
    echo "📦 Installing packages from requirements.txt..."
    pip install -r requirements.txt
    # Step 5.5: Check for outdated packages and offer to upgrade
    echo "🔎 Checking for outdated packages..."
    OUTDATED_LIST=$(python3 - <<'PY'
import json, subprocess, sys
try:
    out = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'])
    data = json.loads(out.decode())
    if not data:
        sys.exit(0)
    for pkg in data:
        print(f"{pkg['name']}=={pkg['version']} -> {pkg['latest_version']}")
except Exception:
    sys.exit(0)
PY
)

    if [ -z "$OUTDATED_LIST" ]; then
        echo "✅ All installed packages are up-to-date."
    else
        echo "⚠️  The following packages are outdated:" 
        echo "$OUTDATED_LIST"
        read -p "🔄 Upgrade all outdated packages now? (y/n): " upgrade_all
        if [[ "$upgrade_all" =~ ^[Yy]$ ]]; then
            # collect just the package names
            PKG_NAMES=$(python3 - <<'PY'
import json, subprocess, sys
try:
    out = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'])
    data = json.loads(out.decode())
    names = [pkg['name'] for pkg in data]
    print(' '.join(names))
except Exception:
    print('')
PY
)
            if [ -n "$PKG_NAMES" ]; then
                echo "⬆️  Upgrading: $PKG_NAMES"
                pip install --upgrade $PKG_NAMES
                echo "✅ Upgrades finished."
                read -p "🧾 Update requirements.txt from environment now? (y/n): " regen
                if [[ "$regen" =~ ^[Yy]$ ]]; then
                    pip freeze > requirements.txt
                    echo "✅ requirements.txt updated."
                fi
            else
                echo "⚠️  Could not determine package names to upgrade."
            fi
        else
            echo "ℹ️  Skipping upgrades."
        fi
    fi
else
    echo "⚠️  requirements.txt not found."
    echo "📝 Creating a new empty requirements.txt..."
    touch requirements.txt
    echo "# Add your dependencies here" > requirements.txt
    echo "✅ requirements.txt created."
fi

# Step 6: Ask user if they want to update requirements.txt
read -p "🔄 Do you want to generate a new requirements.txt from current environment? (y/n): " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    echo "🧾 Generating requirements.txt..."
    pip freeze > requirements.txt
    echo "✅ requirements.txt updated."
fi

# Step 7: Finish
echo "✨ Setup complete!"
echo "To activate this environment later, run: source .ml-venv/bin/activate"
