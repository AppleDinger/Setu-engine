#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================"
echo "      SETU: Network Analytics Engine - Setup         "
echo "======================================================"

# 1. Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed. Please install it and try again."
    exit 1
fi

# 2. Create Virtual Environment
echo -e "\n📦 Creating virtual environment (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch with CUDA hooks (Targeting RTX 4060)
echo -e "\n⚡ Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install standard dependencies from requirements.txt
echo -e "\n📋 Installing core pipeline dependencies (spaCy, Pandas, NetworkX)..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found! Installing explicit packages..."
    pip install spacy pandas networkx
fi

# 6. Download the spaCy Language Model
echo -e "\n🧠 Downloading spaCy English model (en_core_web_sm)..."
python3 -m spacy download en_core_web_sm

# 7. Clone/Link the Dataset Asset (Decoupled Architecture Flex)
echo -e "\n📁 Checking for Setu-dataset storage asset..."
if [ ! -d "../Setu-dataset" ]; then
    echo "ℹ️ Setu-dataset directory not found in parent path."
    echo "🔄 Cloning raw corpus asset from GitHub..."
    git clone https://github.com/AppleDinger/Setu-dataset.git ../Setu-dataset
else
    echo "✅ Found existing Setu-dataset directory."
fi

echo "======================================================"
echo "🎉 Setup Complete! Pipeline is ready to run."
echo "To activate the environment, run: source .venv/bin/activate"
echo "======================================================"