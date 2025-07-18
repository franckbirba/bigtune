#!/bin/bash
# BigTune - Unified RunPod Training Script
# Automatically detects environment and sets up training
set -e

echo "🚀 BigTune Training Setup"
echo "========================"

# Detect environment
if [ -f "/usr/local/bin/jupyter" ] || [ -f "/opt/conda/bin/jupyter" ]; then
    echo "📦 Detected: axolotl-cloud image (pre-configured)"
    ENVIRONMENT="axolotl-cloud"
elif nvidia-smi &>/dev/null; then
    echo "🐳 Detected: CUDA environment (needs setup)"
    ENVIRONMENT="cuda"
else
    echo "💻 Detected: CPU environment"
    ENVIRONMENT="cpu"
fi

# Move to workspace
cd /workspace
echo "📂 Working directory: $(pwd)"

# Disable Jupyter if present to free resources
if [ "$ENVIRONMENT" = "axolotl-cloud" ]; then
    echo "🔇 Disabling Jupyter to free resources..."
    export JUPYTER_DISABLE=1
    pkill -f jupyter || true
fi

# Setup based on environment
if [ "$ENVIRONMENT" = "cuda" ]; then
    echo "⚙️  Setting up CUDA environment..."
    
    # System packages
    apt update
    apt install -y git python3 python3-pip python3-dev build-essential cmake ninja-build
    
    # Python packages
    pip3 install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    echo "🔥 Installing PyTorch with CUDA..."
    pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Clone and install axolotl
    echo "📥 Installing axolotl..."
    git clone https://github.com/OpenAccess-AI-Collective/axolotl.git /workspace/axolotl
    cd /workspace/axolotl
    
    # Fix Python 3.10 compatibility
    echo "🔧 Fixing Python 3.10 compatibility..."
    python3 -c "
import re
with open('src/axolotl/logging_config.py', 'r') as f:
    content = f.read()
content = re.sub(r'logging\.getLevelNamesMapping\(\)', '{\"CRITICAL\": 50, \"ERROR\": 40, \"WARNING\": 30, \"INFO\": 20, \"DEBUG\": 10, \"NOTSET\": 0}', content)
with open('src/axolotl/logging_config.py', 'w') as f:
    f.write(content)
"
    
    # Install axolotl
    pip3 install -e .
    cd /workspace
fi

# Hugging Face authentication
if [ -n "$HF_TOKEN" ]; then
    echo "🤗 Authenticating with Hugging Face..."
    echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
else
    echo "⚠️  HF_TOKEN not set - you may need to login manually"
fi

# Copy training files
echo "📋 Setting up training configuration..."
if [ -d "/workspace/llm-builder" ]; then
    cp -r /workspace/llm-builder/config /workspace/
    cp -r /workspace/llm-builder/datasets /workspace/
else
    echo "❌ llm-builder directory not found!"
    exit 1
fi

# Determine config file to use
CONFIG_FILE=${CONFIG_FILE:-"config/positivity-lora.yaml"}
FULL_CONFIG_PATH="/workspace/$CONFIG_FILE"

# Check if config file exists
if [ ! -f "$FULL_CONFIG_PATH" ]; then
    echo "❌ Configuration file not found: $FULL_CONFIG_PATH"
    echo "Available config files:"
    find /workspace/config -name "*.yaml" 2>/dev/null || echo "No YAML files found in /workspace/config"
    exit 1
fi

# Parse output directory from config file
OUTPUT_DIR=$(python3 -c "
import yaml
try:
    with open('$FULL_CONFIG_PATH', 'r') as f:
        config = yaml.safe_load(f)
    print(config.get('output_dir', 'output/lora-adapter'))
except Exception as e:
    print('output/lora-adapter')
")

# Clean previous training output
echo "🧹 Cleaning previous training output..."
rm -rf "/workspace/$OUTPUT_DIR"

# Start training
echo "🎯 Starting LoRA training..."
echo "Configuration: $FULL_CONFIG_PATH"
echo "Output directory: /workspace/$OUTPUT_DIR"
echo ""

# Run training with proper module path
python3 -m axolotl.cli.train "$FULL_CONFIG_PATH"

echo ""
echo "✅ Training completed!"
echo "📁 Output saved to: /workspace/$OUTPUT_DIR"