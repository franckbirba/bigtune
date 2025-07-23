#!/usr/bin/env python3
"""
Convert merged model to GGUF format for better LM Studio performance
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_llamacpp():
    """Setup llama.cpp for conversion"""
    llamacpp_dir = Path("./llama.cpp")
    
    if not llamacpp_dir.exists():
        print("📥 Cloning llama.cpp...")
        result = subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to clone: {result.stderr}")
            return False
    
    # Install requirements
    req_file = llamacpp_dir / "requirements.txt"
    if req_file.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    
    return llamacpp_dir

def convert_to_gguf():
    """Convert to GGUF with different quantization levels"""
    
    llamacpp_dir = setup_llamacpp()
    if not llamacpp_dir:
        return False
    
    convert_script = llamacpp_dir / "convert_hf_to_gguf.py"
    model_path = "./merged_model"
    output_dir = Path("./gguf_models")
    output_dir.mkdir(exist_ok=True)
    
    # Different quantization levels for different memory requirements
    # Updated for current llama.cpp supported formats
    quantizations = [
        ("q8_0", "8-bit (medium, ~7GB)"),
        ("f16", "16-bit (large, ~14GB)"),
        ("bf16", "BF16 (full precision, ~14GB)"),
        ("f32", "32-bit (highest quality, ~28GB)")
    ]
    
    print("🔄 Converting to GGUF formats...")
    
    for quant_type, description in quantizations:
        print(f"\n📦 Creating {description}...")
        output_file = output_dir / f"mistral-7b-positivity-{quant_type}.gguf"
        
        cmd = [
            sys.executable, str(convert_script),
            model_path,
            "--outtype", quant_type,
            "--outfile", str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_gb = output_file.stat().st_size / (1024**3)
            print(f"✅ {quant_type}: {output_file.name} ({size_gb:.1f}GB)")
        else:
            print(f"❌ {quant_type} failed: {result.stderr}")
    
    return True

def install_in_lmstudio():
    """Install GGUF models in LM Studio"""
    gguf_dir = Path("./gguf_models")
    lms_dir = Path("/Users/franckbirba/.lmstudio/models/custom")
    
    if not gguf_dir.exists():
        print("❌ No GGUF models found")
        return
    
    for gguf_file in gguf_dir.glob("*.gguf"):
        # Create model directory for each quantization
        model_name = gguf_file.stem  # filename without extension
        model_dir = lms_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy GGUF file
        dest_file = model_dir / gguf_file.name
        result = subprocess.run(["cp", str(gguf_file), str(dest_file)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to copy {gguf_file.name}: {result.stderr}")
            continue
        
        size_gb = gguf_file.stat().st_size / (1024**3)
        print(f"📁 Installed: {model_name} ({size_gb:.1f}GB)")

def main():
    print("🚀 Converting to GGUF for LM Studio")
    print("=" * 40)
    
    if not Path("./merged_model").exists():
        print("❌ Merged model not found. Run the merge script first!")
        return
    
    # Convert to GGUF
    if convert_to_gguf():
        print("\n📥 Installing in LM Studio...")
        install_in_lmstudio()
        
        print("\n🎉 Done! Your models are now available in LM Studio:")
        print("   - helpdesk-support-agent-4b-q8_0 (recommended for most systems)")
        print("   - helpdesk-support-agent-4b-f16 (better quality)")
        print("   - helpdesk-support-agent-4b-bf16 (high quality)")
        print("   - helpdesk-support-agent-4b-f32 (highest quality, needs lots of RAM)")
        
        print("\n💡 Try loading the q8_0 version first - it should work on most systems!")
        print("   Restart LM Studio to see the new models.")
    
if __name__ == "__main__":
    main()