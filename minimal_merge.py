#!/usr/bin/env python3
"""
Minimal merge script with basic LoRA config
"""

import json
import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

# Import configuration
try:
    from bigtune.config import config
except ImportError:
    # Fallback for direct script execution
    import os
    from dotenv import load_dotenv
    
    # Load .env file
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    class FallbackConfig:
        BASE_MODEL = os.getenv('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
        CONFIG_FILE = os.getenv('CONFIG_FILE', 'config/positivity-lora.yaml')
    
    config = FallbackConfig()

def load_training_config():
    """Load the training configuration YAML file"""
    # Look for config file in llm-builder directory first, then current directory
    config_paths = [
        Path("llm-builder") / config.CONFIG_FILE,
        Path(config.CONFIG_FILE),
        Path("llm-builder/config/positivity-lora.yaml")  # fallback
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    
    raise FileNotFoundError(f"Could not find config file. Tried: {[str(p) for p in config_paths]}")

def get_config_values():
    """Extract relevant values from training config"""
    training_config = load_training_config()
    
    base_model = training_config.get('base_model', config.BASE_MODEL)
    output_dir = training_config.get('output_dir', './output/lora-adapter')
    
    # Convert relative paths to absolute from project root
    if not Path(output_dir).is_absolute():
        output_dir = Path(".") / output_dir
    
    return base_model, str(output_dir)

def create_minimal_config():
    """Create a minimal LoRA config that's compatible with older peft"""
    base_model, output_dir = get_config_values()
    adapter_config_path = Path(output_dir) / "adapter_config.json"
    
    # Read the original config to get the important values
    with open(adapter_config_path, 'r') as f:
        original_config = json.load(f)
    
    # Create minimal config with only essential fields
    minimal_config = {
        "base_model_name_or_path": original_config["base_model_name_or_path"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": original_config["lora_alpha"],
        "lora_dropout": original_config["lora_dropout"],
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": original_config["r"],
        "target_modules": original_config["target_modules"],
        "task_type": "CAUSAL_LM"
    }
    
    # Save minimal config
    with open(adapter_config_path, 'w') as f:
        json.dump(minimal_config, f, indent=2)
    
    print(f"✅ Created minimal adapter config at {adapter_config_path}")

def merge_model():
    """Merge LoRA with base model"""
    base_model, output_dir = get_config_values()
    
    print(f"🔄 Loading base model: {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    print(f"🔄 Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    print(f"🔄 Loading LoRA adapters from: {output_dir}")
    model = PeftModel.from_pretrained(base_model_obj, output_dir)
    
    print("🔄 Merging LoRA with base model...")
    merged_model = model.merge_and_unload()
    
    print("💾 Saving merged model...")
    merged_model.save_pretrained("./merged_model", safe_serialization=True)
    tokenizer.save_pretrained("./merged_model")
    
    print("✅ Merged model saved to ./merged_model")

def main():
    print("🚀 Minimal LoRA merge")
    print("=" * 30)
    
    try:
        base_model, output_dir = get_config_values()
        print(f"📋 Configuration:")
        print(f"   Base model: {base_model}")
        print(f"   LoRA output: {output_dir}")
        print()
        
        create_minimal_config()
        merge_model()
        
        print("\n🎉 Success! Your merged model is ready in './merged_model/'")
        print("\n📖 To use in LM Studio:")
        print("1. Open LM Studio")
        print("2. Click 'Load Model from Folder'")
        print("3. Select './merged_model/' directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()