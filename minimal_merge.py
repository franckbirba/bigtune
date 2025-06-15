#!/usr/bin/env python3
"""
Minimal merge script with basic LoRA config
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

def create_minimal_config():
    """Create a minimal LoRA config that's compatible with older peft"""
    
    # Read the original config to get the important values
    with open("./output/lora-positivity-rewriter/adapter_config.json", 'r') as f:
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
    with open("./output/lora-positivity-rewriter/adapter_config.json", 'w') as f:
        json.dump(minimal_config, f, indent=2)
    
    print("✅ Created minimal adapter config")

def merge_model():
    """Merge LoRA with base model"""
    print("🔄 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    print("🔄 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, "./output/lora-positivity-rewriter")
    
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