"""
BigTune Configuration Management
Loads configuration from .env file and environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the project root
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'

if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"⚠️  Warning: .env file not found at {env_path}")
    print("   Please copy .env.example to .env and configure your settings")

class Config:
    """Configuration class with .env support"""
    
    # RunPod API Configuration
    RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY', '')
    RUNPOD_IMAGE_NAME = os.getenv('RUNPOD_IMAGE_NAME', 'nvidia/cuda:12.1.1-devel-ubuntu22.04')
    
    # GPU and Resource Configuration
    GPU_TYPE = os.getenv('GPU_TYPE', 'NVIDIA A40')
    MACHINE_SIZE = os.getenv('MACHINE_SIZE', 'petite')
    VOLUME_SIZE_GB = int(os.getenv('VOLUME_SIZE_GB', '50'))
    CONTAINER_DISK_SIZE_GB = int(os.getenv('CONTAINER_DISK_SIZE_GB', '50'))
    MIN_VCPU_COUNT = int(os.getenv('MIN_VCPU_COUNT', '2'))
    MIN_MEMORY_GB = int(os.getenv('MIN_MEMORY_GB', '15'))
    
    # SSH Configuration
    SSH_KEY_PATH = os.path.expanduser(os.getenv('SSH_KEY_PATH', '~/.ssh/runpod_rsa'))
    
    # Project Paths
    DATASET_DIR = os.getenv('DATASET_DIR', './datasets')
    CONFIG_FILE = os.getenv('CONFIG_FILE', 'config/positivity-lora.yaml')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
    VOLUME_NAME = os.getenv('VOLUME_NAME', 'llm-builder')
    
    # Model Configuration
    BASE_MODEL = os.getenv('BASE_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')
    LORA_ALPHA = int(os.getenv('LORA_ALPHA', '16'))
    LORA_DROPOUT = float(os.getenv('LORA_DROPOUT', '0.05'))
    LORA_R = int(os.getenv('LORA_R', '8'))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '2e-4'))
    
    # LM Studio Configuration
    LM_STUDIO_PATH = os.path.expanduser(os.getenv('LM_STUDIO_PATH', '/Users/franckbirba/.lmstudio/models/custom'))
    
    # Hugging Face Configuration
    HF_TOKEN = os.getenv('HF_TOKEN', '')
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.RUNPOD_API_KEY:
            errors.append("RUNPOD_API_KEY is required")
            
        if not Path(cls.SSH_KEY_PATH + '.pub').exists():
            errors.append(f"SSH public key not found: {cls.SSH_KEY_PATH}.pub")
            
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
            
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding sensitive data)"""
        print("📋 BigTune Configuration:")
        print("=" * 40)
        print(f"GPU Type:         {cls.GPU_TYPE}")
        print(f"Machine Size:     {cls.MACHINE_SIZE}")
        print(f"Volume Size:      {cls.VOLUME_SIZE_GB}GB")
        print(f"Base Model:       {cls.BASE_MODEL}")
        print(f"SSH Key:          {cls.SSH_KEY_PATH}")
        print(f"LM Studio Path:   {cls.LM_STUDIO_PATH}")
        print(f"API Key:          {'✅ Configured' if cls.RUNPOD_API_KEY else '❌ Missing'}")
        print(f"HF Token:         {'✅ Configured' if cls.HF_TOKEN else '❌ Missing'}")

# Create a default config instance
config = Config()