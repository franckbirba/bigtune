#!/usr/bin/env python3
"""
Ollama integration for BigTune - Deploy trained models to Ollama
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

class OllamaDeployment:
    """Deploy BigTune models to Ollama"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "bigtune-model"
        self.merged_model_path = "./merged_model"
        
    def create_modelfile(self, 
                        system_prompt: str = None,
                        temperature: float = 0.3,
                        top_p: float = 0.8,
                        is_rag_model: bool = False) -> str:
        """Create Ollama Modelfile for the trained model"""
        
        # Default system prompt
        if not system_prompt:
            if is_rag_model:
                system_prompt = """You are an expert AI assistant with access to documentation.

CRITICAL: When provided with documentation context, use ONLY the exact information from that documentation. Do not modify paths, endpoints, or add information not present in the provided context."""
            else:
                system_prompt = "You are a helpful AI assistant trained on specialized knowledge."
        
        # Template for chat models (supports most fine-tuned models)
        modelfile_content = f'''FROM {self.merged_model_path}

TEMPLATE """<|system|>
{system_prompt}
</s>
<|user|>
{{{{ .Prompt }}</s>
<|assistant|>
"""

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "</s>"
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
'''
        
        return modelfile_content
    
    def create_model(self, 
                     system_prompt: str = None,
                     temperature: float = 0.3,
                     is_rag_model: bool = False,
                     force_recreate: bool = False) -> bool:
        """Create Ollama model from merged model"""
        
        # Check if merged model exists
        if not Path(self.merged_model_path).exists():
            print(f"❌ Merged model not found at {self.merged_model_path}")
            print("Run 'bigtune merge' first to create the merged model")
            return False
        
        # Remove existing model if force recreate
        if force_recreate:
            try:
                subprocess.run(["ollama", "rm", self.model_name], 
                             capture_output=True, check=False)
                print(f"🗑️ Removed existing model: {self.model_name}")
            except Exception:
                pass
        
        # Create temporary Modelfile
        modelfile_content = self.create_modelfile(
            system_prompt=system_prompt,
            temperature=temperature,
            is_rag_model=is_rag_model
        )
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
                f.write(modelfile_content)
                modelfile_path = f.name
            
            # Create model with Ollama
            print(f"🚀 Creating Ollama model: {self.model_name}")
            result = subprocess.run(
                ["ollama", "create", self.model_name, "-f", modelfile_path],
                capture_output=True, text=True
            )
            
            # Clean up temp file
            os.unlink(modelfile_path)
            
            if result.returncode == 0:
                print(f"✅ Successfully created model: {self.model_name}")
                print(f"🎯 Test with: ollama run {self.model_name}")
                return True
            else:
                print(f"❌ Failed to create model: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            return False
    
    def test_model(self, test_prompt: str = "Hello, how are you?") -> Optional[str]:
        """Test the deployed model"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, test_prompt],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"❌ Model test failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Model test timeout")
            return None
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return None
    
    def save_modelfile_template(self, output_path: str = "./Modelfile.template"):
        """Save a template Modelfile for customization"""
        template_content = '''FROM ./merged_model

# Customize your system prompt below
TEMPLATE """<|system|>
You are a helpful AI assistant trained on specialized knowledge.

# For RAG models, add this line:
# CRITICAL: When provided with documentation context, use ONLY the exact information from that documentation.
</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>
"""

# Adjust parameters as needed
PARAMETER temperature 0.3
PARAMETER top_p 0.8
PARAMETER stop "</s>"
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"

# Additional parameters you can set:
# PARAMETER top_k 40
# PARAMETER repeat_penalty 1.1
# PARAMETER num_ctx 2048
'''
        
        with open(output_path, 'w') as f:
            f.write(template_content)
        
        print(f"💾 Saved Modelfile template to: {output_path}")
        print("Customize the template and run: ollama create your-model -f Modelfile.template")


def deploy_to_ollama(model_name: str = None, 
                    config_file: str = None,
                    rag_mode: bool = False,
                    test_prompt: str = None) -> bool:
    """Main function to deploy BigTune model to Ollama"""
    
    # Load config if provided
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    # Extract model name from config or use default
    if not model_name:
        if config.get('model_name'):
            model_name = config['model_name']
        elif config.get('output_dir'):
            model_name = Path(config['output_dir']).name
        else:
            model_name = "bigtune-model"
    
    # Create deployment
    deployment = OllamaDeployment(model_name)
    
    # Create model
    success = deployment.create_model(
        system_prompt=config.get('system_prompt'),
        temperature=config.get('temperature', 0.3),
        is_rag_model=rag_mode,
        force_recreate=True
    )
    
    if not success:
        return False
    
    # Test model if prompt provided
    if test_prompt:
        print(f"\n🧪 Testing model with: '{test_prompt}'")
        response = deployment.test_model(test_prompt)
        if response:
            print(f"🤖 Response: {response}")
        else:
            print("❌ Model test failed")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy BigTune model to Ollama")
    parser.add_argument("--model-name", help="Name for the Ollama model")
    parser.add_argument("--config", help="Path to BigTune config file")
    parser.add_argument("--rag", action="store_true", help="Deploy as RAG model")
    parser.add_argument("--test", help="Test prompt for the deployed model")
    parser.add_argument("--template-only", action="store_true", 
                       help="Just create Modelfile template")
    
    args = parser.parse_args()
    
    if args.template_only:
        deployment = OllamaDeployment()
        deployment.save_modelfile_template()
    else:
        deploy_to_ollama(
            model_name=args.model_name,
            config_file=args.config,
            rag_mode=args.rag,
            test_prompt=args.test
        )