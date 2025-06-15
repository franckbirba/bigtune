#!/usr/bin/env python3
"""
BigTune CLI - LoRA Fine-tuning Pipeline
Orchestrates RunPod training, model merging, and GGUF conversion for LM Studio
"""

import argparse
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
try:
    from .config import config
except ImportError:
    # For direct script execution
    from config import config

class BigTune:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.scripts = {
            'train': self.base_dir / 'launch_runpod_job.py',
            'merge': self.base_dir / 'minimal_merge.py', 
            'convert': self.base_dir / 'convert_to_gguf_simple.py'
        }

    def log(self, message, level="INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def run_script(self, script_name, description):
        """Run a script with monitoring"""
        script_path = self.scripts[script_name]
        
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False
            
        self.log(f"Starting {description}...")
        self.log(f"Running: python {script_path}")
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=False, text=True)
            
            if result.returncode == 0:
                self.log(f"✅ {description} completed successfully")
                return True
            else:
                self.log(f"❌ {description} failed with exit code {result.returncode}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ {description} failed: {e}", "ERROR")
            return False

    def check_prerequisites(self):
        """Check if required directories and files exist"""
        checks = [
            (self.base_dir / "llm-builder" / "config" / "positivity-lora.yaml", "Training config"),
            (self.base_dir / "llm-builder" / "datasets" / "positivity-rewriter.jsonl", "Training dataset"),
        ]
        
        missing = []
        for path, name in checks:
            if not path.exists():
                missing.append(f"{name}: {path}")
        
        if missing:
            self.log("Missing prerequisites:", "ERROR")
            for item in missing:
                self.log(f"  - {item}", "ERROR")
            return False
            
        return True

    def train(self, args):
        """Launch RunPod training job"""
        self.log("🚀 Starting LoRA training on RunPod")
        
        # Validate configuration first
        if not config.validate():
            return False
        
        if not self.check_prerequisites():
            return False
            
        return self.run_script('train', "RunPod training")

    def merge(self, args):
        """Merge LoRA adapters with base model"""
        self.log("🔄 Merging LoRA adapters with base model")
        
        # Check if training output exists
        output_path = self.base_dir / "output" / "lora-positivity-rewriter"
        if not output_path.exists():
            self.log("❌ No LoRA training output found. Run 'bigtune train' first.", "ERROR")
            return False
            
        return self.run_script('merge', "LoRA merging")

    def convert(self, args):
        """Convert merged model to GGUF for LM Studio"""
        self.log("📦 Converting to GGUF formats for LM Studio")
        
        # Check if merged model exists
        merged_path = self.base_dir / "merged_model"
        if not merged_path.exists():
            self.log("❌ No merged model found. Run 'bigtune merge' first.", "ERROR")
            return False
            
        return self.run_script('convert', "GGUF conversion")

    def full_pipeline(self, args):
        """Run complete pipeline: train -> merge -> convert"""
        self.log("🎯 Running full BigTune pipeline")
        self.log("=" * 50)
        
        steps = [
            ("train", "Training LoRA on RunPod"),
            ("merge", "Merging LoRA with base model"), 
            ("convert", "Converting to GGUF for LM Studio")
        ]
        
        for step, description in steps:
            self.log(f"\n📋 Step: {description}")
            if not getattr(self, step)(args):
                self.log(f"❌ Pipeline failed at: {description}", "ERROR")
                return False
                
        self.log("\n🎉 Full pipeline completed successfully!")
        self.log("Your models are ready in LM Studio!")
        return True

    def status(self, args):
        """Check status of pipeline components"""
        self.log("📊 BigTune Pipeline Status")
        self.log("=" * 40)
        
        # Check training output
        training_output = self.base_dir / "output" / "lora-positivity-rewriter"
        training_status = "✅ Complete" if training_output.exists() else "❌ Not found"
        self.log(f"Training:     {training_status}")
        
        # Check merged model
        merged_model = self.base_dir / "merged_model"
        merged_status = "✅ Complete" if merged_model.exists() else "❌ Not found"
        self.log(f"Merge:        {merged_status}")
        
        # Check GGUF models
        gguf_dir = self.base_dir / "gguf_models"
        if gguf_dir.exists():
            gguf_files = list(gguf_dir.glob("*.gguf"))
            gguf_status = f"✅ {len(gguf_files)} models" if gguf_files else "❌ No models"
        else:
            gguf_status = "❌ Not found"
        self.log(f"GGUF:         {gguf_status}")
        
        # Check LM Studio installation
        lms_dir = Path("/Users/franckbirba/.lmstudio/models/custom")
        if lms_dir.exists():
            lms_models = [d for d in lms_dir.iterdir() if d.is_dir() and "mistral-7b-positivity" in d.name]
            lms_status = f"✅ {len(lms_models)} models" if lms_models else "❌ No models"
        else:
            lms_status = "❌ LM Studio not found"
        self.log(f"LM Studio:    {lms_status}")

    def clean(self, args):
        """Clean up intermediate files"""
        self.log("🧹 Cleaning up intermediate files")
        
        cleanup_dirs = [
            "merged_model",
            "gguf_models", 
            "llama.cpp"
        ]
        
        if args.all:
            cleanup_dirs.append("output")
            
        for dir_name in cleanup_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                if args.dry_run:
                    self.log(f"Would remove: {dir_path}")
                else:
                    subprocess.run(["rm", "-rf", str(dir_path)])
                    self.log(f"Removed: {dir_path}")
            else:
                self.log(f"Not found: {dir_path}")

    def config_cmd(self, args):
        """Show and validate configuration"""
        if args.validate:
            self.log("🔍 Validating configuration...")
            if config.validate():
                self.log("✅ Configuration is valid")
                return True
            else:
                return False
        else:
            config.print_config()
            return True

def main():
    parser = argparse.ArgumentParser(
        description="BigTune - LoRA Fine-tuning Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bigtune train              # Train LoRA on RunPod
  bigtune merge              # Merge LoRA with base model
  bigtune convert            # Convert to GGUF for LM Studio
  bigtune full               # Run complete pipeline
  bigtune status             # Check pipeline status
  bigtune clean              # Clean intermediate files
  bigtune clean --all        # Clean everything including training output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LoRA model on RunPod')
    
    # Merge command  
    merge_parser = subparsers.add_parser('merge', help='Merge LoRA adapters with base model')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert merged model to GGUF')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check pipeline status')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up files')
    clean_parser.add_argument('--all', action='store_true', help='Clean training output too')
    clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
        
    args = parser.parse_args()
    
    bigtune = BigTune()
    
    # Map commands to methods
    commands = {
        'train': bigtune.train,
        'merge': bigtune.merge, 
        'convert': bigtune.convert,
        'full': bigtune.full_pipeline,
        'status': bigtune.status,
        'clean': bigtune.clean,
        'config': bigtune.config_cmd
    }
    
    if args.command in commands:
        success = commands[args.command](args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()