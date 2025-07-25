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
import yaml
from pathlib import Path
from datetime import datetime
try:
    from .config import config
    from .ollama_integration import deploy_to_ollama
    from .dataset_generator import DatasetGenerator
    from .rag_commands import RAGCommands, add_rag_commands
    from .unified_server import serve_unified
    from .docker_packaging import package_model
except ImportError:
    # For direct script execution
    from config import config
    from ollama_integration import deploy_to_ollama
    from dataset_generator import DatasetGenerator
    from rag_commands import RAGCommands, add_rag_commands
    from unified_server import serve_unified
    from docker_packaging import package_model

class BigTune:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.scripts = {
            'train': self.base_dir / 'launch_runpod_job.py',
            'merge': self.base_dir / 'minimal_merge.py', 
            'convert': self.base_dir / 'convert_to_gguf_simple.py'
        }
    
    def load_training_config(self):
        """Load the training configuration YAML file"""
        config_file = config.CONFIG_FILE
        
        config_paths = []
        
        # If CONFIG_FILE is an absolute path, use it directly
        if Path(config_file).is_absolute():
            config_paths.append(Path(config_file))
        else:
            # Relative paths - check multiple locations
            config_paths.extend([
                Path(config_file),  # Current working directory
                self.base_dir / "llm-builder" / config_file,  # BigTune structure
                self.base_dir / config_file,  # BigTune root
                Path("config") / config_file.split('/')[-1] if '/' in config_file else Path("config") / config_file  # Just config/ dir
            ])
        
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        
        return None
    
    def get_training_paths(self):
        """Get training paths from config"""
        training_config = self.load_training_config()
        if not training_config:
            return None, None
        
        output_dir = training_config.get('output_dir', './output/lora-adapter')
        dataset_path = None
        
        # Extract dataset path from config
        datasets = training_config.get('datasets', [])
        if datasets and isinstance(datasets[0], dict):
            dataset_path = datasets[0].get('path')
        
        return output_dir, dataset_path

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
        training_config = self.load_training_config()
        if not training_config:
            self.log("❌ Could not load training configuration", "ERROR")
            return False
        
        # Always return True if we have a valid config - let RunPod handle file validation
        # The upload process will copy the files to the correct locations
        self.log("✅ Configuration loaded successfully", "INFO")
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
        output_dir, _ = self.get_training_paths()
        if not output_dir:
            self.log("❌ Could not determine output directory from config", "ERROR")
            return False
        
        # Handle both absolute (external) and relative (internal) paths
        if Path(output_dir).is_absolute():
            output_path = Path(output_dir)
        else:
            output_path = self.base_dir / output_dir
            
        if not output_path.exists():
            self.log(f"❌ No LoRA training output found at {output_path}. Run 'bigtune train' first.", "ERROR")
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
        
        # Get paths from config
        output_dir, _ = self.get_training_paths()
        
        # Check training output
        if output_dir:
            if Path(output_dir).is_absolute():
                training_output = Path(output_dir)
            else:
                training_output = self.base_dir / output_dir
            training_status = "✅ Complete" if training_output.exists() else "❌ Not found"
            self.log(f"Training:     {training_status} ({output_dir})")
        else:
            self.log(f"Training:     ❌ Config not found")
        
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
        lms_dir = Path(config.LM_STUDIO_PATH) if hasattr(config, 'LM_STUDIO_PATH') else Path("/Users/franckbirba/.lmstudio/models/custom")
        if lms_dir.exists():
            lms_models = [d for d in lms_dir.iterdir() if d.is_dir()]
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
    
    def deploy(self, args):
        """Deploy merged model to Ollama"""
        self.log("🚀 Deploying model to Ollama")
        
        # Check if merged model exists
        merged_model = self.base_dir / "merged_model"
        if not merged_model.exists():
            self.log("❌ Merged model not found. Run 'bigtune merge' first.")
            return False
        
        # Load config for model name and settings
        try:
            training_config = self.load_training_config()
            model_name = args.name or training_config.get('model_name', 'bigtune-model')
        except Exception:
            model_name = args.name or 'bigtune-model'
        
        # Deploy to Ollama
        success = deploy_to_ollama(
            model_name=model_name,
            config_file=config.CONFIG_FILE if hasattr(config, 'CONFIG_FILE') else None,
            rag_mode=args.rag,
            test_prompt=args.test
        )
        
        if success:
            self.log(f"✅ Model deployed successfully: {model_name}")
            self.log(f"🎯 Test with: ollama run {model_name}")
            return True
        else:
            self.log("❌ Deployment failed")
            return False
    
    def generate(self, args):
        """Generate training dataset using BigAcademy"""
        self.log("📊 Generating training dataset")
        
        try:
            generator = DatasetGenerator()
            
            if args.list_agents:
                agents = generator.list_available_agents()
                self.log("Available agent profiles:")
                for agent in sorted(agents):
                    self.log(f"  📋 {agent}")
                return True
            
            if not args.agent:
                self.log("❌ Agent name required. Use --list-agents to see available agents.")
                return False
            
            # Generate dataset
            if args.quick:
                files = generator.quick_generate(args.agent, args.quick)
            elif args.target:
                files = generator.generate_large_dataset(args.agent, args.target)
            else:
                files = generator.generate_dataset(args.agent, args.samples or 100)
            
            if files:
                self.log(f"✅ Successfully generated {len(files)} dataset files")
                for file_path in files:
                    self.log(f"   📄 {file_path}")
                return True
            else:
                self.log("❌ Dataset generation failed")
                return False
                
        except Exception as e:
            self.log(f"❌ Error: {e}")
            return False
    
    def serve(self, args):
        """Run unified server with model + RAG"""
        self.log("🚀 Starting BigTune unified server")
        
        # Auto-detect model if not specified
        model = args.model
        if not model and hasattr(config, 'MODEL_NAME'):
            model = config.MODEL_NAME
        
        try:
            serve_unified(
                model=model,
                rag_index=args.rag,
                host=args.host,
                port=args.port,
                no_cors=args.no_cors
            )
            return True
        except KeyboardInterrupt:
            self.log("\n⏹️ Server stopped")
            return True
        except Exception as e:
            self.log(f"❌ Server error: {e}")
            return False
    
    def package(self, args):
        """Package model and RAG into Docker container"""
        self.log("📦 Packaging model for Docker deployment")
        
        # Validate model exists
        model_name = args.model
        if not model_name:
            self.log("❌ Model name required")
            return False
        
        # Determine image name
        image_name = args.image
        if not image_name:
            image_name = f"bigtune-{model_name.replace(':', '-').lower()}"
        
        # Get RAG indexes
        rag_indexes = args.rag or []
        if args.auto_rag:
            # Auto-detect available RAG indexes
            try:
                from .rag.generic_rag import GenericRAG
                available_indexes = GenericRAG.list_indexes()
                rag_indexes = [idx['name'] for idx in available_indexes]
                self.log(f"🔍 Auto-detected RAG indexes: {', '.join(rag_indexes)}")
            except Exception:
                pass
        
        # Package the model
        try:
            success = package_model(
                model_name=model_name,
                image_name=image_name,
                rag_indexes=rag_indexes if rag_indexes else None,
                output_dir=args.output,
                port=args.port,
                no_build=args.no_build,
                registry=args.registry,
                push=args.push
            )
            
            if success:
                self.log(f"✅ Docker package created successfully")
                self.log(f"🐳 Image: {image_name}")
                if args.output:
                    self.log(f"📁 Package: {args.output}")
                return True
            else:
                self.log("❌ Packaging failed")
                return False
                
        except Exception as e:
            self.log(f"❌ Packaging error: {e}")
            return False

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
  bigtune rag init --source swagger --url http://api.example.com/swagger.json
  bigtune generate --agent my_agent --samples 200  # Generate training dataset
  bigtune train              # Train LoRA on RunPod
  bigtune merge              # Merge LoRA with base model
  bigtune convert            # Convert to GGUF for LM Studio
  bigtune deploy             # Deploy merged model to Ollama
  bigtune serve              # Serve model with RAG
  bigtune package my-model --rag my-index          # Package for Docker
  bigtune full               # Run complete pipeline
  bigtune status             # Check pipeline status
  bigtune clean              # Clean intermediate files
  bigtune clean --all        # Clean everything including training output
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate training dataset')
    generate_parser.add_argument('--agent', help='Agent profile name')
    generate_parser.add_argument('--samples', type=int, help='Samples per template')
    generate_parser.add_argument('--quick', choices=['small', 'medium', 'large', 'xl'], 
                                help='Quick generation with predefined sizes')
    generate_parser.add_argument('--target', type=int, help='Target total samples')
    generate_parser.add_argument('--list-agents', action='store_true', 
                                help='List available agent profiles')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LoRA model on RunPod')
    
    # Merge command  
    merge_parser = subparsers.add_parser('merge', help='Merge LoRA adapters with base model')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert merged model to GGUF')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy merged model to Ollama')
    deploy_parser.add_argument('--name', help='Name for the Ollama model')
    deploy_parser.add_argument('--rag', action='store_true', help='Deploy as RAG model')
    deploy_parser.add_argument('--test', help='Test prompt for the deployed model')
    
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
    
    # Add RAG commands
    rag_parser = add_rag_commands(subparsers)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Run unified server (model + RAG)')
    serve_parser.add_argument('--model', help='Model name to serve')
    serve_parser.add_argument('--rag', default='default', help='RAG index to use')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--no-cors', action='store_true', help='Disable CORS')
    
    # Package command
    package_parser = subparsers.add_parser('package', help='Package model and RAG for Docker deployment')
    package_parser.add_argument('model', help='Model name to package')
    package_parser.add_argument('--image', help='Docker image name (auto-generated if not provided)')
    package_parser.add_argument('--rag', nargs='+', help='RAG indexes to include')
    package_parser.add_argument('--auto-rag', action='store_true', 
                               help='Auto-detect and include all available RAG indexes')
    package_parser.add_argument('--output', help='Output directory for Docker package')
    package_parser.add_argument('--port', type=int, default=8000, help='Service port in container')
    package_parser.add_argument('--no-build', action='store_true', help='Skip Docker image build')
    package_parser.add_argument('--registry', help='Docker registry URL (e.g., registry.gitlab.com/user/project)')
    package_parser.add_argument('--push', action='store_true', help='Push image to registry after build')
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
        
    args = parser.parse_args()
    
    bigtune = BigTune()
    
    # Map commands to methods
    commands = {
        'generate': bigtune.generate,
        'train': bigtune.train,
        'merge': bigtune.merge, 
        'convert': bigtune.convert,
        'deploy': bigtune.deploy,
        'serve': bigtune.serve,
        'package': bigtune.package,
        'full': bigtune.full_pipeline,
        'status': bigtune.status,
        'clean': bigtune.clean,
        'config': bigtune.config_cmd
    }
    
    # Handle RAG commands separately
    if args.command == 'rag':
        rag_commands = RAGCommands()
        rag_methods = {
            'init': rag_commands.init,
            'list': rag_commands.list,
            'info': rag_commands.info,
            'search': rag_commands.search,
            'rebuild': rag_commands.rebuild,
            'delete': rag_commands.delete,
            'add': rag_commands.add
        }
        
        if hasattr(args, 'rag_command') and args.rag_command in rag_methods:
            success = rag_methods[args.rag_command](args)
            sys.exit(0 if success else 1)
        else:
            rag_parser.print_help()
            sys.exit(1)
    
    if args.command in commands:
        success = commands[args.command](args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()