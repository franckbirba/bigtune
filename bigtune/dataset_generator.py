#!/usr/bin/env python3
"""
Generic dataset generation utility for BigTune
Industrialized pattern from Console Bocal specific scripts
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import subprocess
import tempfile
import yaml

class DatasetGenerator:
    """Generic dataset generator that works with BigAcademy"""
    
    def __init__(self, bigacademy_path: Optional[str] = None):
        # Auto-detect BigAcademy path
        if not bigacademy_path:
            possible_paths = [
                Path("../bigacademy"),
                Path("/Users/franckbirba/DEV/TEST-CREWAI/bigacademy"),
                Path.cwd() / "bigacademy"
            ]
            
            for path in possible_paths:
                if path.exists() and (path / "generate_axolotl_dataset.py").exists():
                    bigacademy_path = str(path)
                    break
        
        if not bigacademy_path or not Path(bigacademy_path).exists():
            raise FileNotFoundError(
                "BigAcademy not found. Please specify --bigacademy-path or ensure it's in a standard location"
            )
        
        self.bigacademy_path = Path(bigacademy_path)
        self.original_cwd = Path.cwd()
        
        # Add BigAcademy to Python path
        sys.path.insert(0, str(self.bigacademy_path))
    
    def list_available_agents(self) -> List[str]:
        """List available agent profiles in BigAcademy"""
        agents_dir = self.bigacademy_path / "configs" / "agents"
        if not agents_dir.exists():
            return []
        
        agents = []
        for agent_file in agents_dir.glob("*.yaml"):
            agents.append(agent_file.stem)
        
        return agents
    
    def list_available_templates(self, agent_name: str) -> List[str]:
        """List available templates for an agent"""
        try:
            # Change to BigAcademy directory for imports
            os.chdir(self.bigacademy_path)
            
            # Import the generator
            from generate_axolotl_dataset import load_agent_config
            
            agent_config = load_agent_config(agent_name)
            if agent_config and 'templates' in agent_config:
                return list(agent_config['templates'].keys())
            
            return []
            
        except Exception as e:
            print(f"Error loading agent templates: {e}")
            return []
        finally:
            os.chdir(self.original_cwd)
    
    def generate_dataset(self, 
                        agent_name: str,
                        samples_per_template: int = 100,
                        specific_templates: Optional[List[str]] = None,
                        output_prefix: str = "generated") -> List[str]:
        """Generate dataset using BigAcademy"""
        
        try:
            # Change to BigAcademy directory for generation
            os.chdir(self.bigacademy_path)
            
            # Import the generator
            from generate_axolotl_dataset import generate_axolotl_datasets
            
            print(f"🚀 Generating dataset for agent: {agent_name}")
            print(f"📊 Samples per template: {samples_per_template}")
            
            if specific_templates:
                print(f"🎯 Using specific templates: {', '.join(specific_templates)}")
                # TODO: Add template filtering support to BigAcademy
                print("⚠️  Template filtering not yet implemented in BigAcademy")
            
            # Generate the datasets
            files = generate_axolotl_datasets(agent_name, samples_per_template)
            
            if files:
                print(f"✅ Successfully generated {len(files) * samples_per_template} training samples!")
                print("Dataset files:")
                for file_path in files:
                    print(f"   📄 {file_path}")
                return files
            else:
                print("❌ Dataset generation failed")
                return []
                
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            return []
        finally:
            os.chdir(self.original_cwd)
    
    def generate_large_dataset(self,
                              agent_name: str, 
                              target_samples: int = 1000) -> List[str]:
        """Generate a large dataset by calculating optimal samples per template"""
        
        templates = self.list_available_templates(agent_name)
        if not templates:
            print(f"❌ No templates found for agent: {agent_name}")
            return []
        
        samples_per_template = max(1, target_samples // len(templates))
        
        print(f"🎯 Target samples: {target_samples}")
        print(f"📋 Available templates: {len(templates)}")
        print(f"🔢 Samples per template: {samples_per_template}")
        
        return self.generate_dataset(agent_name, samples_per_template)
    
    def quick_generate(self, 
                      agent_name: str,
                      size: str = "medium") -> List[str]:
        """Quick dataset generation with predefined sizes"""
        
        size_map = {
            "small": 50,
            "medium": 200,
            "large": 500,
            "xl": 1000
        }
        
        samples = size_map.get(size, 200)
        print(f"🚀 Quick generation: {size} dataset ({samples} samples per template)")
        
        return self.generate_dataset(agent_name, samples)
    
    def create_training_script(self, 
                              agent_name: str,
                              samples_per_template: int,
                              output_file: str = "generate_dataset.py") -> str:
        """Create a standalone dataset generation script"""
        
        script_content = f'''#!/usr/bin/env python3
"""
Generated dataset creation script for {agent_name}
Created by BigTune DatasetGenerator
"""

import sys
import os
from pathlib import Path

# Add BigAcademy to path
bigacademy_path = Path("{self.bigacademy_path}")
sys.path.insert(0, str(bigacademy_path))

# Change to BigAcademy directory for relative paths
os.chdir(bigacademy_path)

# Import the generator
from generate_axolotl_dataset import generate_axolotl_datasets

# Generate dataset
print("🚀 Generating dataset for {agent_name} with {samples_per_template} samples per template...")
files = generate_axolotl_datasets("{agent_name}", samples_per_template={samples_per_template})

if files:
    total_samples = len(files) * {samples_per_template}
    print(f"\\n✅ Successfully generated {{total_samples}} training samples!")
    print("Dataset files:")
    for file_path in files:
        print(f"   📄 {{file_path}}")
else:
    print("❌ Dataset generation failed")

print("\\n✅ Dataset generation complete!")
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_file, 0o755)
        
        print(f"💾 Created dataset generation script: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="BigTune Dataset Generator - Industrialized dataset creation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available agents
  python dataset_generator.py --list-agents
  
  # Generate medium dataset
  python dataset_generator.py --agent console_bocal_support --quick medium
  
  # Generate large custom dataset
  python dataset_generator.py --agent my_agent --samples 500
  
  # Create standalone generation script
  python dataset_generator.py --agent my_agent --create-script --samples 200
        """
    )
    
    # BigAcademy path
    parser.add_argument('--bigacademy-path', 
                       help='Path to BigAcademy installation')
    
    # Agent selection
    parser.add_argument('--agent', 
                       help='Agent profile name to generate data for')
    
    # List operations
    parser.add_argument('--list-agents', action='store_true',
                       help='List available agent profiles')
    parser.add_argument('--list-templates', 
                       help='List templates for specified agent')
    
    # Generation options
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per template (default: 100)')
    parser.add_argument('--quick', choices=['small', 'medium', 'large', 'xl'],
                       help='Quick generation with predefined sizes')
    parser.add_argument('--target', type=int,
                       help='Target total samples (auto-calculate per template)')
    
    # Output options
    parser.add_argument('--create-script', action='store_true',
                       help='Create standalone generation script')
    parser.add_argument('--script-name', default='generate_dataset.py',
                       help='Name for generated script')
    
    args = parser.parse_args()
    
    try:
        generator = DatasetGenerator(args.bigacademy_path)
        
        # List operations
        if args.list_agents:
            agents = generator.list_available_agents()
            print("Available agent profiles:")
            for agent in sorted(agents):
                print(f"  📋 {agent}")
            return
        
        if args.list_templates:
            templates = generator.list_available_templates(args.list_templates)
            print(f"Templates for {args.list_templates}:")
            for template in templates:
                print(f"  🎯 {template}")
            return
        
        # Generation operations
        if not args.agent:
            print("❌ Agent name required for generation operations")
            parser.print_help()
            return
        
        # Create script
        if args.create_script:
            samples = args.samples
            if args.target:
                templates = generator.list_available_templates(args.agent)
                samples = max(1, args.target // len(templates)) if templates else args.samples
            
            generator.create_training_script(
                args.agent, 
                samples,
                args.script_name
            )
            return
        
        # Generate dataset
        files = []
        if args.quick:
            files = generator.quick_generate(args.agent, args.quick)
        elif args.target:
            files = generator.generate_large_dataset(args.agent, args.target)
        else:
            files = generator.generate_dataset(args.agent, args.samples)
        
        if files:
            print(f"\n🎉 Dataset generation successful!")
            print(f"📁 Generated {len(files)} dataset files")
            print(f"🔗 Ready for BigTune training!")
        else:
            print("\n❌ Dataset generation failed")
            sys.exit(1)
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()