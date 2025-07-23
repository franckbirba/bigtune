#!/usr/bin/env python3
"""
RAG configuration management for BigTune
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RAGConfig:
    """RAG configuration structure"""
    name: str
    description: str = ""
    sources: List[Dict[str, Any]] = field(default_factory=list)
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'sources': self.sources,
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        return cls(**data)
    
    def save(self, path: Optional[str] = None) -> str:
        """Save configuration to file"""
        if not path:
            config_dir = Path.home() / ".bigtune" / "rag_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            path = config_dir / f"{self.name}.yaml"
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        return str(path)
    
    @classmethod
    def load(cls, name_or_path: str) -> 'RAGConfig':
        """Load configuration from file"""
        path = Path(name_or_path)
        
        # If not a path, look in default location
        if not path.exists():
            config_dir = Path.home() / ".bigtune" / "rag_configs"
            path = config_dir / f"{name_or_path}.yaml"
        
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {name_or_path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)


class RAGConfigManager:
    """Manage RAG configurations"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".bigtune" / "rag_configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def create_config(self, name: str, description: str = "") -> RAGConfig:
        """Create a new RAG configuration"""
        config = RAGConfig(name=name, description=description)
        config.save()
        return config
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations"""
        configs = []
        
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                config = RAGConfig.load(config_file.stem)
                configs.append({
                    'name': config.name,
                    'description': config.description,
                    'sources': len(config.sources),
                    'created_at': config.created_at
                })
            except Exception as e:
                configs.append({
                    'name': config_file.stem,
                    'error': str(e)
                })
        
        return configs
    
    def get_config(self, name: str) -> RAGConfig:
        """Get a specific configuration"""
        return RAGConfig.load(name)
    
    def delete_config(self, name: str) -> bool:
        """Delete a configuration"""
        config_path = self.config_dir / f"{name}.yaml"
        if config_path.exists():
            config_path.unlink()
            return True
        return False
    
    def add_source_to_config(self, 
                           config_name: str,
                           source_type: str,
                           source_path: str,
                           **kwargs) -> RAGConfig:
        """Add a source to existing configuration"""
        config = self.get_config(config_name)
        
        source = {
            'type': source_type,
            'path': source_path,
            'added_at': datetime.now().isoformat()
        }
        source.update(kwargs)
        
        config.sources.append(source)
        config.save()
        
        return config


# Predefined configurations for common use cases
PREDEFINED_CONFIGS = {
    'api_documentation': {
        'description': 'Configuration for API documentation RAG',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 300,
        'chunk_overlap': 50,
        'sources': []
    },
    
    'codebase_assistant': {
        'description': 'Configuration for codebase assistance',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 500,
        'chunk_overlap': 100,
        'sources': []
    },
    
    'knowledge_base': {
        'description': 'Configuration for general knowledge base',
        'embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': 400,
        'chunk_overlap': 75,
        'sources': []
    }
}


def create_predefined_config(template: str, name: str) -> RAGConfig:
    """Create a config from predefined template"""
    if template not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown template: {template}")
    
    config_data = PREDEFINED_CONFIGS[template].copy()
    config_data['name'] = name
    
    config = RAGConfig.from_dict(config_data)
    config.save()
    
    return config


# Example usage in YAML format for users
EXAMPLE_CONFIG = """
# Example RAG configuration file
name: my_api_rag
description: RAG for my API documentation

# Document sources
sources:
  - type: swagger
    path: http://api.example.com/swagger.json
    
  - type: github
    path: myorg/myrepo
    branch: main
    files: 
      - "docs/**/*.md"
      - "README.md"
    
  - type: local
    path: ./documentation
    recursive: true
    extensions: [.md, .txt]

# Embedding configuration
embedding_model: all-MiniLM-L6-v2
chunk_size: 400
chunk_overlap: 50
"""


if __name__ == "__main__":
    # Example usage
    manager = RAGConfigManager()
    
    # Create a new config
    config = manager.create_config(
        "console_bocal_rag",
        "RAG configuration for Console Bocal API"
    )
    
    # Add sources
    config = manager.add_source_to_config(
        "console_bocal_rag",
        "swagger",
        "http://localhost:5050/api/swagger.json"
    )
    
    print(f"Created config: {config.name}")
    print(f"Sources: {len(config.sources)}")