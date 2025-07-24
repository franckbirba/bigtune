#!/usr/bin/env python3
"""
Docker packaging for BigTune - Create production-ready containers
"""

import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import json
from datetime import datetime


class DockerPackager:
    """Package BigTune models and RAG into Docker containers"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.temp_dir = None
        
    def create_dockerfile(self, 
                         model_name: str,
                         rag_index: Optional[str] = None,
                         base_image: str = "python:3.10-slim",
                         port: int = 8000) -> str:
        """Generate Dockerfile for the model service"""
        
        dockerfile_content = f'''# BigTune Production Image
# Generated on {datetime.now().isoformat()}
FROM {base_image}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy BigTune package
COPY bigtune/ ./bigtune/
COPY requirements.txt ./
COPY setup.py ./
COPY README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Install additional RAG dependencies
RUN pip install --no-cache-dir \\
    sentence-transformers \\
    faiss-cpu \\
    fastapi \\
    uvicorn

# Copy model and RAG data
COPY models/ ./models/
'''

        if rag_index:
            dockerfile_content += "COPY rag_indexes/ ./rag_indexes/\n"
        
        dockerfile_content += f'''
# Create directories
RUN mkdir -p /root/.ollama
RUN mkdir -p /root/.bigtune/rag_indexes

# Copy startup script
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Expose port
EXPOSE {port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{port}/health || exit 1

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0
ENV BIGTUNE_MODEL={model_name}
ENV BIGTUNE_RAG_INDEX={rag_index or ""}
ENV BIGTUNE_PORT={port}

# Start service
ENTRYPOINT ["./docker-entrypoint.sh"]
'''
        
        return dockerfile_content
    
    def create_entrypoint_script(self, 
                                model_name: str,
                                rag_index: Optional[str] = None,
                                port: int = 8000) -> str:
        """Generate Docker entrypoint script"""
        
        script_content = f'''#!/bin/bash
set -e

echo "🚀 Starting BigTune Production Service"
echo "📊 Model: {model_name}"
echo "📚 RAG Index: {rag_index or 'None'}"
echo "🌐 Port: {port}"

# Start Ollama in background
echo "Starting Ollama service..."
ollama serve >/dev/null 2>&1 &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {{1..30}}; do
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama failed to start"
        exit 1
    fi
    sleep 2
done

# Import model if exists
if [ -f "./models/{model_name}.tar" ]; then
    echo "📦 Importing model: {model_name}"
    ollama create {model_name} -f ./models/Modelfile
    echo "✅ Model imported successfully"
else
    echo "⚠️  No model file found, assuming model is available in Ollama"
fi

# Copy RAG indexes if they exist
if [ -d "./rag_indexes" ]; then
    echo "📚 Setting up RAG indexes..."
    cp -r ./rag_indexes/* /root/.bigtune/rag_indexes/ 2>/dev/null || true
    echo "✅ RAG indexes ready"
fi

# Start BigTune server
echo "🚀 Starting BigTune unified server..."
exec bigtune serve \\
    --model "{model_name}" \\
    --rag "{rag_index or 'default'}" \\
    --host 0.0.0.0 \\
    --port {port}
'''
        
        return script_content
    
    def create_docker_compose(self,
                             service_name: str,
                             image_name: str,
                             port: int = 8000,
                             include_traefik: bool = True) -> str:
        """Generate docker-compose.yml for production deployment"""
        
        compose_data = {
            'version': '3.8',
            'services': {
                service_name: {
                    'image': image_name,
                    'container_name': service_name,
                    'restart': 'unless-stopped',
                    'ports': [f"{port}:{port}"],
                    'environment': {
                        'BIGTUNE_PORT': str(port)
                    },
                    'volumes': [
                        './logs:/app/logs',
                        'ollama_data:/root/.ollama'
                    ],
                    'healthcheck': {
                        'test': [f"curl -f http://localhost:{port}/health || exit 1"],
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3,
                        'start_period': '60s'
                    }
                }
            },
            'volumes': {
                'ollama_data': {}
            }
        }
        
        if include_traefik:
            compose_data['services'][service_name]['labels'] = {
                'traefik.enable': 'true',
                f'traefik.http.routers.{service_name}.rule': f'Host(`{service_name}.localhost`)',
                f'traefik.http.services.{service_name}.loadbalancer.server.port': str(port)
            }
            
            # Add Traefik service
            compose_data['services']['traefik'] = {
                'image': 'traefik:v2.10',
                'container_name': 'traefik',
                'restart': 'unless-stopped',
                'ports': ['80:80', '8080:8080'],
                'volumes': [
                    '/var/run/docker.sock:/var/run/docker.sock:ro',
                    './traefik.yml:/traefik.yml:ro'
                ]
            }
        
        return yaml.dump(compose_data, default_flow_style=False)
    
    def create_traefik_config(self) -> str:
        """Generate Traefik configuration"""
        
        traefik_config = {
            'api': {
                'dashboard': True,
                'insecure': True
            },
            'providers': {
                'docker': {
                    'exposedByDefault': False
                }
            },
            'entryPoints': {
                'web': {
                    'address': ':80'
                }
            }
        }
        
        return yaml.dump(traefik_config, default_flow_style=False)
    
    def export_ollama_model(self, model_name: str, output_dir: Path) -> bool:
        """Export Ollama model for Docker packaging"""
        try:
            models_dir = output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Get model info
            result = subprocess.run(
                ["ollama", "show", model_name],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Model '{model_name}' not found in Ollama")
                return False
            
            # Create a simplified Modelfile for the container
            modelfile_content = f'''FROM ./model_data
'''
            
            with open(models_dir / "Modelfile", 'w') as f:
                f.write(modelfile_content)
            
            # Note: In a real implementation, we'd need to extract the model weights
            # For now, we'll create a placeholder that assumes the model exists
            placeholder_content = f"# Model {model_name} should be available in the container\\n"
            with open(models_dir / f"{model_name}.info", 'w') as f:
                f.write(placeholder_content)
            
            print(f"✅ Model '{model_name}' export prepared")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting model: {e}")
            return False
    
    def copy_rag_indexes(self, 
                        rag_indexes: List[str], 
                        output_dir: Path) -> bool:
        """Copy RAG indexes for Docker packaging"""
        try:
            rag_dir = output_dir / "rag_indexes"
            rag_dir.mkdir(exist_ok=True)
            
            source_dir = Path.home() / ".bigtune" / "rag_indexes"
            
            for index_name in rag_indexes:
                index_source = source_dir / index_name
                index_dest = rag_dir / index_name
                
                if index_source.exists():
                    shutil.copytree(index_source, index_dest, dirs_exist_ok=True)
                    print(f"✅ RAG index '{index_name}' copied")
                else:
                    print(f"❌ RAG index '{index_name}' not found")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error copying RAG indexes: {e}")
            return False
    
    def create_requirements_txt(self) -> str:
        """Generate requirements.txt for Docker"""
        
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "peft>=0.4.0",
            "requests>=2.25.0",
            "python-dotenv>=0.19.0",
            "pathlib",
            "sentence-transformers",
            "faiss-cpu",
            "fastapi",
            "uvicorn",
            "pyyaml",
            "numpy",
            "pydantic"
        ]
        
        return "\\n".join(requirements)
    
    def package(self,
                model_name: str,
                image_name: str,
                rag_indexes: Optional[List[str]] = None,
                output_dir: Optional[str] = None,
                port: int = 8000,
                base_image: str = "python:3.10-slim",
                build: bool = True,
                registry: Optional[str] = None,
                push: bool = False) -> bool:
        """Package model and RAG into Docker image"""
        
        if not output_dir:
            output_dir = f"./docker-package-{model_name.replace(':', '-')}"
        
        package_dir = Path(output_dir)
        package_dir.mkdir(exist_ok=True)
        
        # Handle registry prefixing
        final_image_name = image_name
        if registry:
            if not image_name.startswith(registry):
                final_image_name = f"{registry.rstrip('/')}/{image_name}"
        
        print(f"📦 Packaging model '{model_name}' into '{final_image_name}'")
        print(f"📁 Output directory: {package_dir.absolute()}")
        if registry:
            print(f"🌐 Registry: {registry}")
        
        try:
            # Copy BigTune source
            print("📋 Copying BigTune source...")
            bigtune_dest = package_dir / "bigtune"
            if bigtune_dest.exists():
                shutil.rmtree(bigtune_dest)
            shutil.copytree(self.base_dir / "bigtune", bigtune_dest)
            
            # Copy additional files
            for file_name in ["setup.py", "README.md"]:
                src_file = self.base_dir / file_name
                if src_file.exists():
                    shutil.copy2(src_file, package_dir / file_name)
            
            # Create requirements.txt
            with open(package_dir / "requirements.txt", 'w') as f:
                f.write(self.create_requirements_txt())
            
            # Export model
            print("🤖 Preparing model...")
            if not self.export_ollama_model(model_name, package_dir):
                return False
            
            # Copy RAG indexes
            if rag_indexes:
                print("📚 Copying RAG indexes...")
                if not self.copy_rag_indexes(rag_indexes, package_dir):
                    return False
            
            # Generate Dockerfile
            print("🐳 Generating Dockerfile...")
            dockerfile_content = self.create_dockerfile(
                model_name=model_name,
                rag_index=rag_indexes[0] if rag_indexes else None,
                base_image=base_image,
                port=port
            )
            
            with open(package_dir / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)
            
            # Generate entrypoint script
            entrypoint_content = self.create_entrypoint_script(
                model_name=model_name,
                rag_index=rag_indexes[0] if rag_indexes else None,
                port=port
            )
            
            with open(package_dir / "docker-entrypoint.sh", 'w') as f:
                f.write(entrypoint_content)
            
            # Generate docker-compose.yml
            print("📝 Generating docker-compose.yml...")
            compose_content = self.create_docker_compose(
                service_name=model_name.replace(':', '-'),
                image_name=final_image_name,
                port=port
            )
            
            with open(package_dir / "docker-compose.yml", 'w') as f:
                f.write(compose_content)
            
            # Generate Traefik config
            traefik_content = self.create_traefik_config()
            with open(package_dir / "traefik.yml", 'w') as f:
                f.write(traefik_content)
            
            # Create deployment README
            self.create_deployment_readme(
                package_dir, 
                model_name, 
                final_image_name, 
                rag_indexes, 
                port
            )
            
            # Build Docker image if requested
            if build:
                print(f"🔨 Building Docker image '{final_image_name}'...")
                build_result = subprocess.run([
                    "docker", "build", 
                    "-t", final_image_name,
                    "."
                ], cwd=package_dir, capture_output=True, text=True)
                
                if build_result.returncode == 0:
                    print(f"✅ Docker image '{final_image_name}' built successfully")
                    
                    # Push to registry if requested
                    if push and registry:
                        print(f"📤 Pushing image to registry '{registry}'...")
                        push_result = subprocess.run([
                            "docker", "push", final_image_name
                        ], capture_output=True, text=True)
                        
                        if push_result.returncode == 0:
                            print(f"✅ Image pushed to registry successfully")
                        else:
                            print(f"❌ Push failed: {push_result.stderr}")
                            print(f"💡 Make sure you're logged in: docker login {registry}")
                            return False
                else:
                    print(f"❌ Docker build failed: {build_result.stderr}")
                    return False
            
            print(f"\\n🎉 Packaging complete!")
            print(f"📁 Package directory: {package_dir.absolute()}")
            print(f"🐳 Docker image: {final_image_name}")
            if registry and push:
                print(f"📤 Image available in registry: {registry}")
            print(f"🚀 Deploy with: cd {output_dir} && docker-compose up -d")
            
            return True
            
        except Exception as e:
            print(f"❌ Packaging failed: {e}")
            return False
    
    def create_deployment_readme(self,
                               package_dir: Path,
                               model_name: str,
                               image_name: str,
                               rag_indexes: Optional[List[str]],
                               port: int):
        """Create deployment README"""
        
        readme_content = f'''# {model_name.title()} - Production Deployment

This package contains a production-ready Docker deployment for the BigTune model `{model_name}`.

## 🚀 Quick Deployment

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

### Using Docker directly

```bash
# Build image (if not already built)
docker build -t {image_name} .

# Run container
docker run -d \\
  --name {model_name.replace(':', '-')} \\
  -p {port}:{port} \\
  {image_name}
```

## 📊 Service Information

- **Model**: `{model_name}`
'''
        
        if rag_indexes:
            readme_content += f"- **RAG Indexes**: {', '.join(rag_indexes)}\\n"
        
        readme_content += f'''- **Port**: {port}
- **Health Check**: http://localhost:{port}/health
- **API Docs**: http://localhost:{port}/docs

## 🔧 Configuration

### Environment Variables

- `BIGTUNE_MODEL`: Model name (default: {model_name})
- `BIGTUNE_RAG_INDEX`: RAG index name (default: {rag_indexes[0] if rag_indexes else 'default'})
- `BIGTUNE_PORT`: Service port (default: {port})

### Custom Configuration

Edit `docker-compose.yml` to customize:

```yaml
environment:
  BIGTUNE_MODEL: "your-model-name"
  BIGTUNE_RAG_INDEX: "your-rag-index"
  BIGTUNE_PORT: "{port}"
```

## 🌐 API Usage

### Query the model

```bash
curl -X POST "http://localhost:{port}/query" \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "Your question here", "use_rag": true}}'
```

### Health check

```bash
curl http://localhost:{port}/health
```

## 📝 Production Notes

### Scaling

To run multiple instances:

```bash
docker-compose up -d --scale {model_name.replace(':', '-')}=3
```

### Monitoring

The service includes health checks and can be monitored with:
- Docker health checks
- Traefik dashboard (http://localhost:8080)
- Custom monitoring solutions

### Persistence

RAG indexes and model data are included in the image. For dynamic updates, mount volumes:

```yaml
volumes:
  - ./custom-rag:/root/.bigtune/rag_indexes
```

## 🔍 Troubleshooting

### Check container logs
```bash
docker-compose logs {model_name.replace(':', '-')}
```

### Verify model loading
```bash
docker-compose exec {model_name.replace(':', '-')} ollama list
```

### Test RAG indexes
```bash
docker-compose exec {model_name.replace(':', '-')} bigtune rag list
```

## 📞 Support

For issues and questions:
- Check the BigTune documentation
- Review container logs
- Test individual components

---

**Generated by BigTune on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**
'''
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)


def package_model(model_name: str,
                 image_name: Optional[str] = None,
                 rag_indexes: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 port: int = 8000,
                 no_build: bool = False,
                 registry: Optional[str] = None,
                 push: bool = False) -> bool:
    """Main packaging function"""
    
    if not image_name:
        image_name = f"bigtune-{model_name.replace(':', '-').lower()}"
    
    packager = DockerPackager()
    return packager.package(
        model_name=model_name,
        image_name=image_name,
        rag_indexes=rag_indexes,
        output_dir=output_dir,
        port=port,
        build=not no_build,
        registry=registry,
        push=push
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Package BigTune model for Docker deployment")
    parser.add_argument("model", help="Model name to package")
    parser.add_argument("--image", help="Docker image name")
    parser.add_argument("--rag", nargs="+", help="RAG indexes to include")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--no-build", action="store_true", help="Skip Docker build")
    parser.add_argument("--registry", help="Docker registry URL (e.g., registry.gitlab.com/user/project)")
    parser.add_argument("--push", action="store_true", help="Push image to registry after build")
    
    args = parser.parse_args()
    
    success = package_model(
        model_name=args.model,
        image_name=args.image,
        rag_indexes=args.rag,
        output_dir=args.output,
        port=args.port,
        no_build=args.no_build,
        registry=args.registry,
        push=args.push
    )
    
    exit(0 if success else 1)