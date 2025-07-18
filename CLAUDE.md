# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BigTune is a LoRA fine-tuning pipeline CLI tool that orchestrates training on RunPod and deployment to LM Studio. The system automates the complete workflow from training LoRA adapters to converting models for local deployment.

## Development Commands

### Installation and Setup
```bash
# Install the package in development mode
pip install -e .

# Copy and configure environment file
cp .env.example .env
# Edit .env with your RunPod API key and other settings
```

### Core CLI Commands
```bash
# Validate configuration
bigtune config --validate

# Run individual pipeline steps
bigtune train              # Train LoRA on RunPod
bigtune merge              # Merge LoRA with base model  
bigtune convert            # Convert to GGUF for LM Studio

# Run complete pipeline
bigtune full               # Execute all steps sequentially

# Utility commands
bigtune status             # Check pipeline status
bigtune clean              # Clean intermediate files
bigtune clean --all        # Clean everything including training output
```

### Testing
No formal test framework is configured. Use manual testing with the CLI commands above.

## Architecture Overview

### Core Components

1. **CLI Interface** (`bigtune/cli.py`): Main orchestration layer that manages the three-stage pipeline
2. **Configuration Management** (`bigtune/config.py`): Handles .env file loading and environment variable management
3. **RunPod Orchestration** (`launch_runpod_job.py`): Manages cloud GPU provisioning and training execution
4. **Model Merging** (`minimal_merge.py`): Combines LoRA adapters with base models using PEFT
5. **GGUF Conversion** (`convert_to_gguf_simple.py`): Converts models to GGUF format for LM Studio

### Pipeline Flow

1. **Training Phase**: Deploys to RunPod with automatic environment detection and setup via `runpod_train.sh`
2. **Merging Phase**: Downloads training artifacts and merges LoRA adapters with base model
3. **Conversion Phase**: Creates quantized GGUF models at different sizes (q4_0, q5_0, q8_0, f16)

### Key Directories

- `bigtune/`: Python package with CLI and configuration
- `llm-builder/`: Training configuration and datasets for axolotl
- `llm-builder/config/`: YAML configuration files for training
- `llm-builder/datasets/`: Training datasets in JSONL format
- `output/`: Training artifacts and LoRA adapters
- `gguf_models/`: Converted GGUF models for LM Studio

### Configuration System

The project uses a hierarchical configuration system:
1. `.env` file for environment-specific settings
2. `bigtune/config.py` for centralized configuration management
3. `llm-builder/config/*.yaml` for axolotl training parameters

### RunPod Integration

The system automatically detects RunPod environments:
- `axolotl-cloud`: Pre-configured images with axolotl installed
- `cuda`: Generic CUDA images requiring setup
- Automatic dependency installation and environment preparation

### Model Support

Currently configured for Mistral-7B-Instruct-v0.2 with LoRA fine-tuning parameters:
- LoRA rank (r): 8
- LoRA alpha: 16
- Learning rate: 2e-4
- Target modules: All linear projections (q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj)

## Important Implementation Notes

- All scripts can run independently or as part of the orchestrated pipeline
- Configuration validation is performed before expensive operations
- RunPod jobs include automatic cleanup and resource management
- GGUF conversion supports multiple quantization levels for different memory requirements
- LM Studio integration includes automatic model installation paths