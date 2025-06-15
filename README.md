# BigTune - LoRA Fine-tuning Pipeline

A professional CLI tool for training LoRA adapters on RunPod and deploying them to LM Studio.

## Quick Start

1. **Configure your environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

2. **Install the package**:
   ```bash
   pip install -e .
   ```

3. **Run the full pipeline**:
   ```bash
   bigtune full
   ```

## Commands

- `bigtune config` - Show current configuration
- `bigtune config --validate` - Validate configuration
- `bigtune train` - Train LoRA on RunPod
- `bigtune merge` - Merge LoRA with base model
- `bigtune convert` - Convert to GGUF for LM Studio
- `bigtune full` - Run complete pipeline
- `bigtune status` - Check pipeline status
- `bigtune clean` - Clean up intermediate files

## Configuration

All configuration is managed through the `.env` file:

- **RunPod API**: Set your API key and preferred GPU type
- **SSH Keys**: Configure your SSH key path
- **Model Settings**: Set LoRA parameters and base model
- **Paths**: Configure dataset and output directories

## Pipeline Steps

1. **Training**: Deploys to RunPod, trains LoRA adapters
2. **Merging**: Combines LoRA with base model
3. **Conversion**: Creates GGUF files for LM Studio
4. **Installation**: Installs models in LM Studio

## File Structure

- `bigtune/` - Python package with CLI
- `llm-builder/` - Training configuration and datasets
- `runpod_train.sh` - Unified training script for RunPod
- `launch_runpod_job.py` - RunPod orchestration
- `minimal_merge.py` - LoRA merging script
- `convert_to_gguf_simple.py` - GGUF conversion script

## Requirements

- Python 3.8+
- RunPod account and API key
- SSH key pair for RunPod access
- LM Studio (for local model deployment)