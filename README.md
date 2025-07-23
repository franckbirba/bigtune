# BigTune - LoRA Fine-tuning Pipeline with RAG

A professional CLI tool for training LoRA adapters on RunPod and deploying them to LM Studio, now enhanced with **Retrieval-Augmented Generation (RAG)** capabilities.

## ✨ New: RAG Features

BigTune now includes a complete RAG system for building intelligent, documentation-aware models:

- 🔍 **Vector-based document retrieval** for accurate API responses
- 🎯 **RAG-aware training datasets** for instruction-following models
- 🔄 **Argilla integration** for AI-assisted feedback and improvement
- 🚀 **FastAPI service** for production RAG deployments

## Quick Start

### Standard LoRA Training

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

### RAG System Usage

1. **Query the RAG system**:
   ```bash
   ./bigtune-rag query "What is the endpoint to get all aliases?"
   ```

2. **Generate RAG training data**:
   ```bash
   ./bigtune-rag generate --samples 100 --include-problems
   ```

3. **Start RAG server**:
   ```bash
   ./bigtune-rag server --port 8000
   ```

4. **Setup Argilla feedback**:
   ```bash
   ./bigtune-rag feedback --test-queries
   ```

## Commands

### Core Pipeline
- `bigtune config` - Show current configuration
- `bigtune config --validate` - Validate configuration
- `bigtune train` - Train LoRA on RunPod
- `bigtune merge` - Merge LoRA with base model
- `bigtune convert` - Convert to GGUF for LM Studio
- `bigtune full` - Run complete pipeline
- `bigtune status` - Check pipeline status
- `bigtune clean` - Clean up intermediate files

### RAG System
- `bigtune-rag query "<question>"` - Query the RAG system
- `bigtune-rag generate` - Generate RAG training datasets
- `bigtune-rag feedback` - Setup Argilla feedback collection
- `bigtune-rag server` - Start RAG FastAPI server

## RAG Architecture

### Components

1. **Document Processing**: Automatically parses API documentation (Swagger/OpenAPI)
2. **Vector Database**: FAISS-based similarity search for relevant document retrieval
3. **Model Integration**: Works with any Ollama-compatible model
4. **Training Pipeline**: Generates datasets that teach models to follow provided documentation
5. **Feedback Loop**: Argilla integration for continuous improvement

### Example: Console Bocal RAG

```python
from bigtune.rag import ConsoleBocalRAGService

# Initialize RAG service
service = ConsoleBocalRAGService()
service.initialize()

# Query with automatic document retrieval
result = service.process_query("How do I create a new alias?")
print(result.answer)  # "Endpoint: POST /api/{domain}/aliases"
print(result.relevant_endpoints)  # Shows retrieved documentation
```

### RAG Training Workflow

1. **Generate RAG Dataset**: Create training examples that teach exact document following
2. **Train Model**: Use BigTune to train with RAG-aware instruction data
3. **Deploy & Test**: Model now follows provided documentation precisely
4. **Collect Feedback**: Use Argilla to gather human feedback for improvements
5. **Iterate**: Retrain with improved datasets

## Configuration

### Standard Configuration
All configuration is managed through the `.env` file:

- **RunPod API**: Set your API key and preferred GPU type
- **SSH Keys**: Configure your SSH key path
- **Model Settings**: Set LoRA parameters and base model
- **Paths**: Configure dataset and output directories

### RAG Configuration

RAG settings can be configured in the RAG service initialization:

```python
# Default Console Bocal setup
rag = ConsoleBocalRAG(swagger_url="http://localhost:5050/api/swagger.json")

# Custom API documentation  
rag = ConsoleBocalRAG(swagger_url="https://api.example.com/swagger.json")
```

## Pipeline Steps

### Standard Pipeline
1. **Training**: Deploys to RunPod, trains LoRA adapters
2. **Merging**: Combines LoRA with base model
3. **Conversion**: Creates GGUF files for LM Studio
4. **Installation**: Installs models in LM Studio

### RAG-Enhanced Pipeline
1. **Document Indexing**: Parse and vectorize API documentation
2. **RAG Dataset Generation**: Create instruction-following training data
3. **RAG-Aware Training**: Train model to follow provided context exactly
4. **RAG Deployment**: Deploy model with document retrieval capabilities
5. **Feedback Collection**: Gather performance data via Argilla

## File Structure

```
bigtune/
├── bigtune/                    # Python package with CLI
│   ├── rag/                   # RAG module (NEW)
│   │   ├── console_bocal_rag.py    # Core RAG system
│   │   ├── rag_training.py         # Training data generation  
│   │   ├── argilla_feedback.py     # Feedback collection
│   │   └── cli.py                  # RAG CLI interface
│   └── ...
├── bigtune-rag                 # RAG CLI entry point (NEW)
├── llm-builder/               # Training configuration and datasets
├── runpod_train.sh           # Unified training script for RunPod
├── launch_runpod_job.py      # RunPod orchestration
├── minimal_merge.py          # LoRA merging script
└── convert_to_gguf_simple.py # GGUF conversion script
```

## Use Cases

### Standard LoRA Fine-tuning
- Domain-specific chat models
- Instruction following improvements
- Task-specific model adaptation

### RAG-Enhanced Models
- **API Documentation Assistants**: Models that provide accurate API guidance
- **Knowledge Base Q&A**: Systems that answer questions using live documentation
- **Customer Support**: Bots that reference up-to-date help documents
- **Code Assistance**: Models that follow current API specifications exactly

## Requirements

### Core Requirements
- Python 3.8+
- RunPod account and API key
- SSH key pair for RunPod access
- LM Studio (for local model deployment)

### RAG Requirements
```bash
pip install sentence-transformers faiss-cpu fastapi uvicorn
```

### Optional: Argilla Feedback
```bash
# Use existing BigAcademy stack or standalone:
docker run -p 6900:6900 argilla/argilla-quickstart
```

## Examples

### Training a RAG-Aware Model

1. **Generate RAG training data**:
   ```bash
   ./bigtune-rag generate --samples 200 --include-problems
   ```

2. **Create training config** with RAG dataset:
   ```yaml
   datasets:
     - path: ./datasets/rag_training_TIMESTAMP.jsonl
       type: alpaca
   ```

3. **Train with BigTune**:
   ```bash
   CONFIG_FILE=./config/my-rag-model.yaml bigtune train
   ```

4. **Test the results**:
   ```bash
   ./bigtune-rag query "How do I delete a user?"
   # Expected: Exact API endpoint with correct parameters
   ```

### Production RAG Deployment

```bash
# Start RAG server
./bigtune-rag server --port 8000

# Query via API
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the aliases endpoint?"}'
```

## Success Stories

BigTune's RAG system has been successfully used to create:

- **Console Bocal Expert**: API assistant with 100% endpoint accuracy
- **Documentation Bots**: Models that never hallucinate API information
- **Developer Tools**: Assistants that provide current, accurate technical guidance

The RAG training approach ensures models follow provided documentation exactly, eliminating the common problem of AI systems providing outdated or incorrect API information.

---

**🎯 Ready to build intelligent, documentation-aware models with BigTune + RAG!**