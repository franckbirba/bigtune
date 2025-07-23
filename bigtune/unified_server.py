#!/usr/bin/env python3
"""
Unified server for BigTune - Serves model + RAG together
"""

import os
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import subprocess
from pathlib import Path
from .rag.generic_rag import GenericRAG


class QueryRequest(BaseModel):
    """Request model for queries"""
    question: str
    use_rag: bool = True
    rag_index: str = "default"
    model_name: Optional[str] = None
    top_k: int = 3
    temperature: float = 0.3
    max_tokens: int = 500


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    model_used: str
    rag_used: bool
    sources: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_available: bool
    rag_indexes: List[str]
    default_model: Optional[str]


class UnifiedServer:
    """Unified server combining model inference and RAG"""
    
    def __init__(self, 
                 default_model: Optional[str] = None,
                 default_rag_index: Optional[str] = None,
                 enable_cors: bool = True):
        
        self.app = FastAPI(
            title="BigTune Unified Server",
            description="Serves BigTune models with RAG capabilities",
            version="1.0.0"
        )
        
        # Configuration
        self.default_model = default_model or self._find_default_model()
        self.default_rag_index = default_rag_index or "default"
        self.rag_instances = {}
        
        # Setup CORS if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Register routes
        self._register_routes()
    
    def _find_default_model(self) -> Optional[str]:
        """Find a default model from Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        # Prefer bigtune models
                        if 'bigtune' in model_name or 'console-bocal' in model_name:
                            return model_name
                        # Return first model as fallback
                        if not hasattr(self, '_first_model'):
                            self._first_model = model_name
                
                return getattr(self, '_first_model', None)
                
        except Exception:
            return None
    
    def _get_rag_instance(self, index_name: str) -> Optional[GenericRAG]:
        """Get or create RAG instance"""
        if index_name not in self.rag_instances:
            try:
                rag = GenericRAG(index_name)
                if rag.documents:  # Only cache if has documents
                    self.rag_instances[index_name] = rag
                else:
                    return None
            except Exception:
                return None
        
        return self.rag_instances.get(index_name)
    
    def _query_ollama(self, 
                     prompt: str, 
                     model: str,
                     temperature: float = 0.3,
                     max_tokens: int = 500) -> str:
        """Query Ollama model"""
        try:
            # Build command - Ollama doesn't support temperature via CLI
            # We'll include it in the prompt for models that support it
            cmd = [
                "ollama", "run",
                model,
                prompt
            ]
            
            # Run Ollama
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"Ollama error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Model query timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def _build_rag_prompt(self, question: str, documents: List[Any]) -> str:
        """Build prompt with RAG context"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"DOCUMENT {i}:")
            context_parts.append(f"Source: {doc.source_type} - {doc.title}")
            context_parts.append(doc.content)
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert assistant with access to documentation.

CRITICAL: Use ONLY the information provided in the documentation below. Do not modify paths, endpoints, or add information not present in the provided context.

DOCUMENTATION:
{context}

QUESTION: {question}

Please provide a precise answer based solely on the documentation provided above."""
        
        return prompt, context
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check server health and available resources"""
            # Get available RAG indexes
            rag_indexes = [idx['name'] for idx in GenericRAG.list_indexes()]
            
            # Check if model is available
            model_available = False
            if self.default_model:
                try:
                    # Quick test
                    self._query_ollama("Hi", self.default_model, max_tokens=10)
                    model_available = True
                except Exception:
                    pass
            
            return HealthResponse(
                status="healthy",
                model_available=model_available,
                rag_indexes=rag_indexes,
                default_model=self.default_model
            )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest):
            """Process a query with optional RAG"""
            
            # Determine model to use
            model_name = request.model_name or self.default_model
            if not model_name:
                raise HTTPException(
                    status_code=400, 
                    detail="No model specified and no default model available"
                )
            
            # Process with or without RAG
            if request.use_rag:
                # Get RAG instance
                rag = self._get_rag_instance(request.rag_index)
                if not rag:
                    raise HTTPException(
                        status_code=404,
                        detail=f"RAG index '{request.rag_index}' not found or empty"
                    )
                
                # Search for relevant documents
                documents = rag.search(request.question, k=request.top_k)
                
                if documents:
                    # Build RAG prompt
                    prompt, context = self._build_rag_prompt(request.question, documents)
                    
                    # Query model with context
                    answer = self._query_ollama(
                        prompt, 
                        model_name,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    
                    # Prepare sources
                    sources = [
                        {
                            'title': doc.title,
                            'source_type': doc.source_type,
                            'source_path': doc.source_path
                        }
                        for doc in documents
                    ]
                    
                    return QueryResponse(
                        answer=answer,
                        model_used=model_name,
                        rag_used=True,
                        sources=sources,
                        context=context
                    )
                else:
                    # No relevant documents, fallback to direct query
                    answer = self._query_ollama(
                        request.question,
                        model_name,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens
                    )
                    
                    return QueryResponse(
                        answer=answer,
                        model_used=model_name,
                        rag_used=False,
                        sources=None,
                        context=None
                    )
            else:
                # Direct query without RAG
                answer = self._query_ollama(
                    request.question,
                    model_name,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                return QueryResponse(
                    answer=answer,
                    model_used=model_name,
                    rag_used=False,
                    sources=None,
                    context=None
                )
        
        @self.app.get("/models")
        async def list_models():
            """List available Ollama models"""
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    models = []
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 2:
                                models.append({
                                    'name': parts[0],
                                    'size': parts[1] if len(parts) > 1 else 'unknown'
                                })
                    
                    return {"models": models, "default": self.default_model}
                else:
                    return {"models": [], "error": "Could not list models"}
                    
            except Exception as e:
                return {"models": [], "error": str(e)}
        
        @self.app.get("/rag/indexes")
        async def list_rag_indexes():
            """List available RAG indexes"""
            indexes = GenericRAG.list_indexes()
            return {
                "indexes": indexes,
                "default": self.default_rag_index
            }
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API info"""
            return {
                "service": "BigTune Unified Server",
                "version": "1.0.0",
                "endpoints": {
                    "query": "POST /query - Query with model and optional RAG",
                    "health": "GET /health - Health check",
                    "models": "GET /models - List available models",
                    "rag_indexes": "GET /rag/indexes - List RAG indexes"
                },
                "default_model": self.default_model,
                "default_rag_index": self.default_rag_index
            }
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        print(f"🚀 Starting BigTune Unified Server")
        print(f"📊 Default Model: {self.default_model or 'None'}")
        print(f"📚 Default RAG Index: {self.default_rag_index}")
        print(f"🌐 Server: http://{host}:{port}")
        print(f"📖 API Docs: http://{host}:{port}/docs")
        
        uvicorn.run(self.app, host=host, port=port)


def serve_unified(model: Optional[str] = None,
                 rag_index: Optional[str] = None,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 no_cors: bool = False):
    """Main entry point for unified server"""
    server = UnifiedServer(
        default_model=model,
        default_rag_index=rag_index,
        enable_cors=not no_cors
    )
    server.run(host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BigTune Unified Server")
    parser.add_argument("--model", help="Default model name")
    parser.add_argument("--rag", help="Default RAG index", default="default")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-cors", action="store_true", help="Disable CORS")
    
    args = parser.parse_args()
    
    serve_unified(
        model=args.model,
        rag_index=args.rag,
        host=args.host,
        port=args.port,
        no_cors=args.no_cors
    )