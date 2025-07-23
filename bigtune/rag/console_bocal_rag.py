#!/usr/bin/env python3
"""
Console Bocal RAG System
Retrieval-Augmented Generation for accurate Console Bocal API responses
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


@dataclass
class EndpointDoc:
    """Structured endpoint documentation"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: Dict[str, Any]
    tags: List[str]
    operation_id: str
    content: str  # Full text representation for embedding


class ConsoleBocalRAG:
    """RAG system for Console Bocal API documentation"""
    
    def __init__(self, swagger_url: str = "http://localhost:5050/api/swagger.json"):
        self.swagger_url = swagger_url
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.endpoints: List[EndpointDoc] = []
        self.index: Optional[faiss.IndexFlatL2] = None
        self.embeddings: Optional[np.ndarray] = None
        
    def fetch_swagger_data(self) -> Dict[str, Any]:
        """Fetch swagger documentation from Console Bocal API"""
        try:
            response = requests.get(self.swagger_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching swagger data: {e}")
            return {}
    
    def parse_swagger_to_endpoints(self, swagger_data: Dict[str, Any]) -> List[EndpointDoc]:
        """Parse swagger JSON into structured endpoint documentation"""
        endpoints = []
        
        base_path = swagger_data.get('basePath', '')
        paths = swagger_data.get('paths', {})
        
        for path, path_data in paths.items():
            # Handle path-level parameters
            path_params = path_data.get('parameters', [])
            
            for method, method_data in path_data.items():
                if method == 'parameters':
                    continue
                    
                # Build full endpoint path
                full_path = f"{base_path}{path}".replace('//', '/')
                
                # Extract parameters (path + method level)
                parameters = path_params + method_data.get('parameters', [])
                
                # Create content for embedding
                content = self._build_endpoint_content(
                    full_path, method.upper(), method_data, parameters
                )
                
                endpoint = EndpointDoc(
                    path=full_path,
                    method=method.upper(),
                    summary=method_data.get('summary', ''),
                    description=method_data.get('description', ''),
                    parameters=parameters,
                    responses=method_data.get('responses', {}),
                    tags=method_data.get('tags', []),
                    operation_id=method_data.get('operationId', ''),
                    content=content
                )
                endpoints.append(endpoint)
        
        return endpoints
    
    def _build_endpoint_content(self, path: str, method: str, method_data: Dict, parameters: List) -> str:
        """Build comprehensive text content for embedding"""
        content_parts = [
            f"Endpoint: {method} {path}",
            f"Summary: {method_data.get('summary', '')}",
            f"Description: {method_data.get('description', '')}",
            f"Tags: {', '.join(method_data.get('tags', []))}",
        ]
        
        # Add parameter information
        if parameters:
            param_info = []
            for param in parameters:
                param_desc = f"{param.get('name', 'unknown')} ({param.get('in', 'unknown')})"
                if param.get('required'):
                    param_desc += " [required]"
                if param.get('description'):
                    param_desc += f": {param.get('description')}"
                param_info.append(param_desc)
            content_parts.append(f"Parameters: {'; '.join(param_info)}")
        
        # Add response information
        responses = method_data.get('responses', {})
        if responses:
            response_info = []
            for code, resp_data in responses.items():
                resp_desc = f"{code}: {resp_data.get('description', 'No description')}"
                response_info.append(resp_desc)
            content_parts.append(f"Responses: {'; '.join(response_info)}")
        
        return "\n".join(content_parts)
    
    def build_vector_database(self) -> None:
        """Create FAISS vector database from endpoint documentation"""
        if not self.endpoints:
            print("No endpoints loaded. Call load_documentation() first.")
            return
        
        # Generate embeddings for all endpoint content
        texts = [endpoint.content for endpoint in self.endpoints]
        print(f"Generating embeddings for {len(texts)} endpoints...")
        
        embeddings = self.embedding_model.encode(texts)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
        print(f"Vector database created with {len(self.endpoints)} endpoints")
    
    def load_documentation(self) -> bool:
        """Load and process Console Bocal API documentation"""
        print("Fetching Console Bocal swagger documentation...")
        swagger_data = self.fetch_swagger_data()
        
        if not swagger_data:
            print("Failed to fetch swagger data")
            return False
        
        print("Parsing swagger data into endpoint documentation...")
        self.endpoints = self.parse_swagger_to_endpoints(swagger_data)
        print(f"Loaded {len(self.endpoints)} endpoints")
        
        self.build_vector_database()
        return True
    
    def save_index(self, filepath: str) -> None:
        """Save the vector index and endpoints to disk"""
        if self.index is None:
            print("No index to save")
            return
            
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save endpoints and embeddings
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump({
                'endpoints': self.endpoints,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Index saved to {filepath}.*")
    
    def load_index(self, filepath: str) -> bool:
        """Load the vector index and endpoints from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load endpoints and embeddings
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.endpoints = data['endpoints']
                self.embeddings = data['embeddings']
            
            print(f"Index loaded from {filepath}.*")
            return True
        except FileNotFoundError:
            print(f"Index files not found at {filepath}.*")
            return False
    
    def retrieve_relevant_docs(self, query: str, k: int = 3) -> List[EndpointDoc]:
        """Retrieve top-k most relevant endpoint documentation"""
        if self.index is None:
            return []
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search for similar endpoints
        distances, indices = self.index.search(query_embedding, k)
        
        # Return corresponding endpoints
        relevant_docs = []
        for idx in indices[0]:
            if idx < len(self.endpoints):
                relevant_docs.append(self.endpoints[idx])
        
        return relevant_docs
    
    def format_context(self, docs: List[EndpointDoc]) -> str:
        """Format retrieved documents as context for the LLM"""
        if not docs:
            return "No relevant documentation found."
        
        context_parts = []
        for doc in docs:
            context = f"""
ENDPOINT: {doc.method} {doc.path}
SUMMARY: {doc.summary}
DESCRIPTION: {doc.description}
"""
            if doc.parameters:
                param_list = []
                for param in doc.parameters:
                    param_str = f"- {param.get('name', 'unknown')}"
                    if param.get('required'):
                        param_str += " (required)"
                    if param.get('description'):
                        param_str += f": {param.get('description')}"
                    param_list.append(param_str)
                context += f"PARAMETERS:\n{chr(10).join(param_list)}\n"
            
            context_parts.append(context.strip())
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)


# RAG-enhanced query interface
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    context: str
    relevant_endpoints: List[str]


class ConsoleBocalRAGService:
    """FastAPI service wrapper for Console Bocal RAG"""
    
    def __init__(self):
        self.rag = ConsoleBocalRAG()
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "console-bocal-expert"
        
    def initialize(self) -> bool:
        """Initialize the RAG system"""
        # Try to load existing index
        index_path = "/Users/franckbirba/DEV/TEST-CREWAI/bigtune/console_bocal_index"
        
        if not self.rag.load_index(index_path):
            print("Building new index from API...")
            if not self.rag.load_documentation():
                return False
            self.rag.save_index(index_path)
        
        return True
    
    def query_ollama(self, prompt: str) -> str:
        """Query the Console Bocal expert model via Ollama"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No response generated')
            
        except requests.RequestException as e:
            return f"Error querying model: {e}"
    
    def process_query(self, question: str, top_k: int = 3) -> QueryResponse:
        """Process a query using RAG + Console Bocal expert model"""
        # Retrieve relevant documentation
        relevant_docs = self.rag.retrieve_relevant_docs(question, top_k)
        context = self.rag.format_context(relevant_docs)
        
        # Build enhanced prompt
        system_prompt = """You are an expert Console Bocal API assistant. Answer ONLY based on the provided documentation.

CRITICAL INSTRUCTIONS:
- Copy the EXACT endpoint paths from the documentation below
- Use the EXACT HTTP methods shown in the documentation
- Do NOT modify or guess endpoint paths
- If asking about aliases, the correct endpoint is GET /api/{domain}/aliases (NOT /email/aliases)
- Always reference the actual API documentation provided

DOCUMENTATION PROVIDED BELOW - USE ONLY THIS INFORMATION:"""
        
        full_prompt = f"""{system_prompt}

{context}

User Question: {question}

Answer:"""
        
        # Query the model
        answer = self.query_ollama(full_prompt)
        
        # Extract endpoint paths for response
        endpoint_paths = [f"{doc.method} {doc.path}" for doc in relevant_docs]
        
        return QueryResponse(
            answer=answer,
            context=context,
            relevant_endpoints=endpoint_paths
        )


# FastAPI application
app = FastAPI(title="Console Bocal RAG API", version="1.0.0")
rag_service = ConsoleBocalRAGService()


@app.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup"""
    print("Initializing Console Bocal RAG service...")
    if not rag_service.initialize():
        print("Failed to initialize RAG service")
        exit(1)
    print("RAG service ready!")


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the Console Bocal expert with RAG enhancement"""
    try:
        result = rag_service.process_query(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "endpoints": len(rag_service.rag.endpoints)}


if __name__ == "__main__":
    # CLI mode for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Console Bocal RAG System")
    parser.add_argument("--server", action="store_true", help="Start FastAPI server")
    parser.add_argument("--build-index", action="store_true", help="Build vector index")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    
    args = parser.parse_args()
    
    if args.server:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.build_index:
        rag = ConsoleBocalRAG()
        if rag.load_documentation():
            rag.save_index("/Users/franckbirba/DEV/TEST-CREWAI/bigtune/console_bocal_index")
    elif args.query:
        service = ConsoleBocalRAGService()
        if service.initialize():
            result = service.process_query(args.query)
            print(f"Answer: {result.answer}")
            print(f"Relevant endpoints: {result.relevant_endpoints}")
    else:
        print("Use --server, --build-index, or --query <question>")