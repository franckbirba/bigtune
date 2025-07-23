#!/usr/bin/env python3
"""
Generic RAG system for BigTune - Support any documentation source
"""

import os
import json
import yaml
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime
import hashlib
import glob
import subprocess

@dataclass
class Document:
    """Generic document structure for RAG"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    source_type: str
    source_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'metadata': self.metadata,
            'source_type': self.source_type,
            'source_path': self.source_path
        }


class DocumentLoader(ABC):
    """Abstract base class for document loaders"""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> List[Document]:
        """Load documents from source"""
        pass
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source type identifier"""
        pass


class SwaggerLoader(DocumentLoader):
    """Load API documentation from Swagger/OpenAPI"""
    
    def get_source_type(self) -> str:
        return "swagger"
    
    def load(self, source: str, **kwargs) -> List[Document]:
        """Load Swagger documentation"""
        documents = []
        
        # Load swagger JSON
        if source.startswith(('http://', 'https://')):
            response = requests.get(source)
            swagger_data = response.json()
        else:
            with open(source, 'r') as f:
                swagger_data = json.load(f)
        
        base_path = swagger_data.get('basePath', '')
        
        # Extract endpoints
        for path, methods in swagger_data.get('paths', {}).items():
            for method, endpoint_data in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch']:
                    doc_id = hashlib.md5(f"{method}:{path}".encode()).hexdigest()
                    
                    # Build content
                    content_parts = [
                        f"ENDPOINT: {method.upper()} {base_path}{path}",
                        f"SUMMARY: {endpoint_data.get('summary', 'No summary')}",
                        f"DESCRIPTION: {endpoint_data.get('description', 'No description')}"
                    ]
                    
                    # Add parameters
                    params = endpoint_data.get('parameters', [])
                    if params:
                        content_parts.append("\nPARAMETERS:")
                        for param in params:
                            param_line = f"- {param.get('name')} ({param.get('in')}"
                            if param.get('required'):
                                param_line += ", required"
                            param_line += f"): {param.get('description', 'No description')}"
                            content_parts.append(param_line)
                    
                    # Add responses
                    responses = endpoint_data.get('responses', {})
                    if responses:
                        content_parts.append("\nRESPONSES:")
                        for code, resp_data in responses.items():
                            content_parts.append(f"- {code}: {resp_data.get('description', 'No description')}")
                    
                    doc = Document(
                        id=doc_id,
                        title=f"{method.upper()} {path}",
                        content="\n".join(content_parts),
                        metadata={
                            'method': method.upper(),
                            'path': path,
                            'full_path': f"{base_path}{path}",
                            'tags': endpoint_data.get('tags', []),
                            'operationId': endpoint_data.get('operationId')
                        },
                        source_type="swagger",
                        source_path=source
                    )
                    documents.append(doc)
        
        return documents


class MarkdownLoader(DocumentLoader):
    """Load documentation from Markdown files"""
    
    def get_source_type(self) -> str:
        return "markdown"
    
    def load(self, source: str, recursive: bool = True, **kwargs) -> List[Document]:
        """Load Markdown files from directory or file"""
        documents = []
        
        source_path = Path(source)
        if source_path.is_file():
            files = [source_path]
        else:
            pattern = "**/*.md" if recursive else "*.md"
            files = list(source_path.glob(pattern))
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first H1 or filename
            lines = content.split('\n')
            title = file_path.stem
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            
            doc = Document(
                id=doc_id,
                title=title,
                content=content,
                metadata={
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'relative_path': str(file_path.relative_to(source_path.parent))
                },
                source_type="markdown",
                source_path=str(file_path)
            )
            documents.append(doc)
        
        return documents


class CodeLoader(DocumentLoader):
    """Load documentation from code files and docstrings"""
    
    def get_source_type(self) -> str:
        return "code"
    
    def load(self, source: str, extensions: List[str] = None, recursive: bool = True, **kwargs) -> List[Document]:
        """Load code files and extract documentation"""
        documents = []
        
        if not extensions:
            extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs']
        
        source_path = Path(source)
        files = []
        
        if source_path.is_file():
            files = [source_path]
        else:
            for ext in extensions:
                pattern = f"**/*{ext}" if recursive else f"*{ext}"
                files.extend(source_path.glob(pattern))
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract meaningful documentation (functions, classes, etc.)
                # This is simplified - could use AST parsing for better results
                doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
                
                doc = Document(
                    id=doc_id,
                    title=file_path.name,
                    content=content,
                    metadata={
                        'file_path': str(file_path),
                        'extension': file_path.suffix,
                        'language': self._detect_language(file_path.suffix)
                    },
                    source_type="code",
                    source_path=str(file_path)
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _detect_language(self, extension: str) -> str:
        """Detect programming language from extension"""
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php'
        }
        return lang_map.get(extension, 'unknown')


class GitHubLoader(DocumentLoader):
    """Load documentation from GitHub repositories"""
    
    def get_source_type(self) -> str:
        return "github"
    
    def load(self, source: str, branch: str = "main", paths: List[str] = None, **kwargs) -> List[Document]:
        """Load files from GitHub repository"""
        documents = []
        
        # Clone or update repo
        repo_name = source.split('/')[-1]
        temp_dir = Path(f"/tmp/bigtune_rag_{repo_name}")
        
        if temp_dir.exists():
            # Update existing clone
            subprocess.run(["git", "-C", str(temp_dir), "pull"], capture_output=True)
        else:
            # Clone repo
            clone_url = f"https://github.com/{source}.git"
            subprocess.run(["git", "clone", clone_url, str(temp_dir)], capture_output=True)
        
        # Checkout branch
        subprocess.run(["git", "-C", str(temp_dir), "checkout", branch], capture_output=True)
        
        # Load specific paths or entire repo
        if paths:
            for path_pattern in paths:
                for file_path in temp_dir.glob(path_pattern):
                    if file_path.is_file():
                        documents.extend(self._load_file(file_path, source, temp_dir))
        else:
            # Load README and docs by default
            for pattern in ["README*", "readme*", "docs/**/*", "*.md"]:
                for file_path in temp_dir.glob(pattern):
                    if file_path.is_file():
                        documents.extend(self._load_file(file_path, source, temp_dir))
        
        return documents
    
    def _load_file(self, file_path: Path, repo: str, repo_dir: Path) -> List[Document]:
        """Load a single file from repo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc_id = hashlib.md5(f"{repo}:{file_path}".encode()).hexdigest()
            
            return [Document(
                id=doc_id,
                title=file_path.name,
                content=content,
                metadata={
                    'repo': repo,
                    'file_path': str(file_path.relative_to(repo_dir)),
                    'github_url': f"https://github.com/{repo}/blob/main/{file_path.relative_to(repo_dir)}"
                },
                source_type="github",
                source_path=f"{repo}:{file_path.relative_to(repo_dir)}"
            )]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []


class GenericRAG:
    """Generic RAG system supporting multiple document sources"""
    
    def __init__(self, index_name: str = "default"):
        self.index_name = index_name
        self.index_dir = Path.home() / ".bigtune" / "rag_indexes" / index_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents: List[Document] = []
        self.embeddings = None
        self.index = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Register loaders
        self.loaders = {
            'swagger': SwaggerLoader(),
            'markdown': MarkdownLoader(),
            'code': CodeLoader(),
            'github': GitHubLoader()
        }
        
        # Load existing index if available
        self.load_index()
    
    def add_source(self, source_type: str, source_path: str, **kwargs) -> int:
        """Add documents from a source"""
        if source_type not in self.loaders:
            raise ValueError(f"Unknown source type: {source_type}")
        
        loader = self.loaders[source_type]
        new_docs = loader.load(source_path, **kwargs)
        
        # Add to documents
        self.documents.extend(new_docs)
        
        print(f"✅ Added {len(new_docs)} documents from {source_type}: {source_path}")
        return len(new_docs)
    
    def build_index(self) -> None:
        """Build FAISS index from documents"""
        if not self.documents:
            print("❌ No documents to index")
            return
        
        print(f"🔨 Building index for {len(self.documents)} documents...")
        
        # Generate embeddings
        texts = [doc.content for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        # Save index
        self.save_index()
        print(f"✅ Index built and saved to {self.index_dir}")
    
    def save_index(self) -> None:
        """Save index and documents to disk"""
        if self.index is None:
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        
        # Save documents and metadata
        with open(self.index_dir / "documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save index metadata
        metadata = {
            'name': self.index_name,
            'created_at': datetime.now().isoformat(),
            'num_documents': len(self.documents),
            'sources': {}
        }
        
        # Count documents by source
        for doc in self.documents:
            source = doc.source_type
            metadata['sources'][source] = metadata['sources'].get(source, 0) + 1
        
        with open(self.index_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_index(self) -> bool:
        """Load existing index from disk"""
        index_file = self.index_dir / "index.faiss"
        docs_file = self.index_dir / "documents.pkl"
        
        if not index_file.exists() or not docs_file.exists():
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            print(f"✅ Loaded index '{self.index_name}' with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents"""
        if self.index is None or not self.documents:
            return []
        
        # Embed query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        # Return relevant documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the index"""
        metadata_file = self.index_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        return {
            'name': self.index_name,
            'num_documents': len(self.documents) if self.documents else 0,
            'status': 'loaded' if self.index else 'not loaded'
        }
    
    @staticmethod
    def list_indexes() -> List[Dict[str, Any]]:
        """List all available indexes"""
        indexes_dir = Path.home() / ".bigtune" / "rag_indexes"
        indexes = []
        
        if indexes_dir.exists():
            for index_dir in indexes_dir.iterdir():
                if index_dir.is_dir():
                    metadata_file = index_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            indexes.append(json.load(f))
                    else:
                        indexes.append({
                            'name': index_dir.name,
                            'status': 'unknown'
                        })
        
        return indexes
    
    @staticmethod
    def delete_index(index_name: str) -> bool:
        """Delete an index"""
        index_dir = Path.home() / ".bigtune" / "rag_indexes" / index_name
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir)
            print(f"✅ Deleted index: {index_name}")
            return True
        else:
            print(f"❌ Index not found: {index_name}")
            return False