#!/usr/bin/env python3
"""
RAG commands for BigTune CLI
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from .rag.generic_rag import GenericRAG


class RAGCommands:
    """Handle RAG-related commands for BigTune"""
    
    def __init__(self):
        self.rag = None
    
    def init(self, args) -> bool:
        """Initialize a new RAG index from various sources"""
        index_name = args.name or "default"
        
        # Create or load RAG instance
        self.rag = GenericRAG(index_name)
        
        # Add source based on type
        try:
            if args.source == "swagger":
                self.rag.add_source("swagger", args.url or args.path)
                
            elif args.source == "github":
                kwargs = {}
                if args.branch:
                    kwargs['branch'] = args.branch
                if args.files:
                    kwargs['paths'] = args.files.split(',')
                self.rag.add_source("github", args.repo, **kwargs)
                
            elif args.source == "local":
                source_type = "markdown"
                if args.code:
                    source_type = "code"
                    
                kwargs = {'recursive': args.recursive}
                if args.extensions:
                    kwargs['extensions'] = args.extensions.split(',')
                    
                self.rag.add_source(source_type, args.path, **kwargs)
                
            else:
                print(f"❌ Unknown source type: {args.source}")
                return False
            
            # Build index
            self.rag.build_index()
            
            print(f"✅ RAG index '{index_name}' created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating RAG index: {e}")
            return False
    
    def list(self, args) -> bool:
        """List available RAG indexes"""
        indexes = GenericRAG.list_indexes()
        
        if not indexes:
            print("No RAG indexes found")
            return True
        
        print("Available RAG indexes:")
        print("-" * 50)
        
        for idx in indexes:
            print(f"📚 {idx['name']}")
            if 'created_at' in idx:
                print(f"   Created: {idx['created_at']}")
            if 'num_documents' in idx:
                print(f"   Documents: {idx['num_documents']}")
            if 'sources' in idx:
                sources = ", ".join([f"{k}({v})" for k, v in idx['sources'].items()])
                print(f"   Sources: {sources}")
            print()
        
        return True
    
    def info(self, args) -> bool:
        """Show information about a RAG index"""
        index_name = args.name or "default"
        rag = GenericRAG(index_name)
        
        info = rag.get_info()
        
        print(f"RAG Index: {info['name']}")
        print("-" * 30)
        
        if 'created_at' in info:
            print(f"Created: {info['created_at']}")
        print(f"Documents: {info.get('num_documents', 0)}")
        
        if 'sources' in info:
            print("\nDocument Sources:")
            for source, count in info['sources'].items():
                print(f"  - {source}: {count} documents")
        
        return True
    
    def search(self, args) -> bool:
        """Search in a RAG index"""
        index_name = args.index or "default"
        rag = GenericRAG(index_name)
        
        if not rag.documents:
            print(f"❌ No documents in index '{index_name}'")
            return False
        
        # Perform search
        results = rag.search(args.query, k=args.top_k)
        
        if not results:
            print("No relevant documents found")
            return True
        
        print(f"Found {len(results)} relevant documents:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"--- Document {i} ---")
            print(f"Title: {doc.title}")
            print(f"Source: {doc.source_type} - {doc.source_path}")
            print(f"\nContent:")
            
            # Show first 500 chars of content
            content_preview = doc.content[:500]
            if len(doc.content) > 500:
                content_preview += "..."
            print(content_preview)
            print()
        
        return True
    
    def rebuild(self, args) -> bool:
        """Rebuild a RAG index"""
        index_name = args.name or "default"
        rag = GenericRAG(index_name)
        
        if not rag.documents:
            print(f"❌ No documents to rebuild in index '{index_name}'")
            return False
        
        print(f"🔨 Rebuilding index '{index_name}'...")
        rag.build_index()
        
        return True
    
    def delete(self, args) -> bool:
        """Delete a RAG index"""
        index_name = args.name
        
        if not args.force:
            response = input(f"Are you sure you want to delete index '{index_name}'? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled")
                return True
        
        return GenericRAG.delete_index(index_name)
    
    def add(self, args) -> bool:
        """Add more sources to existing RAG index"""
        index_name = args.index or "default"
        rag = GenericRAG(index_name)
        
        # Similar to init but adds to existing index
        try:
            if args.source == "swagger":
                rag.add_source("swagger", args.url or args.path)
            elif args.source == "github":
                kwargs = {}
                if args.branch:
                    kwargs['branch'] = args.branch
                if args.files:
                    kwargs['paths'] = args.files.split(',')
                rag.add_source("github", args.repo, **kwargs)
            elif args.source == "local":
                source_type = "markdown"
                if args.code:
                    source_type = "code"
                kwargs = {'recursive': args.recursive}
                if args.extensions:
                    kwargs['extensions'] = args.extensions.split(',')
                rag.add_source(source_type, args.path, **kwargs)
            
            # Rebuild index
            rag.build_index()
            return True
            
        except Exception as e:
            print(f"❌ Error adding source: {e}")
            return False


def add_rag_commands(subparsers):
    """Add RAG commands to BigTune CLI"""
    
    # Main RAG command
    rag_parser = subparsers.add_parser('rag', help='Manage RAG indexes')
    rag_subparsers = rag_parser.add_subparsers(dest='rag_command', help='RAG commands')
    
    # rag init - Create new index
    init_parser = rag_subparsers.add_parser('init', help='Initialize a new RAG index')
    init_parser.add_argument('--name', help='Index name (default: default)')
    init_parser.add_argument('--source', required=True, 
                           choices=['swagger', 'github', 'local'],
                           help='Source type')
    
    # Source-specific options
    init_parser.add_argument('--url', help='URL for swagger source')
    init_parser.add_argument('--path', help='Path for local/swagger source')
    init_parser.add_argument('--repo', help='GitHub repo (owner/name)')
    init_parser.add_argument('--branch', default='main', help='GitHub branch')
    init_parser.add_argument('--files', help='File patterns (comma-separated)')
    init_parser.add_argument('--recursive', action='store_true', 
                           help='Recursive search for local files')
    init_parser.add_argument('--code', action='store_true',
                           help='Index code files instead of markdown')
    init_parser.add_argument('--extensions', help='File extensions for code (comma-separated)')
    
    # rag list - List indexes
    list_parser = rag_subparsers.add_parser('list', help='List available RAG indexes')
    
    # rag info - Show index info
    info_parser = rag_subparsers.add_parser('info', help='Show RAG index information')
    info_parser.add_argument('name', nargs='?', help='Index name')
    
    # rag search - Search in index
    search_parser = rag_subparsers.add_parser('search', help='Search in RAG index')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--index', default='default', help='Index name')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    # rag rebuild - Rebuild index
    rebuild_parser = rag_subparsers.add_parser('rebuild', help='Rebuild RAG index')
    rebuild_parser.add_argument('name', nargs='?', help='Index name')
    
    # rag delete - Delete index
    delete_parser = rag_subparsers.add_parser('delete', help='Delete RAG index')
    delete_parser.add_argument('name', help='Index name to delete')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # rag add - Add sources to existing index
    add_parser = rag_subparsers.add_parser('add', help='Add sources to existing index')
    add_parser.add_argument('--index', default='default', help='Index name')
    add_parser.add_argument('--source', required=True,
                          choices=['swagger', 'github', 'local'],
                          help='Source type')
    # Copy source options from init
    add_parser.add_argument('--url', help='URL for swagger source')
    add_parser.add_argument('--path', help='Path for local/swagger source')
    add_parser.add_argument('--repo', help='GitHub repo (owner/name)')
    add_parser.add_argument('--branch', default='main', help='GitHub branch')
    add_parser.add_argument('--files', help='File patterns (comma-separated)')
    add_parser.add_argument('--recursive', action='store_true',
                          help='Recursive search for local files')
    add_parser.add_argument('--code', action='store_true',
                          help='Index code files instead of markdown')
    add_parser.add_argument('--extensions', help='File extensions for code (comma-separated)')
    
    return rag_parser