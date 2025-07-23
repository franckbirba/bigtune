"""
BigTune RAG (Retrieval-Augmented Generation) Module

This module provides RAG capabilities for BigTune, allowing models to be enhanced
with real-time document retrieval for more accurate and up-to-date responses.

Main components:
- console_bocal_rag: Complete RAG system for Console Bocal API documentation
- rag_training: Tools for generating RAG-aware training datasets
- argilla_feedback: AI-assisted feedback collection via Argilla

Example usage:
    from bigtune.rag import ConsoleBocalRAG
    
    rag = ConsoleBocalRAG()
    if rag.load_documentation():
        result = rag.query("What is the endpoint to get all aliases?")
        print(result.answer)
"""

from .console_bocal_rag import ConsoleBocalRAG, ConsoleBocalRAGService
from .rag_training import RAGTrainingGenerator
from .argilla_feedback import ArgillaaRAGFeedback

__all__ = [
    'ConsoleBocalRAG',
    'ConsoleBocalRAGService', 
    'RAGTrainingGenerator',
    'ArgillaaRAGFeedback'
]

__version__ = '1.0.0'