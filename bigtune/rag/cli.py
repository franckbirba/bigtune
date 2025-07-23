#!/usr/bin/env python3
"""
BigTune RAG CLI Interface

Command-line interface for RAG functionality in BigTune.
Provides easy access to RAG query, training, and feedback features.
"""

import argparse
import sys
from pathlib import Path

from .console_bocal_rag import ConsoleBocalRAGService
from .rag_training import RAGTrainingGenerator
from .argilla_feedback import ArgillaaRAGFeedback


def cmd_query(args):
    """Handle RAG query command"""
    print(f"🔍 Querying Console Bocal RAG: {args.question}")
    
    service = ConsoleBocalRAGService()
    if not service.initialize():
        print("❌ Failed to initialize RAG service")
        return 1
        
    result = service.process_query(args.question, args.top_k)
    
    print(f"\n💬 Answer:")
    print(result.answer)
    print(f"\n📚 Relevant endpoints:")
    for endpoint in result.relevant_endpoints:
        print(f"  • {endpoint}")
        
    if args.show_context:
        print(f"\n📄 Retrieved context:")
        print(result.context[:500] + "..." if len(result.context) > 500 else result.context)
    
    return 0


def cmd_generate_training(args):
    """Handle training data generation"""
    print(f"🎯 Generating RAG training data...")
    
    generator = RAGTrainingGenerator()
    dataset = generator.generate_rag_dataset(samples_per_template=args.samples)
    
    # Add problematic case examples
    if args.include_problems:
        problematic = generator.create_rag_examples_for_problematic_cases()
        dataset.extend(problematic * args.problem_repeat)
    
    filepath = generator.save_dataset(dataset, "rag_training_cli")
    
    print(f"✅ Generated {len(dataset)} training samples")
    print(f"💾 Saved to: {filepath}")
    
    return 0


def cmd_setup_feedback(args):
    """Handle Argilla feedback setup"""
    print("🔄 Setting up Argilla feedback system...")
    
    feedback = ArgillaaRAGFeedback()
    
    if not feedback.setup_workspace_and_dataset():
        print("❌ Failed to setup Argilla workspace")
        return 1
    
    if args.test_queries:
        test_questions = [
            "What is the endpoint to get all aliases for a domain?",
            "How do I create a new alias?",
            "What endpoint deletes a user?",
            "How can I send SMS via the API?"
        ]
        
        print(f"🧪 Testing with {len(test_questions)} sample queries...")
        feedback.test_model_and_collect_feedback(test_questions)
    
    print("✅ Argilla feedback system ready")
    print(f"🌐 Access at: http://localhost:6900")
    
    return 0


def cmd_server(args):
    """Handle FastAPI server startup"""
    print(f"🚀 Starting RAG server on port {args.port}...")
    
    try:
        import uvicorn
        from .console_bocal_rag import app
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="BigTune RAG - Retrieval-Augmented Generation for Console Bocal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bigtune-rag query "What is the aliases endpoint?"
  bigtune-rag generate --samples 100 --include-problems
  bigtune-rag feedback --test-queries
  bigtune-rag server --port 8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask the RAG system')
    query_parser.add_argument('--top-k', type=int, default=3, help='Number of documents to retrieve')
    query_parser.add_argument('--show-context', action='store_true', help='Show retrieved context')
    query_parser.set_defaults(func=cmd_query)
    
    # Generate training data command  
    gen_parser = subparsers.add_parser('generate', help='Generate RAG training data')
    gen_parser.add_argument('--samples', type=int, default=50, help='Samples per template')
    gen_parser.add_argument('--include-problems', action='store_true', help='Include problematic cases')
    gen_parser.add_argument('--problem-repeat', type=int, default=5, help='Times to repeat problem cases')
    gen_parser.set_defaults(func=cmd_generate_training)
    
    # Argilla feedback command
    feedback_parser = subparsers.add_parser('feedback', help='Setup Argilla feedback system')
    feedback_parser.add_argument('--test-queries', action='store_true', help='Run test queries')
    feedback_parser.set_defaults(func=cmd_setup_feedback)
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start RAG FastAPI server')
    server_parser.add_argument('--port', type=int, default=8000, help='Server port')
    server_parser.set_defaults(func=cmd_server)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())