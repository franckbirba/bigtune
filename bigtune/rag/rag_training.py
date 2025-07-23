#!/usr/bin/env python3
"""
Generate RAG-aware training data for Console Bocal model
Focus on teaching the model to follow provided documentation exactly
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime
from .console_bocal_rag import ConsoleBocalRAG, EndpointDoc

class RAGTrainingGenerator:
    """Generate training data that teaches RAG tool usage"""
    
    def __init__(self):
        self.rag = ConsoleBocalRAG()
        self.templates = self._load_templates()
        
    def _load_templates(self) -> List[Dict[str, str]]:
        """Define training templates for RAG instruction-following"""
        return [
            {
                "name": "exact_endpoint_extraction",
                "instruction": "Extract the EXACT endpoint from the provided documentation. Do not modify or guess.",
                "context_template": "ENDPOINT: {method} {path}\nSUMMARY: {summary}\nDESCRIPTION: {description}",
                "question_templates": [
                    "What is the endpoint for {action}?",
                    "Which endpoint should I use to {action}?", 
                    "What's the exact API path for {action}?",
                    "How do I {action}?"
                ],
                "answer_template": "The exact endpoint is: {method} {path}"
            },
            {
                "name": "parameter_extraction",
                "instruction": "List the required parameters from the documentation exactly as shown.",
                "context_template": "ENDPOINT: {method} {path}\nPARAMETERS:\n{parameters}",
                "question_templates": [
                    "What parameters are required for this endpoint?",
                    "Which parameters do I need to provide?",
                    "What are the required fields?"
                ],
                "answer_template": "Required parameters:\n{parameters}"
            },
            {
                "name": "method_identification", 
                "instruction": "Identify the HTTP method from the documentation.",
                "context_template": "ENDPOINT: {method} {path}\nSUMMARY: {summary}",
                "question_templates": [
                    "What HTTP method should I use?",
                    "Is this GET, POST, PUT, or DELETE?",
                    "Which HTTP verb is correct?"
                ],
                "answer_template": "The HTTP method is: {method}"
            },
            {
                "name": "description_based_matching",
                "instruction": "Match the user's intent to the correct endpoint based on the description.",
                "context_template": "Available endpoints:\n{multiple_endpoints}",
                "question_templates": [
                    "I want to {action}, which endpoint should I use?",
                    "How can I {action}?",
                    "What endpoint lets me {action}?"
                ],
                "answer_template": "Based on the documentation, use: {method} {path}"
            },
            {
                "name": "full_api_response",
                "instruction": "Provide complete API guidance using only the provided documentation.",
                "context_template": "ENDPOINT: {method} {path}\nSUMMARY: {summary}\nDESCRIPTION: {description}\nPARAMETERS:\n{parameters}",
                "question_templates": [
                    "How do I {action}?",
                    "Can you help me {action}?",
                    "I need to {action}, what should I do?"
                ],
                "answer_template": "To {action}, use the endpoint: {method} {path}\n\nRequired parameters:\n{parameters}\n\nDescription: {description}"
            }
        ]
    
    def load_console_bocal_data(self) -> bool:
        """Load Console Bocal API data"""
        index_path = "/Users/franckbirba/DEV/TEST-CREWAI/bigtune/console_bocal_index"
        
        if not self.rag.load_index(index_path):
            print("Building new index...")
            if not self.rag.load_documentation():
                return False
            self.rag.save_index(index_path)
        
        print(f"Loaded {len(self.rag.endpoints)} endpoints")
        return True
    
    def _format_parameters(self, endpoint: EndpointDoc) -> str:
        """Format parameters for training examples"""
        if not endpoint.parameters:
            return "None"
        
        param_lines = []
        for param in endpoint.parameters:
            line = f"- {param.get('name', 'unknown')}"
            if param.get('required'):
                line += " (required)"
            if param.get('description'):
                line += f": {param.get('description')}"
            param_lines.append(line)
        return "\n".join(param_lines)
    
    def _generate_action_from_endpoint(self, endpoint: EndpointDoc) -> str:
        """Generate natural language action from endpoint"""
        path_parts = endpoint.path.split('/')
        method = endpoint.method.lower()
        
        # Extract meaningful action from path and method
        if 'aliases' in endpoint.path:
            if method == 'get' and endpoint.path.endswith('/aliases'):
                return "list all aliases for a domain"
            elif method == 'get' and '{alias}' in endpoint.path:
                return "get alias details"
            elif method == 'post' and endpoint.path.endswith('/aliases'):
                return "create a new alias"
            elif method == 'delete' and '{alias}' in endpoint.path:
                return "delete an alias"
        
        elif 'users' in endpoint.path:
            if method == 'get' and endpoint.path.endswith('/users'):
                return "list all users"
            elif method == 'post' and endpoint.path.endswith('/users'):
                return "create a new user"
        
        # Fallback to summary
        return endpoint.summary.lower() if endpoint.summary else f"{method} operation"
    
    def generate_rag_training_sample(self, endpoint: EndpointDoc, template: Dict[str, str]) -> Dict[str, Any]:
        """Generate a single RAG training sample"""
        action = self._generate_action_from_endpoint(endpoint)
        parameters = self._format_parameters(endpoint)
        
        # Build context based on template
        if template["name"] == "description_based_matching":
            # For this template, we need multiple endpoints
            similar_endpoints = [ep for ep in self.rag.endpoints 
                               if ep.tags == endpoint.tags][:3]
            context_lines = []
            for ep in similar_endpoints:
                context_lines.append(f"ENDPOINT: {ep.method} {ep.path}")
                context_lines.append(f"SUMMARY: {ep.summary}")
                context_lines.append("")
            context = "\n".join(context_lines)
        else:
            context = template["context_template"].format(
                method=endpoint.method,
                path=endpoint.path,
                summary=endpoint.summary,
                description=endpoint.description,
                parameters=parameters
            )
        
        # Random question
        question_template = random.choice(template["question_templates"])
        question = question_template.format(action=action)
        
        # Generate answer
        answer = template["answer_template"].format(
            method=endpoint.method,
            path=endpoint.path,
            action=action,
            parameters=parameters,
            description=endpoint.description
        )
        
        # Create training sample in Alpaca format
        system_prompt = f"""You are a Console Bocal API expert. {template['instruction']}

CRITICAL: Use ONLY the information provided in the documentation below. Do not modify endpoints or guess information.

DOCUMENTATION:
{context}"""
        
        return {
            "instruction": system_prompt,
            "input": question,
            "output": answer,
            "template_type": template["name"],
            "endpoint": f"{endpoint.method} {endpoint.path}",
            "tags": endpoint.tags
        }
    
    def generate_rag_dataset(self, samples_per_template: int = 50) -> List[Dict[str, Any]]:
        """Generate complete RAG training dataset"""
        if not self.load_console_bocal_data():
            return []
        
        dataset = []
        
        for template in self.templates:
            print(f"Generating {samples_per_template} samples for {template['name']}...")
            
            # Select diverse endpoints for this template
            selected_endpoints = random.sample(
                self.rag.endpoints, 
                min(samples_per_template, len(self.rag.endpoints))
            )
            
            for endpoint in selected_endpoints:
                try:
                    sample = self.generate_rag_training_sample(endpoint, template)
                    dataset.append(sample)
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    continue
        
        print(f"Generated {len(dataset)} total training samples")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str) -> None:
        """Save dataset in JSONL format for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/Users/franckbirba/DEV/TEST-CREWAI/bigacademy/datasets/rag_training_{timestamp}.jsonl"
        
        with open(filepath, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Dataset saved to: {filepath}")
        return filepath
    
    def create_rag_examples_for_problematic_cases(self) -> List[Dict[str, Any]]:
        """Create specific examples for known problematic cases"""
        problematic_examples = [
            {
                "instruction": """You are a Console Bocal API expert. Extract the EXACT endpoint from the provided documentation. Do not modify or guess.

CRITICAL: Use ONLY the information provided in the documentation below. Do not modify endpoints or guess information.

DOCUMENTATION:
ENDPOINT: GET /api/{domain}/aliases
SUMMARY: Get the alias list
DESCRIPTION: Get the alias list. Now supports pagination, search, and sorting via query parameters.""",
                "input": "What is the endpoint to get all aliases for a domain?",
                "output": "The exact endpoint is: GET /api/{domain}/aliases",
                "template_type": "exact_endpoint_extraction",
                "endpoint": "GET /api/{domain}/aliases",
                "tags": ["<domain>/aliases"]
            },
            {
                "instruction": """You are a Console Bocal API expert. Extract the EXACT endpoint from the provided documentation. Do not modify or guess.

CRITICAL: Use ONLY the information provided in the documentation below. Do not modify endpoints or guess information.

DOCUMENTATION:
ENDPOINT: GET /api/{domain}/aliases
SUMMARY: Get the alias list
DESCRIPTION: Get the alias list.""",
                "input": "How do I list email aliases for a domain?",
                "output": "The exact endpoint is: GET /api/{domain}/aliases",
                "template_type": "exact_endpoint_extraction",
                "endpoint": "GET /api/{domain}/aliases", 
                "tags": ["<domain>/aliases"]
            },
            {
                "instruction": """You are a Console Bocal API expert. Extract the EXACT endpoint from the provided documentation. Do not modify or guess.

CRITICAL: Use ONLY the information provided in the documentation below. Do not modify endpoints or guess information.

DOCUMENTATION:
ENDPOINT: POST /api/{domain}/aliases
SUMMARY: Create an alias
DESCRIPTION: Create an Alias""",
                "input": "What endpoint creates a new alias?",
                "output": "The exact endpoint is: POST /api/{domain}/aliases",
                "template_type": "exact_endpoint_extraction",
                "endpoint": "POST /api/{domain}/aliases",
                "tags": ["<domain>/aliases"]
            }
        ]
        
        return problematic_examples


def main():
    """Generate RAG training dataset"""
    generator = RAGTrainingGenerator()
    
    # Generate comprehensive dataset
    dataset = generator.generate_rag_dataset(samples_per_template=30)
    
    # Add specific problematic case examples
    problematic_examples = generator.create_rag_examples_for_problematic_cases()
    dataset.extend(problematic_examples * 5)  # Repeat problematic cases
    
    # Shuffle and save
    random.shuffle(dataset)
    filepath = generator.save_dataset(dataset, "rag_training")
    
    print(f"\n✅ RAG Training Dataset Ready!")
    print(f"📁 Location: {filepath}")
    print(f"📊 Total samples: {len(dataset)}")
    print(f"🎯 Focus: Teaching exact endpoint extraction from RAG context")
    
    # Show sample
    print(f"\n📋 Sample training example:")
    sample = random.choice(dataset)
    print(f"Template: {sample['template_type']}")
    print(f"Input: {sample['input']}")
    print(f"Output: {sample['output'][:100]}...")

if __name__ == "__main__":
    main()