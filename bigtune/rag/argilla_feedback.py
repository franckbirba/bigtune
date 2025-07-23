#!/usr/bin/env python3
"""
Argilla integration for Console Bocal RAG feedback and improvement
AI-assisted feedback loop for model responses
"""

import argilla as rg
import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from .console_bocal_rag import ConsoleBocalRAGService
import openai
import os

class ArgillaaRAGFeedback:
    """Argilla-based feedback system for Console Bocal RAG"""
    
    def __init__(self, argilla_url: str = "http://localhost:6900", api_key: str = "bigacademy-api-key"):
        self.argilla_url = argilla_url
        self.api_key = api_key
        self.workspace_name = "console-bocal-rag"
        self.dataset_name = "rag-responses-feedback"
        self.rag_service = ConsoleBocalRAGService()
        
        # Initialize Argilla with BigAcademy credentials
        try:
            rg.init(
                api_url=argilla_url,
                api_key=api_key
            )
            print(f"✅ Connected to BigAcademy Argilla at {argilla_url}")
        except Exception as e:
            print(f"❌ Failed to connect to Argilla: {e}")
            print(f"Make sure BigAcademy stack is running: cd /Users/franckbirba/DEV/TEST-CREWAI/bigacademy && docker-compose up -d")
    
    def setup_workspace_and_dataset(self) -> bool:
        """Set up Argilla workspace and dataset for RAG feedback"""
        try:
            # Create workspace if it doesn't exist
            try:
                workspace = rg.Workspace(name=self.workspace_name)
            except Exception:
                workspace = rg.Workspace.create(name=self.workspace_name)
                print(f"✅ Created workspace: {self.workspace_name}")
            
            # Define the dataset schema for RAG feedback
            settings = rg.Settings(
                fields=[
                    rg.TextField(name="question", title="User Question"),
                    rg.TextField(name="retrieved_context", title="Retrieved Documentation"),
                    rg.TextField(name="model_response", title="Model Response"),
                    rg.TextField(name="expected_endpoint", title="Correct Endpoint"),
                ],
                questions=[
                    # Human feedback questions
                    rg.RatingQuestion(
                        name="accuracy",
                        title="How accurate is the model response?",
                        values=[1, 2, 3, 4, 5],
                        description="1=Completely wrong, 5=Perfect"
                    ),
                    rg.RatingQuestion(
                        name="endpoint_correctness", 
                        title="Did the model use the exact endpoint from documentation?",
                        values=[1, 2, 3, 4, 5],
                        description="1=Wrong endpoint, 5=Exact match"
                    ),
                    rg.MultiLabelQuestion(
                        name="issues",
                        title="What issues are present?",
                        labels=[
                            "wrong_endpoint", "missing_api_prefix", "wrong_method", 
                            "added_parameters", "incomplete_response", "hallucination"
                        ]
                    ),
                    rg.TextQuestion(
                        name="corrected_response",
                        title="Provide the corrected response",
                        required=False
                    )
                ],
                metadata=[
                    rg.TermsMetadataProperty(name="template_type"),
                    rg.TermsMetadataProperty(name="endpoint_category"),
                    rg.FloatMetadataProperty(name="confidence_score"),
                ]
            )
            
            # Create or get dataset
            try:
                dataset = rg.Dataset(name=self.dataset_name, workspace=workspace)
                print(f"✅ Using existing dataset: {self.dataset_name}")
            except Exception:
                dataset = rg.Dataset(
                    name=self.dataset_name,
                    workspace=workspace,
                    settings=settings
                )
                dataset.create()
                print(f"✅ Created dataset: {self.dataset_name}")
            
            self.dataset = dataset
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Argilla: {e}")
            return False
    
    def generate_ai_feedback(self, question: str, context: str, response: str, expected_endpoint: str) -> Dict[str, Any]:
        """Generate AI-assisted feedback for model responses"""
        
        # Use OpenAI to analyze the response quality
        analysis_prompt = f"""Analyze this Console Bocal API model response for accuracy and adherence to documentation.

USER QUESTION: {question}

PROVIDED DOCUMENTATION:
{context}

MODEL RESPONSE: {response}

EXPECTED CORRECT ENDPOINT: {expected_endpoint}

Please analyze:
1. Did the model use the EXACT endpoint from documentation?
2. Are there any hallucinations or made-up information?
3. Rate accuracy from 1-5 (5=perfect)
4. What specific issues exist?
5. Provide a corrected response if needed.

Return analysis in JSON format:
{{
    "endpoint_correct": true/false,
    "accuracy_rating": 1-5,
    "issues": ["list", "of", "issues"],
    "hallucinations": "description of any hallucinations",
    "corrected_response": "improved response if needed"
}}"""
        
        try:
            # Use OpenAI for analysis (you can replace with any LLM)
            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an API documentation expert who analyzes model responses for accuracy."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1
            )
            
            ai_analysis = json.loads(completion.choices[0].message.content)
            return ai_analysis
            
        except Exception as e:
            print(f"AI feedback generation failed: {e}")
            return {
                "endpoint_correct": "unknown",
                "accuracy_rating": 3,
                "issues": ["ai_analysis_failed"],
                "hallucinations": "Could not analyze",
                "corrected_response": ""
            }
    
    def test_model_and_collect_feedback(self, test_questions: List[str]) -> None:
        """Test model responses and collect them in Argilla for feedback"""
        
        if not hasattr(self, 'dataset'):
            print("❌ Dataset not set up. Run setup_workspace_and_dataset() first.")
            return
        
        # Initialize RAG service
        if not self.rag_service.initialize():
            print("❌ Failed to initialize RAG service")
            return
        
        records = []
        
        for question in test_questions:
            print(f"📝 Testing: {question}")
            
            # Get model response through RAG
            try:
                rag_result = self.rag_service.process_query(question)
                
                # Extract expected endpoint from retrieved docs
                expected_endpoint = rag_result.relevant_endpoints[0] if rag_result.relevant_endpoints else "Unknown"
                
                # Generate AI feedback
                ai_feedback = self.generate_ai_feedback(
                    question, 
                    rag_result.context, 
                    rag_result.answer, 
                    expected_endpoint
                )
                
                # Create Argilla record
                record = rg.Record(
                    fields={
                        "question": question,
                        "retrieved_context": rag_result.context[:2000],  # Truncate for display
                        "model_response": rag_result.answer,
                        "expected_endpoint": expected_endpoint
                    },
                    metadata={
                        "template_type": "rag_response",
                        "endpoint_category": expected_endpoint.split()[1].split('/')[2] if len(expected_endpoint.split()) > 1 else "unknown",
                        "confidence_score": float(ai_feedback.get("accuracy_rating", 3))
                    },
                    suggestions=[
                        rg.Suggestion(
                            question_name="accuracy", 
                            value=ai_feedback.get("accuracy_rating", 3)
                        ),
                        rg.Suggestion(
                            question_name="endpoint_correctness",
                            value=5 if ai_feedback.get("endpoint_correct") else 2
                        ),
                        rg.Suggestion(
                            question_name="issues",
                            value=ai_feedback.get("issues", [])
                        ),
                        rg.Suggestion(
                            question_name="corrected_response",
                            value=ai_feedback.get("corrected_response", "")
                        )
                    ]
                )
                
                records.append(record)
                
            except Exception as e:
                print(f"❌ Error testing question '{question}': {e}")
                continue
        
        # Upload records to Argilla
        if records:
            self.dataset.records.log(records)
            print(f"✅ Uploaded {len(records)} records to Argilla for feedback")
            print(f"🌐 View at: {self.argilla_url}/dataset/{self.dataset_name}/annotation")
    
    def export_feedback_for_training(self) -> str:
        """Export annotated feedback as new training data"""
        
        # Get all annotated records
        annotated_records = []
        for record in self.dataset.records():
            if record.responses:  # Has human feedback
                annotated_records.append(record)
        
        if not annotated_records:
            print("❌ No annotated records found")
            return ""
        
        # Convert to training format
        training_data = []
        for record in annotated_records:
            response = record.responses[0]  # Get first response
            
            # Only use high-quality corrections
            if (response.values.get("accuracy", 0) >= 4 and 
                response.values.get("corrected_response")):
                
                training_sample = {
                    "instruction": f"""You are a Console Bocal API expert. Answer based ONLY on the provided documentation.

DOCUMENTATION:
{record.fields["retrieved_context"]}""",
                    "input": record.fields["question"],
                    "output": response.values["corrected_response"],
                    "feedback_score": response.values.get("accuracy", 0),
                    "issues_fixed": response.values.get("issues", []),
                    "source": "argilla_feedback"
                }
                training_data.append(training_sample)
        
        # Save training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/Users/franckbirba/DEV/TEST-CREWAI/bigacademy/datasets/argilla_feedback_training_{timestamp}.jsonl"
        
        with open(filepath, 'w') as f:
            for sample in training_data:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Exported {len(training_data)} corrected samples to: {filepath}")
        return filepath


def main():
    """Set up Argilla feedback system and test with sample questions"""
    
    feedback_system = ArgillaaRAGFeedback()
    
    # Setup workspace and dataset
    if not feedback_system.setup_workspace_and_dataset():
        print("❌ Failed to setup Argilla workspace")
        return
    
    # Test questions that we know are problematic
    test_questions = [
        "What is the endpoint to get all aliases for a domain?",
        "How do I create a new alias?",
        "What endpoint lists users?",
        "How can I delete an alias?",
        "What's the API for getting alias details?",
        "How do I list access groups?",
        "What endpoint manages user cards?",
        "How do I send SMS via the API?"
    ]
    
    print("🚀 Testing model responses and collecting feedback...")
    feedback_system.test_model_and_collect_feedback(test_questions)
    
    print(f"\n✅ Argilla Feedback System Ready!")
    print(f"🌐 Open: http://localhost:6900/dataset/{feedback_system.dataset_name}/annotation")
    print(f"📋 Review and annotate {len(test_questions)} model responses")
    print(f"🔄 Run export_feedback_for_training() to create improved training data")

if __name__ == "__main__":
    main()