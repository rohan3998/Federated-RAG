#!/usr/bin/env python3
"""Test script for OpenAI integration in FedRAG."""

import os
from fedrag.llm_querier import LLMQuerier

def test_openai_integration():
    """Test the OpenAI integration with a simple medical question."""
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print("🧪 Testing OpenAI Integration")
    print("=" * 50)
    
    # Initialize the LLM querier with OpenAI
    try:
        llm_querier = LLMQuerier(model_name="gpt-3.5-turbo")  # Use cheaper model for testing
        print("✅ LLMQuerier initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLMQuerier: {e}")
        return False
    
    # Test with a simple medical question
    question = "What are the main symptoms of diabetes?"
    documents = [
        "Diabetes mellitus is characterized by high blood glucose levels. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
        "Type 2 diabetes often presents with fatigue, blurred vision, and slow-healing wounds. Many patients also experience increased hunger."
    ]
    options = {"A": "Based on medical literature"}
    dataset_name = "medical"
    
    print(f"🔍 Question: {question}")
    print("📚 Testing with sample medical documents...")
    
    try:
        prompt, answer = llm_querier.answer(question, documents, options, dataset_name, max_new_tokens=150)
        
        print("\n" + "=" * 50)
        print("📖 RESULTS")
        print("=" * 50)
        print(f"🤖 Generated Answer: {answer}")
        print("=" * 50)
        print("✅ OpenAI integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during answer generation: {e}")
        return False

if __name__ == "__main__":
    success = test_openai_integration()
    if success:
        print("\n🎉 All tests passed! OpenAI integration is working correctly.")
    else:
        print("\n💥 Tests failed. Please check your OpenAI API key and internet connection.") 