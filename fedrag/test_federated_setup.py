#!/usr/bin/env python3
"""
Test script to verify the 5-client federated setup with split textbooks corpus.
This script simulates what happens when 5 clients with different corpus parts respond to queries.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedrag.retriever import Retriever
from fedrag.corpus_splitter import CorpusSplitter


def test_individual_parts():
    """Test that each corpus part can be queried individually."""
    print("Testing individual corpus parts...")
    
    retriever = Retriever()
    splitter = CorpusSplitter()
    
    # Get available parts
    parts = splitter.list_available_parts()
    if len(parts) != 5:
        print(f"Error: Expected 5 parts, found {len(parts)}: {parts}")
        return False
    
    test_query = "What are the symptoms of cardiovascular disease?"
    
    all_results = {}
    total_docs = 0
    
    for part in parts:
        print(f"\nTesting {part}...")
        try:
            results = retriever.query_faiss_index(part, test_query, knn=3)
            print(f"  ✓ Retrieved {len(results)} documents")
            
            # Show some sample results
            for i, (doc_id, doc_data) in enumerate(list(results.items())[:2]):
                print(f"    {i+1}. Score: {doc_data['score']:.4f}")
                print(f"       Title: {doc_data['title'][:80]}...")
                print(f"       Content: {doc_data['content'][:100]}...")
            
            all_results[part] = results
            total_docs += len(results)
            
        except Exception as e:
            print(f"  ✗ Error querying {part}: {e}")
            return False
    
    print(f"\n✓ All {len(parts)} parts working correctly!")
    print(f"✓ Total documents retrieved: {total_docs}")
    return True


def test_federated_simulation():
    """Simulate a federated query across all 5 clients."""
    print("\n" + "="*60)
    print("SIMULATING FEDERATED QUERY")
    print("="*60)
    
    retriever = Retriever()
    splitter = CorpusSplitter()
    parts = splitter.list_available_parts()
    
    test_query = "What is the treatment for hypertension?"
    knn = 2  # Each client returns top-2 documents
    
    # Simulate what each client would return
    all_documents = []
    all_scores = []
    
    print(f"Query: {test_query}")
    print(f"Requesting top-{knn} documents from each of {len(parts)} clients...\n")
    
    for i, part in enumerate(parts, 1):
        print(f"Client {i} ({part}):")
        try:
            results = retriever.query_faiss_index(part, test_query, knn=knn)
            
            for doc_id, doc_data in results.items():
                all_documents.append(doc_data['content'])
                all_scores.append(doc_data['score'])
                print(f"  Doc: {doc_data['title'][:50]}... (score: {doc_data['score']:.4f})")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nFederated Results Summary:")
    print(f"  Total documents from all clients: {len(all_documents)}")
    print(f"  Score range: {min(all_scores):.4f} - {max(all_scores):.4f}")
    
    # Show which client had the best scores
    best_score_idx = all_scores.index(min(all_scores))  # Lower L2 distance is better
    client_num = (best_score_idx // knn) + 1
    print(f"  Best scoring document came from Client {client_num}")
    
    return True


def main():
    print("FedRAG 5-Client Setup Test")
    print("=" * 30)
    
    # Test individual parts
    if not test_individual_parts():
        print("\n❌ Individual part testing failed!")
        sys.exit(1)
    
    # Test federated simulation
    if not test_federated_simulation():
        print("\n❌ Federated simulation failed!")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\nYour 5-client federated setup is ready!")
    print("\nNext steps:")
    print("1. Run with Flower: flwr run")
    print("2. Or start 5 separate client processes")
    print("3. Each client will automatically be assigned a different corpus part")
    
    print(f"\nCorpus distribution:")
    splitter = CorpusSplitter()
    parts = splitter.list_available_parts()
    for i, part in enumerate(parts, 1):
        print(f"  Client {i}: {part}")


if __name__ == "__main__":
    main() 