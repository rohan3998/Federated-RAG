"""fedrag: A Flower Federated RAG app."""

import os
import shutil
import numpy as np
from pathlib import Path
from typing import List

from fedrag.retriever import Retriever

class CorpusSplitter:
    """Handles splitting corpus datasets into multiple parts for federated clients."""
    
    def __init__(self, corpus_dir: str = None):
        if not corpus_dir:
            self.corpus_dir = os.path.join(os.path.dirname(__file__), "../data/corpus")
        else:
            self.corpus_dir = corpus_dir
    
    def split_textbooks_corpus(self, num_parts: int = 5) -> List[str]:
        """
        Split the textbooks corpus into multiple parts for different clients.
        
        Args:
            num_parts: Number of parts to split the corpus into
            
        Returns:
            List of part names (e.g., ['textbooks_part1', 'textbooks_part2', ...])
        """
        textbooks_dir = os.path.join(self.corpus_dir, "textbooks")
        chunk_dir = os.path.join(textbooks_dir, "chunk")
        
        if not os.path.exists(chunk_dir):
            raise RuntimeError("Textbooks corpus not found. Please download it first.")
        
        # Get all textbook files
        chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.jsonl')]
        chunk_files.sort()  # Ensure consistent ordering
        
        print(f"Found {len(chunk_files)} textbook files to split into {num_parts} parts")
        
        # Distribute files across parts as evenly as possible
        files_per_part = len(chunk_files) // num_parts
        remainder = len(chunk_files) % num_parts
        
        part_names = []
        start_idx = 0
        
        for part_idx in range(num_parts):
            part_name = f"textbooks_part{part_idx + 1}"
            part_names.append(part_name)
            part_dir = os.path.join(self.corpus_dir, part_name)
            part_chunk_dir = os.path.join(part_dir, "chunk")
            
            # Create directories
            Path(part_chunk_dir).mkdir(parents=True, exist_ok=True)
            
            # Calculate how many files this part gets
            current_part_size = files_per_part + (1 if part_idx < remainder else 0)
            end_idx = start_idx + current_part_size
            
            # Copy files to this part
            part_files = chunk_files[start_idx:end_idx]
            print(f"Part {part_idx + 1}: {len(part_files)} files")
            print(f"  Files: {', '.join(part_files)}")
            
            for file_name in part_files:
                src_path = os.path.join(chunk_dir, file_name)
                dst_path = os.path.join(part_chunk_dir, file_name)
                shutil.copy2(src_path, dst_path)
            
            start_idx = end_idx
        
        return part_names
    
    def build_indices_for_parts(self, part_names: List[str], batch_size: int = 32):
        """
        Build FAISS indices for each corpus part.
        
        Args:
            part_names: List of part names to build indices for
            batch_size: Batch size for processing embeddings
        """
        retriever = Retriever()
        
        for part_name in part_names:
            print(f"Building FAISS index for {part_name}...")
            retriever.build_faiss_index(
                dataset_name=part_name,
                batch_size=batch_size
            )
            print(f"Built FAISS index for {part_name}")
            
            # Test the index with a sample query
            sample_query = "What are the complications of a cardiovascular disease?"
            try:
                results = retriever.query_faiss_index(part_name, sample_query, knn=2)
                print(f"Test query for {part_name} returned {len(results)} results")
            except Exception as e:
                print(f"Warning: Test query failed for {part_name}: {e}")
    
    def cleanup_parts(self, part_names: List[str]):
        """
        Remove the split corpus parts (useful for cleanup).
        
        Args:
            part_names: List of part names to remove
        """
        for part_name in part_names:
            part_dir = os.path.join(self.corpus_dir, part_name)
            if os.path.exists(part_dir):
                shutil.rmtree(part_dir)
                print(f"Removed {part_name}")
    
    def list_available_parts(self) -> List[str]:
        """
        List all available textbook parts.
        
        Returns:
            List of part names found in the corpus directory
        """
        part_names = []
        for item in os.listdir(self.corpus_dir):
            if item.startswith("textbooks_part") and os.path.isdir(os.path.join(self.corpus_dir, item)):
                part_names.append(item)
        return sorted(part_names)


if __name__ == "__main__":
    # Example usage
    splitter = CorpusSplitter()
    
    print("Splitting textbooks corpus into 5 parts...")
    part_names = splitter.split_textbooks_corpus(num_parts=5)
    
    print(f"\nCreated parts: {part_names}")
    
    print("\nBuilding FAISS indices for all parts...")
    splitter.build_indices_for_parts(part_names)
    
    print("\nCorpus splitting and indexing completed!")
    print("\nTo use these parts, update your configuration:")
    print(f'clients-corpus-names = "{"|".join(part_names)}"') 