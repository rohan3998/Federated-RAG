#!/usr/bin/env python3
"""
Script to split the textbooks corpus into 5 parts for federated learning setup.
This will create textbooks_part1, textbooks_part2, ..., textbooks_part5 directories.
"""

import argparse
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fedrag.corpus_splitter import CorpusSplitter


def main():
    parser = argparse.ArgumentParser(
        description="Split textbooks corpus into multiple parts for federated clients"
    )
    parser.add_argument(
        "--num-parts", 
        type=int, 
        default=5,
        help="Number of parts to split the corpus into (default: 5)"
    )
    parser.add_argument(
        "--skip-indexing", 
        action="store_true",
        help="Skip building FAISS indices (only split the corpus)"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true",
        help="Remove existing corpus parts before creating new ones"
    )
    parser.add_argument(
        "--list-parts", 
        action="store_true",
        help="List existing corpus parts and exit"
    )
    
    args = parser.parse_args()
    
    splitter = CorpusSplitter()
    
    if args.list_parts:
        parts = splitter.list_available_parts()
        if parts:
            print("Existing corpus parts:")
            for part in parts:
                print(f"  - {part}")
        else:
            print("No corpus parts found.")
        return
    
    if args.cleanup:
        existing_parts = splitter.list_available_parts()
        if existing_parts:
            print("Cleaning up existing parts...")
            splitter.cleanup_parts(existing_parts)
        else:
            print("No existing parts to cleanup.")
    
    print(f"Splitting textbooks corpus into {args.num_parts} parts...")
    try:
        part_names = splitter.split_textbooks_corpus(num_parts=args.num_parts)
        print(f"\nSuccessfully created parts: {part_names}")
        
        if not args.skip_indexing:
            print("\nBuilding FAISS indices for all parts...")
            print("This may take several minutes...")
            splitter.build_indices_for_parts(part_names)
            print("FAISS indexing completed!")
        else:
            print("\nSkipped FAISS indexing. Run the following to build indices later:")
            print("python split_corpus.py --skip-indexing")
        
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("\nTo use these parts with 5 clients, update your configuration:")
        print(f'clients-corpus-names = "{"|".join(part_names)}"')
        print("\nOr update pyproject.toml:")
        print(f'clients-corpus-names = "{"|".join(part_names)}"')
        
        print(f"\nNow you can run the federated system with 5 clients:")
        print("flwr run --run-config-overrides clients-corpus-names=\"{}\"".format("|".join(part_names)))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 