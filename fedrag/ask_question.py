#!/usr/bin/env python3
"""
Simple script to ask questions to the FedRAG system.
This script makes it easy to run federated queries without dealing with the Flower command directly.
"""

import subprocess
import sys
import argparse


def ask_question(question: str):
    """Run the FedRAG system with a specific question."""
    
    print("🚀 Starting FedRAG with your question...")
    print(f"❓ Question: {question}")
    print("-" * 60)
    
    # Create a temporary config file to avoid TOML parsing issues
    import tempfile
    import os
    
    # Create temporary config content
    config_content = f'user-question = """{question}"""'
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        temp_config_file = f.name
    
    try:
        # Build the command using the temporary config file
        cmd = [
            'flwr', 'run',
            '--run-config', temp_config_file
        ]
        
        # Run the federated system
        result = subprocess.run(cmd, check=True, text=True)
        print("\n✅ FedRAG completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running FedRAG: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ Error: 'flwr' command not found. Please make sure Flower is installed.")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_config_file)
        except:
            pass


def interactive_mode():
    """Run in interactive mode where user can ask multiple questions."""
    
    print("🎯 FedRAG Interactive Question Asking")
    print("=" * 40)
    print("Ask medical questions to the federated RAG system!")
    print("Type 'quit', 'exit', or 'q' to stop.\n")
    
    while True:
        try:
            question = input("🔍 Enter your medical question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
                
            success = ask_question(question)
            
            if not success:
                print("Would you like to try another question? (y/n)")
                continue_choice = input().strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions to the FedRAG system"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Medical question to ask (if not provided, will run in interactive mode)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.interactive or not args.question:
        interactive_mode()
    else:
        ask_question(args.question)


if __name__ == "__main__":
    main() 