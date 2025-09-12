"""
Alternative question loader - loads from JSON file instead of hardcoded data
Use this if you want to load questions from a separate JSON file
"""

import json
import os

def load_questions_from_file(file_path="questions.json"):
    """
    Load questions from JSON file
    
    Args:
        file_path: Path to the JSON file containing questions
        
    Returns:
        List of question dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"✅ Loaded {len(questions)} questions from {file_path}")
        return questions
    except FileNotFoundError:
        print(f"❌ Questions file {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return []

# Example usage in main.py:
# Replace the hardcoded questions_bank with:
# questions_bank = load_questions_from_file("questions.json")
