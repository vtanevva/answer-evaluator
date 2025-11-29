#!/usr/bin/env python3
"""
Test script for the new chunking functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.text_processing import TextProcessor

def test_chunking():
    """Test the chunking functionality"""
    processor = TextProcessor()
    
    # Test answer
    answer = """
    Photosynthesis is the process by which plants convert sunlight into energy. 
    This process occurs in the chloroplasts of plant cells. 
    The main components are carbon dioxide, water, and sunlight. 
    The result is glucose and oxygen. 
    This process is essential for life on Earth.
    """
    
    print("Testing answer chunking...")
    print(f"Original answer: {answer.strip()}")
    print()
    
    # Test chunking
    chunks = processor.chunk_answer(answer, chunk_size=3)
    print(f"Number of chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk}")
    print()
    
    # Test n-gram similarity
    text1 = "photosynthesis converts sunlight energy"
    text2 = "plants use sunlight to make energy"
    
    similarity = processor.calculate_ngram_similarity(text1, text2, n=2)
    print(f"N-gram similarity between '{text1}' and '{text2}': {similarity:.3f}")
    
    # Test with different n-gram sizes
    for n in [1, 2, 3]:
        similarity = processor.calculate_ngram_similarity(text1, text2, n=n)
        print(f"N-gram similarity (n={n}): {similarity:.3f}")

if __name__ == "__main__":
    test_chunking()
