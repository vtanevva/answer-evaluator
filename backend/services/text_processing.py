"""
Text processing utilities for answer evaluation
"""

import re
from typing import List, Set
from core.config import settings


class TextProcessor:
    """
    Text processing service for normalizing and analyzing text
    
    This class handles:
    - Text normalization (lowercase, remove punctuation)
    - Stopword removal
    - Basic stemming
    - Token overlap calculation
    """
    
    def __init__(self):
        """Initialize text processor with configuration"""
        self._stopwords = set(settings.text_processing.stopwords)
        self._stemming_suffixes = settings.text_processing.stemming_suffixes
        self._min_token_length = settings.text_processing.min_token_length
    
    def normalize_text(self, text: str) -> List[str]:
        """
        Normalize text by converting to lowercase, removing punctuation,
        filtering stopwords, and applying basic stemming
        
        Args:
            text: Input text to normalize
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        normalized_text = text.lower()
        
        # Remove punctuation and keep only alphanumeric characters
        normalized_text = re.sub(r"[^a-z0-9\s]", " ", normalized_text)
        
        # Split into tokens and filter stopwords
        tokens = [
            token for token in normalized_text.split() 
            if token and token not in self._stopwords
        ]
        
        # Apply basic stemming
        stemmed_tokens = []
        for token in tokens:
            stemmed_token = self._apply_stemming(token)
            stemmed_tokens.append(stemmed_token)
        
        return stemmed_tokens
    
    def _apply_stemming(self, token: str) -> str:
        """
        Apply basic stemming by removing common suffixes
        
        Args:
            token: Token to stem
            
        Returns:
            Stemmed token
        """
        if len(token) <= self._min_token_length:
            return token
            
        for suffix in self._stemming_suffixes:
            if token.endswith(suffix):
                return token[:-len(suffix)]
                
        return token
    
    def calculate_token_overlap(self, user_tokens: List[str], key_point_tokens: List[str]) -> float:
        """
        Calculate the overlap between user tokens and key point tokens
        
        Args:
            user_tokens: Normalized tokens from user answer
            key_point_tokens: Normalized tokens from key point
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        user_token_set = set(user_tokens)
        key_point_token_set = set(key_point_tokens)
        
        if not key_point_token_set:
            return 0.0
            
        intersection = user_token_set & key_point_token_set
        return len(intersection) / len(key_point_token_set)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple punctuation-based splitting
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        raw_sentences = re.split(r"[\.!?\n]+", text)
        sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
        
        # If no sentences found, return the original text
        if not sentences:
            sentences = [text]
            
        return sentences
