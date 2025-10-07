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
        # Polarity lexicons to prevent opposite-meaning matches (expandable)
        self._positive_polarity = {
            "advantage", "advantages", "advantageous",
            "benefit", "benefits", "beneficial",
            "pro", "pros",
            "strength", "strengths",
            "upside"
        }
        self._negative_polarity = {
            "disadvantage", "disadvantages", "disadvantageous",
            "drawback", "drawbacks",
            "con", "cons",
            "weakness", "weaknesses",
            "harm", "harmful",
            "downside"
        }
        # Negating prefixes and common antonym pairs to catch direction conflicts
        self._negating_prefixes = {
            "dis", "un", "in", "im", "ir", "il", "non", "de", "anti", "counter", "mis"
        }
        self._antonym_pairs = {
            ("charge", "discharge"),
            ("connect", "disconnect"),
            ("agree", "disagree"),
            ("like", "dislike"),
            ("activate", "deactivate"),
            ("approve", "disapprove"),
            ("inflate", "deflate"),
            ("engage", "disengage"),
            ("assemble", "disassemble"),
            ("appear", "disappear"),
            ("overcrowding", "undercrowding")
        }
        # Directional prefix pairs (opposites over the same root)
        self._directional_prefix_pairs = [
            ("over", "under"),
            ("hyper", "hypo"),
            ("up", "down")
        ]
        # Verb-based opposites (including different forms)
        self._verb_opposites = {
            # Base forms and -ed/-ing variants
            ("increase", "decrease"),
            ("increased", "decreased"),
            ("increasing", "decreasing"),
            ("rises", "falls"),
            ("rising", "falling"),
            ("rose", "fell"),
            ("grow", "shrink"),
            ("growing", "shrinking"),
            ("grew", "shrank"),
            ("expand", "contract"),
            ("expanding", "contracting"),
            ("expanded", "contracted"),
            ("strengthen", "weaken"),
            ("strengthening", "weakening"),
            ("strengthened", "weakened"),
            ("improve", "worsen"),
            ("improving", "worsening"),
            ("improved", "worsened")
        }
        # Causal indicators (words that link cause and effect)
        self._causal_words = {
            "leads", "lead", "leading",
            "causes", "cause", "causing", "caused",
            "results", "result", "resulting", "resulted",
            "affects", "affect", "affecting", "affected",
            "impacts", "impact", "impacting", "impacted",
            "influences", "influence", "influencing", "influenced",
            "boosts", "boost", "boosting", "boosted",
            "reduces", "reduce", "reducing", "reduced",
            "increases", "increase", "increasing", "increased",
            "decreases", "decrease", "decreasing", "decreased"
        }
    
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
    
    def _detect_polarity(self, tokens: List[str]) -> str:
        """
        Detect coarse polarity from tokens using lightweight lexicons.
        Returns one of: "positive", "negative", "neutral".
        """
        if any(token in self._positive_polarity for token in tokens):
            return "positive"
        if any(token in self._negative_polarity for token in tokens):
            return "negative"
        return "neutral"

    def has_polarity_conflict(self, user_tokens: List[str], key_point_tokens: List[str]) -> bool:
        """
        True if both sides express opposite polarity (positive vs negative).
        Neutral on either side does not trigger a conflict.
        """
        user_pol = self._detect_polarity(user_tokens)
        key_pol = self._detect_polarity(key_point_tokens)
        return (user_pol != "neutral" and key_pol != "neutral" and user_pol != key_pol)

    def _is_negated_pair(self, a: str, b: str) -> bool:
        """
        Detect if a and b differ only by a negating prefix (e.g., charge vs discharge).
        """
        if a == b:
            return False
        for prefix in self._negating_prefixes:
            if a.startswith(prefix) and a[len(prefix):] == b:
                return True
            if b.startswith(prefix) and b[len(prefix):] == a:
                return True
        return False

    def get_semantic_relationship(self, tokens: List[str]) -> tuple[str, str, str]:
        """
        Extract subject-verb-object relationship from tokens.
        Returns (subject, verb, object) or ("", "", "") if not found.
        """
        # Simple extraction - take first matching patterns
        subject = ""
        verb = ""
        obj = ""
        
        for token in tokens:
            # Find the main verb (action/causal word)
            if token in self._causal_words or any(token in pair for pair in self._verb_opposites):
                verb = token
                break
        
        if verb:
            # Look for subject before verb
            verb_idx = tokens.index(verb)
            subject = " ".join(tokens[:verb_idx]) if verb_idx > 0 else ""
            # Look for object after verb
            obj = " ".join(tokens[verb_idx + 1:]) if verb_idx < len(tokens) - 1 else ""
            
        return (subject.strip(), verb.strip(), obj.strip())

    def has_direction_conflict(self, user_tokens: List[str], key_point_tokens: List[str]) -> bool:
        """
        True if semantic relationships conflict (e.g., opposite claims about same subject).
        """
        # First check basic token-level conflicts
        user_set = set(user_tokens)
        kp_set = set(key_point_tokens)
        
        # Extract semantic relationships
        user_subj, user_verb, user_obj = self.get_semantic_relationship(user_tokens)
        kp_subj, kp_verb, kp_obj = self.get_semantic_relationship(key_point_tokens)
        
        # If we found semantic relationships in both
        if user_verb and kp_verb:
            # Check if verbs are opposites
            verbs_opposite = ((user_verb, kp_verb) in self._verb_opposites or 
                            (kp_verb, user_verb) in self._verb_opposites)
            
            # Check if they're talking about the same thing
            similar_subject = (user_subj and kp_subj and 
                             (user_subj in kp_subj or kp_subj in user_subj))
            
            # If opposite claims about same subject, it's a conflict
            if verbs_opposite and similar_subject:
                return True
        
        # Fall back to token-level checks
        for u in user_set:
            for k in kp_set:
                if (u, k) in self._antonym_pairs or (k, u) in self._antonym_pairs:
                    return True
                if self._is_negated_pair(u, k):
                    return True
                # Check directional opposite prefixes with same root
                for pos_pref, neg_pref in self._directional_prefix_pairs:
                    if u.startswith(pos_pref) and k.startswith(neg_pref) and len(u) > len(pos_pref) and len(k) > len(neg_pref):
                        if u[len(pos_pref):] == k[len(neg_pref):]:
                            return True
                    if k.startswith(pos_pref) and u.startswith(neg_pref) and len(k) > len(pos_pref) and len(u) > len(neg_pref):
                        if k[len(pos_pref):] == u[len(neg_pref):]:
                            return True
        return False

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
