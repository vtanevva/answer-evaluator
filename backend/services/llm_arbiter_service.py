"""
LLM Arbiter Service for high-uncertainty grading cases

Uses Llama 3.1 8B via Groq API for ultra-low-cost verification.
Only triggers for top 10-15% most uncertain cases.

Cost: ~€0.00012 per verification (well within budget constraints)
"""

from typing import Dict, Optional
import os
import httpx
from core.config import settings


class LLMArbiterService:
    """
    Service for LLM-based arbitration of uncertain grading cases
    
    Triggers only when:
    - Uncertainty score exceeds threshold (top 15%)
    - Cosine and NLI strongly disagree
    - NLI contradiction with high cosine similarity
    
    Uses Llama 3.1 8B (Groq) for speed and cost-efficiency.
    """
    
    def __init__(self):
        """Initialize LLM arbiter with Groq API"""
        self.enabled = getattr(settings.grading, 'llm_arbiter_enabled', False)
        
        if not self.enabled:
            self.client = None
            return
        
        self.provider = getattr(settings.grading, 'llm_arbiter_provider', 'groq')
        self.model = getattr(settings.grading, 'llm_arbiter_model', 'llama-3.1-8b-instant')
        self.threshold = getattr(settings.grading, 'llm_arbiter_threshold', 0.85)
        self.timeout = getattr(settings.grading, 'llm_arbiter_timeout', 3)
        self.max_tokens = getattr(settings.grading, 'llm_arbiter_max_tokens', 150)
        
        # Initialize API client
        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.api_base = "https://api.groq.com/openai/v1"
            if not self.api_key:
                print("⚠️ GROQ_API_KEY not found, LLM arbiter disabled")
                self.enabled = False
                return
        elif self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_base = "https://api.openai.com/v1"
        else:
            print(f"⚠️ Unsupported LLM provider: {self.provider}")
            self.enabled = False
            return
        
        self.client = httpx.AsyncClient(timeout=self.timeout)
        
        print(f"✅ LLM Arbiter initialized")
        print(f"   Provider: {self.provider}")
        print(f"   Model: {self.model}")
        print(f"   Trigger threshold: {self.threshold} (top ~15% uncertain cases)")
        print(f"   Cost per call: ~€0.00012")
    
    def should_trigger(self, uncertainty_score: float) -> bool:
        """
        Determine if case should trigger LLM arbitration
        
        Args:
            uncertainty_score: Calculated uncertainty (0-1)
            
        Returns:
            True if uncertainty exceeds threshold (top 15% cases)
        """
        return self.enabled and uncertainty_score >= self.threshold
    
    async def verify_answer(
        self,
        question_text: str,
        key_point: str,
        student_answer: str,
        cosine_similarity: float,
        nli_entailment: float,
        nli_contradiction: bool
    ) -> Dict[str, any]:
        """
        Use LLM to arbitrate uncertain grading case
        
        Args:
            question_text: The original question
            key_point: The key point being evaluated
            student_answer: Student's answer text
            cosine_similarity: Embedding similarity score
            nli_entailment: NLI entailment score
            nli_contradiction: Whether NLI detected contradiction
            
        Returns:
            Dictionary with:
            - is_correct: LLM's decision (bool)
            - confidence: LLM's confidence (0-1)
            - reasoning: Brief explanation
        """
        if not self.enabled:
            return {
                "is_correct": False,
                "confidence": 0.0,
                "reasoning": "LLM arbiter disabled"
            }
        
        # Construct concise prompt for Llama 3.1 8B
        prompt = f"""You are grading a student answer. Determine if the student's answer correctly addresses the key point.

Question: {question_text}

Key Point to Check: {key_point}

Student Answer: {student_answer}

Context:
- Cosine Similarity: {cosine_similarity:.2f}
- NLI Entailment: {nli_entailment:.2f}
- NLI Detected Contradiction: {nli_contradiction}

Task: Does the student's answer correctly cover this key point?

Respond in JSON format:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief 1-sentence explanation"
}}"""

        try:
            # Call Groq API (OpenAI-compatible)
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise grading assistant. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,  # Low temperature for consistency
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"}  # JSON mode
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse LLM response
            import json
            llm_output = json.loads(result["choices"][0]["message"]["content"])
            
            return {
                "is_correct": llm_output.get("is_correct", False),
                "confidence": llm_output.get("confidence", 0.5),
                "reasoning": llm_output.get("reasoning", "No reasoning provided")
            }
            
        except Exception as e:
            print(f"⚠️ LLM arbiter error: {e}")
            # Fallback to NLI decision on error
            return {
                "is_correct": nli_entailment > 0.5 and not nli_contradiction,
                "confidence": 0.0,
                "reasoning": f"LLM error: {str(e)}"
            }
    
    async def grade_holistically(
        self,
        question_text: str,
        key_points: list,
        student_answer: str,
        sentences: list = None
    ) -> Dict[str, any]:
        """
        Use LLM to grade the ENTIRE answer holistically like a teacher would.
        
        PRODUCTION-LEVEL: Provides both full answer AND sentence breakdown
        for maximum context understanding.
        
        This is the most accurate method - the LLM reads the whole answer
        and determines which key points are covered, understanding context,
        implications, and paraphrasing naturally.
        
        Args:
            question_text: The original question
            key_points: List of key points to check
            student_answer: Student's complete answer text
            sentences: Pre-split sentences for detailed analysis
            
        Returns:
            Dictionary with:
            - covered: List of booleans for each key point
            - score: Overall percentage (0-100)
            - reasoning: Brief explanation for each key point
        """
        if not self.enabled:
            return None
        
        # Split into sentences if not provided
        if sentences is None:
            sentences = [s.strip() for s in student_answer.split('.') if s.strip()]
        
        # Format key points for prompt
        key_points_text = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(key_points)])
        
        # Format sentence breakdown
        sentences_text = "\n".join([f"  [{i+1}] \"{s}\"" for i, s in enumerate(sentences)])
        
        prompt = f"""You are an expert teacher grading a student's short answer. Your job is to determine which key points from the marking scheme are covered in the student's answer.

QUESTION: {question_text}

MARKING SCHEME - Key Points (each worth equal marks):
{key_points_text}

STUDENT'S COMPLETE ANSWER:
"{student_answer}"

SENTENCE-BY-SENTENCE BREAKDOWN:
{sentences_text}

GRADING INSTRUCTIONS:
1. Read the COMPLETE answer first to understand the overall meaning
2. Then check each sentence to see which key points are addressed
3. A key point is COVERED if:
   - The student addresses the concept directly, OR
   - The student implies it clearly through context, OR
   - The student uses synonyms/paraphrasing (e.g., "jobs" = "employment" = "work opportunities")
4. A key point is NOT COVERED if:
   - It's completely absent from the answer, OR
   - The student contradicts it (says the opposite)
5. Be GENEROUS - if there's a reasonable interpretation that covers the point, mark it as covered

For each key point, respond with:
- true if covered (found in any sentence or implied by overall meaning)
- false if not covered

Respond ONLY with valid JSON:
{{
  "covered": [true, false, true, ...],
  "reasoning": ["sentence [X]: 'quote' addresses this", "missing: no mention of concept Y", ...]
}}"""

        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a fair and accurate grading assistant. Evaluate student answers holistically. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"}
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            import json
            llm_output = json.loads(result["choices"][0]["message"]["content"])
            
            covered = llm_output.get("covered", [False] * len(key_points))
            reasoning = llm_output.get("reasoning", ["No reason"] * len(key_points))
            
            # Calculate score
            score = (sum(covered) / len(key_points)) * 100 if key_points else 0
            
            print(f"   🤖 LLM Holistic Grading: {sum(covered)}/{len(key_points)} key points covered ({score:.0f}%)")
            for i, (kp, cov, reason) in enumerate(zip(key_points, covered, reasoning)):
                status = "✅" if cov else "❌"
                print(f"      {status} KP{i+1}: {reason}")
            
            return {
                "covered": covered,
                "score": score,
                "reasoning": reasoning,
                "method": "llm_holistic"
            }
            
        except Exception as e:
            print(f"⚠️ LLM holistic grading error: {e}")
            return None
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
