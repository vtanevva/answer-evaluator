"""
LLM-based similarity verification utilities.

This module introduces an additional verification step for borderline
embedding similarity scores. Chunks whose similarity falls within the
configurable mid-range band (default 0.8 < similarity < 0.855) will be 

at 0.75 it doesnt work as llm verification doesnt work correclty

validated with an LLM to determine whether they should be treated as true
matches.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from core.config import settings

# Default LLM verification threshold for borderline matches
DEFAULT_LLM_THRESHOLD = 0.80


@dataclass
class LLMVerificationDecision:
    """Represents the decision returned by the LLM verification step."""

    is_similar: bool
    explanation: str
    raw_response: str


@dataclass
class LLMVerificationResult:
    """Outcome of verifying a single chunk."""

    adjusted_score: float
    llm_used: bool
    reason: str
    decision: Optional[LLMVerificationDecision] = None


class LLMVerificationService:
    """
    Applies an LLM-based verification step for borderline similarity scores.

    High-similarity chunks (>= high threshold) keep their original score.
    Low-similarity chunks (<= lower threshold) keep their original score.
    Only chunks with similarity in (lower threshold, high threshold) are double-checked.
    """

    def __init__(
        self,
        openai_client: Any,
        *,
        high_similarity: Optional[float] = None,
        llm_threshold: Optional[float] = None,
    ) -> None:
        similarity_config = settings.grading.similarity_thresholds

        self._client = openai_client
        self._high_similarity = (
            high_similarity
            if high_similarity is not None
            else similarity_config.high_similarity
        )
        self._llm_threshold = (
            llm_threshold if llm_threshold is not None else DEFAULT_LLM_THRESHOLD
        )

        if self._llm_threshold >= self._high_similarity:
            raise ValueError(
                "LLM verification threshold must be less than the high similarity threshold."
            )

    def verify_chunk(
        self,
        chunk_text: str,
        reference_chunks: Iterable[str],
        similarity_score: float,
    ) -> LLMVerificationResult:
        """
        Verify a single chunk, optionally invoking the LLM.

        Args:
            chunk_text: The candidate chunk to verify.
            reference_chunks: Reference chunks from the ground-truth answer.
            similarity_score: The original cosine similarity score.

        Returns:
            LLMVerificationResult with the adjusted score and LLM metadata.
        """

        # Clearly correct matches keep their original score.
        if similarity_score >= self._high_similarity:
            return LLMVerificationResult(
                adjusted_score=similarity_score,
                llm_used=False,
                reason="above_high_threshold",
            )

        # Scores at or below the LLM threshold keep their original score.
        if similarity_score <= self._llm_threshold:
            return LLMVerificationResult(
                adjusted_score=similarity_score,
                llm_used=False,
                reason="at_or_below_llm_threshold",
            )

        # Borderline chunk – attempt an LLM verification.
        decision = self._llm_decide(chunk_text, list(reference_chunks))

        if decision is None:
            return LLMVerificationResult(
                adjusted_score=similarity_score,
                llm_used=False,
                reason="llm_unavailable",
            )

        reason = (
            "llm_confirmed_similarity" if decision.is_similar else "llm_rejected_similarity"
        )

        return LLMVerificationResult(
            adjusted_score=similarity_score,
            llm_used=True,
            reason=reason,
            decision=decision,
        )

    def verify_chunks(
        self,
        chunk_scores: Iterable[Dict[str, Any]],
        reference_chunks: Iterable[str],
    ) -> List[LLMVerificationResult]:
        """
        Run verification across multiple chunks.

        Args:
            chunk_scores: Iterable of dicts with at least `text` (or `chunk`)
                and `similarity` keys.
            reference_chunks: Iterable of ground-truth chunks.

        Returns:
            A list of verification results in input order.
        """

        reference_list = list(reference_chunks)
        results: List[LLMVerificationResult] = []

        for chunk in chunk_scores:
            candidate_text = (
                chunk.get("text")
                or chunk.get("chunk")
                or chunk.get("candidate")
                or ""
            )
            similarity = float(chunk.get("similarity", 0.0))

            result = self.verify_chunk(candidate_text, reference_list, similarity)
            results.append(result)

        return results

    def _llm_decide(
        self, chunk_text: str, reference_chunks: List[str]
    ) -> Optional[LLMVerificationDecision]:
        """
        Ask the LLM whether the candidate chunk matches any reference chunk.
        """
        if not self._client:
            return None

        if not reference_chunks:
            return LLMVerificationDecision(
                is_similar=False,
                explanation="No reference chunks provided to compare against.",
                raw_response="NO - no reference chunks provided",
            )

        prompt = self._build_prompt(chunk_text, reference_chunks)

        try:
            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator that determines whether a candidate "
                            "chunk from a student's answer matches any reference chunk."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.0,
            )

            message = response.choices[0].message.content.strip()
        except Exception as exc:
            return LLMVerificationDecision(
                is_similar=False,
                explanation=f"LLM call failed: {exc}",
                raw_response=f"ERROR: {exc}",
            )

        normalized = message.lower()
        is_affirmative = normalized.startswith("yes") or '"is_similar": true' in normalized

        return LLMVerificationDecision(
            is_similar=is_affirmative,
            explanation=message,
            raw_response=message,
        )

    @staticmethod
    def _build_prompt(candidate_chunk: str, reference_chunks: List[str]) -> str:
        """
        Construct the verification prompt for the LLM.
        """
        references_formatted = "\n".join(
            f"{idx + 1}. {reference.strip()}" for idx, reference in enumerate(reference_chunks)
        )

        prompt_lines = [
            "Decide if the candidate chunk conveys the same meaning as ANY of the reference chunks.",
            "Answer with 'YES - explanation' or 'NO - explanation'.",
            "",
            f"Candidate chunk:\n{candidate_chunk.strip()}",
            "",
            "Reference chunks:",
            references_formatted,
        ]

        return "\n".join(prompt_lines)

