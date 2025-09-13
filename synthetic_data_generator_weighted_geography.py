#!/usr/bin/env python3
"""
Weighted Synthetic Data Generator for Geography (Secondary School)
- Numeric question IDs
- Weighted key points format
- Uses gpt-4o-mini
"""

import os
import json
import time
import random
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from config_weighted_geography import *

# Load environment variables from .env file
load_dotenv()

# Import OpenAI with backward compatibility
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    OPENAI_NEW_VERSION = True
except ImportError:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    OPENAI_NEW_VERSION = False

# Global token tracking
total_input_tokens = 0
total_output_tokens = 0
total_cost_usd = 0.0
seen_questions: set[str] = set()

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1000) * GPT4O_MINI_INPUT_COST_PER_1K_TOKENS
    output_cost = (output_tokens / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS
    return input_cost + output_cost

def update_token_tracking(input_tokens: int, output_tokens: int):
    global total_input_tokens, total_output_tokens, total_cost_usd
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    cost = calculate_cost(input_tokens, output_tokens)
    total_cost_usd += cost
    return cost

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from model output; tolerate verbosity/code fences."""
    text = text.strip()
    # remove code fences if present
    if text.startswith("```") and text.endswith("```"):
        text = text.strip('`')
        # after removing fences there might be a language tag
        if "\n" in text:
            text = text.split("\n", 1)[1]
    # try direct
    try:
        return json.loads(text)
    except Exception:
        # try substring between first { and last }
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            pass
    raise ValueError("Invalid JSON from model")


def generate_weighted_question(question_id: int, prior_examples: List[str] | None = None) -> Tuple[Dict[str, Any], float]:
    question_type = random.choice(QUESTION_TYPES)
    prior_hint = "\nPreviously generated questions (avoid repeating phrasing):\n- " + "\n- ".join(prior_examples[:10]) if prior_examples else ""
    prompt = f"""
    Create a {SUBJECT} question suitable for {GRADE_LEVEL} students ({STUDENT_AGE_RANGE}) about {question_type} and provide 2-4 key points with weights.
    Return your response as a JSON object with this exact structure:
    {{
        "question_text": "Your {SUBJECT} question here",
        "key_points": [
            {{"text": "First key point", "weight": 1}},
            {{"text": "Second key point", "weight": 1}},
            {{"text": "Third key point", "weight": 1}}
        ]
    }}
    Requirements:
    - Each key point must be a short, specific chunk (5–15 words).
    - Points distinct and non-overlapping for easy grading.
    - 2–4 points total, weight 1 each.
    - Do NOT repeat a question text or its phrasing used earlier in this session. Vary subtopics and wording.
    Examples:
    - "Factors influencing population density"
    - "Warm ocean currents raise coastal temperatures"
    - "Plates converge at subduction zones"
    {prior_hint}
    """
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = update_token_tracking(input_tokens, output_tokens)
            result = _parse_json_response(response.choices[0].message.content)
        else:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = update_token_tracking(input_tokens, output_tokens)
            result = _parse_json_response(response.choices[0].message.content)
        result["question_id"] = question_id
        return result, cost
    except Exception:
        fallback = {
            "question_id": question_id,
            "question_text": "Explain the causes of earthquakes.",
            "key_points": [
                {"text": "Movement along fault lines", "weight": 1},
                {"text": "Release of built-up tectonic stress", "weight": 1},
                {"text": "Plate boundary interactions", "weight": 1}
            ]
        }
        return fallback, 0.0

def save_weighted_data(records: List[Dict[str, Any]], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def print_progress_weighted(question_num: int, total_questions: int, current_cost: float):
    if SHOW_PROGRESS:
        pct = (question_num / total_questions) * 100
        print(f"📊 Progress: {question_num}/{total_questions} ({pct:.1f}%) - Cost: ${current_cost:.4f}")

def main():
    print("🚀 Starting WEIGHTED Geography data generation...")
    print("📊 Configuration:")
    print(f"   - Subject: {SUBJECT}")
    print(f"   - Grade level: {GRADE_LEVEL}")
    print(f"   - Questions to generate: {M_QUESTIONS}")
    print(f"   - Output file: {OUTPUT_FILE}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Question types: {', '.join(QUESTION_TYPES)}")
    print("-" * 60)

    all_records: List[Dict[str, Any]] = []
    start = time.time()
    try:
        for qid in range(1, M_QUESTIONS + 1):
            if SHOW_PROGRESS:
                print(f"📝 Generating weighted question {qid}/{M_QUESTIONS}...")
            # retry up to 4 times for valid & unique question_text
            data = None
            cost_accum = 0.0
            for attempt in range(4):
                candidate, cost = generate_weighted_question(qid, prior_examples=list(seen_questions))
                cost_accum += cost
                qt = candidate.get("question_text", "").strip().lower()
                if qt and qt not in seen_questions:
                    data = candidate
                    seen_questions.add(qt)
                    break
                time.sleep(0.2)
            if data is None:
                # last resort: make unique suffix
                candidate, _ = generate_weighted_question(qid)
                qt = candidate.get("question_text", "Untitled question")
                candidate["question_text"] = f"{qt} (variant {qid})"
                data = candidate
                seen_questions.add(candidate["question_text"].lower())
            print(f"   Question: {data['question_text'][:60]}{'...' if len(data['question_text'])>60 else ''}")
            print(f"   Key points: {len(data['key_points'])}")
            print(f"   Question generation cost: ${cost_accum:.4f}")
            all_records.append(data)
            print_progress_weighted(qid, M_QUESTIONS, total_cost_usd)
            time.sleep(API_DELAY)
            print()
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrupted, saving {len(all_records)} questions so far...")

    print(f"💾 Saving {len(all_records)} questions to {OUTPUT_FILE}...")
    save_weighted_data(all_records, OUTPUT_FILE)
    dur = time.time() - start
    print("✅ Generation complete!")
    print("📊 Final Statistics:")
    print(f"   - Total questions: {len(all_records)}")
    print(f"   - Time taken: {dur:.1f} seconds")
    print(f"   - Questions per second: {len(all_records)/dur:.1f}")
    print(f"   - Output file: {OUTPUT_FILE}")
    print("💰 Cost Analysis:")
    print(f"   - Total input tokens: {total_input_tokens:,}")
    print(f"   - Total output tokens: {total_output_tokens:,}")
    print(f"   - Total tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"   - Total cost: ${total_cost_usd:.4f}")
    if all_records:
        print("📄 Sample record:")
        print(json.dumps(all_records[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in environment.")
        print("Create backend/.env with your key or set the env var.")
        raise SystemExit(1)
    main()
