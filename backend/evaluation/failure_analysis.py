"""
Detailed analysis of specific failure cases to understand what's going wrong
"""

import sys
import os
sys.path.append('..')

from evaluation_script import EvaluationBenchmark
import json

def analyze_failure_cases():
    """Analyze specific cases where the system fails"""
    
    print("🔍 DETAILED FAILURE ANALYSIS")
    print("=" * 80)
    
    # Initialize evaluation
    eval_benchmark = EvaluationBenchmark()
    
    # Load the annotated questions to see the actual incorrect answers
    with open('annotated_questions.json', 'r') as f:
        questions = json.load(f)
    
    print("📊 ANALYZING FAILING QUESTIONS:")
    print()
    
    # Focus on questions with low accuracy
    failing_questions = [1, 2, 4, 7]  # These had 50-62.5% accuracy
    
    for q_id in failing_questions:
        question = next(q for q in questions if q["question_id"] == q_id)
        print(f"📝 QUESTION {q_id}: {question['question_text']}")
        print("-" * 60)
        
        print("Key Points:")
        for i, kp in enumerate(question["key_points"]):
            print(f"  {i+1}. {kp['text']}")
        
        print("\n❌ INCORRECT ANSWERS (that are getting marked as correct):")
        
        for i, incorrect_answer in enumerate(question["incorrect_answers"]):
            print(f"\n  Incorrect Answer {i+1}:")
            print(f"  Text: '{incorrect_answer}'")
            
            try:
                result = eval_benchmark._grading_service.grade_answer(q_id, incorrect_answer)
                print(f"  Score: {result.score}%")
                print(f"  Hit Key Points: {result.hit_key_points}")
                
                # Analyze why this got a high score
                if result.score >= 60:  # Classification threshold
                    print(f"  🚨 PROBLEM: This incorrect answer got {result.score}% (>= 60%)")
                    print(f"  Why it passed:")
                    for kp_text in result.hit_key_points:
                        print(f"    - Matched: '{kp_text}'")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    analyze_failure_cases()
