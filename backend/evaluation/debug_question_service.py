"""
Debug script to check what's happening with the question service
"""

import sys
import os
sys.path.append('..')

from evaluation_script import EvaluationBenchmark

print('Testing EvaluationBenchmark...')
try:
    eval = EvaluationBenchmark()
    print(f'✅ Questions loaded: {len(eval._questions)}')
    
    if len(eval._questions) > 0:
        q1 = eval._questions[0]
        print(f'Question 1: {q1["question_text"]}')
        print(f'Question 1 key points:')
        for i, kp in enumerate(q1["key_points"]):
            print(f'  {i+1}. {kp["text"]}')
        
        # Check what the grading service sees
        print(f'\nChecking grading service...')
        all_q = eval._grading_service._question_service.get_all_questions()
        print(f'Grading service sees {len(all_q)} questions')
        
        if len(all_q) > 0:
            gq1 = all_q[0]
            print(f'Grading service Question 1: {gq1["question_text"]}')
            print(f'Grading service Question 1 key points:')
            for i, kp in enumerate(gq1["key_points"]):
                print(f'  {i+1}. {kp["text"]}')
        
        # Check the question service properties
        print(f'\nQuestion service properties:')
        print(f'_is_loaded: {eval._question_service._is_loaded}')
        print(f'_questions_bank length: {len(eval._question_service._questions_bank)}')
        print(f'_questions_by_id length: {len(eval._question_service._questions_by_id)}')
        
        if len(eval._question_service._questions_bank) > 0:
            print(f'First question in _questions_bank: {eval._question_service._questions_bank[0]["question_text"]}')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
