"""
Evaluation benchmark script for the Answer Evaluator system

This script evaluates the system performance using hardcoded correct and incorrect answers
for questions from the data directory. It calculates precision, recall, F1, and accuracy metrics.
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import statistics
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, that's ok

# Add backend to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.grading_service import GradingService
from services.question_service import QuestionService
from core.config import settings
from services.embedding_service import EmbeddingService
import openai


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    question_id: int
    question_text: str
    correct_answers: List[str]
    incorrect_answers: List[str]
    correct_predictions: List[bool]  # True if correctly identified as correct
    incorrect_predictions: List[bool]  # True if correctly identified as incorrect
    threshold_used: float


@dataclass
class MetricsResult:
    """Container for calculated metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_predictions: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class EvaluationBenchmark:
    """
    Benchmark evaluation system with hardcoded answers
    """
    
    def __init__(self):
        """Initialize the benchmark system"""
        self._load_questions_from_data_files()
        if len(self._questions) == 0:
            print("❌ No questions loaded")
            return
        self._initialize_grading_service()
        self._setup_hardcoded_answers()
        
    def _load_questions_from_data_files(self) -> None:
        """Load questions from annotated questions file"""
        # Load from our annotated questions file with realistic student answers
        annotated_file = Path(__file__).parent / "annotated_questions.json"
        self._questions = []
        
        if annotated_file.exists():
            with open(annotated_file, 'r', encoding='utf-8') as f:
                self._questions = json.load(f)
                print(f"✅ Loaded {len(self._questions)} annotated questions with realistic student answers")
        else:
            print("❌ Annotated questions file not found")
        
        print(f"📊 Total questions loaded: {len(self._questions)}")
    
    def _initialize_grading_service(self) -> None:
        """Initialize the evaluation service"""
        # Set up OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found!")
            print("   Current environment variables containing 'OPENAI':")
            for key, value in os.environ.items():
                if 'OPENAI' in key.upper():
                    print(f"   {key} = {value[:10]}...")
            print("\n   Please set your API key:")
            print("   - Environment: export OPENAI_API_KEY='your-key'")
            print("   - Or create .env file with: OPENAI_API_KEY=your-key")
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        print(f"✅ OpenAI API key found")
        openai.api_key = openai_api_key
        
        # Create question service with our loaded questions
        self._question_service = QuestionService()
        self._question_service._questions = self._questions  # Override with our data
        
        # Create evaluation service
        self._grading_service = GradingService(
            self._question_service, 
            openai
        )
        
        print("🔄 Precomputing embeddings for evaluation...")
        self._grading_service.precompute_embeddings()
        print("✅ Evaluation service initialized")
    
    def _setup_hardcoded_answers(self) -> None:
        """Setup answers from the annotated questions file"""
        self._hardcoded_answers: Dict[int, Dict[str, List[str]]] = {}
        
        for question in self._questions:
            question_id = question["question_id"]
            
            # Use the realistic student answers from the annotated file
            correct_answers = question.get("correct_answers", [])
            incorrect_answers = question.get("incorrect_answers", [])
            
            self._hardcoded_answers[question_id] = {
                "correct": correct_answers,
                "incorrect": incorrect_answers
            }
            
            print(f"📝 Question {question_id}: {len(correct_answers)} correct + {len(incorrect_answers)} incorrect answers")
    
    def _evaluate_answer_and_classify(self, question_id: int, answer: str, expected_correct: bool) -> bool:
        """
        Evaluate an answer and determine if it was classified correctly
        
        Args:
            question_id: ID of the question
            answer: The answer text to evaluate
            expected_correct: Whether this answer should be classified as correct
            
        Returns:
            True if the classification was correct, False otherwise
        """
        try:
            evaluation_result = self._grading_service.evaluate_answer(question_id, answer)
            
            # Use configurable classification threshold instead of hardcoded 50%
            classification_threshold = 60.0  # Slightly higher threshold for better precision
            predicted_correct = evaluation_result.score >= classification_threshold
            
            # Return True if prediction matches expectation
            return predicted_correct == expected_correct
            
        except Exception as e:
            print(f"❌ Error evaluating answer for question {question_id}: {e}")
            return False
    
    def run_evaluation(self) -> List[EvaluationResult]:
        """
        Run the complete evaluation benchmark
        
        Returns:
            List of evaluation results for each question
        """
        print("\n🔬 Starting evaluation benchmark...")
        print(f"📊 Testing {len(self._questions)} questions with 8 answers each")
        print(f"🎯 Using similarity thresholds: High={settings.evaluation.similarity_thresholds.high_similarity}, Mid={settings.evaluation.similarity_thresholds.mid_similarity}")
        print("="*80)
        
        evaluation_results = []
        
        for i, question in enumerate(self._questions, 1):
            question_id = question["question_id"]
            question_text = question["question_text"]
            
            print(f"\n📝 Question {i}/{len(self._questions)}: {question_text[:60]}...")
            
            if question_id not in self._hardcoded_answers:
                print(f"⚠️ No hardcoded answers for question {question_id}, skipping...")
                continue
            
            answers_data = self._hardcoded_answers[question_id]
            correct_answers = answers_data["correct"]
            incorrect_answers = answers_data["incorrect"]
            
            # Evaluate correct answers
            print("  ✅ Evaluating correct answers...")
            correct_predictions = []
            for j, answer in enumerate(correct_answers, 1):
                is_correct_classification = self._evaluate_answer_and_classify(
                    question_id, answer, expected_correct=True
                )
                correct_predictions.append(is_correct_classification)
                status = "✓" if is_correct_classification else "✗"
                print(f"    {status} Correct answer {j}: {'Correctly classified' if is_correct_classification else 'Misclassified'}")
            
            # Evaluate incorrect answers
            print("  ❌ Evaluating incorrect answers...")
            incorrect_predictions = []
            for j, answer in enumerate(incorrect_answers, 1):
                is_correct_classification = self._evaluate_answer_and_classify(
                    question_id, answer, expected_correct=False
                )
                incorrect_predictions.append(is_correct_classification)
                status = "✓" if is_correct_classification else "✗"
                print(f"    {status} Incorrect answer {j}: {'Correctly classified' if is_correct_classification else 'Misclassified'}")
            
            # Store results
            result = EvaluationResult(
                question_id=question_id,
                question_text=question_text,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
                correct_predictions=correct_predictions,
                incorrect_predictions=incorrect_predictions,
                threshold_used=settings.evaluation.similarity_thresholds.high_similarity
            )
            evaluation_results.append(result)
            
            # Show question summary
            correct_classified = sum(correct_predictions)
            incorrect_classified = sum(incorrect_predictions)
            total_correct = correct_classified + incorrect_classified
            question_accuracy = total_correct / 8 * 100
            print(f"  📊 Question accuracy: {question_accuracy:.1f}% ({total_correct}/8)")
        
        return evaluation_results
    
    def calculate_metrics(self, evaluation_results: List[EvaluationResult]) -> MetricsResult:
        """
        Calculate precision, recall, F1, and accuracy from evaluation results
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Calculated metrics
        """
        true_positives = 0  # Correct answers correctly classified as correct
        false_positives = 0  # Incorrect answers incorrectly classified as correct
        true_negatives = 0  # Incorrect answers correctly classified as incorrect
        false_negatives = 0  # Correct answers incorrectly classified as incorrect
        
        for result in evaluation_results:
            # Process correct answers (should be classified as positive)
            for prediction in result.correct_predictions:
                if prediction:
                    true_positives += 1
                else:
                    false_negatives += 1
            
            # Process incorrect answers (should be classified as negative)
            for prediction in result.incorrect_predictions:
                if prediction:
                    true_negatives += 1
                else:
                    false_positives += 1
        
        total_predictions = true_positives + false_positives + true_negatives + false_negatives
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0.0
        
        return MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            total_predictions=total_predictions,
            true_positives=true_positives,
            false_positives=false_positives,
            true_negatives=true_negatives,
            false_negatives=false_negatives
        )
    
    def display_results(self, evaluation_results: List[EvaluationResult], metrics: MetricsResult) -> None:
        """
        Display evaluation results in a nice format
        
        Args:
            evaluation_results: List of evaluation results
            metrics: Calculated metrics
        """
        print("\n" + "="*80)
        print("🎯 EVALUATION BENCHMARK RESULTS")
        print("="*80)
        
        # Configuration info
        print(f"\n📋 CONFIGURATION:")
        print(f"  🎯 Cosine Similarity Threshold (High): {settings.evaluation.similarity_thresholds.high_similarity}")
        print(f"  🎯 Cosine Similarity Threshold (Mid):  {settings.evaluation.similarity_thresholds.mid_similarity}")
        print(f"  📏 Minimum Lexical Overlap:          {settings.evaluation.similarity_thresholds.min_lexical_overlap}")
        print(f"  📊 Classification Threshold:          60% (answers >= 60% score considered correct)")
        
        # Overall metrics
        print(f"\n📊 OVERALL METRICS:")
        print(f"  🎯 Accuracy:   {metrics.accuracy:.3f} ({metrics.accuracy*100:.1f}%)")
        print(f"  🎯 Precision:  {metrics.precision:.3f} ({metrics.precision*100:.1f}%)")
        print(f"  🎯 Recall:     {metrics.recall:.3f} ({metrics.recall*100:.1f}%)")
        print(f"  🎯 F1-Score:   {metrics.f1_score:.3f} ({metrics.f1_score*100:.1f}%)")
        
        # Confusion matrix
        print(f"\n📈 CONFUSION MATRIX:")
        print(f"  📊 Total Predictions: {metrics.total_predictions}")
        print(f"  ✅ True Positives:    {metrics.true_positives:3d} (Correct answers correctly identified)")
        print(f"  ❌ False Positives:   {metrics.false_positives:3d} (Incorrect answers wrongly identified as correct)")
        print(f"  ✅ True Negatives:    {metrics.true_negatives:3d} (Incorrect answers correctly identified)")
        print(f"  ❌ False Negatives:   {metrics.false_negatives:3d} (Correct answers wrongly identified as incorrect)")
        
        # Per-question breakdown
        print(f"\n📝 PER-QUESTION BREAKDOWN:")
        print("-" * 80)
        
        for i, result in enumerate(evaluation_results, 1):
            correct_classified = sum(result.correct_predictions)
            incorrect_classified = sum(result.incorrect_predictions)
            total_correct = correct_classified + incorrect_classified
            accuracy = total_correct / 8 * 100
            
            print(f"Q{i:2d}: {result.question_text[:50]:<50} | Accuracy: {accuracy:5.1f}% ({total_correct}/8)")
        
        # Summary statistics
        question_accuracies = []
        for result in evaluation_results:
            correct_classified = sum(result.correct_predictions)
            incorrect_classified = sum(result.incorrect_predictions)
            total_correct = correct_classified + incorrect_classified
            accuracy = total_correct / 8 * 100
            question_accuracies.append(accuracy)
        
        if question_accuracies:
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"  📊 Mean Question Accuracy:    {statistics.mean(question_accuracies):.1f}%")
            print(f"  📊 Median Question Accuracy:  {statistics.median(question_accuracies):.1f}%")
            print(f"  📊 Min Question Accuracy:     {min(question_accuracies):.1f}%")
            print(f"  📊 Max Question Accuracy:     {max(question_accuracies):.1f}%")
            if len(question_accuracies) > 1:
                print(f"  📊 Std Dev:                   {statistics.stdev(question_accuracies):.1f}%")
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)


def main():
    """Main function to run the evaluation benchmark"""
    try:
        print("🚀 Starting Answer Evaluator Benchmark")
        print("="*80)
        
        # Initialize benchmark
        benchmark = EvaluationBenchmark()
        
        # Run evaluation
        evaluation_results = benchmark.run_evaluation()
        
        if not evaluation_results:
            print("❌ No evaluation results generated. Check questions and answers setup.")
            return
        
        # Calculate metrics
        metrics = benchmark.calculate_metrics(evaluation_results)
        
        # Display results
        benchmark.display_results(evaluation_results, metrics)
        
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
