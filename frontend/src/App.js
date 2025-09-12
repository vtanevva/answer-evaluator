import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// API configuration - points to FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

function App() {
  // State management
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch a random question from the backend
  const fetchQuestion = async () => {
    setIsLoading(true);
    setError(null);
    setEvaluation(null);
    setUserAnswer('');
    
    try {
      const response = await axios.get(`${API_BASE_URL}/question`);
      setCurrentQuestion(response.data);
      console.log('✅ Question loaded:', response.data);
    } catch (err) {
      console.error('❌ Error fetching question:', err);
      setError('Failed to load question. Make sure the backend is running on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  // Submit answer for evaluation
  const submitAnswer = async () => {
    if (!userAnswer.trim()) {
      setError('Please enter an answer before submitting.');
      return;
    }

    if (!currentQuestion) {
      setError('No question loaded.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/answer`, {
        question_id: currentQuestion.question_id,
        user_answer: userAnswer
      });
      
      setEvaluation(response.data);
      console.log('✅ Answer evaluated:', response.data);
    } catch (err) {
      console.error('❌ Error evaluating answer:', err);
      setError('Failed to evaluate answer. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Load first question on component mount
  useEffect(() => {
    fetchQuestion();
  }, []);

  // Handle Enter key press in textarea
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      submitAnswer();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Answer Evaluator
          </h1>
          <p className="text-gray-600">
            Test your knowledge with AI-powered answer evaluation
          </p>
        </div>

        {/* Main Chat Interface */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          {/* Question Section */}
          {currentQuestion && (
            <div className="mb-6">
              <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                <h3 className="text-lg font-semibold text-blue-900 mb-2">
                  Question
                </h3>
                <p className="text-blue-800">
                  {currentQuestion.question_text}
                </p>
              </div>
            </div>
          )}

          {/* User Answer Input */}
          <div className="mb-6">
            <label htmlFor="answer" className="block text-sm font-medium text-gray-700 mb-2">
              Your Answer
            </label>
            <textarea
              id="answer"
              value={userAnswer}
              onChange={(e) => setUserAnswer(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your answer here... (Ctrl+Enter to submit)"
              className="w-full h-32 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              disabled={isLoading}
            />
            <p className="text-xs text-gray-500 mt-1">
              Press Ctrl+Enter to submit your answer
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 mb-6">
            <button
              onClick={submitAnswer}
              disabled={isLoading || !userAnswer.trim()}
              className="flex-1 bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Evaluating...' : 'Submit Answer'}
            </button>
            
            <button
              onClick={fetchQuestion}
              disabled={isLoading}
              className="flex-1 bg-gray-600 text-white px-6 py-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Loading...' : 'Next Question'}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 bg-red-50 border-l-4 border-red-400 p-4 rounded">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {/* Evaluation Results */}
          {evaluation && (
            <div className="space-y-4">
              {/* Score Display */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">Score</h3>
                  <span className={`text-2xl font-bold ${
                    evaluation.score >= 80 ? 'text-green-600' :
                    evaluation.score >= 50 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {evaluation.score}%
                  </span>
                </div>
                
                {/* Progress Bar */}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      evaluation.score >= 80 ? 'bg-green-500' :
                      evaluation.score >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${evaluation.score}%` }}
                  ></div>
                </div>
              </div>

              {/* Feedback */}
              <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
                <h3 className="text-lg font-semibold text-blue-900 mb-2">Feedback</h3>
                <p className="text-blue-800">{evaluation.feedback}</p>
              </div>

              {/* Hit Key Points */}
              {evaluation.hit_key_points.length > 0 && (
                <div className="bg-green-50 border-l-4 border-green-400 p-4 rounded">
                  <h3 className="text-lg font-semibold text-green-900 mb-2">
                    ✅ Key Points You Covered
                  </h3>
                  <ul className="list-disc list-inside space-y-1">
                    {evaluation.hit_key_points.map((point, index) => (
                      <li key={index} className="text-green-800">{point}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Missing Key Points */}
              {evaluation.missing_key_points.length > 0 && (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded">
                  <h3 className="text-lg font-semibold text-yellow-900 mb-2">
                    ⚠️ Key Points You Missed
                  </h3>
                  <ul className="list-disc list-inside space-y-1">
                    {evaluation.missing_key_points.map((point, index) => (
                      <li key={index} className="text-yellow-800">{point}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="text-gray-600 mt-2">Processing...</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Powered by OpenAI embeddings and FastAPI</p>
          <p>Questions are evaluated using cosine similarity with precomputed key point embeddings</p>
        </div>
      </div>
    </div>
  );
}

export default App;
