import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// API configuration - points to FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState('answer'); // 'answer', 'add', or 'view'
  
  // State management for answering questions
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State management for adding questions
  const [newQuestionText, setNewQuestionText] = useState('');
  const [keyPoints, setKeyPoints] = useState([{ text: '', weight: 1 }]);
  const [isAddingQuestion, setIsAddingQuestion] = useState(false);
  const [addQuestionSuccess, setAddQuestionSuccess] = useState(null);
  const [addQuestionError, setAddQuestionError] = useState(null);
  
  // State management for viewing questions
  const [allQuestions, setAllQuestions] = useState([]);
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);
  const [deletingQuestionId, setDeletingQuestionId] = useState(null);
  const [viewQuestionsError, setViewQuestionsError] = useState(null);
  const [selectedCategoryFilter, setSelectedCategoryFilter] = useState('all');

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

  // Add Question Functions
  const addKeyPoint = () => {
    setKeyPoints([...keyPoints, { text: '', weight: 1 }]);
  };

  const removeKeyPoint = (index) => {
    if (keyPoints.length > 1) {
      setKeyPoints(keyPoints.filter((_, i) => i !== index));
    }
  };

  const updateKeyPoint = (index, field, value) => {
    const updated = [...keyPoints];
    updated[index] = { ...updated[index], [field]: value };
    setKeyPoints(updated);
  };

  const submitNewQuestion = async () => {
    // Validate form
    if (!newQuestionText.trim()) {
      setAddQuestionError('Question text cannot be empty.');
      return;
    }

    const validKeyPoints = keyPoints.filter(kp => kp.text.trim() !== '');
    if (validKeyPoints.length === 0) {
      setAddQuestionError('At least one key point is required.');
      return;
    }

    setIsAddingQuestion(true);
    setAddQuestionError(null);
    setAddQuestionSuccess(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/question`, {
        question_text: newQuestionText.trim(),
        key_points: validKeyPoints.map(kp => ({
          text: kp.text.trim(),
          weight: kp.weight || 1
        }))
      });

      setAddQuestionSuccess(`Question added successfully! Question ID: ${response.data.question_id}`);
      
      // Reset form
      setNewQuestionText('');
      setKeyPoints([{ text: '', weight: 1 }]);
      
      // Clear success message after 5 seconds
      setTimeout(() => {
        setAddQuestionSuccess(null);
      }, 5000);
      
      // Refresh questions list if on view tab
      if (activeTab === 'view') {
        fetchAllQuestions();
      }
    } catch (err) {
      console.error('❌ Error adding question:', err);
      setAddQuestionError(
        err.response?.data?.detail || 
        'Failed to add question. Please try again.'
      );
    } finally {
      setIsAddingQuestion(false);
    }
  };

  // View Questions Functions
  const fetchAllQuestions = async () => {
    setIsLoadingQuestions(true);
    setViewQuestionsError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/questions`);
      setAllQuestions(response.data.questions);
    } catch (err) {
      console.error('❌ Error fetching questions:', err);
      setViewQuestionsError('Failed to load questions. Please try again.');
    } finally {
      setIsLoadingQuestions(false);
    }
  };

  const deleteQuestion = async (questionId) => {
    if (!window.confirm(`Are you sure you want to delete question ${questionId}? This action cannot be undone.`)) {
      return;
    }

    setDeletingQuestionId(questionId);
    setViewQuestionsError(null);

    try {
      await axios.delete(`${API_BASE_URL}/question/${questionId}`);
      
      // Remove from local state
      setAllQuestions(allQuestions.filter(q => q.question_id !== questionId));
      
      // Show success message temporarily
      const successMsg = `Question ${questionId} deleted successfully`;
      alert(successMsg);
    } catch (err) {
      console.error('❌ Error deleting question:', err);
      setViewQuestionsError(
        err.response?.data?.detail || 
        `Failed to delete question ${questionId}. Please try again.`
      );
    } finally {
      setDeletingQuestionId(null);
    }
  };

  // Fetch questions when view tab is opened
  useEffect(() => {
    if (activeTab === 'view') {
      fetchAllQuestions();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

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

        {/* Tab Navigation */}
        <div className="mb-6 flex gap-4 justify-center">
          <button
            onClick={() => setActiveTab('answer')}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              activeTab === 'answer'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Answer Questions
          </button>
          <button
            onClick={() => setActiveTab('add')}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              activeTab === 'add'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Add Question
          </button>
          <button
            onClick={() => setActiveTab('view')}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              activeTab === 'view'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            View Questions
          </button>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          {activeTab === 'answer' ? (
            /* Answer Questions Tab */
            <>
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
          </>
          ) : activeTab === 'add' ? (
            /* Add Question Tab */
            <div>
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Add New Question</h2>
              
              {/* Question Text Input */}
              <div className="mb-6">
                <label htmlFor="questionText" className="block text-sm font-medium text-gray-700 mb-2">
                  Question Text
                </label>
                <textarea
                  id="questionText"
                  value={newQuestionText}
                  onChange={(e) => setNewQuestionText(e.target.value)}
                  placeholder="Enter the question here..."
                  className="w-full h-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  disabled={isAddingQuestion}
                />
              </div>

              {/* Key Points Section */}
              <div className="mb-6">
                <div className="flex items-center justify-between mb-3">
                  <label className="block text-sm font-medium text-gray-700">
                    Key Points (Answer Chunks)
                  </label>
                  <button
                    onClick={addKeyPoint}
                    disabled={isAddingQuestion}
                    className="px-3 py-1 text-sm bg-green-600 text-white rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  >
                    + Add Key Point
                  </button>
                </div>
                
                <div className="space-y-3">
                  {keyPoints.map((keyPoint, index) => (
                    <div key={index} className="flex gap-3 items-start">
                      <div className="flex-1">
                        <textarea
                          value={keyPoint.text}
                          onChange={(e) => updateKeyPoint(index, 'text', e.target.value)}
                          placeholder={`Key point ${index + 1}...`}
                          className="w-full h-20 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                          disabled={isAddingQuestion}
                        />
                      </div>
                      <div className="w-24">
                        <label className="block text-xs text-gray-600 mb-1">Weight</label>
                        <input
                          type="number"
                          min="1"
                          value={keyPoint.weight}
                          onChange={(e) => updateKeyPoint(index, 'weight', parseInt(e.target.value) || 1)}
                          className="w-full px-2 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          disabled={isAddingQuestion}
                        />
                      </div>
                      <button
                        onClick={() => removeKeyPoint(index)}
                        disabled={isAddingQuestion || keyPoints.length === 1}
                        className="mt-6 px-3 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Success Message */}
              {addQuestionSuccess && (
                <div className="mb-6 bg-green-50 border-l-4 border-green-400 p-4 rounded">
                  <p className="text-green-800">{addQuestionSuccess}</p>
                </div>
              )}

              {/* Error Message */}
              {addQuestionError && (
                <div className="mb-6 bg-red-50 border-l-4 border-red-400 p-4 rounded">
                  <p className="text-red-800">{addQuestionError}</p>
                </div>
              )}

              {/* Submit Button */}
              <button
                onClick={submitNewQuestion}
                disabled={isAddingQuestion || !newQuestionText.trim()}
                className="w-full bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
              >
                {isAddingQuestion ? 'Adding Question...' : 'Add Question to Database'}
              </button>

              {/* Info Note */}
              <div className="mt-6 p-4 bg-blue-50 rounded-md">
                <p className="text-sm text-blue-800">
                  <strong>Note:</strong> The question will be added to the database and embeddings will be computed and stored in the vector database. This may take a few moments.
                </p>
              </div>
            </div>
          ) : (
            /* View Questions Tab */
            <div>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">All Questions</h2>
                <button
                  onClick={fetchAllQuestions}
                  disabled={isLoadingQuestions}
                  className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoadingQuestions ? 'Loading...' : '🔄 Refresh'}
                </button>
              </div>

              {/* Category Filter Buttons */}
              {!isLoadingQuestions && allQuestions.length > 0 && (
                <div className="mb-6 flex flex-wrap gap-2">
                  <button
                    onClick={() => setSelectedCategoryFilter('all')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      selectedCategoryFilter === 'all'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    All
                  </button>
                  <button
                    onClick={() => setSelectedCategoryFilter('biology')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      selectedCategoryFilter === 'biology'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Biology
                  </button>
                  <button
                    onClick={() => setSelectedCategoryFilter('geography')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      selectedCategoryFilter === 'geography'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Geography
                  </button>
                  <button
                    onClick={() => setSelectedCategoryFilter('economics')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      selectedCategoryFilter === 'economics'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    Economics
                  </button>
                  <button
                    onClick={() => setSelectedCategoryFilter('user')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                      selectedCategoryFilter === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    User Added
                  </button>
                </div>
              )}

              {/* Error Message */}
              {viewQuestionsError && (
                <div className="mb-6 bg-red-50 border-l-4 border-red-400 p-4 rounded">
                  <p className="text-red-800">{viewQuestionsError}</p>
                </div>
              )}

              {/* Loading State */}
              {isLoadingQuestions ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                  <p className="text-gray-600 mt-2">Loading questions...</p>
                </div>
              ) : allQuestions.length === 0 ? (
                <div className="text-center py-12 bg-gray-50 rounded-lg">
                  <p className="text-gray-600 text-lg">No questions found.</p>
                  <p className="text-gray-500 mt-2">Add a question using the "Add Question" tab.</p>
                </div>
              ) : (
                <div>
                  {(() => {
                    // Group questions by category
                    const groupedQuestions = allQuestions.reduce((acc, question) => {
                      const category = question.category || 'unknown';
                      if (!acc[category]) {
                        acc[category] = [];
                      }
                      acc[category].push(question);
                      return acc;
                    }, {});

                    // Define category order and display names
                    const categoryOrder = ['biology', 'geography', 'economics', 'user', 'fallback'];
                    const categoryLabels = {
                      'biology': 'Biology',
                      'geography': 'Geography',
                      'economics': 'Economics',
                      'user': 'User Added',
                      'fallback': 'Fallback',
                      'unknown': 'Unknown'
                    };

                    // Sort categories by predefined order
                    const sortedCategories = Object.keys(groupedQuestions).sort((a, b) => {
                      const indexA = categoryOrder.indexOf(a);
                      const indexB = categoryOrder.indexOf(b);
                      if (indexA === -1 && indexB === -1) return a.localeCompare(b);
                      if (indexA === -1) return 1;
                      if (indexB === -1) return -1;
                      return indexA - indexB;
                    });

                    // Filter categories based on selected filter
                    const filteredCategories = selectedCategoryFilter === 'all' 
                      ? sortedCategories 
                      : sortedCategories.filter(cat => cat === selectedCategoryFilter);

                    return (
                      <div className="space-y-8">
                        {filteredCategories.length === 0 ? (
                          <div className="text-center py-12 bg-gray-50 rounded-lg">
                            <p className="text-gray-600 text-lg">No questions found in this category.</p>
                          </div>
                        ) : (
                          filteredCategories.map((category) => {
                          const questions = groupedQuestions[category];
                          const categoryLabel = categoryLabels[category] || category.charAt(0).toUpperCase() + category.slice(1);
                          
                          return (
                            <div key={category} className="border border-gray-200 rounded-lg p-6">
                              <div className="mb-4 pb-3 border-b border-gray-200">
                                <h3 className="text-xl font-bold text-gray-900 capitalize">
                                  {categoryLabel}
                                </h3>
                                <p className="text-sm text-gray-500 mt-1">
                                  {questions.length} question{questions.length !== 1 ? 's' : ''}
                                </p>
                              </div>
                              
                              <div className="space-y-4">
                                {questions.map((question) => (
                                  <div
                                    key={question.question_id}
                                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow bg-gray-50"
                                  >
                                    <div className="flex items-start justify-between">
                                      <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-2">
                                          <span className="text-sm font-semibold text-gray-500">
                                            ID: {question.question_id}
                                          </span>
                                        </div>
                                        <h4 className="text-lg font-semibold text-gray-900 mb-3">
                                          {question.question_text}
                                        </h4>
                                        
                                        {/* Key Points */}
                                        <div className="mt-3">
                                          <p className="text-sm font-medium text-gray-700 mb-2">Key Points:</p>
                                          <ul className="list-disc list-inside space-y-1">
                                            {question.key_points && question.key_points.map((kp, idx) => (
                                              <li key={idx} className="text-sm text-gray-600">
                                                <span className="font-medium">({kp.weight || 1})</span> {kp.text}
                                              </li>
                                            ))}
                                          </ul>
                                        </div>
                                      </div>
                                      
                                      {/* Delete Button */}
                                      <button
                                        onClick={() => deleteQuestion(question.question_id)}
                                        disabled={deletingQuestionId === question.question_id}
                                        className="ml-4 px-4 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                                      >
                                        {deletingQuestionId === question.question_id ? 'Deleting...' : '🗑️ Delete'}
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}

              {/* Total Count */}
              {!isLoadingQuestions && allQuestions.length > 0 && (
                <div className="mt-6 text-center text-sm text-gray-500">
                  Total: {allQuestions.length} question{allQuestions.length !== 1 ? 's' : ''}
                </div>
              )}
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
