### Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp ../.env.example .env
# Edit .env and add your API keys (see .env.example for details)

# Run the server
python main.py
```

The backend will start on `http://localhost:8000`

**Required Environment Variables:**
- `PINECONE_API_KEY`: Your Pinecone API key for vector storage (required)
- `OPENAI_API_KEY`: Your OpenAI API key (optional - only if using `type: "openai"` in embeddings config)
- `GROQ_API_KEY`: Your Groq API key (optional - only if `llm_arbiter_enabled: true` in grading config)

**Note**: NLI models (DeBERTa) and local embedding models (GTE-Multilingual) are automatically downloaded from HuggingFace on first use.

**Unified Embeddings Interface:**
The system uses a unified embeddings interface that supports any embedding model with easy switching through configuration.

**Supported Models:**
1. **OpenAI** (text-embedding-ada-002): High-quality embeddings, requires API credits
2. **GTE-Multilingual** (Alibaba-NLP/gte-multilingual-base): Free local model, supports 100+ languages
3. **MiniLM** (sentence-transformers/all-MiniLM-L6-v2): Fast, lightweight, free
4. **Any HuggingFace Model**: Use any sentence-transformer compatible model

To switch models, edit `backend/settings.yaml`:

```yaml
embeddings:
  model: "Alibaba-NLP/gte-multilingual-base"  # Model name or path
  type: "sentence-transformer"  # "openai" or "sentence-transformer"
  dimensions: 768  # Must match model's output dimensions
```

**Quick Reference:**
- See `backend/MODEL_SWITCHING_GUIDE.md` for common model configurations
- See `backend/EMBEDDINGS_INTERFACE.md` for detailed architecture documentation

## 🧠 Grading Architecture

### 4-Tier Hybrid System

The system uses a sophisticated **4-tier hybrid architecture** that combines cosine similarity, NLI (Natural Language Inference), and LLM reasoning for accurate and cost-efficient grading:

```
User Answer → Validation → Embeddings → Cosine Similarity → Tier Decision
                                                                    ├─ TIER 1 (Fast): ≥92% → AUTO PASS ✅ | <60% → AUTO FAIL ❌
                                                                    ├─ TIER 2 (LLM): 78-92% → Groq Llama 8B verification
                                                                    ├─ TIER 3 (NLI): 70-78% → DeBERTa contradiction check
                                                                    └─ TIER 4 (Critical): High disagreement → LLM arbiter
                                                                             ↓
                                                                    Score + Feedback → Response
```

#### Tier Breakdown

| Tier | Trigger | Method | Cost | Frequency |
|------|---------|--------|------|-----------|
| **1 - Fast Track** | Cosine ≥92% or <60% | Cosine only | $0 | ~40-50% |
| **2 - LLM Zone** | Cosine 78-92% | Groq Llama 8B | $0.00012 | ~20-30% |
| **3 - NLI Zone** | Cosine 70-78% | DeBERTa NLI | $0 | ~20-30% |
| **4 - Critical** | Model disagreement | LLM Arbiter | $0.00012 | <5% |

**Average cost per answer**: ~$0.00003-$0.00005 (250x cheaper than pure GPT-4)

### Grading Modes

Configure grading mode in `backend/settings.yaml`:

#### 1. **Hybrid Mode** (Recommended - Default)
```yaml
grading:
  grading_method: "hybrid"
  tier1_auto_pass: 0.92
  tier2_llm_min: 0.78
  tier3_nli_min: 0.70
  llm_arbiter_enabled: true
```
- Uses all 4 tiers
- Best accuracy (92-95%)
- Lowest cost
- Fast for obvious cases

#### 2. **NLI-First Mode**
```yaml
grading:
  grading_method: "nli"
  nli_entailment_threshold: 0.65
  llm_arbiter_enabled: true
```
- NLI as primary grader
- No cosine tiers
- Better at paraphrasing
- Free (local models)

#### 3. **Pure Embedding Mode**
```yaml
grading:
  grading_method: "embedding"
  similarity_thresholds:
    high_similarity: 0.855
```
- Legacy mode
- Cosine similarity only
- Fastest but less accurate

### Key Features

✅ **Full-Context Evaluation**: Analyzes both individual sentences and complete answer
✅ **Negation Detection**: Catches "not", "never", explicit contradictions
✅ **Paraphrase Understanding**: NLI models understand synonyms naturally
✅ **Cost Optimization**: Fast-track paths avoid expensive operations
✅ **Confidence Scoring**: Tracks uncertainty for active learning


### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000`

## 🔧 Pinecone Setup

### 1. Create Pinecone Account
1. Go to [Pinecone](https://www.pinecone.io/) and create an account
2. Create a new project
3. Get your API key from the dashboard

### 2. Configure Index
The application will automatically create a Pinecone index with the following settings:
- **Index Name**: `answer-evaluator`
- **Dimensions**: `1536` (OpenAI text-embedding-ada-002)
- **Metric**: `cosine`
- **Cloud**: `AWS us-east-1`

### 3. Management Scripts
The backend includes utility scripts for managing the Pinecone index:

```bash
# Test Pinecone connection
python test_pinecone_integration.py

# Clear all vectors from the index
python clear_pinecone_index.py

# Delete and recreate the index with correct dimensions
python recreate_pinecone_index.py
```

## 🔧 How It Works

### 1. Vector Database Integration
The system now uses **Pinecone** as the vector database for storing and retrieving embeddings:

```python
# At startup, the backend:
# 1. Connects to Pinecone index
# 2. Loads questions from hardcoded JSON
# 3. Checks if embeddings exist in Pinecone
# 4. If not, computes embeddings using OpenAI API and stores in Pinecone
# 5. Uses Pinecone for fast similarity searches

def precompute_embeddings():
    for question in questions_bank:
        for key_point in question["key_points"]:
            embedding = get_embedding(key_point["text"])
            # Store in Pinecone with metadata
```

### 2. Answer Grading
```python
# When a user submits an answer:
# 1. Get embedding for user's answer
# 2. Compare with each key point embedding using cosine similarity
# 3. Mark key points as "hit" or "missing" based on threshold
# 4. Calculate score and generate feedback

def grade_answer(question_id, user_answer):
    user_embedding = get_embedding(user_answer)
    
    for key_point_embedding in key_point_embeddings[question_id]:
        similarity = compute_cosine_similarity(user_embedding, key_point_embedding)
        if similarity >= 0.75:  # Threshold
            # Mark as hit
        else:
            # Mark as missing
```

### 3. Cosine Similarity
```python
# Cosine similarity measures the angle between two vectors
# Formula: cos(θ) = (A · B) / (||A|| × ||B||)

def compute_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)
```

## 📊 API Endpoints

### GET /question
Returns a random question from the question bank.

**Response:**
```json
{
  "question_id": 1,
  "question_text": "What is inflation?"
}
```

### POST /answer
Evaluates a user's answer against the question's key points.

**Request:**
```json
{
  "question_id": 1,
  "user_answer": "Inflation is when prices go up"
}
```

**Response:**
```json
{
  "score": 66.7,
  "hit_key_points": ["General increase in prices"],
  "missing_key_points": ["Reduction of purchasing power"],
  "feedback": "Partial - missing 1 key point(s). Good start!"
}
```

## 🎛️ Configuration

### Similarity Threshold
Adjust the similarity threshold in `backend/main.py`:
```python
similarity_threshold = 0.75  # Adjust this value (0.0 to 1.0)
```

### Question Bank
Modify the hardcoded questions in `backend/main.py`:
```python
questions_bank = [
    {
        "question_id": 1,
        "question_text": "Your question here",
        "key_points": [
            {"text": "Key point 1", "weight": 1},
            {"text": "Key point 2", "weight": 1}
        ]
    }
]
```

## 🔮 Scaling for Production

### 1. Vector Database Integration
Replace in-memory storage with a vector database:

```python
# Instead of storing in key_point_embeddings dict:
# Store in Pinecone, Weaviate, or similar vector DB

import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-api-key", environment="your-env")
index = pinecone.Index("answer-evaluator")

# Store embeddings
index.upsert([(f"kp_{question_id}_{i}", embedding) for i, embedding in enumerate(embeddings)])

# Query embeddings
results = index.query(queries=[user_embedding], top_k=5)
```

### 2. Database Integration
Replace hardcoded questions with a database:

```python
# Use SQLAlchemy or similar ORM
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True)
    question_text = Column(String)
    key_points = Column(JSON)
```

### 3. Caching
Add Redis for caching embeddings:

```python
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Cache embeddings
def get_cached_embedding(text):
    cached = redis_client.get(f"embedding:{text}")
    if cached:
        return json.loads(cached)
    
    embedding = get_embedding(text)
    redis_client.setex(f"embedding:{text}", 3600, json.dumps(embedding))
    return embedding
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

## 📈 Performance Considerations

1. **Embedding Caching**: Precompute and cache embeddings to avoid API calls
2. **Batch Processing**: Process multiple questions at once
3. **Vector Database**: Use specialized vector DBs for large-scale similarity search
4. **Async Processing**: Use async/await for better concurrency

## 🔒 Security

1. **API Key Protection**: Store OpenAI API key in environment variables
2. **Input Validation**: Validate and sanitize user inputs
3. **Rate Limiting**: Implement rate limiting for API endpoints
4. **CORS Configuration**: Configure CORS properly for production

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**: Check if port 8000 is available
2. **API key errors**: Verify OpenAI API key is set correctly
3. **CORS errors**: Ensure frontend is running on localhost:3000
4. **Embedding errors**: Check OpenAI API quota and network connection

### Debug Mode
Enable debug logging in `backend/main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

MIT License - feel free to use and modify for your projects!