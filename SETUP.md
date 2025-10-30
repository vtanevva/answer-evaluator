# Answer Evaluator Setup Guide

This guide will help you set up the Answer Evaluator application on any PC.

## Prerequisites

- Python 3.11+ installed
- Node.js 16+ installed
- Git installed

## Quick Setup

### 1. Clone and Navigate
```bash
git clone <repository-url>
cd answer-evaluator
```

### 2. Backend Setup

#### Create Virtual Environment
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Download NLTK Data
```bash
python -c "import nltk; nltk.download('stopwords')"
```

#### Configure Environment Variables
Create a `.env` file in the `backend` directory:
```env
# Required: Get from OpenAI Platform (https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required: Get from Pinecone (https://www.pinecone.io/)
PINECONE_API_KEY=your-pinecone-api-key-here

# Optional
PORT=8080
HOST=0.0.0.0
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### Start Backend (Terminal 1)
```bash
cd backend
# Activate virtual environment if not active
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

uvicorn main:app --reload
```
Backend will run on: http://127.0.0.1:8000

### Start Frontend (Terminal 2)
```bash
cd frontend
npm start
```
Frontend will run on: http://localhost:3000

## API Keys Setup

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add it to your `.env` file

### Pinecone API Key
1. Go to https://www.pinecone.io/
2. Create a free account
3. Create a new project
4. Get your API key from the dashboard
5. Add it to your `.env` file

## Current Configuration

### Embedding Provider
- **Current**: GTE-Multilingual (local, free)
- **Alternative**: OpenAI (requires credits)
- **Switch**: Change `provider` in `backend/settings.yaml`

### Vector Database
- **Pinecone**: Automatic index creation per provider/dimension
- **Index Naming**: `answer-evaluator-{provider}-{dimensions}`
- **Current Index**: `answer-evaluator-gte-multilingual-768`

## Features Working

✅ **Multi-provider embeddings** (OpenAI + GTE-Multilingual)  
✅ **Pinecone vector storage** with automatic index management  
✅ **On-demand embedding computation** when cache is empty  
✅ **Provider-specific index naming** (non-destructive switching)  
✅ **Real-time answer evaluation** with semantic similarity  
✅ **Question bank management** (50 questions loaded)  
✅ **CORS configured** for frontend-backend communication  

## Troubleshooting

### Backend Issues

**Module not found errors:**
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

**Pinecone API key issues:**
- Verify `.env` file exists in `backend` directory
- Check API key is valid and not expired
- Ensure no spaces around the `=` in `.env`

**Port already in use:**
```bash
# Change port in settings.yaml or run on different port
uvicorn main:app --reload --port 8001
```

### Frontend Issues

**npm install fails:**
```bash
# Clear cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
- Ensure backend is running on http://127.0.0.1:8000
- Check CORS settings in `backend/settings.yaml`

### Performance Notes

- **First evaluation per question**: Slower (computes embeddings on-demand)
- **Subsequent evaluations**: Fast (uses computed embeddings)
- **GPU acceleration**: Available if CUDA-compatible GPU detected
- **Precompute all embeddings**: Set `precompute_embeddings=true` in settings.yaml

## File Structure
```
answer-evaluator/
├── backend/
│   ├── .env                 # Environment variables (create this)
│   ├── requirements.txt     # Python dependencies
│   ├── settings.yaml        # App configuration
│   ├── main.py             # FastAPI app entry point
│   ├── core/config.py      # Configuration management
│   ├── services/           # Business logic
│   ├── routes/             # API endpoints
│   └── models/             # Data models
├── frontend/
│   ├── package.json        # Node.js dependencies
│   ├── src/                # React components
│   └── public/             # Static files
├── myenv/                  # Python virtual environment
└── SETUP.md               # This file
```

## Security Notes

- ⚠️ Never commit `.env` files to version control
- ⚠️ Keep API keys secure and rotate them regularly
- ⚠️ Use environment variables for production deployment