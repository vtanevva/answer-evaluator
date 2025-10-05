# CI/CD Pipeline Configuration

## Required GitHub Secrets

To enable the full CI/CD pipeline, add these secrets to your GitHub repository:

### Required Secrets:
- `OPENAI_API_KEY`: Your OpenAI API key for running evaluations

### How to Add Secrets:
1. Go to your repository on GitHub
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add the secret name and value

## Workflow Overview

### 🧪 Test Job
- **Triggers**: Pull requests and pushes to main
- **Matrix**: Tests on Python 3.10, and 3.11
- **Steps**:
  - Code checkout
  - Python setup with caching
  - Dependency installation
  - NLTK data download
  - Linting with flake8
  - Unit tests with pytest
  - Coverage reporting

### 🎯 Evaluation Job
- **Triggers**: Pull requests only
- **Dependencies**: Requires test job to pass
- **Steps**:
  - Runs the evaluation benchmark
  - Parses performance metrics
  - Posts results as PR comment
  - Uploads detailed results as artifact

### 🔒 Security Scan Job
- **Triggers**: Pull requests and pushes to main
- **Dependencies**: Requires test job to pass
- **Steps**:
  - Runs Bandit security analysis
  - Uploads scan results

### ✅ Build Status Job
- **Triggers**: After all other jobs complete
- **Purpose**: Final status check for branch protection
- **Behavior**: Fails if critical jobs fail

## Branch Protection Rules

The pipeline includes automatic branch protection setup:

### Protected Branch: `main`
- **Required Status Checks**:
  - Run Tests (Python 3.10, 3.11)
  - Build Status
- **Required Reviews**: 1 approving review
- **Settings**:
  - Dismiss stale reviews when new commits are pushed
  - Block force pushes
  - Block branch deletion

### Manual Setup
Run the branch protection workflow manually:
1. Go to Actions tab
2. Select "Branch Protection Setup"
3. Click "Run workflow"

## PR Comments

The pipeline automatically comments on PRs with:
- ✅ Evaluation benchmark results
- 📊 Performance metrics (Accuracy, Precision, Recall, F1-Score)
- 🔗 Links to workflow runs and artifacts
- ⚠️ Warnings if evaluation fails

## Artifacts

The following artifacts are preserved for 30 days:
- `evaluation-results-{run_number}`: Complete evaluation output
- `security-scan-results`: Bandit security scan results

## Local Testing

Before pushing, you can run the same checks locally:

```bash
# Install dependencies
cd backend
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run linting
python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run tests
python -m pytest tests/unit/ -v

# Run evaluation (requires OPENAI_API_KEY)
export OPENAI_API_KEY="your-key"
python evaluation/evaluation_script.py
```

## Troubleshooting

### Common Issues:

1. **Missing OPENAI_API_KEY**:
   - Evaluation will be skipped with a warning
   - Add the secret to repository settings

2. **Test Failures**:
   - PR cannot be merged until tests pass
   - Check the Actions tab for detailed logs

3. **Evaluation Failures**:
   - Evaluation failure doesn't block PR merge
   - Check artifact for detailed error logs

4. **Branch Protection Not Working**:
   - Run the branch protection setup workflow manually
   - Ensure you have admin access to the repository

### Status Check Names:
- `Run Tests (3.10)` - Python 3.10 tests  
- `Run Tests (3.11)` - Python 3.11 tests
- `Run Evaluation Benchmark` - Evaluation results
- `Security Scan` - Security analysis
- `Build Status` - Overall pipeline status

## Performance Benchmarks

The evaluation benchmark tests the answer evaluation system with:
- Realistic student answers (correct and incorrect)
- Multiple question types (biology, geography, economics)
- Standard ML metrics (precision, recall, F1, accuracy)
- Configurable similarity thresholds

Target performance thresholds:
- **Accuracy**: > 80%
- **Precision**: > 85% 
- **Recall**: > 75%
- **F1-Score**: > 80%
