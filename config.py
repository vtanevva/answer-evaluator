"""
Configuration file for the synthetic data generator
Modify these settings to customize your data generation
"""

# Data Generation Settings
M_QUESTIONS = 50  # Number of different questions to generate
N_ANSWERS_PER_QUESTION = 100  # Number of student answers per question (100 samples total)

# Output Settings
OUTPUT_FORMAT = "jsonl"  # "jsonl" or "csv"
OUTPUT_FILE = "synthetic_biology_data.jsonl"

# API Settings
MODEL_NAME = "gpt-4o-mini"  # OpenAI model to use
TEMPERATURE_QUESTION = 0.8  # Temperature for question generation (0.0-2.0)
TEMPERATURE_ANSWERS = 0.9   # Temperature for answer generation (0.0-2.0)
MAX_TOKENS_QUESTION = 500   # Max tokens for question generation
MAX_TOKENS_ANSWERS = 2000   # Max tokens for answer generation

# Cost tracking
GPT4O_MINI_INPUT_COST_PER_1K_TOKENS = 0.00015  # $0.00015 per 1K input tokens
GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS = 0.0006  # $0.0006 per 1K output tokens

# Subject Settings
SUBJECT = "biology"  # Subject for questions
GRADE_LEVEL = "middle school"  # Target grade level
STUDENT_AGE_RANGE = "ages 12-14"  # Age range for students

# Generation Settings
API_DELAY = 0.5  # Delay between API calls (seconds)
SHOW_PROGRESS = True  # Whether to show progress updates

# Question Types (you can add more)
QUESTION_TYPES = [
    "animals",
    "plants", 
    "ecosystems",
    "human body",
    "genetics",
    "evolution",
    "cell biology",
    "environmental science",
    "butterflies"
]
