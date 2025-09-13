"""
Configuration for the weighted synthetic data generator (Geography - Secondary School)
"""

# Data Generation Settings
M_QUESTIONS = 50  # Number of different questions to generate

# Output Settings
OUTPUT_FILE = "weighted_geography_questions.json"

# API Settings
MODEL_NAME = "gpt-4o-mini"  # OpenAI model to use
TEMPERATURE_QUESTION = 0.8  # Temperature for question generation (0.0-2.0)
MAX_TOKENS_QUESTION = 500   # Max tokens for question generation

# Cost tracking (same as biology)
GPT4O_MINI_INPUT_COST_PER_1K_TOKENS = 0.00015
GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS = 0.0006

# Subject Settings
SUBJECT = "geography"
GRADE_LEVEL = "secondary school"
STUDENT_AGE_RANGE = "ages 14-18"

# Generation Settings
API_DELAY = 0.5
SHOW_PROGRESS = True

# Question Types (Geography topics)
QUESTION_TYPES = [
    "physical geography",
    "human geography",
    "climate and weather",
    "biomes and ecosystems",
    "rivers and hydrology",
    "coastal processes",
    "plate tectonics",
    "earthquakes and volcanoes",
    "population and migration",
    "urbanization",
    "development and globalization",
    "resources and energy",
    "agriculture",
    "transport and trade",
    "maps and GIS"
]
