"""
Configuration management using simple classes and YAML
"""

import os
import yaml
from typing import List, Dict, Any
from dataclasses import dataclass, field
from nltk.corpus import stopwords

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Answer Evaluator"
    description: str = "Evaluate student answers using embeddings"


@dataclass
class CorsConfig:
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = True
    allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class OpenAIConfig:
    model_name: str = "text-embedding-ada-002"
    embedding_dimensions: int = 1536


@dataclass
class EmbeddingConfig:
    provider: str = "openai"  # Options: "openai", "gte-multilingual"
    openai_model: str = "text-embedding-ada-002"
    openai_dimensions: int = 1536
    gte_model: str = "Alibaba-NLP/gte-multilingual-base"
    gte_dimensions: int = 768
    
    @property
    def current_model(self) -> str:
        """Get the current model name based on provider"""
        if self.provider == "openai":
            return self.openai_model
        elif self.provider == "gte-multilingual":
            return self.gte_model
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")
    
    @property
    def current_dimensions(self) -> int:
        """Get the current embedding dimensions based on provider"""
        if self.provider == "openai":
            return self.openai_dimensions
        elif self.provider == "gte-multilingual":
            return self.gte_dimensions
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    environment: str = "gcp-starter"
    index_name: str = "answer-evaluator"
    dimension: int = 1536
    metric: str = "cosine"


@dataclass
class SimilarityThresholds:
    high_similarity: float = 0.88
    mid_similarity: float = 0.83
    min_lexical_overlap: float = 0.5


@dataclass
class FeedbackMessages:
    perfect_score: str = "Correct! You covered all the key points."
    partial_score: str = "Partial - missing {missing_count} key point(s). Good start!"
    low_score: str = "Incorrect - try again. Review the material and provide a more complete answer."
    empty_answer: str = "Please try again. Even if you're unsure, try to explain what you think might be the answer."
    short_answer: str = "Your answer is too short. Please provide a more detailed explanation with at least a few words."


@dataclass
class AnswerValidation:
    min_answer_length: int = 10
    min_word_count: int = 2
    invalid_answers: List[str] = field(default_factory=lambda: ["i don't know", "i don't know.", "dont know"])


@dataclass
class GradingConfig:
    precompute_embeddings: bool = True
    similarity_thresholds: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    feedback_messages: FeedbackMessages = field(default_factory=FeedbackMessages)
    answer_validation: AnswerValidation = field(default_factory=AnswerValidation)


@dataclass
class QuestionsConfig:
    default_file_path: str = "questions.json"
    fallback_questions: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TextProcessingConfig:
    additional_stopwords: str = "the a an and or of in on at to for from as by is are was were be being been it its this that those these with without into over under which who whom whose what when where how why can could should would may might must do does did doing done have has had having not no nor if then else than also more most much many few several such same other another vs versus per each both either neither all any some most mostly mainly typically usually about around roughly approximately like include including e.g. i.e."
    stopwords: List[str] = field(default_factory=list)
    stemming_suffixes: List[str] = field(default_factory=lambda: ["ing", "ed", "es", "s"])
    min_token_length: int = 4

    def __post_init__(self):
        # Get NLTK stopwords and combine with additional stopwords
        nltk_stopwords = stopwords.words('english')
        additional_words = self.additional_stopwords.split()
        self.stopwords = nltk_stopwords + additional_words

@dataclass
class Settings:
    """Main application settings"""
    server: ServerConfig = field(default_factory=ServerConfig)
    cors: CorsConfig = field(default_factory=CorsConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    grading: GradingConfig = field(default_factory=GradingConfig)
    questions: QuestionsConfig = field(default_factory=QuestionsConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)


def _create_server_config(data: Dict[str, Any]) -> ServerConfig:
    """Create ServerConfig with validation and defaults"""
    try:
        return ServerConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid server configuration: {e}")
        print("📋 Using default server settings")
        return ServerConfig()


def _create_cors_config(data: Dict[str, Any]) -> CorsConfig:
    """Create CorsConfig with validation and defaults"""
    try:
        return CorsConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid CORS configuration: {e}")
        print("📋 Using default CORS settings")
        return CorsConfig()


def _create_openai_config(data: Dict[str, Any]) -> OpenAIConfig:
    """Create OpenAIConfig with validation and defaults"""
    try:
        return OpenAIConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid OpenAI configuration: {e}")
        print("📋 Using default OpenAI settings")
        return OpenAIConfig()


def _create_grading_config(data: Dict[str, Any]) -> GradingConfig:
    """Create GradingConfig with validation and graceful fallbacks"""
    try:
        # Extract simple fields with defaults
        precompute_embeddings = data.get('precompute_embeddings', True)

        
        # Create nested configs with error handling
        similarity_thresholds = SimilarityThresholds()
        if 'similarity_thresholds' in data:
            try:
                similarity_thresholds = SimilarityThresholds(**data['similarity_thresholds'])
            except Exception as e:
                print(f"⚠️ Invalid similarity thresholds: {e}")
                print("📋 Using default similarity thresholds")
        
        feedback_messages = FeedbackMessages()
        if 'feedback_messages' in data:
            try:
                feedback_messages = FeedbackMessages(**data['feedback_messages'])
            except Exception as e:
                print(f"⚠️ Invalid feedback messages: {e}")
                print("📋 Using default feedback messages")
        
        answer_validation = AnswerValidation()
        if 'answer_validation' in data:
            try:
                answer_validation = AnswerValidation(**data['answer_validation'])
            except Exception as e:
                print(f"⚠️ Invalid answer validation settings: {e}")
                print("📋 Using default answer validation")
        
        return GradingConfig(
            precompute_embeddings=precompute_embeddings,
            similarity_thresholds=similarity_thresholds,
            feedback_messages=feedback_messages,
            answer_validation=answer_validation
        )
    except Exception as e:
        print(f"⚠️ Invalid grading configuration: {e}")
        print("📋 Using default grading settings")
        return GradingConfig()


def _create_questions_config(data: Dict[str, Any]) -> QuestionsConfig:
    """Create QuestionsConfig with validation and defaults"""
    try:
        return QuestionsConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid questions configuration: {e}")
        print("📋 Using default questions settings")
        return QuestionsConfig()


def _create_text_processing_config(data: Dict[str, Any]) -> TextProcessingConfig:
    """Create TextProcessingConfig with validation and defaults"""
    try:
        return TextProcessingConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid text processing configuration: {e}")
        print("📋 Using default text processing settings")
        return TextProcessingConfig()


def _create_embedding_config(data: Dict[str, Any]) -> EmbeddingConfig:
    """Create EmbeddingConfig with validation and defaults"""
    try:
        # Only accept known keys; unknown keys will be ignored by dataclass
        return EmbeddingConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid embedding configuration: {e}")
        print("📋 Using default embedding settings")
        return EmbeddingConfig()


def _create_pinecone_config(data: Dict[str, Any]) -> PineconeConfig:
    """Create PineconeConfig with validation and defaults"""
    try:
        return PineconeConfig(**data)
    except Exception as e:
        print(f"⚠️ Invalid Pinecone configuration: {e}")
        print("📋 Using default Pinecone settings")
        return PineconeConfig()


def create_config_from_dict(data: Dict[str, Any]) -> Settings:
    """
    Create Settings object from dictionary data 
    
    This function validates each configuration section individually and falls back
    to defaults when invalid data is encountered, ensuring the application can
    always start with a valid configuration.
    
    Args:
        data: Dictionary with configuration data
        
    Returns:
        Settings object with validated configuration
    """
    if not isinstance(data, dict):
        print("⚠️ Configuration data is not a dictionary, using defaults")
        return Settings()
    
    settings = Settings()
    
    config_sections = [
        ('server', _create_server_config, 'server'),
        ('cors', _create_cors_config, 'cors'), 
        ('openai', _create_openai_config, 'openai'),
        ('embeddings', _create_embedding_config, 'embeddings'),
        ('pinecone', _create_pinecone_config, 'pinecone'),
        ('grading', _create_grading_config, 'grading'),
        ('questions', _create_questions_config, 'questions'),
        ('text_processing', _create_text_processing_config, 'text_processing')
    ]
    
    for section_key, create_func, attr_name in config_sections:
        if section_key in data:
            try:
                setattr(settings, attr_name, create_func(data[section_key]))
            except Exception as e:
                print(f"⚠️ Failed to process {section_key} configuration: {e}")
                print(f"📋 Using default {section_key} settings")
                # Default config is already set in Settings()
    
    return settings


def _validate_yaml_file_path(yaml_file_path: str) -> str:
    """
    Validate and resolve the YAML file path
    
    Args:
        yaml_file_path: Path to validate
        
    Returns:
        Resolved absolute path
    """
    if not yaml_file_path:
        raise ValueError("YAML file path cannot be empty")
    
    # Convert to absolute path if relative
    if not os.path.isabs(yaml_file_path):
        yaml_file_path = os.path.abspath(yaml_file_path)
    
    return yaml_file_path


def _load_yaml_data(file_path: str) -> Dict[str, Any]:
    """
    Load and parse YAML data from file
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Parsed YAML data as dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise IOError(f"Configuration file is not readable: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
            
        if yaml_data is None:
            print(f"⚠️ Configuration file is empty: {file_path}")
            return {}
            
        if not isinstance(yaml_data, dict):
            raise ValueError(f"Configuration file must contain a YAML dictionary, got {type(yaml_data)}")
            
        return yaml_data
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {file_path}: {e}")
    except UnicodeDecodeError as e:
        raise IOError(f"Cannot decode configuration file {file_path}: {e}")


def load_settings_from_yaml(yaml_file_path: str = "settings.yaml") -> Settings:
    """
    Load settings from YAML file with comprehensive error handling
    
    This function gracefully handles various error scenarios:
    - Missing configuration file
    - Invalid YAML syntax
    - Invalid configuration values
    - File permission issues
    - Empty or malformed files
    
    Args:
        yaml_file_path: Path to the YAML configuration file
        
    Returns:
        Settings object with loaded configuration (defaults used for any errors)
    """
    try:
        # Validate and resolve file path
        resolved_path = _validate_yaml_file_path(yaml_file_path)
        
        # Load YAML data
        yaml_data = _load_yaml_data(resolved_path)
        
        # Create settings from YAML data
        if yaml_data:
            settings = create_config_from_dict(yaml_data)
            print(f"✅ Successfully loaded configuration from {resolved_path}")
            
            # Log any sections that weren't found in the file
            expected_sections = ['server', 'cors', 'openai', 'grading', 'questions', 'text_processing']
            missing_sections = [section for section in expected_sections if section not in yaml_data]
            if missing_sections:
                print(f"📋 Using defaults for missing sections: {', '.join(missing_sections)}")
            
            return settings
        else:
            print(f"⚠️ Configuration file is empty, using all defaults: {resolved_path}")
            return Settings()
            
    except FileNotFoundError as e:
        print(f"⚠️ Configuration file not found: {e}")
        print("📋 Using all default settings")
        
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        print("📋 Using all default settings")
        
    except IOError as e:
        print(f"❌ File access error: {e}")
        print("📋 Using all default settings")
        
    except ValueError as e:
        print(f"❌ Configuration validation error: {e}")
        print("📋 Using all default settings")
        
    except Exception as e:
        print(f"❌ Unexpected error loading configuration: {e}")
        print("📋 Using all default settings")
    
    # Return default settings for any error
    return Settings()


def validate_settings(settings: Settings) -> None:
    """
    Validate loaded settings and provide helpful warnings
    
    Args:
        settings: Settings object to validate
    """
    # Validate server settings
    if settings.server.port < 1 or settings.server.port > 65535:
        print(f"⚠️ Invalid port number: {settings.server.port}, should be 1-65535")
    
    # Validate grading settings
    if not (0.0 <= settings.grading.similarity_thresholds.high_similarity <= 1.0):
        print(f"⚠️ High similarity threshold should be 0.0-1.0, got: {settings.grading.similarity_thresholds.high_similarity}")
    
    if not (0.0 <= settings.grading.similarity_thresholds.mid_similarity <= 1.0):
        print(f"⚠️ Mid similarity threshold should be 0.0-1.0, got: {settings.grading.similarity_thresholds.mid_similarity}")
    
    if settings.grading.similarity_thresholds.high_similarity <= settings.grading.similarity_thresholds.mid_similarity:
        print(f"⚠️ High similarity threshold ({settings.grading.similarity_thresholds.high_similarity}) should be greater than mid similarity ({settings.grading.similarity_thresholds.mid_similarity})")
    
    if settings.grading.answer_validation.min_answer_length < 1:
        print(f"⚠️ Minimum answer length should be at least 1, got: {settings.grading.answer_validation.min_answer_length}")
    
    if settings.grading.answer_validation.min_word_count < 1:
        print(f"⚠️ Minimum word count should be at least 1, got: {settings.grading.answer_validation.min_word_count}")
    
    # Validate questions settings
    if not settings.questions.default_file_path:
        print("⚠️ Questions file path is empty")
    
    # Validate text processing settings
    if settings.text_processing.min_token_length < 1:
        print(f"⚠️ Minimum token length should be at least 1, got: {settings.text_processing.min_token_length}")


def get_configuration_summary(settings: Settings) -> str:
    """
    Generate a human-readable summary of the current configuration
    
    Args:
        settings: Settings object to summarize
        
    Returns:
        Configuration summary string
    """
    summary = []
    summary.append("📋 Configuration Summary:")
    summary.append(f"  🌐 Server: {settings.server.host}:{settings.server.port}")
    summary.append(f"  🤖 OpenAI Model: {settings.openai.model_name}")
    summary.append(f"  💾 Embedding Cache: {'Enabled' if not settings.grading.precompute_embeddings else 'Disabled (fresh computation)'}")
    summary.append(f"  📄 Questions File: {settings.questions.default_file_path}")
    summary.append(f"  🎯 Similarity Thresholds: High={settings.grading.similarity_thresholds.high_similarity}, Mid={settings.grading.similarity_thresholds.mid_similarity}")
    
    return "\n".join(summary)


# Global settings instance with validation
settings = load_settings_from_yaml(
    os.path.join(os.path.dirname(__file__), "..", "settings.yaml")
)

# Validate the loaded settings
validate_settings(settings)

