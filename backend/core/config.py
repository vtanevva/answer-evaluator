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
@dataclass
class EmbeddingConfig:
    """Unified embeddings configuration supporting any model"""
    # Model identifier (API model name or local model path)
    model: str = "text-embedding-ada-002"
    
    # Model type: "openai" or "sentence-transformer"
    type: str = "openai"
    
    # Embedding dimensions
    dimensions: int = 1536
    
    @property
    def is_local(self) -> bool:
        """Check if this is a local model"""
        return self.type == "sentence-transformer"
    
    @property
    def is_remote(self) -> bool:
        """Check if this is a remote API model"""
        return self.type == "openai"


@dataclass
class PineconeConfig:
    api_key: str = os.getenv("PINECONE_API_KEY", "")
    environment: str = "gcp-starter"
    index_name: str = "answer-evaluator"
    dimension: int = 1536
    metric: str = "cosine"


@dataclass
class SimilarityThresholds:
    high_similarity: float = 0.855
    mid_similarity: float = 0.80
    llm_verification_threshold: float = 0.8
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
    violent_answers: List[str] = field(default_factory=lambda: ["fuck"])
    invalid_answers: List[str] = field(default_factory=lambda: ["i don't know", "i don't know.", "dont know"])
    feedback_message: str = "{invalid_answer} is not a valid answer to the question. Try again."


@dataclass
class GradingConfig:
    # Grading method: "hybrid", "nli", or "embedding"
    grading_method: str = "embedding"
    
    # NLI-specific settings
    nli_model: str = "microsoft/deberta-v3-small"
    nli_entailment_threshold: float = 0.6
    nli_contradiction_threshold: float = 0.7
    
    # Hybrid mode thresholds (three-tier verification)
    auto_pass_threshold: float = 0.85      # >= 85% similarity = auto-pass (no NLI)
    nli_verify_threshold: float = 0.70     # 70-85% similarity = NLI verification
    nli_deep_check_threshold: float = 0.70 # < 70% similarity = NLI deep check
    
    # LLM Arbiter settings
    llm_arbiter_enabled: bool = False
    llm_arbiter_provider: str = "groq"
    llm_arbiter_model: str = "llama-3.1-8b-instant"
    llm_holistic_mode: str = "never"        # "never", "always", or "fallback"
    
    # Embedding-specific settings
    precompute_embeddings: bool = True
    similarity_thresholds: SimilarityThresholds = field(default_factory=SimilarityThresholds)
    
    # Common settings
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
class ConfidenceThresholds:
    high: float = 0.8
    medium: float = 0.6
    low: float = 0.4


@dataclass
class MethodWeights:
    semantic_distance: float = 0.4
    contextual_analysis: float = 0.3
    known_pattern: float = 0.2
    negation_pattern: float = 0.2
    llm_validation: float = 0.3


@dataclass
class AntonymDetectionConfig:
    semantic_distance_threshold: float = 0.3
    context_similarity_threshold: float = 0.7
    confidence_thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    use_llm_validation: bool = True
    llm_validation_threshold: float = 0.5
    method_weights: MethodWeights = field(default_factory=MethodWeights)
    antonym_penalty_multiplier: float = 0.2
    min_confidence_for_penalty: str = "medium"

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
    antonym_detection: AntonymDetectionConfig = field(default_factory=AntonymDetectionConfig)


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
        grading_method = data.get('grading_method', 'embedding')
        nli_model = data.get('nli_model', 'microsoft/deberta-v3-small')
        nli_entailment_threshold = data.get('nli_entailment_threshold', 0.6)
        nli_contradiction_threshold = data.get('nli_contradiction_threshold', 0.7)
        
        # Hybrid mode thresholds
        auto_pass_threshold = data.get('auto_pass_threshold', 0.85)
        nli_verify_threshold = data.get('nli_verify_threshold', 0.70)
        nli_deep_check_threshold = data.get('nli_deep_check_threshold', 0.70)
        
        precompute_embeddings = data.get('precompute_embeddings', True)
        
        # LLM Arbiter settings
        llm_arbiter_enabled = data.get('llm_arbiter_enabled', False)
        llm_arbiter_provider = data.get('llm_arbiter_provider', 'groq')
        llm_arbiter_model = data.get('llm_arbiter_model', 'llama-3.1-8b-instant')
        llm_holistic_mode = data.get('llm_holistic_mode', 'never')

        
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
            grading_method=grading_method,
            nli_model=nli_model,
            nli_entailment_threshold=nli_entailment_threshold,
            nli_contradiction_threshold=nli_contradiction_threshold,
            auto_pass_threshold=auto_pass_threshold,
            nli_verify_threshold=nli_verify_threshold,
            nli_deep_check_threshold=nli_deep_check_threshold,
            precompute_embeddings=precompute_embeddings,
            llm_arbiter_enabled=llm_arbiter_enabled,
            llm_arbiter_provider=llm_arbiter_provider,
            llm_arbiter_model=llm_arbiter_model,
            llm_holistic_mode=llm_holistic_mode,
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

def _create_antonym_detection_config(data: Dict[str, Any]) -> AntonymDetectionConfig:
    """Create AntonymDetectionConfig with validation and defaults"""
    try:
        # Extract nested configs with error handling
        confidence_thresholds = ConfidenceThresholds()
        if 'confidence_thresholds' in data:
            try:
                confidence_thresholds = ConfidenceThresholds(**data['confidence_thresholds'])
            except Exception as e:
                print(f"⚠️ Invalid confidence thresholds: {e}")
                print("📋 Using default confidence thresholds")
        
        method_weights = MethodWeights()
        if 'method_weights' in data:
            try:
                method_weights = MethodWeights(**data['method_weights'])
            except Exception as e:
                print(f"⚠️ Invalid method weights: {e}")
                print("📋 Using default method weights")
        
        # Extract simple fields with defaults
        return AntonymDetectionConfig(
            semantic_distance_threshold=data.get('semantic_distance_threshold', 0.3),
            context_similarity_threshold=data.get('context_similarity_threshold', 0.7),
            confidence_thresholds=confidence_thresholds,
            use_llm_validation=data.get('use_llm_validation', True),
            llm_validation_threshold=data.get('llm_validation_threshold', 0.5),
            method_weights=method_weights,
            antonym_penalty_multiplier=data.get('antonym_penalty_multiplier', 0.2),
            min_confidence_for_penalty=data.get('min_confidence_for_penalty', 'medium')
        )
    except Exception as e:
        print(f"⚠️ Invalid antonym detection configuration: {e}")
        print("📋 Using default antonym detection settings")
        return AntonymDetectionConfig()


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
        ('text_processing', _create_text_processing_config, 'text_processing'),
        ('antonym_detection', _create_antonym_detection_config, 'antonym_detection')
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
            print(f"Successfully loaded configuration from {resolved_path}")
            
            # Log any sections that weren't found in the file
            expected_sections = [
                'server',
                'cors',
                'openai',
                'embeddings',
                'pinecone',
                'grading',
                'questions',
                'text_processing',
                'antonym_detection'
            ]
            missing_sections = [section for section in expected_sections if section not in yaml_data]
            if missing_sections:
                print(f"Using defaults for missing sections: {', '.join(missing_sections)}")
            
            return settings
        else:
            print(f"Warning: Configuration file is empty, using all defaults: {resolved_path}")
            return Settings()
            
    except FileNotFoundError as e:
        print(f"Warning: Configuration file not found: {e}")
        print("Using all default settings")
        
    except yaml.YAMLError as e:
        print(f"Error: YAML parsing error: {e}")
        print("Using all default settings")
        
    except IOError as e:
        print(f"Error: File access error: {e}")
        print("Using all default settings")
        
    except ValueError as e:
        print(f"Error: Configuration validation error: {e}")
        print("Using all default settings")
        
    except Exception as e:
        print(f"Error: Unexpected error loading configuration: {e}")
        print("Using all default settings")
    
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
    if not (0.0 <= settings.grading.similarity_thresholds.llm_verification_threshold <= 1.0):
        print(f"⚠️ LLM verification threshold should be 0.0-1.0, got: {settings.grading.similarity_thresholds.llm_verification_threshold}")
    
    if settings.grading.similarity_thresholds.mid_similarity < settings.grading.similarity_thresholds.llm_verification_threshold:
        print(
            "⚠️ Mid similarity threshold "
            f"({settings.grading.similarity_thresholds.mid_similarity}) should be "
            "greater than or equal to the LLM verification threshold "
            f"({settings.grading.similarity_thresholds.llm_verification_threshold})"
        )
    
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
    summary.append(
        "  🎯 Similarity Thresholds: "
        f"High={settings.grading.similarity_thresholds.high_similarity}, "
        f"Mid={settings.grading.similarity_thresholds.mid_similarity}, "
        f"LLM={settings.grading.similarity_thresholds.llm_verification_threshold}"
    )
    
    return "\n".join(summary)


# Global settings instance with validation
settings = load_settings_from_yaml(
    os.path.join(os.path.dirname(__file__), "..", "settings.yaml")
)

# Validate the loaded settings
validate_settings(settings)

