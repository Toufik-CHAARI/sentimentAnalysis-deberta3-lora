"""
Configuration settings for the Sentiment Analysis application.
"""

import os
from pathlib import Path
from typing import Any, Dict

# Default model configuration
DEFAULT_MODEL_DIR = "deberta3_lora_400000k/merged"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128

# Dataset configuration
SENTIMENT140_DATASET = "kazanova/sentiment140"
SENTIMENT140_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
DEFAULT_SAMPLE_SIZE = 50_000

# Label mapping
LABEL_MAPPING = {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"}

REVERSE_LABEL_MAPPING = {"😞 Negative Sentiment": 0, "😊 Positive Sentiment": 1}

# Sentiment140 original labels (0=negative, 4=positive)
SENTIMENT140_LABEL_MAPPING = {0: "😞 Negative Sentiment", 4: "😊 Positive Sentiment"}

# Visualization settings
PLOT_COLORS = {
    "primary": "#1f77b4",
    "negative": "#DC143C",  # Crimson
    "positive": "#2E8B57",  # Sea Green
    "background": "white",
}

# Accessibility settings
WCAG_COMPLIANT = True
HIGH_CONTRAST_COLORS = True

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / DEFAULT_MODEL_DIR


# Environment variables
def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""
    return {
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        "dvc_bucket": os.getenv("DVC_BUCKET"),
    }


# Streamlit page config
PAGE_CONFIG = {
    "page_title": "Sentiment140 Dashboard — DeBERTa v3",
    "page_icon": "📊",
    "layout": "wide",
}
