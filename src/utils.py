"""
Utility functions for the Sentiment Analysis application.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if "random_tweets" not in st.session_state:
        st.session_state.random_tweets = None
        st.session_state.tweet_selection = None
        st.session_state.refresh_counter = 0

    if "sentiment140_df" not in st.session_state:
        st.session_state.sentiment140_df = None


def format_tweet_display(tweet_id: int, tweet_text: str, max_length: int = 80) -> str:
    """
    Format tweet for display in dropdown.

    Args:
        tweet_id: Tweet ID
        tweet_text: Tweet text
        max_length: Maximum length for display

    Returns:
        Formatted string for display
    """
    display_text = tweet_text[:max_length] + "..." if len(tweet_text) > max_length else tweet_text
    return f"ID: {tweet_id} | {display_text}"


def get_true_label_display(target: int) -> str:
    """
    Get display string for true label.

    Args:
        target: Target value (0 or 4 for Sentiment140)

    Returns:
        Formatted label string
    """
    if target == 0:
        return "😞 Negative"
    elif target == 4:
        return "😊 Positive"
    else:
        return f"Unknown ({target})"


def calculate_model_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray
) -> Dict[str, float]:
    """
    Calculate model performance metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities

    Returns:
        Dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc": roc_auc_score(y_true, y_probs[:, 1]),
    }

    return metrics


def create_evaluation_results(
    df: pd.DataFrame, y_pred: np.ndarray, y_probs: np.ndarray, model_service
) -> pd.DataFrame:
    """
    Create detailed evaluation results dataframe.

    Args:
        df: Original dataframe
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        model_service: Model service instance

    Returns:
        DataFrame with evaluation results
    """
    results = df.copy()
    results["pred"] = y_pred
    results["pred_label"] = model_service.to_label_name(y_pred)
    results["prob_negative"] = y_probs[:, 0]
    results["prob_positive"] = y_probs[:, 1]

    return results


def validate_csv_upload(df: pd.DataFrame, required_columns: List[str]) -> tuple[bool, str]:
    """
    Validate uploaded CSV file.

    Args:
        df: Uploaded dataframe
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df.empty:
        return False, "The uploaded file is empty."

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"

    return True, ""


def get_file_size_info(num_rows: int) -> str:
    """
    Get file size information for display.

    Args:
        num_rows: Number of rows in dataset

    Returns:
        Formatted size information string
    """
    size_mb = num_rows * 0.0001  # Rough estimate
    return f"{num_rows:,} rows (~{size_mb:.1f} MB in memory)"


def create_download_button(
    data: pd.DataFrame, filename: str, button_text: str = "⬇️ Download Results (CSV)"
) -> None:
    """
    Create download button for dataframe.

    Args:
        data: DataFrame to download
        filename: Filename for download
        button_text: Button text
    """
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(button_text, data=csv_data, file_name=filename, mime="text/csv")


def display_model_info(model_service) -> None:
    """
    Display model information in sidebar.

    Args:
        model_service: Model service instance
    """
    model_info = model_service.get_model_info()

    if model_info["loaded"]:
        st.sidebar.success("Model loaded ✅")
        with st.sidebar.expander("Model Info"):
            st.write(f"**Device:** {model_info['device']}")
            st.write(f"**Model Type:** {model_info['model_type']}")
            st.write(f"**Number of Labels:** {model_info['num_labels']}")
            st.write("**Labels:**")
            for label_id, label_name in model_info["labels"].items():
                st.write(f"  - {label_id}: {label_name}")
    else:
        st.sidebar.error("Model not loaded")


def get_accessibility_footer() -> str:
    """
    Get accessibility footer text.

    Returns:
        Footer text for accessibility compliance
    """
    return (
        "♿ **WCAG AA Compliant Dashboard** - High contrast colors, colorblind-friendly palettes, "
        "text alternatives, and screen reader support. Includes exploratory analysis, "
        "prediction and evaluation."
    )


def format_confidence_display(confidence: float) -> str:
    """
    Format confidence score for display.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        Formatted confidence string
    """
    return f"{confidence:.3f}"


def get_probability_breakdown_text(probs: np.ndarray) -> str:
    """
    Get textual description of probability breakdown for screen readers.

    Args:
        probs: Probability array [negative, positive]

    Returns:
        Textual description
    """
    neg_prob = probs[0]
    pos_prob = probs[1]

    return (
        f"📊 **Textual alternative for screen readers:** "
        f"NEGATIVE sentiment probability: {neg_prob:.3f} ({neg_prob*100:.1f}%), "
        f"POSITIVE sentiment probability: {pos_prob:.3f} ({pos_prob*100:.1f}%)"
    )


def get_confusion_matrix_text(cm: np.ndarray) -> str:
    """
    Get textual description of confusion matrix for screen readers.

    Args:
        cm: Confusion matrix

    Returns:
        Textual description
    """
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return (
        f"📊 Confusion matrix showing prediction accuracy. "
        f"True Negatives: {tn}, False Positives: {fp}, "
        f"False Negatives: {fn}, True Positives: {tp}"
    )
