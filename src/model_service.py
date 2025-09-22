"""
Model service for handling DeBERTa v3 LoRA model operations.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, LABEL_MAPPING


class ModelService:
    """Service class for model operations."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(
        self, model_dir: str
    ) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """
        Load the DeBERTa v3 LoRA model and tokenizer.

        Args:
            model_dir: Path to the model directory

        Returns:
            Tuple of (tokenizer, model)

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        # Ensure we're using absolute path for local files
        if not os.path.isabs(model_dir):
            model_dir = os.path.abspath(model_dir)

        # Check if the directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )

        # Update labels to use English with emoticons
        if self.model is not None:
            self.model.config.id2label = LABEL_MAPPING  # type: ignore
            self.model.config.label2id = {v: k for k, v in LABEL_MAPPING.items()}  # type: ignore

        # Type assertions to satisfy mypy
        assert self.tokenizer is not None
        assert self.model is not None
        return self.tokenizer, self.model

    @torch.inference_mode()
    def predict_batch(
        self, texts: List[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sentiment for a batch of texts.

        Args:
            texts: List of text strings to predict
            batch_size: Batch size for processing

        Returns:
            Tuple of (probabilities, predictions)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")

        # Type assertions after null checks
        assert self.model is not None
        assert self.tokenizer is not None

        self.model.eval()  # type: ignore
        self.model.to(self.device)  # type: ignore

        all_probs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer(  # type: ignore
                chunk,
                padding=True,
                truncation=True,
                max_length=DEFAULT_MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits  # type: ignore
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            all_probs.append(probs)

        probs = np.vstack(all_probs) if all_probs else np.zeros((0, 2), dtype=np.float32)
        preds = probs.argmax(axis=-1)
        return probs, preds

    def predict_single(self, text: str) -> Tuple[np.ndarray, int, str]:
        """
        Predict sentiment for a single text.

        Args:
            text: Text string to predict

        Returns:
            Tuple of (probabilities, prediction_id, label_name)
        """
        probs, preds = self.predict_batch([text])
        pred_id = preds[0]
        label = self.to_label_name(np.array([pred_id]))[0]
        return probs[0], pred_id, label

    def to_label_name(self, ids: np.ndarray) -> List[str]:
        """
        Convert prediction IDs to label names.

        Args:
            ids: Array of prediction IDs

        Returns:
            List of label names
        """
        if self.model is None:
            id2label = LABEL_MAPPING
        else:
            id2label = getattr(self.model.config, "id2label", LABEL_MAPPING)  # type: ignore

        return [str(id2label.get(int(i), int(i))) for i in ids]

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False}

        return {
            "loaded": True,
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "num_labels": self.model.config.num_labels,  # type: ignore
            "labels": self.model.config.id2label,  # type: ignore
        }
