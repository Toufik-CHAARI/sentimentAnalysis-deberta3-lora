"""
Data service for handling dataset operations and data processing.
"""

import time
from typing import Optional

import kagglehub
import pandas as pd

from .config import (
    DEFAULT_SAMPLE_SIZE,
    SENTIMENT140_COLUMNS,
    SENTIMENT140_DATASET,
    SENTIMENT140_LABEL_MAPPING,
)


class DataService:
    """Service class for data operations."""

    def __init__(self):
        self._cached_data: Optional[pd.DataFrame] = None
        self._cached_random_tweets: Optional[pd.DataFrame] = None

    def load_sentiment140(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        load_full: bool = False,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load Sentiment140 dataset from KaggleHub with optional sampling or full dataset.

        Args:
            sample_size: Number of samples to load (ignored if load_full=True)
            load_full: Whether to load the full dataset
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with Sentiment140 data
        """
        if use_cache and self._cached_data is not None:
            return self._cached_data

        # Download latest version of Sentiment140 dataset
        path = kagglehub.dataset_download(SENTIMENT140_DATASET)
        csv_file = f"{path}/training.1600000.processed.noemoticon.csv"

        if load_full:
            # Load the full 1.6M dataset
            df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=SENTIMENT140_COLUMNS)
        else:
            # Load with sampling for memory efficiency
            df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=SENTIMENT140_COLUMNS)
            if sample_size < len(df_raw):
                df_raw = df_raw.sample(sample_size, random_state=42)

        if use_cache:
            self._cached_data = df_raw

        return df_raw

    def load_random_tweets(self, num_tweets: int = 30) -> pd.DataFrame:
        """
        Load a random sample of tweets from Sentiment140 for prediction testing.

        Args:
            num_tweets: Number of random tweets to load

        Returns:
            DataFrame with random tweets
        """
        # Download latest version of Sentiment140 dataset
        path = kagglehub.dataset_download(SENTIMENT140_DATASET)
        csv_file = f"{path}/training.1600000.processed.noemoticon.csv"

        # Load a larger sample to get good random selection
        df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=SENTIMENT140_COLUMNS)

        # Get random sample with current timestamp as seed for true randomness
        random_seed = int(time.time() * 1000) % 2**32  # Use current timestamp as seed
        random_tweets = df_raw.sample(num_tweets, random_state=random_seed)

        return random_tweets

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process dataframe for analysis (add text length, label mapping, etc.).

        Args:
            df: Input dataframe

        Returns:
            Processed dataframe
        """
        df_processed = df.copy()

        # Add text length column
        if "text" in df_processed.columns:
            df_processed["text_len"] = df_processed["text"].astype(str).str.split().apply(len)

        # Map Sentiment140 labels (0=negative, 4=positive) to readable labels
        if "target" in df_processed.columns:
            df_processed["label"] = df_processed["target"].map(SENTIMENT140_LABEL_MAPPING)

        return df_processed

    def get_text_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get text statistics from dataframe.

        Args:
            df: Dataframe with text data

        Returns:
            Dictionary with text statistics
        """
        if "text_len" not in df.columns:
            df = self.process_dataframe(df)

        stats = {
            "total_texts": len(df),
            "avg_length": df["text_len"].mean(),
            "median_length": df["text_len"].median(),
            "min_length": df["text_len"].min(),
            "max_length": df["text_len"].max(),
            "std_length": df["text_len"].std(),
        }

        return stats

    def get_label_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get label distribution from dataframe.

        Args:
            df: Dataframe with label data

        Returns:
            DataFrame with label counts
        """
        if "label" not in df.columns:
            df = self.process_dataframe(df)

        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]

        return label_counts

    def get_word_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Get word frequency from dataframe.

        Args:
            df: Dataframe with text data
            top_n: Number of top words to return

        Returns:
            DataFrame with word frequencies
        """
        from collections import Counter

        all_words = " ".join(df["text"].astype(str).tolist()).lower().split()
        freq = Counter(all_words)
        common = pd.DataFrame(freq.most_common(top_n), columns=["word", "freq"])

        return common

    def clear_cache(self):
        """Clear cached data."""
        self._cached_data = None
        self._cached_random_tweets = None
