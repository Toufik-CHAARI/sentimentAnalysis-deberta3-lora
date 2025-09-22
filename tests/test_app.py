"""
Unit tests for the Sentiment Analysis DeBERTa v3 LoRA application.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules
from src.data_service import DataService
from src.model_service import ModelService
from src.utils import (
    calculate_model_metrics,
    create_download_button,
    create_evaluation_results,
    display_model_info,
    format_confidence_display,
    format_tweet_display,
    get_accessibility_footer,
    get_confusion_matrix_text,
    get_file_size_info,
    get_probability_breakdown_text,
    get_true_label_display,
    initialize_session_state,
    validate_csv_upload,
)
from src.visualization_service import VisualizationService


class TestModelService:
    """Test ModelService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_service = ModelService()

    @patch("src.model_service.AutoTokenizer.from_pretrained")
    @patch("src.model_service.AutoModelForSequenceClassification.from_pretrained")
    @patch("os.path.exists")
    def test_load_model_success(self, mock_exists, mock_model, mock_tokenizer):
        """Test successful model loading."""
        # Mock file system
        mock_exists.return_value = True

        # Mock the model and tokenizer
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock()
        mock_model.return_value.config.id2label = {}
        mock_model.return_value.config.label2id = {}

        # Test model loading
        tokenizer, model = self.model_service.load_model("test_model_path")

        # Assertions
        assert tokenizer is not None
        assert model is not None
        # The path gets converted to absolute path, so we check the call was made
        assert mock_tokenizer.call_count == 1
        assert mock_model.call_count == 1
        assert "test_model_path" in str(mock_tokenizer.call_args)
        assert "test_model_path" in str(mock_model.call_args)

    @patch("os.path.exists")
    def test_load_model_directory_not_found(self, mock_exists):
        """Test model loading when directory doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            self.model_service.load_model("nonexistent_path")

    @patch("src.model_service.AutoTokenizer.from_pretrained")
    @patch("src.model_service.AutoModelForSequenceClassification.from_pretrained")
    @patch("os.path.exists")
    def test_predict_batch_success(self, mock_exists, mock_model, mock_tokenizer):
        """Test successful batch prediction."""
        # Setup mocks
        mock_exists.return_value = True
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock()
        mock_model.return_value.config.id2label = {}
        mock_model.return_value.config.label2id = {}

        # Mock model methods
        mock_model.return_value.eval = Mock()
        mock_model.return_value.to = Mock()
        mock_model.return_value.return_value.logits = Mock()

        # Mock tokenizer
        mock_tokenizer.return_value.return_value = {"input_ids": Mock(), "attention_mask": Mock()}

        # Load model first
        self.model_service.load_model("test_model_path")

        # Mock torch operations
        with patch("torch.softmax") as mock_softmax, patch("torch.inference_mode"):
            mock_softmax.return_value.detach.return_value.cpu.return_value.numpy.return_value = (
                np.array([[0.3, 0.7]])
            )

            # Test prediction
            texts = ["test text"]
            probs, preds = self.model_service.predict_batch(texts)

            assert len(probs) == 1
            assert len(preds) == 1
            assert preds[0] == 1  # argmax of [0.3, 0.7]

    @patch("src.model_service.AutoTokenizer.from_pretrained")
    @patch("src.model_service.AutoModelForSequenceClassification.from_pretrained")
    @patch("os.path.exists")
    def test_predict_single_success(self, mock_exists, mock_model, mock_tokenizer):
        """Test successful single prediction."""
        # Setup mocks
        mock_exists.return_value = True
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_model.return_value.config = Mock()
        mock_model.return_value.config.id2label = {
            0: "😞 Negative Sentiment",
            1: "😊 Positive Sentiment",
        }
        mock_model.return_value.config.label2id = {
            "😞 Negative Sentiment": 0,
            "😊 Positive Sentiment": 1,
        }

        # Mock model methods
        mock_model.return_value.eval = Mock()
        mock_model.return_value.to = Mock()
        mock_model.return_value.return_value.logits = Mock()

        # Mock tokenizer
        mock_tokenizer.return_value.return_value = {"input_ids": Mock(), "attention_mask": Mock()}

        # Load model first
        self.model_service.load_model("test_model_path")

        # Mock torch operations
        with patch("torch.softmax") as mock_softmax, patch("torch.inference_mode"):
            mock_softmax.return_value.detach.return_value.cpu.return_value.numpy.return_value = (
                np.array([[0.3, 0.7]])
            )

            # Test prediction
            probs, pred_id, label = self.model_service.predict_single("test text")

            assert len(probs) == 2
            assert pred_id == 1
            assert label == "😊 Positive Sentiment"

    def test_predict_batch_no_model(self):
        """Test batch prediction without loaded model."""
        with pytest.raises(ValueError, match="Model and tokenizer must be loaded first"):
            self.model_service.predict_batch(["test text"])

    def test_to_label_name_with_model(self):
        """Test label name conversion with loaded model."""
        # Mock model with config
        mock_model = Mock()
        mock_model.config.id2label = {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"}
        self.model_service.model = mock_model

        ids = np.array([0, 1])
        labels = self.model_service.to_label_name(ids)

        assert labels == ["😞 Negative Sentiment", "😊 Positive Sentiment"]

    def test_to_label_name_without_model(self):
        """Test label name conversion without loaded model."""
        ids = np.array([0, 1])
        labels = self.model_service.to_label_name(ids)

        assert labels == ["😞 Negative Sentiment", "😊 Positive Sentiment"]

    def test_get_model_info_loaded(self):
        """Test get model info when model is loaded."""
        # Mock model
        mock_model = Mock()
        mock_model.config.num_labels = 2
        mock_model.config.id2label = {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"}
        self.model_service.model = mock_model

        info = self.model_service.get_model_info()

        assert info["loaded"] is True
        assert "device" in info
        assert info["num_labels"] == 2
        assert "labels" in info

    def test_get_model_info_not_loaded(self):
        """Test get model info when model is not loaded."""
        info = self.model_service.get_model_info()

        assert info["loaded"] is False


class TestDataService:
    """Test DataService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_service = DataService()

    @patch("src.data_service.kagglehub.dataset_download")
    @patch("src.data_service.pd.read_csv")
    def test_load_sentiment140_sample(self, mock_read_csv, mock_download):
        """Test loading Sentiment140 sample data."""
        # Mock kagglehub download
        mock_download.return_value = "/mock/path"

        # Mock pandas read_csv
        mock_df = pd.DataFrame(
            {
                "target": [0, 4, 0, 4],
                "text": [
                    "negative tweet",
                    "positive tweet",
                    "another negative",
                    "another positive",
                ],
            }
        )
        mock_read_csv.return_value = mock_df

        # Test loading
        df = self.data_service.load_sentiment140(sample_size=4, load_full=False)

        # Assertions
        assert len(df) == 4
        assert "target" in df.columns
        assert "text" in df.columns
        mock_download.assert_called_once_with("kazanova/sentiment140")

    @patch("src.data_service.kagglehub.dataset_download")
    @patch("src.data_service.pd.read_csv")
    def test_load_random_tweets(self, mock_read_csv, mock_download):
        """Test loading random tweets."""
        # Mock kagglehub download
        mock_download.return_value = "/mock/path"

        # Mock pandas read_csv
        mock_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "target": [0, 4, 0, 4, 0],
                "text": ["tweet1", "tweet2", "tweet3", "tweet4", "tweet5"],
            }
        )
        mock_read_csv.return_value = mock_df

        # Test loading
        df = self.data_service.load_random_tweets(num_tweets=3)

        # Assertions
        assert len(df) == 3
        assert "id" in df.columns
        assert "target" in df.columns
        assert "text" in df.columns

    def test_process_dataframe(self):
        """Test dataframe processing."""
        df = pd.DataFrame(
            {
                "target": [0, 4, 0, 4],
                "text": ["short", "this is a longer text", "tiny", "another longer text here"],
            }
        )

        processed_df = self.data_service.process_dataframe(df)

        # Assertions
        assert "text_len" in processed_df.columns
        assert "label" in processed_df.columns
        assert processed_df["text_len"].tolist() == [1, 5, 1, 4]
        assert processed_df["label"].iloc[0] == "😞 Negative Sentiment"
        assert processed_df["label"].iloc[1] == "😊 Positive Sentiment"

    @patch("src.data_service.kagglehub.dataset_download")
    @patch("src.data_service.pd.read_csv")
    def test_load_sentiment140_with_cache(self, mock_read_csv, mock_download):
        """Test loading Sentiment140 with caching."""
        # Mock kagglehub download
        mock_download.return_value = "/mock/path"

        # Mock pandas read_csv
        mock_df = pd.DataFrame(
            {
                "target": [0, 4, 0, 4],
                "text": [
                    "negative tweet",
                    "positive tweet",
                    "another negative",
                    "another positive",
                ],
            }
        )
        mock_read_csv.return_value = mock_df

        # First load - should cache
        df1 = self.data_service.load_sentiment140(sample_size=4, load_full=False, use_cache=True)
        assert len(df1) == 4

        # Second load - should use cache (no new kagglehub call)
        df2 = self.data_service.load_sentiment140(sample_size=4, load_full=False, use_cache=True)
        assert len(df2) == 4
        assert mock_download.call_count == 1  # Only called once due to caching

    @patch("src.data_service.kagglehub.dataset_download")
    @patch("src.data_service.pd.read_csv")
    def test_load_sentiment140_full_dataset(self, mock_read_csv, mock_download):
        """Test loading full Sentiment140 dataset."""
        # Mock kagglehub download
        mock_download.return_value = "/mock/path"

        # Mock pandas read_csv with larger dataset
        mock_df = pd.DataFrame(
            {
                "target": [0, 4] * 1000,  # 2000 rows
                "text": ["tweet"] * 2000,
            }
        )
        mock_read_csv.return_value = mock_df

        # Test loading full dataset
        df = self.data_service.load_sentiment140(sample_size=100, load_full=True)
        assert len(df) == 2000  # Should load all rows when load_full=True

    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        df = pd.DataFrame(
            {
                "text": ["short", "this is a longer text", "tiny", "another longer text here"],
                "target": [0, 4, 0, 4],
            }
        )

        # Process dataframe first
        df_processed = self.data_service.process_dataframe(df)
        stats = self.data_service.get_text_statistics(df_processed)

        assert stats["total_texts"] == 4
        assert stats["avg_length"] == 2.75  # (1+5+1+4)/4
        assert stats["median_length"] == 2.5
        assert stats["min_length"] == 1
        assert stats["max_length"] == 5
        assert "std_length" in stats

    def test_get_text_statistics_auto_process(self):
        """Test text statistics with auto-processing."""
        df = pd.DataFrame(
            {
                "text": ["short", "this is a longer text"],
                "target": [0, 4],
            }
        )

        # Should automatically process if text_len not present
        stats = self.data_service.get_text_statistics(df)

        assert stats["total_texts"] == 2
        assert stats["avg_length"] == 3.0  # (1+5)/2

    def test_get_label_distribution(self):
        """Test label distribution calculation."""
        df = pd.DataFrame(
            {
                "target": [0, 4, 0, 4, 0],
                "text": ["tweet"] * 5,
            }
        )

        # Process dataframe first
        df_processed = self.data_service.process_dataframe(df)
        label_counts = self.data_service.get_label_distribution(df_processed)

        assert len(label_counts) == 2
        assert "label" in label_counts.columns
        assert "count" in label_counts.columns
        assert label_counts["count"].sum() == 5

    def test_get_label_distribution_auto_process(self):
        """Test label distribution with auto-processing."""
        df = pd.DataFrame(
            {
                "target": [0, 4, 0],
                "text": ["tweet"] * 3,
            }
        )

        # Should automatically process if label not present
        label_counts = self.data_service.get_label_distribution(df)

        assert len(label_counts) == 2
        assert "label" in label_counts.columns
        assert "count" in label_counts.columns

    def test_get_word_frequency(self):
        """Test word frequency calculation."""
        df = pd.DataFrame(
            {
                "text": ["hello world", "world is great", "hello there"],
                "target": [0, 4, 0],
            }
        )

        word_freq = self.data_service.get_word_frequency(df, top_n=5)

        assert len(word_freq) <= 5
        assert "word" in word_freq.columns
        assert "freq" in word_freq.columns
        assert (
            word_freq["freq"].sum() == 7
        )  # Total word count: hello(2), world(2), is(1), great(1), there(1)

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Set some cached data
        self.data_service._cached_data = pd.DataFrame({"test": [1, 2, 3]})
        self.data_service._cached_random_tweets = pd.DataFrame({"test": [4, 5, 6]})

        # Clear cache
        self.data_service.clear_cache()

        assert self.data_service._cached_data is None
        assert self.data_service._cached_random_tweets is None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_tweet_display(self):
        """Test tweet display formatting."""
        result = format_tweet_display(
            12345, "This is a very long tweet that should be truncated", 20
        )
        assert "ID: 12345" in result
        assert "This is a very long" in result

        result_short = format_tweet_display(12345, "Short tweet", 20)
        assert "Short tweet" in result_short
        assert "..." not in result_short

    def test_get_true_label_display(self):
        """Test true label display formatting."""
        assert get_true_label_display(0) == "😞 Negative"
        assert get_true_label_display(4) == "😊 Positive"
        assert get_true_label_display(99) == "Unknown (99)"

    def test_calculate_model_metrics(self):
        """Test model metrics calculation."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])

        metrics = calculate_model_metrics(y_true, y_pred, y_probs)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert 0 <= metrics["auc"] <= 1

    def test_validate_csv_upload(self):
        """Test CSV upload validation."""
        # Valid CSV
        df_valid = pd.DataFrame({"text": ["tweet1", "tweet2"], "label": [0, 1]})
        is_valid, error = validate_csv_upload(df_valid, ["text"])
        assert is_valid
        assert error == ""

        # Missing required column
        df_invalid = pd.DataFrame({"other": ["data1", "data2"]})
        is_valid, error = validate_csv_upload(df_invalid, ["text"])
        assert not is_valid
        assert "Missing required columns" in error

        # Empty CSV
        df_empty = pd.DataFrame()
        is_valid, error = validate_csv_upload(df_empty, ["text"])
        assert not is_valid
        assert "empty" in error.lower()

    def test_get_file_size_info(self):
        """Test file size info formatting."""
        result = get_file_size_info(1000)
        assert "1,000 rows" in result
        assert "0.1 MB" in result

    def test_get_probability_breakdown_text(self):
        """Test probability breakdown text generation."""
        probs = np.array([0.3, 0.7])
        result = get_probability_breakdown_text(probs)
        assert "NEGATIVE sentiment probability: 0.300" in result
        assert "POSITIVE sentiment probability: 0.700" in result
        assert "30.0%" in result
        assert "70.0%" in result

    def test_get_confusion_matrix_text(self):
        """Test confusion matrix text generation."""
        cm = np.array([[10, 2], [3, 15]])
        result = get_confusion_matrix_text(cm)
        assert "True Negatives: 10" in result
        assert "False Positives: 2" in result
        assert "False Negatives: 3" in result
        assert "True Positives: 15" in result

    def test_initialize_session_state(self):
        """Test session state initialization."""
        # This test mainly ensures the function doesn't crash
        # Streamlit session state is complex to mock properly
        try:
            initialize_session_state()
            # If we get here without exception, the test passes
            assert True
        except Exception as e:
            # We expect some warnings/errors when running outside streamlit context
            # This is acceptable for testing purposes
            assert "ScriptRunContext" in str(e) or "Session state" in str(e)

    def test_create_evaluation_results(self):
        """Test evaluation results creation."""
        # Mock model service
        mock_model_service = Mock()
        mock_model_service.to_label_name.return_value = [
            "😞 Negative Sentiment",
            "😊 Positive Sentiment",
        ]

        df = pd.DataFrame({"text": ["tweet1", "tweet2"], "label": [0, 1]})
        y_pred = np.array([0, 1])
        y_probs = np.array([[0.8, 0.2], [0.3, 0.7]])

        results = create_evaluation_results(df, y_pred, y_probs, mock_model_service)

        assert "pred" in results.columns
        assert "pred_label" in results.columns
        assert "prob_negative" in results.columns
        assert "prob_positive" in results.columns
        assert len(results) == 2
        assert results["prob_negative"].tolist() == [0.8, 0.3]
        assert results["prob_positive"].tolist() == [0.2, 0.7]

    @patch("streamlit.download_button")
    def test_create_download_button(self, mock_download_button):
        """Test download button creation."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        create_download_button(df, "test.csv", "Download Test")

        # Verify download_button was called
        mock_download_button.assert_called_once()
        call_args = mock_download_button.call_args
        assert call_args[0][0] == "Download Test"  # button_text
        assert call_args[1]["file_name"] == "test.csv"
        assert call_args[1]["mime"] == "text/csv"

    @patch("streamlit.sidebar")
    def test_display_model_info_loaded(self, mock_sidebar):
        """Test display model info when model is loaded."""
        # Mock model service
        mock_model_service = Mock()
        mock_model_service.get_model_info.return_value = {
            "loaded": True,
            "device": "cpu",
            "model_type": "TestModel",
            "num_labels": 2,
            "labels": {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"},
        }

        # Mock streamlit components
        mock_sidebar.success = Mock()
        mock_sidebar.expander.return_value.__enter__ = Mock()
        mock_sidebar.expander.return_value.__exit__ = Mock()
        mock_sidebar.expander.return_value.write = Mock()

        display_model_info(mock_model_service)

        # Verify success message was shown
        mock_sidebar.success.assert_called_once_with("Model loaded ✅")

    @patch("streamlit.sidebar")
    def test_display_model_info_not_loaded(self, mock_sidebar):
        """Test display model info when model is not loaded."""
        # Mock model service
        mock_model_service = Mock()
        mock_model_service.get_model_info.return_value = {"loaded": False}

        # Mock streamlit components
        mock_sidebar.error = Mock()

        display_model_info(mock_model_service)

        # Verify error message was shown
        mock_sidebar.error.assert_called_once_with("Model not loaded")

    def test_get_accessibility_footer(self):
        """Test accessibility footer generation."""
        footer = get_accessibility_footer()

        assert "WCAG AA Compliant" in footer
        assert "screen reader" in footer
        assert "High contrast" in footer

    def test_format_confidence_display(self):
        """Test confidence display formatting."""
        result = format_confidence_display(0.8547)
        assert result == "0.855"  # Rounded to 3 decimal places

        result2 = format_confidence_display(0.1)
        assert result2 == "0.100"


class TestVisualizationService:
    """Test VisualizationService functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.viz_service = VisualizationService()

    def test_create_text_length_histogram(self):
        """Test text length histogram creation."""
        df = pd.DataFrame({"text_len": [1, 2, 3, 4, 5, 1, 2, 3]})
        fig = self.viz_service.create_text_length_histogram(df)

        assert fig is not None
        assert fig.layout.title.text == "Distribution of Sentence Lengths"

    def test_create_label_distribution_chart(self):
        """Test label distribution chart creation."""
        label_counts = pd.DataFrame(
            {"label": ["😞 Negative Sentiment", "😊 Positive Sentiment"], "count": [100, 150]}
        )
        fig = self.viz_service.create_label_distribution_chart(label_counts)

        assert fig is not None
        assert fig.layout.title.text == "Class Distribution"

    def test_create_word_frequency_chart(self):
        """Test word frequency chart creation."""
        word_freq = pd.DataFrame({"word": ["the", "and", "to"], "freq": [100, 80, 60]})
        fig = self.viz_service.create_word_frequency_chart(word_freq)

        assert fig is not None
        assert fig.layout.title.text == "Top 20 Most Frequent Words"

    def test_create_probability_chart(self):
        """Test probability chart creation."""
        probabilities = {"NEGATIVE": 0.3, "POSITIVE": 0.7}
        fig = self.viz_service.create_probability_chart(probabilities)

        assert fig is not None
        assert fig.layout.title.text == "Sentiment Probabilities"

    def test_get_accessibility_caption(self):
        """Test accessibility caption generation."""
        caption = self.viz_service.get_accessibility_caption("histogram")
        assert "📊" in caption
        assert "WCAG" in caption

        caption_with_info = self.viz_service.get_accessibility_caption("bar", "Additional info")
        assert "Additional info" in caption_with_info

    @patch("matplotlib.pyplot.subplots")
    @patch("src.visualization_service.WordCloud")
    def test_create_wordcloud(self, mock_wordcloud, mock_subplots):
        """Test wordcloud creation."""
        # Mock WordCloud
        mock_wc_instance = Mock()
        mock_wordcloud.return_value = mock_wc_instance
        mock_wc_instance.generate.return_value = mock_wc_instance

        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        self.viz_service.create_wordcloud("test text", "Test Title", 50)

        # Verify WordCloud was created with correct parameters
        mock_wordcloud.assert_called_once_with(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=50,
            relative_scaling=0.5,
        )

        # Verify matplotlib setup
        mock_subplots.assert_called_once_with(figsize=(10, 5))
        mock_ax.imshow.assert_called_once()
        mock_ax.axis.assert_called_once_with("off")
        mock_ax.set_title.assert_called_once_with("Test Title", fontsize=14, pad=20)

    def test_create_confusion_matrix(self):
        """Test confusion matrix creation."""
        cm = [[10, 2], [3, 15]]
        fig = self.viz_service.create_confusion_matrix(cm)

        assert fig is not None
        assert fig.layout.title.text == "Confusion Matrix"

    def test_create_confusion_matrix_with_labels(self):
        """Test confusion matrix creation with custom labels."""
        cm = [[10, 2], [3, 15]]
        labels = ["Negative", "Positive"]
        fig = self.viz_service.create_confusion_matrix(cm, labels, "Custom Title")

        assert fig is not None
        assert fig.layout.title.text == "Custom Title"

    def test_create_metrics_display(self):
        """Test metrics display creation."""
        metrics = {"accuracy": 0.85, "f1_score": 0.82, "precision": 0.88}
        fig = self.viz_service.create_metrics_display(metrics, "Test Metrics")

        assert fig is not None
        assert fig.layout.title.text == "Test Metrics"
        # Should have 3 traces for 3 metrics
        assert len(fig.data) == 3

    def test_create_metrics_display_empty(self):
        """Test metrics display with empty metrics."""
        metrics = {}
        fig = self.viz_service.create_metrics_display(metrics)

        assert fig is not None
        assert len(fig.data) == 0  # No traces for empty metrics

    def test_get_accessibility_caption_all_types(self):
        """Test accessibility caption for all chart types."""
        chart_types = [
            "histogram",
            "bar",
            "wordcloud",
            "confusion_matrix",
            "probability",
            "unknown",
        ]

        for chart_type in chart_types:
            caption = self.viz_service.get_accessibility_caption(chart_type)
            assert "📊" in caption
            assert isinstance(caption, str)
            assert len(caption) > 0

    def test_get_accessibility_caption_with_data_info(self):
        """Test accessibility caption with additional data info."""
        caption = self.viz_service.get_accessibility_caption("histogram", "Sample size: 1000")
        assert "Sample size: 1000" in caption
        assert "📊" in caption

    def test_get_default_layout(self):
        """Test default layout generation."""
        layout = self.viz_service._get_default_layout()

        assert "plot_bgcolor" in layout
        assert "paper_bgcolor" in layout
        assert layout["plot_bgcolor"] == "white"
        assert layout["paper_bgcolor"] == "white"


if __name__ == "__main__":
    pytest.main([__file__])
