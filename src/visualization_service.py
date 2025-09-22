"""
Visualization service for creating charts and plots.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

from .config import PLOT_COLORS, WCAG_COMPLIANT


class VisualizationService:
    """Service class for creating visualizations."""

    def __init__(self):
        self.default_layout = self._get_default_layout()

    def _get_default_layout(self) -> Dict[str, Any]:
        """Get default layout settings for WCAG compliance."""
        layout: Dict[str, Any] = {
            "plot_bgcolor": PLOT_COLORS["background"],
            "paper_bgcolor": PLOT_COLORS["background"],
        }

        if WCAG_COMPLIANT:
            layout.update(
                {
                    "font": {"color": "black"},
                    "xaxis": {"gridcolor": "lightgray"},
                    "yaxis": {"gridcolor": "lightgray"},
                }
            )

        return layout

    def create_text_length_histogram(
        self, df: pd.DataFrame, title: str = "Distribution of Sentence Lengths"
    ) -> go.Figure:
        """
        Create histogram for text length distribution.

        Args:
            df: DataFrame with text_len column
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = px.histogram(
            df,
            x="text_len",
            nbins=50,
            title=title,
            text_auto=True,
            color_discrete_sequence=[PLOT_COLORS["primary"]],
        )

        fig.update_layout(
            xaxis_title="Number of Words", yaxis_title="Frequency", **self.default_layout
        )

        return fig

    def create_label_distribution_chart(
        self, label_counts: pd.DataFrame, title: str = "Class Distribution"
    ) -> go.Figure:
        """
        Create bar chart for label distribution.

        Args:
            label_counts: DataFrame with label and count columns
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = px.bar(
            label_counts,
            x="label",
            y="count",
            title=title,
            text="count",
            color_discrete_sequence=[PLOT_COLORS["positive"], PLOT_COLORS["negative"]],
        )

        fig.update_layout(xaxis_title="Class", yaxis_title="Count", **self.default_layout)

        return fig

    def create_word_frequency_chart(
        self, word_freq: pd.DataFrame, title: str = "Top 20 Most Frequent Words"
    ) -> go.Figure:
        """
        Create bar chart for word frequency.

        Args:
            word_freq: DataFrame with word and freq columns
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = px.bar(
            word_freq,
            x="word",
            y="freq",
            title=title,
            text="freq",
            color_discrete_sequence=[PLOT_COLORS["primary"]],
        )

        fig.update_layout(xaxis_title="Word", yaxis_title="Frequency", **self.default_layout)

        return fig

    def create_wordcloud(
        self,
        text: str,
        title: str = "WordCloud: Most Frequent Words in Tweets",
        max_words: int = 100,
    ) -> plt.Figure:
        """
        Create wordcloud visualization.

        Args:
            text: Text to create wordcloud from
            title: Chart title
            max_words: Maximum number of words to display

        Returns:
            Matplotlib figure
        """
        # Create WordCloud with high contrast colors
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",  # Colorblind-friendly palette
            max_words=max_words,
            relative_scaling=0.5,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=20)

        return fig

    def create_confusion_matrix(
        self,
        cm: List[List[int]],
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
    ) -> go.Figure:
        """
        Create confusion matrix heatmap.

        Args:
            cm: Confusion matrix as 2D list
            labels: Class labels
            title: Chart title

        Returns:
            Plotly figure
        """
        if labels is None:
            labels = ["Negative", "Positive"]

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title=title,
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )

        fig.update_layout(**self.default_layout)

        return fig

    def create_probability_chart(
        self, probabilities: Dict[str, float], title: str = "Sentiment Probabilities"
    ) -> go.Figure:
        """
        Create bar chart for sentiment probabilities.

        Args:
            probabilities: Dictionary with sentiment labels and probabilities
            title: Chart title

        Returns:
            Plotly figure
        """
        labels = list(probabilities.keys())
        values = list(probabilities.values())

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                    marker_color=[PLOT_COLORS["negative"], PLOT_COLORS["positive"]],
                )
            ]
        )

        layout = self.default_layout.copy()
        layout.update(
            {
                "title": title,
                "xaxis_title": "Sentiment",
                "yaxis_title": "Probability",
                "yaxis": dict(range=[0, 1]),
            }
        )
        fig.update_layout(**layout)

        return fig

    def create_metrics_display(
        self, metrics: Dict[str, float], title: str = "Model Performance Metrics"
    ) -> go.Figure:
        """
        Create gauge charts for model metrics.

        Args:
            metrics: Dictionary with metric names and values
            title: Chart title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for i, (metric_name, value) in enumerate(metrics.items()):
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    domain={"x": [i / len(metrics), (i + 1) / len(metrics)], "y": [0, 1]},
                    title={"text": metric_name},
                    gauge={
                        "axis": {"range": [None, 1]},
                        "bar": {"color": PLOT_COLORS["primary"]},
                        "steps": [
                            {"range": [0, 0.5], "color": "lightgray"},
                            {"range": [0.5, 0.8], "color": "gray"},
                            {"range": [0.8, 1], "color": "darkgray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 0.9,
                        },
                    },
                )
            )

        fig.update_layout(title=title, **self.default_layout)

        return fig

    def get_accessibility_caption(self, chart_type: str, data_info: str = "") -> str:
        """
        Get accessibility caption for charts.

        Args:
            chart_type: Type of chart
            data_info: Additional data information

        Returns:
            Accessibility caption
        """
        captions = {
            "histogram": (
                "📊 Chart includes numerical values and labeled axes. "
                "Uses high-contrast colors for accessibility (WCAG AA compliant)."
            ),
            "bar": "📊 Bar chart with high-contrast colors and numerical labels for accessibility.",
            "wordcloud": (
                "📊 Visual representation of most frequent words. "
                "See the frequency table above for exact frequencies."
            ),
            "confusion_matrix": "📊 Confusion matrix showing prediction accuracy.",
            "probability": "📊 Probability chart with high-contrast colors for accessibility.",
        }

        base_caption = captions.get(chart_type, "📊 Chart with accessibility features.")

        if data_info:
            return f"{base_caption} {data_info}"

        return base_caption
