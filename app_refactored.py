"""
Refactored Sentiment Analysis DeBERTa v3 LoRA Streamlit Application
Following separation of concerns principle.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

# Import our modular services
from src.config import DEFAULT_MODEL_DIR, PAGE_CONFIG
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
    get_probability_breakdown_text,
    get_true_label_display,
    initialize_session_state,
    validate_csv_upload,
)
from src.visualization_service import VisualizationService

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title=PAGE_CONFIG["page_title"],
    page_icon=PAGE_CONFIG["page_icon"],
    layout="wide",  # Use literal string instead of variable
)
st.title("📊 Sentiment140 — Proof of Concept (DeBERTa v3 + LoRA merged)")


# ----------------------
# Initialize services
# ----------------------
@st.cache_resource(show_spinner=False)
def get_services():
    """Get cached service instances."""
    return {
        "model_service": ModelService(),
        "data_service": DataService(),
        "viz_service": VisualizationService(),
    }


services = get_services()
model_service = services["model_service"]
data_service = services["data_service"]
viz_service = services["viz_service"]

# ----------------------
# Initialize session state
# ----------------------
initialize_session_state()

# ----------------------
# Sidebar — model loading
# ----------------------
st.sidebar.header("⚙️ Model")
model_dir = st.sidebar.text_input("Path to 'merged' folder", value=DEFAULT_MODEL_DIR)

if st.sidebar.button("Load Model", type="primary"):
    try:
        model_service.load_model(model_dir)
        st.session_state.model_loaded = True
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.session_state.model_loaded = False
        st.sidebar.error(f"Loading failed: {e}")

if not st.session_state.model_loaded:
    st.info("🔌 Specify a model path then click **Load Model** in the sidebar.")
    st.stop()

# Display model info
display_model_info(model_service)

# ----------------------
# Tabs
# ----------------------
tabs = st.tabs(
    ["🔎 Exploratory Analysis", "📝 Predict Tweet", "📦 CSV Predictions", "📊 CSV Evaluation"]
)

# ---- Tab 1: Exploratory Data Analysis
with tabs[0]:
    st.subheader("Exploratory Data Analysis (EDA)")

    # Option to load Sentiment140 dataset
    st.subheader("Exploratory Analysis — Sentiment140 (via KaggleHub)")

    # Dataset loading options
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Option 1: Sample (recommended)**")
        sample_size = st.slider("Sample size", 10_000, 200_000, 50_000, step=10_000)
        if st.button("Load Sample", key="load_sample"):
            with st.spinner("Downloading and loading sample..."):
                df = data_service.load_sentiment140(sample_size=sample_size, load_full=False)
                st.session_state.sentiment140_df = df
                st.success(f"✅ Sample loaded with {len(df):,} rows")

    with col2:
        st.write("**Option 2: Full Dataset**")
        st.warning("⚠️ **Warning**: The full dataset has 1.6M rows (~1.6GB in memory)")
        if st.button("Load Full Dataset (1.6M rows)", key="load_full", type="primary"):
            with st.spinner(
                "Downloading and loading full dataset (this may take several minutes)..."
            ):
                df = data_service.load_sentiment140(sample_size=0, load_full=True)
                st.session_state.sentiment140_df = df
                st.success(f"✅ Full dataset loaded with {len(df):,} rows")

    # Check if Sentiment140 data is loaded
    if "sentiment140_df" in st.session_state and st.session_state.sentiment140_df is not None:
        df = st.session_state.sentiment140_df
        st.write("**Sentiment140 Dataset Preview:**")
        st.dataframe(df.head(10))

        # Process data for analysis
        df_processed = data_service.process_dataframe(df)

        # Global Statistics
        st.write("**Statistical Summary of Text Lengths**")
        stats = data_service.get_text_statistics(df_processed)
        st.write(df_processed["text_len"].describe())

        # Interactive Graphs
        fig_len = viz_service.create_text_length_histogram(df_processed)
        st.plotly_chart(fig_len, use_container_width=True)
        st.caption(viz_service.get_accessibility_caption("histogram"))

        # Distribution of labels
        label_counts = data_service.get_label_distribution(df_processed)
        fig_lab = viz_service.create_label_distribution_chart(label_counts)
        st.plotly_chart(fig_lab, use_container_width=True)
        st.caption(viz_service.get_accessibility_caption("bar"))

        # Top frequent words + WordCloud
        word_freq = data_service.get_word_frequency(df_processed, top_n=20)
        fig_freq = viz_service.create_word_frequency_chart(word_freq)
        st.plotly_chart(fig_freq, use_container_width=True)
        st.caption(viz_service.get_accessibility_caption("bar"))
        st.dataframe(word_freq, use_container_width=True)

        # WordCloud
        st.write("**WordCloud of tweets:**")
        st.caption(viz_service.get_accessibility_caption("wordcloud"))

        all_words = " ".join(df_processed["text"].astype(str).tolist()).lower().split()
        fig_wc = viz_service.create_wordcloud(" ".join(all_words))
        st.pyplot(fig_wc)

        # Alternative text for screen readers
        def get_wordcloud_caption():
            return (
                "🔍 **Alternative text for screen readers:**"
                "WordCloud visualization showing the most frequent words"
                "from the tweet dataset. "
                "Larger words indicate higher frequency. Top words include"
                "common terms like 'the', 'and', 'to', etc. "
                "See the frequency table above for exact counts."
            )

        st.caption(get_wordcloud_caption())

    # Original CSV upload functionality
    st.write(
        "**Or upload a sample of the dataset (CSV with a `text` column and optionally `label`):**"
    )
    up_eda = st.file_uploader("Import CSV for exploratory analysis", type=["csv"], key="csv_eda")
    if up_eda is not None:
        try:
            df = pd.read_csv(up_eda)
            st.write("Dataset preview:")
            st.dataframe(df.head(10))

            if "text" in df.columns:
                # Process and analyze uploaded data
                df_processed = data_service.process_dataframe(df)

                # Statistics on sentence length
                st.write("Statistics on sentence length:")
                st.dataframe(df_processed["text_len"].describe().to_frame())

                # Interactive graph: distribution of sentence lengths
                fig_len = viz_service.create_text_length_histogram(df_processed)
                st.plotly_chart(fig_len, use_container_width=True)
                st.caption(viz_service.get_accessibility_caption("histogram"))

                # Word frequency
                word_freq = data_service.get_word_frequency(df_processed, top_n=20)
                fig_freq = viz_service.create_word_frequency_chart(word_freq)
                st.plotly_chart(fig_freq, use_container_width=True)
                st.caption(viz_service.get_accessibility_caption("bar"))
                st.dataframe(word_freq, use_container_width=True)

                # WordCloud
                st.write("**WordCloud of tweets:**")
                st.caption(viz_service.get_accessibility_caption("wordcloud"))

                all_words = " ".join(df_processed["text"].astype(str).tolist()).lower().split()
                fig_wc = viz_service.create_wordcloud(" ".join(all_words))
                st.pyplot(fig_wc)

                def get_upload_caption():
                    return (
                        "🔍 **Alternative text for screen readers:** WordCloud visualization"
                        "showing the most frequent words from the uploaded dataset. "
                        "Larger words indicate higher frequency. "
                        "See the frequency table above for exact counts."
                    )

                st.caption(get_upload_caption())

            if "label" in df.columns:
                st.write("**Class count:**")
                label_counts = data_service.get_label_distribution(df_processed)
                fig_lab = viz_service.create_label_distribution_chart(label_counts)
                st.plotly_chart(fig_lab, use_container_width=True)
                st.caption(viz_service.get_accessibility_caption("bar"))

        except Exception as e:
            st.error(f"CSV reading error: {e}")

# ---- Tab 2: Single prediction
with tabs[1]:
    st.subheader("Predict Individual Tweet")

    # Refresh button for new random tweets
    if st.button("🔄 Refresh Random Tweets", key="refresh_tweets"):
        with st.spinner("Loading new random tweets..."):
            st.session_state.refresh_counter += 1
            st.session_state.random_tweets = data_service.load_random_tweets(30)
            st.session_state.tweet_selection = None
            st.success(
                f"✅ New random tweets loaded! (Refresh #{st.session_state.refresh_counter})"
            )

    # Load random tweets if not already loaded
    if st.session_state.random_tweets is None:
        with st.spinner("Loading random tweets..."):
            st.session_state.random_tweets = data_service.load_random_tweets(30)

    # Create dropdown with random tweets
    if st.session_state.random_tweets is not None:
        # Show first few tweet IDs to verify refresh is working
        first_ids = st.session_state.random_tweets["id"].head(5).tolist()
        st.caption(f"First 5 tweet IDs: {first_ids}")

        # Create display options with tweet ID and truncated text
        tweet_options = []
        for idx, row in st.session_state.random_tweets.iterrows():
            tweet_options.append(format_tweet_display(row["id"], row["text"]))

        # Dropdown selection
        selected_idx = st.selectbox(
            "Choose a tweet to predict:",
            range(len(tweet_options)),
            format_func=lambda x: tweet_options[x],
            key="tweet_dropdown",
        )

        if selected_idx is not None:
            selected_tweet = st.session_state.random_tweets.iloc[selected_idx]
            st.session_state.tweet_selection = selected_tweet["text"]

            # Display selected tweet details
            st.write("**Selected Tweet Details:**")
            st.write(f"**Tweet ID:** {selected_tweet['id']}")
            st.write(f"**Text:** {selected_tweet['text']}")
            true_label = get_true_label_display(selected_tweet["target"])
            st.write(f"**True Label:** {selected_tweet['target']} ({true_label})")

    # Prediction section
    st.write("---")
    st.write("**Make Prediction**")

    # Only predict if a tweet is selected
    if st.session_state.tweet_selection is not None:
        txt = st.session_state.tweet_selection
        st.write("**Selected Tweet for Prediction:**")
        st.write(f"*{txt}*")

        if st.button("🔮 Predict Sentiment", key="predict_one", type="primary"):
            probs, pred_id, label = model_service.predict_single(txt)

            # Display prediction results
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("Prediction", label)
            with col_pred2:
                confidence = max(probs)
                st.metric("Confidence", format_confidence_display(confidence))

            # Show probability breakdown
            st.write("**Probability Breakdown:**")
            prob_data = {"NEGATIVE": probs[0], "POSITIVE": probs[1]}
            fig_prob = viz_service.create_probability_chart(prob_data)
            st.plotly_chart(fig_prob, use_container_width=True)
            st.caption(get_probability_breakdown_text(probs))

            # Show true label and accuracy
            if st.session_state.random_tweets is not None:
                selected_tweet = st.session_state.random_tweets.iloc[
                    st.session_state.tweet_dropdown
                ]
                true_label = get_true_label_display(selected_tweet["target"])
                prediction_correct = pred_id == selected_tweet["target"]
                st.write(f"**True Label:** {true_label}")
                st.write(f"**Prediction Correct:** {'✅ Yes' if prediction_correct else '❌ No'}")
    else:
        st.info("👆 Please select a tweet from the dropdown above to make a prediction.")

# ---- Tab 3: Batch predictions
with tabs[2]:
    st.subheader("Batch Predictions from CSV")
    up = st.file_uploader("Import CSV", type=["csv"], key="csv_pred")
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.write("Preview:")
            st.dataframe(df.head(10))

            # Validate CSV
            is_valid, error_msg = validate_csv_upload(df, ["text"])
            if not is_valid:
                st.error(error_msg)
            else:
                text_col = st.selectbox("Text column", options=df.columns.tolist())
                if st.button("Run Predictions", key="predict_csv"):
                    texts = df[text_col].astype(str).tolist()
                    probs, preds = model_service.predict_batch(texts)
                    out = df.copy()
                    out["pred_label"] = model_service.to_label_name(preds)
                    out["prob_negative"] = probs[:, 0]
                    out["prob_positive"] = probs[:, 1]
                    st.dataframe(out.head(20))
                    create_download_button(out, "predictions.csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ---- Tab 4: Evaluation
with tabs[3]:
    st.subheader("Evaluate on Labeled CSV")
    up2 = st.file_uploader("Import Labeled CSV", type=["csv"], key="csv_eval")
    if up2 is not None:
        try:
            df = pd.read_csv(up2)
            st.dataframe(df.head(10))

            # Validate CSV
            is_valid, error_msg = validate_csv_upload(df, ["text", "label"])
            if not is_valid:
                st.error(error_msg)
            else:
                text_col = st.selectbox("Text column", options=df.columns.tolist(), key="text_eval")
                label_col = st.selectbox(
                    "Label column", options=df.columns.tolist(), key="label_eval"
                )
                if st.button("Evaluate", key="eval_btn"):
                    # Prepare labels (map NEGATIVE/POSITIVE to 0/1)
                    y_true = (
                        df[label_col]
                        .map({"NEGATIVE": 0, "POSITIVE": 1})
                        .fillna(df[label_col])
                        .astype(int)
                        .values
                    )
                    texts = df[text_col].astype(str).tolist()
                    probs, preds = model_service.predict_batch(texts)

                    # Calculate metrics
                    metrics = calculate_model_metrics(y_true, preds, probs)

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    with col2:
                        st.metric("F1-macro", f"{metrics['f1_macro']:.4f}")
                    with col3:
                        st.metric("AUC", f"{metrics['auc']:.4f}")

                    # Confusion matrix
                    cm = confusion_matrix(y_true, preds)
                    st.write("**Confusion Matrix:**")
                    fig_cm = viz_service.create_confusion_matrix(cm)
                    st.plotly_chart(fig_cm)
                    st.caption(get_confusion_matrix_text(cm))

                    # Export detailed results
                    detail = create_evaluation_results(df, preds, probs, model_service)
                    create_download_button(
                        detail, "eval_results.csv", "⬇️ Download Detailed Results (CSV)"
                    )
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.caption(get_accessibility_footer())
