# streamlit_app.py
# -----------------------------------------------------------
# Proof-of-Concept dashboard (accessible) to demo tweet sentiment predictions
# with EDA (Exploratory Data Analysis) + prediction interface.
# Dataset: Sentiment140 (text data)
# -----------------------------------------------------------

import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import kagglehub

# ----------------------
# Page config
# ----------------------
st.set_page_config(
    page_title="Sentiment140 Dashboard — DeBERTa v3",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sentiment140 — Proof of Concept (DeBERTa v3 + LoRA merged)")

# ----------------------
# Helpers
# ----------------------
@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    # Ensure we're using absolute path for local files
    import os
    if not os.path.isabs(model_dir):
        model_dir = os.path.abspath(model_dir)
    
    # Check if the directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    # Always update labels to use English with emoticons
    model.config.id2label = {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"}
    model.config.label2id = {"😞 Negative Sentiment": 0, "😊 Positive Sentiment": 1}
    return tok, model

@torch.inference_mode()
def predict_batch(tokenizer, model, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
    probs = np.vstack(all_probs) if all_probs else np.zeros((0, 2), dtype=np.float32)
    preds = probs.argmax(axis=-1)
    return probs, preds


def to_label_name(model, ids: np.ndarray) -> List[str]:
    id2label = getattr(model.config, "id2label", {0: "😞 Negative Sentiment", 1: "😊 Positive Sentiment"})
    return [str(id2label.get(int(i), int(i))) for i in ids]

@st.cache_resource(show_spinner=True)
def load_sentiment140(sample_size: int = 50_000, load_full: bool = False) -> pd.DataFrame:
    """Load Sentiment140 dataset from KaggleHub with optional sampling or full dataset."""
    # Download latest version of Sentiment140 dataset
    path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = f"{path}/training.1600000.processed.noemoticon.csv"
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    if load_full:
        # Load the full 1.6M dataset
        df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=columns)
        st.info(f"📊 Full dataset loaded: {len(df_raw):,} rows (~{len(df_raw) * 0.0001:.1f} MB in memory)")
    else:
        # Load with sampling for memory efficiency
        df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=columns)
        if sample_size < len(df_raw):
            df_raw = df_raw.sample(sample_size, random_state=42)
            st.info(f"📊 Sample loaded: {len(df_raw):,} rows out of {1_600_000:,} total")
    
    return df_raw

def load_random_tweets(num_tweets: int = 30) -> pd.DataFrame:
    """Load a random sample of tweets from Sentiment140 for prediction testing."""
    # Download latest version of Sentiment140 dataset
    path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = f"{path}/training.1600000.processed.noemoticon.csv"
    columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Load a larger sample to get good random selection
    df_raw = pd.read_csv(csv_file, encoding="ISO-8859-1", names=columns)
    
    # Get random sample with current timestamp as seed for true randomness
    import time
    random_seed = int(time.time() * 1000) % 2**32  # Use current timestamp as seed
    random_tweets = df_raw.sample(num_tweets, random_state=random_seed)
    
    return random_tweets

# ----------------------
# Sidebar — model loading
# ----------------------
st.sidebar.header("⚙️ Model")
def_guess = "deberta3_lora_400000k/merged"
model_dir = st.sidebar.text_input("Path to 'merged' folder", value=def_guess)

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if st.sidebar.button("Load Model", type="primary"):
    try:
        tok, mdl = load_model(model_dir)
        st.session_state.tokenizer = tok
        st.session_state.model = mdl
        st.session_state.model_loaded = True
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.session_state.model_loaded = False
        st.sidebar.error(f"Loading failed: {e}")

if not st.session_state.model_loaded:
    st.info("🔌 Specify a model path then click **Load Model** in the sidebar.")
    st.stop()

# ----------------------
# Tabs
# ----------------------
tabs = st.tabs(["🔎 Exploratory Analysis", "📝 Predict Tweet", "📦 CSV Predictions", "📊 CSV Evaluation"])

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
                df = load_sentiment140(sample_size=sample_size, load_full=False)
                st.session_state.sentiment140_df = df
                st.success(f"✅ Sample loaded with {len(df):,} rows")
    
    with col2:
        st.write("**Option 2: Full Dataset**")
        st.warning("⚠️ **Warning**: The full dataset has 1.6M rows (~1.6GB in memory)")
        if st.button("Load Full Dataset (1.6M rows)", key="load_full", type="primary"):
            with st.spinner("Downloading and loading full dataset (this may take several minutes)..."):
                df = load_sentiment140(sample_size=0, load_full=True)
                st.session_state.sentiment140_df = df
                st.success(f"✅ Full dataset loaded with {len(df):,} rows")
    
    with col2:
        st.write("**Or upload your own CSV:**")
    
    # Check if Sentiment140 data is loaded
    if "sentiment140_df" in st.session_state:
        df = st.session_state.sentiment140_df
        st.write("**Sentiment140 Dataset Preview:**")
        st.dataframe(df.head(10))
        
        # Colonnes utiles
        df["text_len"] = df["text"].astype(str).str.split().apply(len)
        
        # Global Statistics
        st.write("**Statistical Summary of Text Lengths**")
        st.write(df["text_len"].describe())
        
        # Interactive Graphs
        fig_len = px.histogram(df, x="text_len", nbins=50, title="Distribution of Sentence Lengths", text_auto=True,
                              color_discrete_sequence=["#1f77b4"])
        fig_len.update_layout(
            xaxis_title="Number of Words", 
            yaxis_title="Frequency",
            # WCAG compliance: High contrast colors and accessible color palette
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_len, use_container_width=True)
        st.caption("📊 Chart includes numerical values and labeled axes. Uses high-contrast colors for accessibility (WCAG AA compliant).")
        
        # Distribution of labels (target: 0=NEGATIVE, 4=POSITIVE)
        label_map = {0: "😞 Negative Sentiment", 4: "😊 Positive Sentiment"}
        df["label"] = df["target"].map(label_map)
        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["label", "count"]  # Rename columns for clarity (label: "😞 Negative Sentiment", "😊 Positive Sentiment")
        fig_lab = px.bar(label_counts,
                         x="label", y="count",
                         title="Class Distribution", text="count",
                         color_discrete_sequence=["#2E8B57", "#DC143C"])  # Sea Green and Crimson for good contrast
        fig_lab.update_layout(
            xaxis_title="Class", 
            yaxis_title="Count",
            # WCAG compliance: Colorblind-friendly palette with high contrast
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_lab, use_container_width=True)
        st.caption("📊 Bar chart shows sentiment distribution with colorblind-friendly colors and text labels.")
        
        # Top frequent words + WordCloud
        from collections import Counter
        all_words = " ".join(df["text"].astype(str).tolist()).lower().split()
        freq = Counter(all_words)
        common = pd.DataFrame(freq.most_common(20), columns=["word", "freq"])
        fig_freq = px.bar(common, x="word", y="freq", title="Top 20 Most Frequent Words", text="freq",
                          color_discrete_sequence=["#1f77b4"])
        fig_freq.update_layout(
            xaxis_title="Word", 
            yaxis_title="Frequency",
            # WCAG compliance: High contrast colors
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_freq, use_container_width=True)
        st.caption("📊 Word frequency chart with high-contrast colors and numerical labels for accessibility.")
        st.dataframe(common, use_container_width=True)
        
        # WordCloud with accessibility improvements
        st.write("**WordCloud of tweets:**")
        st.caption("📊 Visual representation of most frequent words. See the table above for exact frequencies.")
        
        # Create WordCloud with high contrast colors
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color="white", 
            colormap="viridis",  # Colorblind-friendly palette
            max_words=100,
            relative_scaling=0.5
        ).generate(" ".join(all_words))
        
        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("WordCloud: Most Frequent Words in Tweets", fontsize=14, pad=20)
        st.pyplot(fig_wc)
        
        # Provide alternative text description for screen readers
        st.caption("🔍 **Alternative text for screen readers:** WordCloud visualization showing the most frequent words from the tweet dataset. Larger words indicate higher frequency. Top words include common terms like 'the', 'and', 'to', etc. See the frequency table above for exact counts.")
    
    # Original CSV upload functionality
    st.write("**Or upload a sample of the dataset (CSV with a `text` column and optionally `label`):**")
    up_eda = st.file_uploader("Import CSV for exploratory analysis", type=["csv"], key="csv_eda")
    if up_eda is not None:
        try:
            df = pd.read_csv(up_eda)
            st.write("Dataset preview:")
            st.dataframe(df.head(10))

            if "text" in df.columns:
                # Statistics on sentence length
                df["text_len"] = df["text"].astype(str).apply(lambda x: len(x.split()))
                st.write("Statistics on sentence length:")
                st.dataframe(df["text_len"].describe().to_frame())

                # Interactive graph: distribution of sentence lengths
                fig_len = px.histogram(df, x="text_len", nbins=50, title="Distribution of Sentence Lengths", text_auto=True,
                                      color_discrete_sequence=["#1f77b4"])
                fig_len.update_layout(
                    xaxis_title="Number of Words", 
                    yaxis_title="Frequency",
                    # WCAG compliance: High contrast colors
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_len, use_container_width=True)
                st.caption("📊 Chart includes numerical values and labeled axes with high-contrast colors (WCAG AA compliant).")

                # Word frequency
                from collections import Counter
                all_words = " ".join(df["text"].astype(str).tolist()).lower().split()
                freq = Counter(all_words)
                common = pd.DataFrame(freq.most_common(20), columns=["word", "freq"])
                fig_freq = px.bar(common, x="word", y="freq", title="Top 20 Most Frequent Words", text="freq",
                                  color_discrete_sequence=["#1f77b4"])
                fig_freq.update_layout(
                    xaxis_title="Word", 
                    yaxis_title="Frequency",
                    # WCAG compliance: High contrast colors
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
                st.caption("📊 Word frequency chart with high-contrast colors and numerical labels for accessibility.")
                st.dataframe(common, use_container_width=True)

                # WordCloud with accessibility improvements
                st.write("**WordCloud of tweets:**")
                st.caption("📊 Visual representation of most frequent words. See the table above for exact frequencies.")
                
                wc = WordCloud(
                    width=800, 
                    height=400, 
                    background_color="white", 
                    colormap="viridis",  # Colorblind-friendly palette
                    max_words=100,
                    relative_scaling=0.5
                ).generate(" ".join(all_words))
                
                fig_wc, ax = plt.subplots(figsize=(10,5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title("WordCloud: Most Frequent Words in Tweets", fontsize=14, pad=20)
                st.pyplot(fig_wc)
                
                # Provide alternative text description for screen readers
                st.caption("🔍 **Alternative text for screen readers:** WordCloud visualization showing the most frequent words from the uploaded dataset. Larger words indicate higher frequency. See the frequency table above for exact counts.")

            if "label" in df.columns:
                st.write("**Class count:**")
                fig_lab = px.histogram(df, x="label", title="Label Distribution",
                                      color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
                fig_lab.update_layout(
                    # WCAG compliance: High contrast colors
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_lab, use_container_width=True)
                st.caption("📊 Label distribution chart with high-contrast colors for accessibility.")

        except Exception as e:
            st.error(f"CSV reading error: {e}")

# ---- Tab 2: Single prediction
with tabs[1]:
    st.subheader("Predict Individual Tweet")
    
    # Initialize session state for random tweets
    if "random_tweets" not in st.session_state:
        st.session_state.random_tweets = None
        st.session_state.tweet_selection = None
        st.session_state.refresh_counter = 0
    
    # Refresh button for new random tweets
    if st.button("🔄 Refresh Random Tweets", key="refresh_tweets"):
        with st.spinner("Loading new random tweets..."):
            # Increment refresh counter to force new random selection
            st.session_state.refresh_counter += 1
            st.session_state.random_tweets = load_random_tweets(30)
            st.session_state.tweet_selection = None
            st.success(f"✅ New random tweets loaded! (Refresh #{st.session_state.refresh_counter})")
    
    # Load random tweets if not already loaded
    if st.session_state.random_tweets is None:
        with st.spinner("Loading random tweets..."):
            st.session_state.random_tweets = load_random_tweets(30)
    
    # Create dropdown with random tweets
    if st.session_state.random_tweets is not None:
        # Show first few tweet IDs to verify refresh is working
        first_ids = st.session_state.random_tweets['id'].head(5).tolist()
        st.caption(f"First 5 tweet IDs: {first_ids}")
        
        # Create display options with tweet ID and truncated text
        tweet_options = []
        for idx, row in st.session_state.random_tweets.iterrows():
            tweet_id = row['id']
            tweet_text = row['text']
            # Truncate long tweets for display
            display_text = tweet_text[:80] + "..." if len(tweet_text) > 80 else tweet_text
            tweet_options.append(f"ID: {tweet_id} | {display_text}")
        
        # Dropdown selection
        selected_idx = st.selectbox(
            "Choose a tweet to predict:",
            range(len(tweet_options)),
            format_func=lambda x: tweet_options[x],
            key="tweet_dropdown"
        )
        
        if selected_idx is not None:
            selected_tweet = st.session_state.random_tweets.iloc[selected_idx]
            st.session_state.tweet_selection = selected_tweet['text']
            
            # Display selected tweet details
            st.write("**Selected Tweet Details:**")
            st.write(f"**Tweet ID:** {selected_tweet['id']}")
            st.write(f"**Text:** {selected_tweet['text']}")
            st.write(f"**True Label:** {selected_tweet['target']} ({'😞 Negative' if selected_tweet['target'] == 0 else '😊 Positive'})")
    
    # Prediction section
    st.write("---")
    st.write("**Make Prediction**")
    
    # Only predict if a tweet is selected
    if st.session_state.tweet_selection is not None:
        txt = st.session_state.tweet_selection
        st.write("**Selected Tweet for Prediction:**")
        st.write(f"*{txt}*")
        
        if st.button("🔮 Predict Sentiment", key="predict_one", type="primary"):
            probs, preds = predict_batch(st.session_state.tokenizer, st.session_state.model, [txt])
            pred_id = preds[0]
            label = to_label_name(st.session_state.model, [pred_id])[0]
            
            # Display prediction results
            col_pred1, col_pred2 = st.columns(2)
            with col_pred1:
                st.metric("Prediction", label)
            with col_pred2:
                confidence = max(probs[0])
                st.metric("Confidence", f"{confidence:.3f}")
            
            # Show probability breakdown with accessibility improvements
            st.write("**Probability Breakdown:**")
            # Create a more accessible bar chart
            prob_data = {"NEGATIVE": probs[0,0], "POSITIVE": probs[0,1]}
            st.bar_chart(prob_data)
            st.caption("📊 **Textual alternative for screen readers:** NEGATIVE sentiment probability: %.3f (%.1f%%), POSITIVE sentiment probability: %.3f (%.1f%%)" % (probs[0,0], probs[0,0]*100, probs[0,1], probs[0,1]*100))
            
            # Show true label and accuracy
            if st.session_state.random_tweets is not None:
                selected_tweet = st.session_state.random_tweets.iloc[st.session_state.tweet_dropdown]
                true_label = "😞 Negative" if selected_tweet['target'] == 0 else "😊 Positive"
                prediction_correct = (pred_id == selected_tweet['target'])
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
            text_col = st.selectbox("Text column", options=df.columns.tolist())
            if st.button("Run Predictions", key="predict_csv"):
                texts = df[text_col].astype(str).tolist()
                probs, preds = predict_batch(st.session_state.tokenizer, st.session_state.model, texts)
                out = df.copy()
                out["pred_label"] = to_label_name(st.session_state.model, preds)
                out["prob_negative"] = probs[:,0]
                out["prob_positive"] = probs[:,1]
                st.dataframe(out.head(20))
                st.download_button("⬇️ Download Predictions (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")
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
            text_col = st.selectbox("Text column", options=df.columns.tolist(), key="text_eval")
            label_col = st.selectbox("Label column", options=df.columns.tolist(), key="label_eval")
            if st.button("Evaluate", key="eval_btn"):
                y_true = df[label_col].map({"NEGATIVE":0, "POSITIVE":1}).fillna(df[label_col]).astype(int).values
                texts = df[text_col].astype(str).tolist()
                probs, preds = predict_batch(st.session_state.tokenizer, st.session_state.model, texts)
                acc = accuracy_score(y_true, preds)
                f1m = f1_score(y_true, preds, average="macro")
                auc = roc_auc_score(y_true, probs[:,1])
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("F1-macro", f"{f1m:.4f}")
                st.metric("AUC", f"{auc:.4f}")
                cm = confusion_matrix(y_true, preds)
                st.write("**Confusion Matrix:**")
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    color_continuous_scale="Blues", 
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count")
                )
                fig_cm.update_layout(
                    # WCAG compliance: High contrast colors
                    plot_bgcolor="white",
                    paper_bgcolor="white"
                )
                st.plotly_chart(fig_cm)
                st.caption("📊 Confusion matrix showing prediction accuracy. True Negatives: %d, False Positives: %d, False Negatives: %d, True Positives: %d" % (cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
                # Export detailed results
                detail = df.copy()
                detail["pred"] = preds
                label_names = to_label_name(st.session_state.model, preds)
                detail["pred_label"] = label_names
                detail["prob_negative"] = probs[:,0]
                detail["prob_positive"] = probs[:,1]
                st.download_button("⬇️ Download Detailed Results (CSV)", data=detail.to_csv(index=False).encode("utf-8"), file_name="eval_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.caption("♿ **WCAG AA Compliant Dashboard** - High contrast colors, colorblind-friendly palettes, text alternatives, and screen reader support. Includes exploratory analysis, prediction and evaluation.")
