import streamlit as st
import joblib
import os
import pandas as pd
import datetime
from classifier import predict, train

# Page config
st.set_page_config(
    page_title="Complaint Priority Classifier",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .high-box {
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border-left: 4px solid #ff4b4b;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }
    .medium-box {
        background: linear-gradient(135deg, #ffa50022, #ffa50011);
        border-left: 4px solid #ffa500;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }
    .low-box {
        background: linear-gradient(135deg, #00c85222, #00c85211);
        border-left: 4px solid #00c852;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }
    .metric-card {
        background: #1e2130;
        padding: 15px; border-radius: 10px;
        text-align: center; margin: 5px;
    }
    .stTextArea textarea {
        background-color: #1e2130 !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🎯 Complaint Priority Classifier")
st.markdown("**AI-powered NLP system** that classifies complaints into High, Medium, or Low priority using TF-IDF + Naive Bayes")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown("""
    **Model:** Multinomial Naive Bayes  
    **Features:** TF-IDF (1-2 ngrams)  
    **Accuracy:** 91.67%  
    **Classes:** High | Medium | Low
    """)
    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "91.67%")
        st.metric("Precision", "93%")
    with col2:
        st.metric("Recall", "92%")
        st.metric("F1 Score", "92%")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Classify", "📂 Batch Classify", "🕐 History", "📊 Analytics", "📈 Model Info"])

# ── Tab 1: Single Classification ──
with tab1:
    st.markdown("### Enter a complaint to classify")

    examples = {
        "🔴 High Example": "The entire system is down and no one can log in!",
        "🟡 Medium Example": "The dashboard loads slowly on some pages",
        "🟢 Low Example": "There is a small typo in the footer text"
    }

    selected = st.selectbox("Or pick an example:", ["-- Type your own --"] + list(examples.keys()))

    if selected != "-- Type your own --":
        default_text = examples[selected]
    else:
        default_text = ""

    complaint_text = st.text_area(
        "Complaint Text:",
        value=default_text,
        height=120,
        placeholder="e.g. The payment system is completely broken for all users..."
    )

    translate_enabled = st.toggle(
        "🌐 Enable Multilingual Support (auto-translate to English)",
        key="translate_toggle"
    )

    if st.button("🚀 Classify Complaint", type="primary", use_container_width=True):
        if complaint_text.strip():
            with st.spinner("Analyzing complaint..."):
                result = predict(complaint_text, translate=translate_enabled)

            priority   = result["priority"]
            confidence = result["confidence"]
            scores     = result["scores"]

            emoji     = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
            box_class = {"high": "high-box", "medium": "medium-box", "low": "low-box"}[priority]

            st.markdown(f"""
            <div class="{box_class}">
                <h2>{emoji} {priority.upper()} PRIORITY</h2>
                <p>Confidence: <strong>{confidence:.0%}</strong></p>
                <p><em>"{complaint_text[:100]}..."</em></p>
            </div>
            """, unsafe_allow_html=True)

            # Show translation if used
            if translate_enabled and result.get("translated_text"):
                st.info(f"🌐 Translated to: *{result['translated_text']}*")

            st.markdown("#### 📊 Confidence Breakdown")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 High", f"{scores.get('high', 0):.1%}")
                st.progress(scores.get('high', 0))
            with col2:
                st.metric("🟡 Medium", f"{scores.get('medium', 0):.1%}")
                st.progress(scores.get('medium', 0))
            with col3:
                st.metric("🟢 Low", f"{scores.get('low', 0):.1%}")
                st.progress(scores.get('low', 0))

            # Save to history
            st.session_state.history.append({
                "Time": datetime.datetime.now().strftime("%H:%M:%S"),
                "Complaint": complaint_text[:60] + "..." if len(complaint_text) > 60 else complaint_text,
                "Priority": priority.upper(),
                "Confidence": f"{confidence:.0%}"
            })

        else:
            st.warning("⚠️ Please enter a complaint text first!")

# ── Tab 2: Batch Classification ──
with tab2:
    st.markdown("### Classify multiple complaints at once")

    input_method = st.radio(
        "Choose input method:",
        ["✍️ Type manually", "📁 Upload CSV file"],
        horizontal=True
    )

    if input_method == "✍️ Type manually":
        st.markdown("Enter one complaint per line:")
        batch_text = st.text_area(
            "Complaints (one per line):",
            height=200,
            placeholder="System is down\nMinor typo in footer\nLogin sometimes fails"
        )

        if st.button("🚀 Classify All", type="primary", use_container_width=True):
            if batch_text.strip():
                complaints = [c.strip() for c in batch_text.strip().split("\n") if c.strip()]
                results = []

                with st.spinner(f"Classifying {len(complaints)} complaints..."):
                    for c in complaints:
                        r = predict(c)
                        results.append({
                            "Complaint": c[:60] + "..." if len(c) > 60 else c,
                            "Priority": r["priority"].upper(),
                            "Confidence": f"{r['confidence']:.0%}",
                            "Emoji": {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
                        })

                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                st.markdown("#### 📊 Summary")
                col1, col2, col3 = st.columns(3)
                high_count = sum(1 for r in results if r["Priority"] == "HIGH")
                med_count  = sum(1 for r in results if r["Priority"] == "MEDIUM")
                low_count  = sum(1 for r in results if r["Priority"] == "LOW")
                col1.metric("🔴 High", high_count)
                col2.metric("🟡 Medium", med_count)
                col3.metric("🟢 Low", low_count)
            else:
                st.warning("⚠️ Please enter at least one complaint!")

    else:
        st.markdown("Upload a CSV file with a column containing complaint text:")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown("#### Preview (first 5 rows):")
            st.dataframe(df_upload.head(), use_container_width=True)

            text_columns = df_upload.columns.tolist()
            selected_col = st.selectbox("Select the complaint text column:", text_columns)

            if st.button("🚀 Classify All Rows", type="primary", use_container_width=True):
                complaints = df_upload[selected_col].dropna().tolist()
                results = []

                with st.spinner(f"Classifying {len(complaints)} complaints..."):
                    for c in complaints:
                        r = predict(str(c))
                        results.append({
                            "Complaint": str(c)[:60] + "..." if len(str(c)) > 60 else str(c),
                            "Priority": r["priority"].upper(),
                            "Confidence": f"{r['confidence']:.0%}",
                            "Emoji": {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
                        })

                df_results = pd.DataFrame(results)
                st.markdown("#### 🎯 Classification Results:")
                st.dataframe(df_results, use_container_width=True)

                st.markdown("#### 📊 Summary")
                col1, col2, col3 = st.columns(3)
                high_count = sum(1 for r in results if r["Priority"] == "HIGH")
                med_count  = sum(1 for r in results if r["Priority"] == "MEDIUM")
                low_count  = sum(1 for r in results if r["Priority"] == "LOW")
                col1.metric("🔴 High", high_count)
                col2.metric("🟡 Medium", med_count)
                col3.metric("🟢 Low", low_count)

                st.markdown("#### ⬇️ Download Results")
                csv_download = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_download,
                    file_name="classified_complaints.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ── Tab 3: History ──
with tab3:
    st.markdown("### 🕐 Classification History")

    if len(st.session_state.history) == 0:
        st.info("No complaints classified yet. Go to Single Classify tab and start!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total  = len(st.session_state.history)
        high_c = sum(1 for h in st.session_state.history if h["Priority"] == "HIGH")
        med_c  = sum(1 for h in st.session_state.history if h["Priority"] == "MEDIUM")
        low_c  = sum(1 for h in st.session_state.history if h["Priority"] == "LOW")

        col1.metric("📋 Total", total)
        col2.metric("🔴 High", high_c)
        col3.metric("🟡 Medium", med_c)
        col4.metric("🟢 Low", low_c)

        st.markdown("---")

        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()

# ── Tab 4: Analytics ──
with tab4:
    st.markdown("### 📊 Live Analytics Dashboard")

    if len(st.session_state.history) == 0:
        st.info("No data yet! Classify some complaints first to see analytics.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        st.markdown("#### Overview")
        col1, col2, col3, col4 = st.columns(4)
        total  = len(df_hist)
        high_c = len(df_hist[df_hist["Priority"] == "HIGH"])
        med_c  = len(df_hist[df_hist["Priority"] == "MEDIUM"])
        low_c  = len(df_hist[df_hist["Priority"] == "LOW"])
        col1.metric("📋 Total Classified", total)
        col2.metric("🔴 High Priority", high_c)
        col3.metric("🟡 Medium Priority", med_c)
        col4.metric("🟢 Low Priority", low_c)

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Priority Distribution")
            priority_counts = df_hist["Priority"].value_counts()
            st.bar_chart(priority_counts)

        with col2:
            st.markdown("#### Classification Timeline")
            df_hist["Index"] = range(1, len(df_hist) + 1)
            priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            df_hist["Priority Score"] = df_hist["Priority"].map(priority_map)
            st.line_chart(df_hist.set_index("Index")["Priority Score"])

        st.markdown("---")

        st.markdown("#### Priority Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            pct = f"{(high_c/total*100):.1f}%"
            st.markdown(f"""
            <div style="background:#ff4b4b22;border-left:4px solid #ff4b4b;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#ff4b4b">🔴 {pct}</h3>
            <p>High Priority</p></div>
            """, unsafe_allow_html=True)
        with col2:
            pct = f"{(med_c/total*100):.1f}%"
            st.markdown(f"""
            <div style="background:#ffa50022;border-left:4px solid #ffa500;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#ffa500">🟡 {pct}</h3>
            <p>Medium Priority</p></div>
            """, unsafe_allow_html=True)
        with col3:
            pct = f"{(low_c/total*100):.1f}%"
            st.markdown(f"""
            <div style="background:#00c85222;border-left:4px solid #00c852;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#00c852">🟢 {pct}</h3>
            <p>Low Priority</p></div>
            """, unsafe_allow_html=True)

# ── Tab 5: Model Info ──
with tab5:
    st.markdown("### 🧠 How the Model Works")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Pipeline
                    Raw Text
        ↓
    TF-IDF Vectorizer
    (1-2 ngrams, 5000 features)
        ↓
    Multinomial Naive Bayes
    (alpha=0.1)
        ↓
    Priority + Confidence
        ↓
    Keyword Hybrid Fallback
    (if confidence < 60%)
        ↓
    🔴 High / 🟡 Medium / 🟢 Low
                    """)
    with col2:
        st.markdown("""
        #### Performance Metrics
        | Class | Precision | Recall | F1 |
        |-------|-----------|--------|----|
        | 🔴 High | 100% | 75% | 86% |
        | 🟡 Medium | 80% | 100% | 89% |
        | 🟢 Low | 100% | 100% | 100% |
        | **Overall** | **93%** | **92%** | **92%** |

        #### Dataset
        - 178 labeled complaints
        - 60 High / 58 Medium / 60 Low
        - 80% train / 20% test split
        """)