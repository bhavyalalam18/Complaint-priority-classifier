import streamlit as st
import joblib
import os
import pandas as pd
import datetime
import csv
import random
import re
import smtplib
import io as _io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import Counter
from classifier import predict, train

# ── Initialize ALL session state FIRST ──
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "feedback_saved" not in st.session_state:
    st.session_state.feedback_saved = False

# ── Helper Functions ──
def send_email_alert(to_email: str, complaint_text: str, confidence: float):
    try:
        sender_email    = "bhavyalalam.4@gmail.com"
        sender_password = "pzow pucd rzzi btry"
        msg             = MIMEMultipart("alternative")
        msg["Subject"]  = "🔴 HIGH PRIORITY Complaint Detected!"
        msg["From"]     = sender_email
        msg["To"]       = to_email
        html = f"""
        <html><body>
        <div style="font-family:Arial;max-width:600px;margin:auto;
                    border:2px solid #ff4b4b;border-radius:10px;padding:20px;">
            <h2 style="color:#ff4b4b;">🔴 HIGH PRIORITY Alert</h2>
            <p><strong>A high priority complaint was detected:</strong></p>
            <div style="background:#fee2e2;padding:15px;border-radius:8px;">
                <p><em>"{complaint_text[:200]}..."</em></p>
            </div>
            <p>Confidence: <strong>{confidence:.0%}</strong></p>
            <p style="color:#666;font-size:12px;">Sent by Complaint Priority Classifier</p>
        </div>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        return True
    except Exception:
        return False

def summarize_complaint(text: str) -> str:
    sentences = text.replace("!", ".").replace("?", ".").split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return text[:100]
    return sentences[0] if len(sentences[0]) > 20 else " ".join(sentences[:2])

def get_keywords(text: str) -> list:
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "have", "has", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "shall", "can", "need",
                  "for", "on", "at", "to", "from", "in", "out", "and", "or",
                  "but", "not", "with", "this", "that", "it", "we", "our", "i"}
    words    = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    return [word for word, _ in Counter(keywords).most_common(5)]
def get_severity_score(text: str, priority: str, confidence: float) -> int:
    """Calculate severity score 1-10 based on priority + confidence + keywords."""
    base = {"high": 7, "medium": 4, "low": 1}[priority]
    bonus = 0
    text_lower = text.lower()
    critical_words = ["crash", "breach", "down", "all users", "production",
                      "critical", "emergency", "data loss", "corrupted"]
    bonus += sum(1 for w in critical_words if w in text_lower)
    score = base + min(bonus, 2) + round((confidence - 0.6) * 2)
    return max(1, min(10, score))
def get_response_template(priority: str, severity: int) -> str:
    """Generate a response template based on priority and severity."""
    templates = {
        "high": f"""Dear Customer,

Thank you for reaching out to us. We have received your complaint and want to assure you that this is being treated as a **critical priority** (Severity: {severity}/10).

Our technical team has been immediately notified and is actively investigating this issue. We understand the urgency and impact this is having on your operations.

**Immediate Actions Taken:**
- Issue escalated to senior engineering team
- Investigation started — ETA for update: 1 hour
- All relevant teams have been alerted

We will provide you with a status update within **1 hour**. We sincerely apologize for the inconvenience caused.

Best regards,
Support Team""",

        "medium": f"""Dear Customer,

Thank you for contacting our support team. We have received your complaint (Severity: {severity}/10) and it has been assigned to our support team for investigation.

**Next Steps:**
- Our team will investigate the issue
- You can expect a resolution within **24-48 hours**
- We will keep you updated on progress

We apologize for any inconvenience this may have caused.

Best regards,
Support Team""",

        "low": f"""Dear Customer,

Thank you for your feedback (Severity: {severity}/10). We appreciate you taking the time to report this to us.

Your feedback has been logged and will be reviewed by our product team in our next sprint cycle.

**Timeline:** 
- This will be addressed in our upcoming release
- Expected resolution: **1-2 weeks**

Thank you for helping us improve our product!

Best regards,
Support Team"""
    }
    return templates[priority]
def save_feedback(text: str, predicted: str, feedback: str):
    filepath   = "data/feedback.csv"
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "predicted", "feedback", "time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "text"     : text,
            "predicted": predicted,
            "feedback" : feedback,
            "time"     : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

# ── Page Config ──
st.set_page_config(
    page_title="Complaint Priority Classifier",
    page_icon="🎯",
    layout="wide"
)

# ── Dynamic CSS ──
if st.session_state.dark_mode:
    theme_bg       = "#0f1117"
    theme_card     = "#1e2130"
    theme_text     = "#ffffff"
    theme_border   = "#333333"
    theme_input_bg = "#1e2130"
else:
    theme_bg       = "#ffffff"
    theme_card     = "#f0f2f6"
    theme_text     = "#1a1a2e"
    theme_border   = "#dddddd"
    theme_input_bg = "#f8f9fa"

st.markdown(f"""
<style>
    .main {{ background-color: {theme_bg}; }}
    .high-box {{
        background: linear-gradient(135deg, #ff4b4b22, #ff4b4b11);
        border-left: 4px solid #ff4b4b;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }}
    .medium-box {{
        background: linear-gradient(135deg, #ffa50022, #ffa50011);
        border-left: 4px solid #ffa500;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }}
    .low-box {{
        background: linear-gradient(135deg, #00c85222, #00c85211);
        border-left: 4px solid #00c852;
        padding: 20px; border-radius: 10px; margin: 10px 0;
    }}
    .metric-card {{
        background: {theme_card};
        padding: 15px; border-radius: 10px;
        text-align: center; margin: 5px;
    }}
    .stTextArea textarea {{
        background-color: {theme_input_bg} !important;
        color: {theme_text} !important;
        border: 1px solid {theme_border} !important;
        border-radius: 8px !important;
    }}
    body {{ color: {theme_text} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("# 🎯 Complaint Priority Classifier")
st.markdown("**AI-powered NLP system** that classifies complaints into High, Medium, or Low priority using TF-IDF + Naive Bayes")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ About")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_mode_toggle")
    st.session_state.dark_mode = dark_mode
    st.markdown("""
    **Model:** Multinomial Naive Bayes  
    **Features:** TF-IDF (1-2 ngrams)  
    **Accuracy:** 91.67%  
    **Classes:** High | Medium | Low
    """)
    st.markdown("---")
    st.markdown("## 📧 Email Alerts")
    alert_email    = st.text_input("Alert email address:", placeholder="you@gmail.com", key="alert_email")
    alert_enabled  = st.toggle("🔔 Enable High Priority Alerts", key="email_alert_toggle")
    if alert_enabled and not alert_email:
        st.warning("⚠️ Enter an email address!")
    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "91.67%")
        st.metric("Precision", "93%")
    with col2:
        st.metric("Recall", "92%")
        st.metric("F1 Score", "92%")

# ── Main Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Single Classify", "📂 Batch Classify", "🕐 History",
    "📊 Analytics", "📈 Model Info", "🔄 Retrain", "📝 Summarizer"
])

# ── Tab 1: Single Classification ──
with tab1:
    st.markdown("### Enter a complaint to classify")

    high_examples   = [
        "The entire system is down and no one can log in!",
        "Critical security breach detected in production database!",
        "Payment gateway is completely broken for all customers",
        "All users are locked out, urgent recovery needed",
        "Server crashed and all data is inaccessible",
    ]
    medium_examples = [
        "The dashboard loads slowly on some pages",
        "Email notifications are delayed by a few hours",
        "Search results occasionally showing incorrect data",
        "Mobile app freezes sometimes on Android devices",
        "Password reset email not received by some users",
    ]
    low_examples    = [
        "There is a small typo in the footer text",
        "Button color doesn't match our brand guidelines",
        "Suggestion to add dark mode to the app",
        "Minor spacing issue on the contact form",
        "Tooltip text on settings page could be clearer",
    ]
    examples = {
        "🔴 High Example"  : random.choice(high_examples),
        "🟡 Medium Example": random.choice(medium_examples),
        "🟢 Low Example"   : random.choice(low_examples),
    }

    selected = st.selectbox("Or pick an example:", ["-- Type your own --"] + list(examples.keys()))
    default_text = examples[selected] if selected != "-- Type your own --" else ""

    complaint_text = st.text_area(
        "Complaint Text:", value=default_text, height=120,
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
            emoji      = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
            box_class  = {"high": "high-box", "medium": "medium-box", "low": "low-box"}[priority]

            st.markdown(f"""
            <div class="{box_class}">
                <h2>{emoji} {priority.upper()} PRIORITY</h2>
                <p>Confidence: <strong>{confidence:.0%}</strong></p>
                <p><em>"{complaint_text[:100]}..."</em></p>
            </div>
            """, unsafe_allow_html=True)
            # Severity score
            severity = get_severity_score(complaint_text, priority, confidence)
            severity_color = "#ff4b4b" if severity >= 7 else "#ffa500" if severity >= 4 else "#00c852"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:15px;
                        padding:10px;border-radius:8px;background:{severity_color}22;
                        border-left:4px solid {severity_color};margin:10px 0;">
                <h2 style="color:{severity_color};margin:0;">⚡ {severity}/10</h2>
                <p style="margin:0;">Severity Score</p>
            </div>
            """, unsafe_allow_html=True)

            if translate_enabled and result.get("translated_text"):
                st.info(f"🌐 Translated to: *{result['translated_text']}*")

            if priority == "high" and alert_enabled and alert_email:
                with st.spinner("📧 Sending alert email..."):
                    sent = send_email_alert(alert_email, complaint_text, confidence)
                if sent:
                    st.success(f"📧 Alert email sent to {alert_email}!")
                else:
                    st.error("❌ Email failed. Check Gmail App Password setup.")

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

            severity = get_severity_score(complaint_text, priority, confidence)
            # Response template
            with st.expander("📨 Generate Response Template"):
                template = get_response_template(priority, severity)
                st.text_area("Copy this response:", value=template, 
                            height=200, key="response_template")
                st.download_button(
                    "⬇️ Download Template",
                    data=template,
                    file_name=f"response_{priority}_priority.txt",
                    mime="text/plain"
                )
            st.session_state.history.append({
                "Time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                "Complaint" : complaint_text[:60] + "..." if len(complaint_text) > 60 else complaint_text,
                "Priority"  : priority.upper(),
                "Confidence": f"{confidence:.0%}",
                "Severity"  : f"{severity}/10"
            })
            st.session_state.last_result  = {"text": complaint_text, "priority": priority}
            st.session_state.feedback_saved = False

        else:
            st.warning("⚠️ Please enter a complaint text first!")

    if st.session_state.last_result and not st.session_state.feedback_saved:
        st.markdown("---")
        st.markdown("#### 💬 Was this classification correct?")
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("👍 Yes, correct!", key="feedback_yes"):
                save_feedback(st.session_state.last_result["text"],
                              st.session_state.last_result["priority"], "correct")
                st.session_state.feedback_saved = True
                st.success("✅ Thanks for your feedback!")
                st.rerun()
        with col2:
            if st.button("👎 No, wrong!", key="feedback_no"):
                save_feedback(st.session_state.last_result["text"],
                              st.session_state.last_result["priority"], "incorrect")
                st.session_state.feedback_saved = True
                st.warning("📝 Feedback saved! We'll improve the model.")
                st.rerun()

    if st.session_state.feedback_saved:
        st.markdown("---")
        st.info("✅ Feedback recorded! Classify another complaint.")

# ── Tab 2: Batch Classification ──
with tab2:
    st.markdown("### Classify multiple complaints at once")
    input_method = st.radio("Choose input method:",
                            ["✍️ Type manually", "📁 Upload CSV file"], horizontal=True)

    if input_method == "✍️ Type manually":
        batch_text = st.text_area("Complaints (one per line):", height=200,
                                   placeholder="System is down\nMinor typo in footer\nLogin sometimes fails")
        if st.button("🚀 Classify All", type="primary", use_container_width=True):
            if batch_text.strip():
                complaints = [c.strip() for c in batch_text.strip().split("\n") if c.strip()]
                results    = []
                with st.spinner(f"Classifying {len(complaints)} complaints..."):
                    for c in complaints:
                        r = predict(c)
                        results.append({
                            "Complaint": c[:60] + "..." if len(c) > 60 else c,
                            "Priority" : r["priority"].upper(),
                            "Confidence": f"{r['confidence']:.0%}",
                            "Emoji"    : {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
                        })
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                st.markdown("#### 📊 Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 High",   sum(1 for r in results if r["Priority"] == "HIGH"))
                col2.metric("🟡 Medium", sum(1 for r in results if r["Priority"] == "MEDIUM"))
                col3.metric("🟢 Low",    sum(1 for r in results if r["Priority"] == "LOW"))
            else:
                st.warning("⚠️ Please enter at least one complaint!")
    else:
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.dataframe(df_upload.head(), use_container_width=True)
            selected_col = st.selectbox("Select the complaint text column:", df_upload.columns.tolist())
            if st.button("🚀 Classify All Rows", type="primary", use_container_width=True):
                complaints = df_upload[selected_col].dropna().tolist()
                results    = []
                with st.spinner(f"Classifying {len(complaints)} complaints..."):
                    for c in complaints:
                        r = predict(str(c))
                        results.append({
                            "Complaint": str(c)[:60] + "..." if len(str(c)) > 60 else str(c),
                            "Priority" : r["priority"].upper(),
                            "Confidence": f"{r['confidence']:.0%}",
                            "Emoji"    : {"high": "🔴", "medium": "🟡", "low": "🟢"}[r["priority"]]
                        })
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                st.markdown("#### 📊 Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 High",   sum(1 for r in results if r["Priority"] == "HIGH"))
                col2.metric("🟡 Medium", sum(1 for r in results if r["Priority"] == "MEDIUM"))
                col3.metric("🟢 Low",    sum(1 for r in results if r["Priority"] == "LOW"))
                csv_download = df_results.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results as CSV", data=csv_download,
                                   file_name="classified_complaints.csv", mime="text/csv",
                                   use_container_width=True)


# ── Tab 3: History ──
with tab3:
    st.markdown("### 🕐 Classification History")
    if len(st.session_state.history) == 0:
        st.info("No complaints classified yet. Go to Single Classify tab and start!")
    else:
        total  = len(st.session_state.history)
        high_c = sum(1 for h in st.session_state.history if h["Priority"] == "HIGH")
        med_c  = sum(1 for h in st.session_state.history if h["Priority"] == "MEDIUM")
        low_c  = sum(1 for h in st.session_state.history if h["Priority"] == "LOW")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📋 Total", total)
        col2.metric("🔴 High", high_c)
        col3.metric("🟡 Medium", med_c)
        col4.metric("🟢 Low", low_c)
        st.markdown("---")

        # Search & Filter
        st.markdown("#### 🔍 Search & Filter")
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("🔍 Search complaints:", 
                                          placeholder="Type to search...",
                                          key="history_search")
        with col2:
            filter_priority = st.selectbox("Filter by priority:",
                                            ["All", "HIGH", "MEDIUM", "LOW"],
                                            key="history_filter")

        df_history = pd.DataFrame(st.session_state.history)

        # Apply search
        if search_query:
            df_history = df_history[
                df_history["Complaint"].str.contains(search_query, case=False, na=False)
            ]
        # Apply filter
        if filter_priority != "All":
            df_history = df_history[df_history["Priority"] == filter_priority]

        st.markdown(f"**Showing {len(df_history)} of {total} records**")
        st.dataframe(df_history, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        with col2:
            if len(df_history) > 0:
                csv_hist = df_history.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download History",
                    data=csv_hist,
                    file_name="complaint_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ── Tab 4: Analytics ──
with tab4:
    st.markdown("### 📊 Live Analytics Dashboard")
    if len(st.session_state.history) == 0:
        st.info("No data yet! Classify some complaints first to see analytics.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        total   = len(df_hist)
        high_c  = len(df_hist[df_hist["Priority"] == "HIGH"])
        med_c   = len(df_hist[df_hist["Priority"] == "MEDIUM"])
        low_c   = len(df_hist[df_hist["Priority"] == "LOW"])

        st.markdown("#### Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📋 Total Classified", total)
        col2.metric("🔴 High Priority", high_c)
        col3.metric("🟡 Medium Priority", med_c)
        col4.metric("🟢 Low Priority", low_c)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Priority Distribution")
            st.bar_chart(df_hist["Priority"].value_counts())
        with col2:
            st.markdown("#### Classification Timeline")
            df_hist["Index"] = range(1, len(df_hist) + 1)
            df_hist["Priority Score"] = df_hist["Priority"].map({"HIGH": 3, "MEDIUM": 2, "LOW": 1})
            st.line_chart(df_hist.set_index("Index")["Priority Score"])

        st.markdown("---")
        st.markdown("#### Priority Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div style="background:#ff4b4b22;border-left:4px solid #ff4b4b;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#ff4b4b">🔴 {(high_c/total*100):.1f}%</h3>
            <p>High Priority</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div style="background:#ffa50022;border-left:4px solid #ffa500;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#ffa500">🟡 {(med_c/total*100):.1f}%</h3>
            <p>Medium Priority</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div style="background:#00c85222;border-left:4px solid #00c852;
            padding:15px;border-radius:8px;text-align:center;">
            <h3 style="color:#00c852">🟢 {(low_c/total*100):.1f}%</h3>
            <p>Low Priority</p></div>""", unsafe_allow_html=True)
            st.markdown("---")
        st.markdown("#### 📄 Export Report")
        if st.button("📄 Generate Report", type="secondary", use_container_width=True):
            report = f"""COMPLAINT PRIORITY CLASSIFIER — ANALYTICS REPORT
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

SUMMARY
-------
Total Complaints Classified : {total}
High Priority               : {high_c} ({(high_c/total*100):.1f}%)
Medium Priority             : {med_c} ({(med_c/total*100):.1f}%)
Low Priority                : {low_c} ({(low_c/total*100):.1f}%)

MODEL PERFORMANCE
-----------------
Accuracy  : 91.67%
Precision : 93%
Recall    : 92%
F1 Score  : 92%

RECENT CLASSIFICATIONS
----------------------
"""
            for h in st.session_state.history[-10:]:
                report += f"[{h['Time']}] {h['Priority']:6} ({h['Confidence']}) — {h['Complaint'][:50]}\n"

            st.download_button(
                "⬇️ Download Report (.txt)",
                data=report,
                file_name=f"complaint_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

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

# ── Tab 6: Retrain ──
with tab6:
    st.markdown("### 🔄 Retrain Model with New Data")
    st.markdown("Upload a CSV with `text` and `priority` columns to retrain the model.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Accuracy", "91.67%")
    col2.metric("Training Samples", "178")
    col3.metric("Last Trained", "Today")
    st.markdown("---")

    st.markdown("#### 📁 Option 1 — Upload New Training Data")
    retrain_file = st.file_uploader("Upload CSV file (must have 'text' and 'priority' columns):",
                                     type=["csv"], key="retrain_upload")
    if retrain_file is not None:
        df_new = pd.read_csv(retrain_file)
        st.dataframe(df_new.head(), use_container_width=True)
        if "text" in df_new.columns and "priority" in df_new.columns:
            st.success(f"✅ Valid file! {len(df_new)} samples found.")
            merge_option = st.radio("How to use this data?",
                                     ["➕ Merge with existing data", "🔄 Replace existing data"],
                                     horizontal=True, key="merge_option")
            if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
                with st.spinner("🔄 Retraining model... please wait..."):
                    try:
                        if merge_option == "➕ Merge with existing data":
                            existing_df = pd.read_csv("data/complaints.csv")
                            combined_df = pd.concat([existing_df, df_new], ignore_index=True)
                            combined_df.drop_duplicates(subset=["text"], inplace=True)
                            combined_df.to_csv("data/complaints.csv", index=False)
                            st.info(f"📊 Combined dataset: {len(combined_df)} samples")
                        else:
                            df_new.to_csv("data/complaints.csv", index=False)
                            st.info(f"📊 New dataset: {len(df_new)} samples")
                        import io, sys
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        train("data/complaints.csv")
                        sys.stdout = old_stdout
                        st.success("✅ Model retrained successfully!")
                        st.code(buffer.getvalue())
                    except Exception as e:
                        st.error(f"❌ Retraining failed: {e}")
        else:
            st.error("❌ CSV must have 'text' and 'priority' columns!")

    st.markdown("---")
    st.markdown("#### ⭐ Option 2 — Retrain Using Feedback Data")
    feedback_path = "data/feedback.csv"
    if os.path.exists(feedback_path):
        df_feedback      = pd.read_csv(feedback_path)
        correct_feedback = df_feedback[df_feedback["feedback"] == "correct"]
        st.info(f"📋 Found {len(df_feedback)} feedback entries — {len(correct_feedback)} marked correct")
        if len(correct_feedback) >= 5:
            if st.button("🚀 Retrain with Feedback Data", type="secondary", use_container_width=True):
                with st.spinner("🔄 Retraining with feedback data..."):
                    try:
                        feedback_train = correct_feedback[["text", "predicted"]].copy()
                        feedback_train.columns = ["text", "priority"]
                        existing_df = pd.read_csv("data/complaints.csv")
                        combined_df = pd.concat([existing_df, feedback_train], ignore_index=True)
                        combined_df.drop_duplicates(subset=["text"], inplace=True)
                        combined_df.to_csv("data/complaints.csv", index=False)
                        import io, sys
                        old_stdout = sys.stdout
                        sys.stdout = buffer = io.StringIO()
                        train("data/complaints.csv")
                        sys.stdout = old_stdout
                        st.success("✅ Model retrained with feedback data!")
                        st.code(buffer.getvalue())
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
        else:
            st.warning("⚠️ Need at least 5 correct feedback entries to retrain.")
    else:
        st.info("No feedback data yet. Use 👍 buttons in Single Classify tab first!")

# ── Tab 7: Summarizer ──
with tab7:
    st.markdown("### 📝 Complaint Summarizer")
    st.markdown("Paste a long complaint and get a quick summary + classification!")
    long_complaint = st.text_area("Paste your complaint here:", height=200,
                                   placeholder="Paste a long complaint text here...",
                                   key="summarizer_input")
    if st.button("📝 Summarize & Classify", type="primary", use_container_width=True):
        if long_complaint.strip():
            with st.spinner("Analyzing complaint..."):
                result     = predict(long_complaint)
                priority   = result["priority"]
                confidence = result["confidence"]
                emoji      = {"high": "🔴", "medium": "🟡", "low": "🟢"}[priority]
                box_class  = {"high": "high-box", "medium": "medium-box", "low": "low-box"}[priority]
                summary    = summarize_complaint(long_complaint)
                keywords   = get_keywords(long_complaint)

            st.markdown(f"""
            <div class="{box_class}">
                <h2>{emoji} {priority.upper()} PRIORITY</h2>
                <p>Confidence: <strong>{confidence:.0%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📋 Summary")
                st.info(f"💡 {summary}")
                st.markdown("#### 🔑 Key Terms")
                if keywords:
                    st.markdown(" ".join([f"`{kw}`" for kw in keywords]))
            with col2:
                st.markdown("#### 📊 Stats")
                st.metric("Words",      len(long_complaint.split()))
                st.metric("Characters", len(long_complaint))
                st.metric("Sentences",  len([s for s in long_complaint.split(".") if s.strip()]))

            st.markdown("---")
            st.markdown("#### 📄 Original Complaint")
            st.text_area("Full text:", value=long_complaint, height=150,
                         disabled=True, key="original_display")
        else:
            st.warning("⚠️ Please enter a complaint text!")