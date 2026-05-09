# 🎯 Complaint Priority Classifier

An AI-powered system that automatically classifies customer complaints into **High**, **Medium**, or **Low** priority using Machine Learning (TF-IDF + Naive Bayes).

---

## ✨ Features

### 🌐 Streamlit Web App
- 🔍 **Single Classification** — classify one complaint with confidence scores & severity rating
- 📂 **Batch Classification** — classify multiple complaints at once (manual or CSV upload)
- 🕐 **History** — view, search, filter and export past classifications
- 📊 **Analytics Dashboard** — live charts, priority breakdown, export reports
- 🧠 **Model Info** — pipeline details, performance metrics
- 🔄 **Retrain** — retrain model with new data or feedback
- 📝 **Summarizer** — summarize long complaints + extract keywords
- 🌍 **Multilingual Support** — auto-translate non-English complaints
- 📧 **Email Alerts** — send alerts for high priority complaints
- 🌙 **Dark Mode** — toggle between light and dark themes
- 💬 **Feedback System** — thumbs up/down to improve model

### 🔌 REST API (FastAPI)
- 🔐 **JWT Authentication** with role-based access (Admin / Viewer)
- 🎯 **Single & Batch** complaint classification
- 📁 **CSV Upload** — upload and classify with accuracy comparison
- 📥 **CSV Download** — download classified results
- 🕐 **History** — view past classifications from database
- 📊 **Analytics** — total complaints, priority breakdown, avg confidence
- 💬 **Feedback** — submit and view prediction corrections
- 🔒 **Rate Limiting** — prevent abuse (10/min single, 5/min batch)
- 📝 **Swagger UI** — interactive API documentation

---

## 🚀 Quick Start

### 1 — Clone the repo
```bash
git clone https://github.com/bhavyalalam18/Complaint-priority-classifier.git
cd Complaint-priority-classifier
```

### 2 — Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Setup environment (optional)
```bash
cp .env.example .env
# Edit .env with your own values or use defaults
```

### 5 — Run Streamlit App
```bash
streamlit run app.py
```

### 6 — Run FastAPI
```bash
uvicorn api.main:app --reload
```

---

## 🌐 Access

| Service | URL |
|---------|-----|
| Streamlit App | `http://localhost:8501` |
| FastAPI Swagger | `http://localhost:8000/docs` |

---

## 🔐 API Authentication

### Default Credentials

| Role | Username | Password |
|------|----------|----------|
| Admin | admin | admin123 |
| Viewer | viewer | viewer123 |

### Get Token
**POST** `/auth/login`
```json
{
  "username": "admin",
  "password": "admin123"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

Use the token in all protected requests:
```
Authorization: Bearer <your_token>
```

---

## 📡 API Endpoints

### Health Check
```
GET /classify/health
```

### Classify Single Complaint
```
POST /classify/single
```
```json
{
  "text": "My internet has been down for 3 days",
  "translate": false,
  "notify_email": "support@example.com"
}
```
Response:
```json
{
  "text": "My internet has been down for 3 days",
  "priority": "high",
  "confidence": 0.95,
  "severity": 9,
  "scores": {"high": 0.95, "medium": 0.03, "low": 0.02},
  "source": "model",
  "email_sent": true
}
```

### Classify Batch
```
POST /classify/batch
```
```json
{
  "complaints": [
    {"text": "My internet is down for 3 days", "translate": false},
    {"text": "I want to change my billing address", "translate": false},
    {"text": "Can you send me my last invoice", "translate": false}
  ],
  "notify_email": "support@example.com"
}
```

### Upload CSV
```
POST /classify/upload-csv
```
Upload a CSV with `text` and optional `priority` columns.
Returns accuracy comparison if `priority` column exists.

### Download Classified CSV
```
POST /classify/upload-csv/download
```

### View History
```
GET /classify/history?limit=10&priority=high
```

### Analytics
```
GET /classify/analytics
```
Response:
```json
{
  "overview": {"total": 50, "today": 10, "emails_sent": 5},
  "priority_breakdown": {"high": 20, "medium": 15, "low": 15, "high_percent": 40.0},
  "performance": {"avg_confidence": 87.5, "avg_severity": 6.2},
  "most_active_user": {"username": "admin", "total_requests": 50}
}
```

### Submit Feedback
```
POST /classify/feedback
```
```json
{
  "complaint_id": 1,
  "correct_priority": "medium"
}
```

### View Feedback (Admin only)
```
GET /classify/feedback?limit=10
```

---

## 🗂️ Project Structure

```
complaint-priority-classifier/
├── api/
│   ├── main.py                 # FastAPI app entry point
│   ├── database.py             # SQLite database models
│   ├── routes/
│   │   ├── auth.py             # Login/logout endpoints
│   │   └── classify.py         # Classification endpoints
│   └── middleware/
│       ├── auth.py             # JWT authentication + bcrypt
│       ├── logging.py          # Request logging
│       └── errors.py           # Error handlers
├── models/
│   └── model.joblib            # Trained ML model
├── data/
│   └── complaints.csv          # Training data
├── classifier.py               # ML prediction + training logic
├── predict.py                  # Prediction utilities
├── app.py                      # Streamlit web interface
├── train.py                    # Model training script
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Multinomial Naive Bayes |
| Features | TF-IDF (1-2 ngrams, 5000 features) |
| Accuracy | 91.67% |
| Training samples | 178 |
| Classes | High / Medium / Low |

### Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| 🔴 High | 100% | 75% | 86% |
| 🟡 Medium | 80% | 100% | 89% |
| 🟢 Low | 100% | 100% | 100% |
| **Overall** | **93%** | **92%** | **92%** |

---

## 🔁 Retrain the Model

```bash
python train.py
```

Or use the **Retrain tab** in the Streamlit app to:
- Upload new training data
- Merge with existing data
- Retrain using feedback collected from users

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn (Naive Bayes + TF-IDF) |
| Web UI | Streamlit |
| REST API | FastAPI |
| Authentication | JWT (python-jose) + bcrypt |
| Database | SQLite (SQLAlchemy) |
| Rate Limiting | SlowAPI |
| Translation | deep-translator |
| Server | Uvicorn |
| Docs | Swagger UI |

---

## 📄 License

MIT License — feel free to use and modify.
