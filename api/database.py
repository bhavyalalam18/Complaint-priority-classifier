from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./complaints.db"

engine        = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal  = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base          = declarative_base()

# ── Complaint Table ──
class ComplaintRecord(Base):
    __tablename__ = "complaints"

    id                 = Column(Integer, primary_key=True, index=True)
    text               = Column(String)
    predicted_priority = Column(String)
    actual_priority    = Column(String, nullable=True)
    confidence         = Column(Float)
    severity           = Column(Integer)
    source             = Column(String, default="model")
    username           = Column(String)
    email_sent         = Column(Boolean, default=False)
    translate          = Column(Boolean, default=False)
    created_at         = Column(DateTime, default=datetime.utcnow)

# ── Feedback Table ──
class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id                 = Column(Integer, primary_key=True, index=True)
    complaint_id       = Column(Integer)
    text               = Column(String)
    predicted_priority = Column(String)
    correct_priority   = Column(String)
    username           = Column(String)
    created_at         = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()