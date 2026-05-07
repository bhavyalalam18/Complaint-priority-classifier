from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
from api.middleware.auth import get_current_user, require_admin
from api.middleware.logging import logger
from classifier import predict
from app import send_email_alert

router = APIRouter(prefix="/classify", tags=["Classification"])

# ── Request/Response Models ──
class ClassifyRequest(BaseModel):
    text        : str            = Field(..., min_length=3, max_length=5000,
                                         description="Complaint text to classify")
    translate   : bool           = Field(False, description="Auto-translate to English")
    notify_email: Optional[str]  = Field(None, description="Email to notify if High priority")

class ClassifyResponse(BaseModel):
    text         : str
    priority     : str
    confidence   : float
    severity     : int
    scores       : dict
    source       : str
    email_sent   : bool  # ← tells user if alert was sent

class BatchComplaintItem(BaseModel):
    text        : str           = Field(..., min_length=3, max_length=5000)
    translate   : bool          = False
    notify_email: Optional[str] = None

class BatchRequest(BaseModel):
    complaints  : List[BatchComplaintItem] = Field(..., min_items=1, max_items=100)
    translate   : bool          = False
    notify_email: Optional[str] = Field(None, description="Email to notify for all High priority in batch")

def calculate_severity(text: str, priority: str, confidence: float) -> int:
    base           = {"high": 7, "medium": 4, "low": 1}[priority]
    critical_words = ["crash", "breach", "down", "all users", "critical",
                      "emergency", "data loss", "corrupted", "production"]
    bonus = sum(1 for w in critical_words if w in text.lower())
    score = base + min(bonus, 2) + round((confidence - 0.6) * 2)
    return max(1, min(10, score))

@router.post("/single", response_model=ClassifyResponse)
async def classify_single(
    request     : ClassifyRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Classification request from {current_user['username']}")
        result     = predict(request.text, translate=request.translate)
        severity   = calculate_severity(request.text, result["priority"], result["confidence"])
        email_sent = False

        # ── Send email alert if High priority ──
        if result["priority"] == "high":
            receiver = request.notify_email or os.getenv("EMAIL_RECEIVER")
            if receiver:
                email_sent = send_email_alert(
                    to_email       = receiver,
                    complaint_text = request.text,
                    confidence     = result["confidence"]
                )
                logger.info(f"📧 Alert sent to {receiver} by {current_user['username']}")

        return ClassifyResponse(
            text       = request.text,
            priority   = result["priority"],
            confidence = result["confidence"],
            severity   = severity,
            scores     = result["scores"],
            source     = result.get("source", "model"),
            email_sent = email_sent
        )
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def classify_batch(
    request     : BatchRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        logger.info(f"Batch request ({len(request.complaints)} items) from {current_user['username']}")
        results     = []
        emails_sent = 0

        for item in request.complaints:
            should_translate = item.translate if item.translate else request.translate
            r        = predict(item.text, translate=should_translate)
            severity = calculate_severity(item.text, r["priority"], r["confidence"])

            # ── Send email alert if High priority ──
            item_email_sent = False
            if r["priority"] == "high":
                receiver = item.notify_email or request.notify_email or os.getenv("EMAIL_RECEIVER")
                if receiver:
                    item_email_sent = send_email_alert(
                        to_email       = receiver,
                        complaint_text = item.text,
                        confidence     = r["confidence"]
                    )
                    if item_email_sent:
                        emails_sent += 1

            results.append({
                "text"      : item.text[:60] + "..." if len(item.text) > 60 else item.text,
                "priority"  : r["priority"],
                "confidence": r["confidence"],
                "severity"  : severity,
                "scores"    : r["scores"],
                "email_sent": item_email_sent
            })

        summary = {
            "total"      : len(results),
            "high"       : sum(1 for r in results if r["priority"] == "high"),
            "medium"     : sum(1 for r in results if r["priority"] == "medium"),
            "low"        : sum(1 for r in results if r["priority"] == "low"),
            "emails_sent": emails_sent
        }
        return {"results": results, "summary": summary}
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status" : "healthy",
        "model"  : "complaint-classifier-v1",
        "version": "1.0.0"
    }