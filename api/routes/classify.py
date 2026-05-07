from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from api.middleware.auth import get_current_user, require_admin
from api.middleware.logging import logger
from classifier import predict

router = APIRouter(prefix="/classify", tags=["Classification"])

# ── Request/Response Models ──
class ClassifyRequest(BaseModel):
    text     : str  = Field(..., min_length=3, max_length=5000,
                            description="Complaint text to classify")
    translate: bool = Field(False, description="Auto-translate to English")

class ClassifyResponse(BaseModel):
    text      : str
    priority  : str
    confidence: float
    severity  : int
    scores    : dict
    source    : str

class BatchComplaintItem(BaseModel):
    text     : str  = Field(..., min_length=3, max_length=5000)
    translate: bool = False

class BatchRequest(BaseModel):
    complaints: List[BatchComplaintItem] = Field(..., min_items=1, max_items=100)
    translate : bool = False

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
        result   = predict(request.text, translate=request.translate)
        severity = calculate_severity(request.text, result["priority"], result["confidence"])
        return ClassifyResponse(
            text       = request.text,
            priority   = result["priority"],
            confidence = result["confidence"],
            severity   = severity,
            scores     = result["scores"],
            source     = result.get("source", "model")
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
        results = []
        for item in request.complaints:
            should_translate = item.translate if item.translate else request.translate
            r        = predict(item.text, translate=should_translate)
            severity = calculate_severity(item.text, r["priority"], r["confidence"])
            results.append({
                "text"      : item.text[:60] + "..." if len(item.text) > 60 else item.text,
                "priority"  : r["priority"],
                "confidence": r["confidence"],
                "severity"  : severity,
                "scores"    : r["scores"]
            })
        summary = {
            "total" : len(results),
            "high"  : sum(1 for r in results if r["priority"] == "high"),
            "medium": sum(1 for r in results if r["priority"] == "medium"),
            "low"   : sum(1 for r in results if r["priority"] == "low")
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