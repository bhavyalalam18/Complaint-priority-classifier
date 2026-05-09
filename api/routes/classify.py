from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.orm import Session
import os
import pandas as pd
import io
from api.middleware.auth import get_current_user, require_admin
from api.middleware.logging import logger
from api.database import get_db, ComplaintRecord
from classifier import predict
from app import send_email_alert
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from sqlalchemy import func
from datetime import datetime
from api.database import get_db, ComplaintRecord, FeedbackRecord

limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/classify", tags=["Classification"])

# ── Request/Response Models ──
class ClassifyRequest(BaseModel):
    text        : str           = Field(..., min_length=3, max_length=5000,
                                        description="Complaint text to classify")
    translate   : bool          = Field(False, description="Auto-translate to English")
    notify_email: Optional[str] = Field(None, description="Email to notify if High priority")

class ClassifyResponse(BaseModel):
    text      : str
    priority  : str
    confidence: float
    severity  : int
    scores    : dict
    source    : str
    email_sent: bool

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
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        logger.info(f"Classification request from {current_user['username']}")
        result     = predict(request.text, translate=request.translate)
        severity   = calculate_severity(request.text, result["priority"], result["confidence"])
        email_sent = False

        if result["priority"] == "high":
            receiver = request.notify_email or os.getenv("EMAIL_RECEIVER")
            if receiver:
                email_sent = send_email_alert(
                    to_email       = receiver,
                    complaint_text = request.text,
                    confidence     = result["confidence"]
                )
                logger.info(f"📧 Alert sent to {receiver} by {current_user['username']}")

        # ── Save to database ──
        record = ComplaintRecord(
            text               = request.text,
            predicted_priority = result["priority"],
            confidence         = result["confidence"],
            severity           = severity,
            source             = result.get("source", "model"),
            username           = current_user["username"],
            email_sent         = email_sent,
            translate          = request.translate
        )
        db.add(record)
        db.commit()

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
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        logger.info(f"Batch request ({len(request.complaints)} items) from {current_user['username']}")
        results     = []
        emails_sent = 0

        for item in request.complaints:
            should_translate = item.translate if item.translate else request.translate
            r        = predict(item.text, translate=should_translate)
            severity = calculate_severity(item.text, r["priority"], r["confidence"])

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

            # ── Save to database ──
            record = ComplaintRecord(
                text               = item.text,
                predicted_priority = r["priority"],
                confidence         = r["confidence"],
                severity           = severity,
                source             = r.get("source", "model"),
                username           = current_user["username"],
                email_sent         = item_email_sent,
                translate          = should_translate
            )
            db.add(record)

            results.append({
                "text"      : item.text[:60] + "..." if len(item.text) > 60 else item.text,
                "priority"  : r["priority"],
                "confidence": r["confidence"],
                "severity"  : severity,
                "scores"    : r["scores"],
                "email_sent": item_email_sent
            })

        db.commit()

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

@router.post("/upload-csv")
async def classify_csv(
    file        : UploadFile = File(...),
    current_user: dict       = Depends(get_current_user),
    db          : Session    = Depends(get_db)
):
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))

        if "text" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="CSV must have a 'text' column"
            )

        has_actual = "priority" in df.columns
        logger.info(f"CSV upload: {len(df)} rows from {current_user['username']}")

        results = []
        for _, row in df.iterrows():
            text     = str(row["text"])
            r        = predict(text)
            severity = calculate_severity(text, r["priority"], r["confidence"])

            result = {
                "text"              : text[:80] + "..." if len(text) > 80 else text,
                "predicted_priority": r["priority"],
                "confidence"        : round(r["confidence"], 3),
                "severity"          : severity,
            }

            if has_actual:
                actual                    = str(row["priority"]).lower().strip()
                result["actual_priority"] = actual
                result["match"]           = actual == r["priority"]

            # ── Save to database ──
            record = ComplaintRecord(
                text               = text,
                predicted_priority = r["priority"],
                confidence         = r["confidence"],
                severity           = severity,
                source             = "csv_upload",
                username           = current_user["username"],
                email_sent         = False,
                translate          = False
            )
            db.add(record)
            results.append(result)

        db.commit()

        summary = {
            "total" : len(results),
            "high"  : sum(1 for r in results if r["predicted_priority"] == "high"),
            "medium": sum(1 for r in results if r["predicted_priority"] == "medium"),
            "low"   : sum(1 for r in results if r["predicted_priority"] == "low"),
        }

        if has_actual:
            correct               = sum(1 for r in results if r.get("match"))
            summary["accuracy"]   = round(correct / len(results) * 100, 2)
            summary["correct"]    = correct
            summary["incorrect"]  = len(results) - correct
            summary["mismatches"] = [r for r in results if not r.get("match")]

        return {
            "filename": file.filename,
            "results" : results,
            "summary" : summary
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=422, detail="CSV file is empty")
    except Exception as e:
        logger.error(f"CSV classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-csv/download")
async def classify_csv_download(
    file        : UploadFile = File(...),
    current_user: dict       = Depends(get_current_user)
):
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))

        if "text" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="CSV must have a 'text' column"
            )

        has_actual           = "priority" in df.columns
        predicted_priorities = []
        confidences          = []
        severities           = []
        matches              = []

        for _, row in df.iterrows():
            text     = str(row["text"])
            r        = predict(text)
            severity = calculate_severity(text, r["priority"], r["confidence"])
            predicted_priorities.append(r["priority"])
            confidences.append(round(r["confidence"], 3))
            severities.append(severity)
            if has_actual:
                actual = str(row["priority"]).lower().strip()
                matches.append(actual == r["priority"])

        df["predicted_priority"] = predicted_priorities
        df["confidence"]         = confidences
        df["severity"]           = severities
        if has_actual:
            df["match"] = matches

        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=classified_results.csv"}
        )

    except Exception as e:
       logger.error(f"CSV download error: {e}")

@router.get("/history")
async def get_history(
    limit       : int    = 10,
    priority    : str    = None,
    current_user: dict   = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    query = db.query(ComplaintRecord).order_by(ComplaintRecord.created_at.desc())
    if priority:
        query = query.filter(ComplaintRecord.predicted_priority == priority.lower())
    records = query.limit(limit).all()
    return {
        "total"  : len(records),
        "history": [
            {
                "id"                : r.id,
                "text"              : r.text[:80] + "..." if len(r.text) > 80 else r.text,
                "predicted_priority": r.predicted_priority,
                "confidence"        : r.confidence,
                "severity"          : r.severity,
                "username"          : r.username,
                "email_sent"        : r.email_sent,
                "created_at"        : r.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            for r in records
        ]
    }


@router.get("/health")
async def health_check():
    return {
        "status" : "healthy",
        "model"  : "complaint-classifier-v1",
        "version": "1.0.0"
    }     
# Single classify — 10 requests per minute
@router.post("/single", response_model=ClassifyResponse)
@limiter.limit("10/minute")
async def classify_single(
    request     : Request,
    body        : ClassifyRequest,
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        logger.info(f"Classification request from {current_user['username']}")
        result     = predict(body.text, translate=body.translate)
        severity   = calculate_severity(body.text, result["priority"], result["confidence"])
        email_sent = False

        if result["priority"] == "high":
            receiver = body.notify_email or os.getenv("EMAIL_RECEIVER")
            if receiver:
                email_sent = send_email_alert(
                    to_email       = receiver,
                    complaint_text = body.text,
                    confidence     = result["confidence"]
                )
                logger.info(f"📧 Alert sent to {receiver} by {current_user['username']}")

        record = ComplaintRecord(
            text               = body.text,
            predicted_priority = result["priority"],
            confidence         = result["confidence"],
            severity           = severity,
            source             = result.get("source", "model"),
            username           = current_user["username"],
            email_sent         = email_sent,
            translate          = body.translate
        )
        db.add(record)
        db.commit()

        return ClassifyResponse(
            text       = body.text,
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


# Batch classify — 5 requests per minute
@router.post("/batch")
@limiter.limit("5/minute")
async def classify_batch(
    request     : Request,
    body        : BatchRequest,
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        logger.info(f"Batch request ({len(body.complaints)} items) from {current_user['username']}")
        results     = []
        emails_sent = 0

        for item in body.complaints:
            should_translate = item.translate if item.translate else body.translate
            r        = predict(item.text, translate=should_translate)
            severity = calculate_severity(item.text, r["priority"], r["confidence"])

            item_email_sent = False
            if r["priority"] == "high":
                receiver = item.notify_email or body.notify_email or os.getenv("EMAIL_RECEIVER")
                if receiver:
                    item_email_sent = send_email_alert(
                        to_email       = receiver,
                        complaint_text = item.text,
                        confidence     = r["confidence"]
                    )
                    if item_email_sent:
                        emails_sent += 1

            record = ComplaintRecord(
                text               = item.text,
                predicted_priority = r["priority"],
                confidence         = r["confidence"],
                severity           = severity,
                source             = r.get("source", "model"),
                username           = current_user["username"],
                email_sent         = item_email_sent,
                translate          = should_translate
            )
            db.add(record)

            results.append({
                "text"      : item.text[:60] + "..." if len(item.text) > 60 else item.text,
                "priority"  : r["priority"],
                "confidence": r["confidence"],
                "severity"  : severity,
                "scores"    : r["scores"],
                "email_sent": item_email_sent
            })

        db.commit()

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


# Login — 5 requests per minute
@router.post("/upload-csv")
@limiter.limit("5/minute")
async def classify_csv(
    request     : Request,
    file        : UploadFile = File(...),
    current_user: dict       = Depends(get_current_user),
    db          : Session    = Depends(get_db)
):
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))

        if "text" not in df.columns:
            raise HTTPException(
                status_code=422,
                detail="CSV must have a 'text' column"
            )

        has_actual = "priority" in df.columns
        logger.info(f"CSV upload: {len(df)} rows from {current_user['username']}")

        results = []
        for _, row in df.iterrows():
            text     = str(row["text"])
            r        = predict(text)
            severity = calculate_severity(text, r["priority"], r["confidence"])

            result = {
                "text"              : text[:80] + "..." if len(text) > 80 else text,
                "predicted_priority": r["priority"],
                "confidence"        : round(r["confidence"], 3),
                "severity"          : severity,
            }

            if has_actual:
                actual                    = str(row["priority"]).lower().strip()
                result["actual_priority"] = actual
                result["match"]           = actual == r["priority"]

            record = ComplaintRecord(
                text               = text,
                predicted_priority = r["priority"],
                confidence         = r["confidence"],
                severity           = severity,
                source             = "csv_upload",
                username           = current_user["username"],
                email_sent         = False,
                translate          = False
            )
            db.add(record)
            results.append(result)

        db.commit()

        summary = {
            "total" : len(results),
            "high"  : sum(1 for r in results if r["predicted_priority"] == "high"),
            "medium": sum(1 for r in results if r["predicted_priority"] == "medium"),
            "low"   : sum(1 for r in results if r["predicted_priority"] == "low"),
        }

        if has_actual:
            correct               = sum(1 for r in results if r.get("match"))
            summary["accuracy"]   = round(correct / len(results) * 100, 2)
            summary["correct"]    = correct
            summary["incorrect"]  = len(results) - correct
            summary["mismatches"] = [r for r in results if not r.get("match")]

        return {
            "filename": file.filename,
            "results" : results,
            "summary" : summary
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=422, detail="CSV file is empty")
    except Exception as e:
        logger.error(f"CSV classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))     

@router.get("/analytics")
async def get_analytics(
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        # ── Total complaints ──
        total = db.query(ComplaintRecord).count()

        if total == 0:
            return {"message": "No complaints classified yet"}

        # ── Priority breakdown ──
        high   = db.query(ComplaintRecord).filter(ComplaintRecord.predicted_priority == "high").count()
        medium = db.query(ComplaintRecord).filter(ComplaintRecord.predicted_priority == "medium").count()
        low    = db.query(ComplaintRecord).filter(ComplaintRecord.predicted_priority == "low").count()

        # ── Average confidence ──
        avg_confidence = db.query(func.avg(ComplaintRecord.confidence)).scalar()

        # ── Average severity ──
        avg_severity = db.query(func.avg(ComplaintRecord.severity)).scalar()

        # ── Complaints today ──
        today       = datetime.utcnow().date()
        today_start = datetime(today.year, today.month, today.day)
        today_count = db.query(ComplaintRecord).filter(
            ComplaintRecord.created_at >= today_start
        ).count()

        # ── Most active user ──
        most_active = db.query(
            ComplaintRecord.username,
            func.count(ComplaintRecord.id).label("count")
        ).group_by(ComplaintRecord.username)\
         .order_by(func.count(ComplaintRecord.id).desc())\
         .first()

        # ── Email alerts sent ──
        emails_sent = db.query(ComplaintRecord).filter(
            ComplaintRecord.email_sent == True
        ).count()

        # ── CSV uploads ──
        csv_uploads = db.query(ComplaintRecord).filter(
            ComplaintRecord.source == "csv_upload"
        ).count()

        return {
            "overview": {
                "total"            : total,
                "today"            : today_count,
                "emails_sent"      : emails_sent,
                "csv_uploads"      : csv_uploads
            },
            "priority_breakdown": {
                "high"            : high,
                "medium"          : medium,
                "low"             : low,
                "high_percent"    : round(high / total * 100, 1),
                "medium_percent"  : round(medium / total * 100, 1),
                "low_percent"     : round(low / total * 100, 1)
            },
            "performance": {
                "avg_confidence"  : round(avg_confidence * 100, 1),
                "avg_severity"    : round(avg_severity, 1),
            },
            "most_active_user"    : {
                "username"        : most_active[0] if most_active else None,
                "total_requests"  : most_active[1] if most_active else 0
            }
        }

    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
# ── Feedback Models ──
class FeedbackRequest(BaseModel):
    complaint_id    : int = Field(..., description="ID of the complaint to correct")
    correct_priority: str = Field(..., description="Correct priority: high, medium, low")

@router.post("/feedback")
async def submit_feedback(
    request     : FeedbackRequest,
    current_user: dict    = Depends(get_current_user),
    db          : Session = Depends(get_db)
):
    try:
        # ── Validate priority ──
        valid = ["high", "medium", "low"]
        if request.correct_priority.lower() not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"correct_priority must be one of: {valid}"
            )

        # ── Find complaint ──
        complaint = db.query(ComplaintRecord).filter(
            ComplaintRecord.id == request.complaint_id
        ).first()

        if not complaint:
            raise HTTPException(
                status_code=404,
                detail=f"Complaint with id {request.complaint_id} not found"
            )

        # ── Save feedback ──
        feedback = FeedbackRecord(
            complaint_id       = request.complaint_id,
            text               = complaint.text,
            predicted_priority = complaint.predicted_priority,
            correct_priority   = request.correct_priority.lower(),
            username           = current_user["username"]
        )
        db.add(feedback)

        # ── Update complaint with actual priority ──
        complaint.actual_priority = request.correct_priority.lower()
        db.commit()

        return {
            "message"           : "✅ Feedback submitted successfully",
            "complaint_id"      : request.complaint_id,
            "text"              : complaint.text[:80],
            "predicted_priority": complaint.predicted_priority,
            "correct_priority"  : request.correct_priority.lower(),
            "match"             : complaint.predicted_priority == request.correct_priority.lower()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback")
async def get_feedback(
    limit       : int   = 10,
    current_user: dict  = Depends(require_admin),
    db          : Session = Depends(get_db)
):
    try:
        records = db.query(FeedbackRecord)\
                    .order_by(FeedbackRecord.created_at.desc())\
                    .limit(limit).all()

        correct   = sum(1 for r in records if r.predicted_priority == r.correct_priority)
        incorrect = len(records) - correct

        return {
            "total"   : len(records),
            "correct" : correct,
            "incorrect": incorrect,
            "accuracy": round(correct / len(records) * 100, 1) if records else 0,
            "feedback": [
                {
                    "id"                : r.id,
                    "complaint_id"      : r.complaint_id,
                    "text"              : r.text[:80],
                    "predicted_priority": r.predicted_priority,
                    "correct_priority"  : r.correct_priority,
                    "match"             : r.predicted_priority == r.correct_priority,
                    "username"          : r.username,
                    "created_at"        : r.created_at.strftime("%Y-%m-%d %H:%M:%S")
                }
                for r in records
            ]
        }
    except Exception as e:
        logger.error(f"Get feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))     