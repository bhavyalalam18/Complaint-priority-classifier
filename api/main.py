from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import os
from dotenv import load_dotenv

from api.routes.auth     import router as auth_router
from api.routes.classify import router as classify_router
from api.middleware.errors  import (validation_exception_handler,
                                     general_exception_handler,
                                     http_exception_handler)
from api.middleware.logging import logger

load_dotenv()

# ── App Setup ──
from fastapi.security import HTTPBearer

security = HTTPBearer()

app = FastAPI(
    title       = os.getenv("APP_NAME", "Complaint Priority Classifier API"),
    description = """
## 🎯 Complaint Priority Classifier API

Automatically classify customer complaints into **High**, **Medium**, or **Low** priority.

### Features
- 🔐 JWT Authentication with role-based access
- 🎯 Single & batch complaint classification
- 🌐 Multilingual support
- 📊 Confidence scores & severity rating
- 📝 Swagger UI documentation
    """,
    version     = os.getenv("APP_VERSION", "1.0.0"),
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Exception Handlers ──
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ── Routers ──
app.include_router(auth_router)
app.include_router(classify_router)

# ── Events ──
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Complaint Priority Classifier API starting up...")
    logger.info(f"📚 Docs available at: http://localhost:8000/docs")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 API shutting down...")

# ── Root ──
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🎯 Complaint Priority Classifier API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/classify/health"
    }