from fastapi import Request
from fastapi.responses import JSONResponse
from api.middleware.logging import logger

async def validation_exception_handler(request: Request, exc: Exception):
    logger.error(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "path": str(request.url)
        }
    )

async def http_exception_handler(request: Request, exc: Exception):
    logger.warning(f"HTTP error on {request.url}: {exc}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": str(request.url)
        }
    )