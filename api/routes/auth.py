from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from api.middleware.auth import authenticate_user, create_token
from api.middleware.logging import logger

router = APIRouter(prefix="/auth", tags=["Authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type  : str = "bearer"
    username    : str
    role        : str
    expires_in  : int = 1800

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = authenticate_user(request.username, request.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    token = create_token({"sub": user["username"], "role": user["role"]})
    logger.info(f"User logged in: {user['username']} (role: {user['role']})")
    return LoginResponse(
        access_token=token,
        username=user["username"],
        role=user["role"]
    )

@router.post("/logout")
async def logout():
    return {"message": "Successfully logged out"}