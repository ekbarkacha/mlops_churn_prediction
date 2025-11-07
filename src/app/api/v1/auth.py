"""
Module: api/v1/auth.py
=======================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Authentication routes for user registration, login, and session verification.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from datetime import timedelta
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.app.core.security import verify_api_key, create_access_token
from src.app.core.deps import get_current_user
from src.app.models.user import UserCreate, UserSession
from src.app.models.common import MessageResponse
from src.app.services.user_service import UserService
from src.app.config import ACCESS_TOKEN_EXPIRE_DAYS
from src.utils.logger import get_logger
from src.monitoring.metrics import API_ERRORS

logger = get_logger("customer_churn_api")
router = APIRouter(prefix="", tags=["Authentication"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=MessageResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def register(user: UserCreate, request: Request):
    """
    Register a new user (requires API key).

    Args:
        user: User registration data (username, password, role).
        request: FastAPI request object.

    Returns:
        Success message.

    Raises:
        HTTPException: If username already exists.
    """
    method = request.method
    endpoint = request.url.path
    try:
        user_service = UserService()
        result = user_service.create_user(user)
        logger.info(f"User {user.username} registered successfully.")
        return result
    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Registration failed: {e.detail}")
        raise


@router.post("/token", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and issue JWT token.

    Args:
        request: FastAPI request object.
        form_data: OAuth2 form with username and password.

    Returns:
        JSON response with success message and HttpOnly cookie containing JWT.

    Raises:
        HTTPException: If credentials are invalid or user not approved.
    """
    method = request.method
    endpoint = request.url.path
    try:
        user_service = UserService()
        user = user_service.authenticate_user(form_data.username, form_data.password)

        if not user:
            raise HTTPException(status_code=401, detail="Incorrect username or password")

        if not user.get("approved", False):
            raise HTTPException(status_code=403, detail="User is not approved. Please contact an admin.")

        # Create JWT token with role
        role = user.get("role", "agent")
        token = create_access_token(
            data={"sub": user["username"], "role": role},
            expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
        )

        # Set token as HttpOnly cookie
        response = JSONResponse(content={"msg": "Login successful"})
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="Lax",
            max_age=ACCESS_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )

        logger.info(f"User {form_data.username} logged in successfully.")
        return response

    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.warning(f"Login failed for {form_data.username}: {e.detail}")
        raise


@router.get("/verify-session", response_model=UserSession, dependencies=[Depends(verify_api_key)])
async def verify_session(current_user: dict = Depends(get_current_user)):
    """
    Verify if the user's session (JWT token) is valid.

    Args:
        current_user: Current user from JWT token.

    Returns:
        User session information (username, role).
    """
    return UserSession(
        username=current_user["username"],
        role=current_user["role"]
    )
