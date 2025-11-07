"""
Module: core/deps.py
=====================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
FastAPI dependency functions for authentication, authorization,
and request context.
"""
from fastapi import HTTPException, Cookie, Depends, Request
from typing import Optional
from src.app.core.security import decode_access_token
from src.app.config import APP_USERS


def get_current_user(access_token: Optional[str] = Cookie(None)) -> dict:
    """
    Dependency to get the current authenticated user from JWT cookie.

    Args:
        access_token: JWT token from HTTP-only cookie.

    Returns:
        Dictionary containing user information (username, role).

    Raises:
        HTTPException: If token is missing or invalid.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    payload = decode_access_token(access_token)
    username: str = payload.get("sub")
    role: str = payload.get("role", APP_USERS.get(1))

    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return {
        "username": username,
        "role": role
    }


def require_role(required_role: str):
    """
    Dependency factory to restrict endpoints to specific user roles.

    Args:
        required_role: The role required to access the endpoint (e.g., 'admin', 'agent').

    Returns:
        A dependency function that validates user role.

    Raises:
        HTTPException: If user lacks required role.
    """
    def role_checker(access_token: Optional[str] = Cookie(None)) -> dict:
        if not access_token:
            raise HTTPException(status_code=401, detail="Missing authentication token")

        payload = decode_access_token(access_token)
        role = payload.get("role", APP_USERS.get(1))

        if role != required_role:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: '{required_role}' role required"
            )

        return payload

    return role_checker


def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Dependency to ensure the current user is an admin.

    Args:
        current_user: Current user from get_current_user dependency.

    Returns:
        User dictionary if user is admin.

    Raises:
        HTTPException: If user is not an admin.
    """
    if current_user["role"] != APP_USERS.get(2):
        raise HTTPException(status_code=403, detail="Admin role required")
    return current_user


def get_request_id(request: Request) -> str:
    """
    Dependency to extract or generate a request ID.

    Args:
        request: FastAPI request object.

    Returns:
        Request ID string.
    """
    import uuid
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))
