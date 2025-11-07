"""
Module: core/security.py
=========================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Security utilities for password hashing, JWT token creation/verification,
and API key validation.
"""
import jwt
from pwdlib import PasswordHash
from fastapi import HTTPException, Header
from datetime import datetime, timedelta, timezone
from src.app.config import SECRET_KEY, ALGORITHM, API_KEY, ADMIN_API_KEY, APP_USERS

pwd_context = PasswordHash.recommended()


# Password hashing functions
def get_password_hash(password: str) -> str:
    """
    Hash a plaintext password.

    Args:
        password: Plaintext password to hash.

    Returns:
        Hashed password string.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against a hashed password.

    Args:
        plain_password: Plaintext password to verify.
        hashed_password: Hashed password to compare against.

    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


# JWT token functions
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dictionary containing token payload (must include 'sub' for username).
        expires_delta: Optional expiration time delta. Defaults to 15 minutes.

    Returns:
        Encoded JWT token string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT access token.

    Args:
        token: JWT token string to decode.

    Returns:
        Decoded token payload as dictionary.

    Raises:
        HTTPException: If token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# API Key validation functions
def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """
    Verify the standard API key.

    Args:
        x_api_key: API key from request header.

    Returns:
        The API key if valid.

    Raises:
        HTTPException: If API key is invalid.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


def verify_admin_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> str:
    """
    Verify the admin API key for privileged operations.

    Args:
        x_api_key: API key from request header.

    Returns:
        The API key if valid.

    Raises:
        HTTPException: If admin API key is invalid.
    """
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Admin API Key")
    return x_api_key
