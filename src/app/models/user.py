"""
Module: user.py
================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
User-related Pydantic schemas for authentication and authorization.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    role: str = Field(default="agent")


class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(default="agent")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v.lower() not in ["agent", "admin"]:
            raise ValueError("Role must be 'agent' or 'admin'")
        return v.lower()


class UserResponse(BaseModel):
    """Schema for user response (without sensitive data)."""
    username: str
    role: str
    approved: bool


class UserInDB(BaseModel):
    """Schema for user stored in database."""
    username: str
    hashed_password: str
    role: str
    approved: bool = False


class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"


class UserSession(BaseModel):
    """Schema for user session verification."""
    username: str
    role: str


class ApproveUserRequest(BaseModel):
    """Schema for approving/disapproving users."""
    username: str
    approve: bool


class ApproveUserResponse(BaseModel):
    """Schema for user approval response."""
    username: str
    approved: bool
    message: str
