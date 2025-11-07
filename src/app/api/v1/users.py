"""
Module: api/v1/users.py
========================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
User management routes for admins to view and approve users.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from src.app.core.security import verify_api_key
from src.app.core.deps import require_role
from src.app.models.user import UserResponse, ApproveUserRequest, ApproveUserResponse
from src.app.services.user_service import UserService
from src.app.config import APP_USERS
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")
router = APIRouter(prefix="/users", tags=["User Management"])


@router.get("", response_model=dict, dependencies=[Depends(verify_api_key)])
async def get_all_users(payload: dict = Depends(require_role(APP_USERS.get(2)))):
    """
    Get list of all registered users.
    Only accessible by Admins.

    Args:
        payload: JWT payload from admin user.

    Returns:
        Dictionary with list of users and total count.
    """
    try:
        user_service = UserService()
        users = user_service.get_all_users()

        safe_users = [
            {
                "username": user.username,
                "role": user.role,
                "approved": user.approved
            }
            for user in users
        ]

        return {"users": safe_users, "total_users": len(safe_users)}

    except Exception as e:
        logger.exception("Failed to load users")
        raise HTTPException(status_code=500, detail="Could not retrieve users")


@router.post("/approve", response_model=ApproveUserResponse, dependencies=[Depends(verify_api_key)])
async def approve_user(
    approve_request: ApproveUserRequest,
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Approve or disapprove a user.
    Only accessible by Admins.

    Args:
        approve_request: Username and approval status.
        payload: JWT payload from admin user.

    Returns:
        Approval status response.
    """
    try:
        user_service = UserService()
        result = user_service.approve_user(
            username=approve_request.username,
            approve=approve_request.approve
        )

        logger.info(f"User {approve_request.username} approval updated by Admin {payload['sub']}")
        return ApproveUserResponse(**result)

    except HTTPException as e:
        raise
    except Exception as e:
        logger.exception("Failed to update user approval")
        raise HTTPException(status_code=500, detail="Could not update user")
