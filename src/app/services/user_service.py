"""
Module: services/user_service.py
==================================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
User management service for CRUD operations and authentication.
"""
import os
import json
from fastapi import HTTPException
from typing import Dict, List
from src.app.core.security import get_password_hash, verify_password, create_access_token
from src.app.models.user import UserCreate, UserResponse, UserInDB, UserSession
from src.app.config import USERS_FILE, APP_USERS
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")


class UserService:
    """Service for managing users."""

    @staticmethod
    def load_users() -> Dict[str, dict]:
        """
        Load users from JSON file.

        Returns:
            Dictionary of users.
        """
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_users(users: Dict[str, dict]) -> None:
        """
        Save users to JSON file.

        Args:
            users: Dictionary of users to save.
        """
        os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

    def create_user(self, user: UserCreate) -> dict:
        """
        Register a new user.

        Args:
            user: User creation data.

        Returns:
            Success message.

        Raises:
            HTTPException: If username already exists.
        """
        users = self.load_users()

        if user.username in users:
            raise HTTPException(status_code=400, detail="Username already exists")

        # Hash password and store user
        hashed_password = get_password_hash(user.password)
        users[user.username] = {
            "username": user.username,
            "hashed_password": hashed_password,
            "role": user.role.lower(),
            "approved": False
        }

        self.save_users(users)
        logger.info(f"User {user.username} registered successfully.")

        return {"msg": "User created successfully"}

    def authenticate_user(self, username: str, password: str) -> dict | bool:
        """
        Authenticate a user by username and password.

        Args:
            username: Username to authenticate.
            password: Plaintext password.

        Returns:
            User dictionary if authenticated, False otherwise.
        """
        users = self.load_users()
        user = users.get(username)

        if not user or not verify_password(password, user["hashed_password"]):
            return False

        return user

    def get_all_users(self) -> List[UserResponse]:
        """
        Get all registered users (without sensitive data).

        Returns:
            List of user response objects.
        """
        users = self.load_users()
        safe_users = [
            UserResponse(
                username=u["username"],
                role=u["role"],
                approved=u.get("approved", False)
            )
            for u in users.values()
        ]
        return safe_users

    def approve_user(self, username: str, approve: bool) -> dict:
        """
        Approve or disapprove a user.

        Args:
            username: Username to approve/disapprove.
            approve: True to approve, False to disapprove.

        Returns:
            Approval status message.

        Raises:
            HTTPException: If user not found.
        """
        users = self.load_users()

        if username not in users:
            raise HTTPException(status_code=404, detail="User not found")

        users[username]["approved"] = approve
        self.save_users(users)

        status = "approved" if approve else "disapproved"
        logger.info(f"User {username} {status}.")

        return {
            "username": username,
            "approved": approve,
            "message": f"User {status} successfully."
        }

    def get_user_by_username(self, username: str) -> dict | None:
        """
        Get a user by username.

        Args:
            username: Username to retrieve.

        Returns:
            User dictionary or None if not found.
        """
        users = self.load_users()
        return users.get(username)
