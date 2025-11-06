"""
Module: schemas.py
==================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
Pydantic schemas for the FastAPI churn prediction application.

Includes:
- User schema for authentication and authorization.
- CustomerInput schema for API requests related to customer churn prediction.

"""
# Imports
from pydantic import BaseModel
from src.app.config import APP_USERS
from typing import Optional

class User(BaseModel):
    username: str
    password: str
    role: str = APP_USERS.get(1)
    approved: Optional[bool] = False

class CustomerInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float