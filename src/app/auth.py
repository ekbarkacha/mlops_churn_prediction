import jwt
from pwdlib import PasswordHash
from fastapi import HTTPException, Cookie,Header
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
from src.app.utils import load_users
from src.app.config import SECRET_KEY,ALGORITHM,API_KEY,APP_USERS

pwd_context = PasswordHash.recommended()


# Allow requests from frontend
def set_middleware(app):
    app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # my frontend domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    
    # Add role in JWT payload
    username = data.get("sub")
    users = load_users()
    user_role = users.get(username, {}).get("role", APP_USERS.get(1))
    to_encode.update({"role": user_role})
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def require_role(required_role: str):
    def role_checker(access_token: str = Cookie(None)):
        if not access_token:
            raise HTTPException(status_code=401, detail="Missing token")
        try:
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
            role = payload.get("role", APP_USERS.get(1))
            if role != required_role:
                raise HTTPException(status_code=403, detail="Forbidden: insufficient permissions")
            return payload
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    return role_checker

# API KEY
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    

