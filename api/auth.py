"""Authentication helpers for the FastAPI backend.

This module provides JWT creation and validation for protected endpoints.
"""
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in the environment")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Create a signed JWT access token.

    Args:
        data: Token payload values.
        expires_delta: Optional custom expiration window.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()

    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate(token: str = Depends(oauth2_scheme)):
    """
    Validate the bearer token used by protected endpoints.

    Returns:
        Decoded token payload if authentication succeeds.

    Raises:
        HTTPException: If the token is missing or invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")