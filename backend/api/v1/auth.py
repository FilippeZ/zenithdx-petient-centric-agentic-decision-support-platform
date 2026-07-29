# backend/api/v1/auth.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Form, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt

from config import settings
from database.connection import get_db_connection

router = APIRouter(tags=["Authentication"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")

def hash_pw(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_pw(raw: str, hashed: str) -> bool:
    if raw == hashed:
        return True
    try:
        return pwd_context.verify(raw, hashed)
    except Exception:
        return False

def create_access_token(data: dict, minutes: int = settings.ACCESS_TOKEN_MINS) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(token: str = Security(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, full_name, username, email, user_type FROM users WHERE username = %s;",
                (username,)
            )
            user = cur.fetchone()
            if not user:
                raise credentials_exception
            return dict(user)
    finally:
        conn.close()

@router.post("/register")
def register(
    full_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    user_type: str = Form(...)  # 'doctor' or 'patient'
):
    if user_type not in ("doctor", "patient"):
        raise HTTPException(status_code=400, detail="Invalid user_type. Must be 'doctor' or 'patient'")

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s OR email = %s;", (username, email))
            if cur.fetchone():
                raise HTTPException(status_code=400, detail="Username or Email already registered")

            hashed = hash_pw(password)
            cur.execute(
                """
                INSERT INTO users (full_name, username, email, hashed_password, user_type)
                VALUES (%s, %s, %s, %s, %s) RETURNING id, full_name, username, email, user_type;
                """,
                (full_name, username, email, hashed, user_type)
            )
            new_user = cur.fetchone()
            conn.commit()
            return {"message": "User registered successfully", "user": dict(new_user)}
    finally:
        conn.close()

@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = %s;", (form.username,))
            user = cur.fetchone()
            if not user or not verify_pw(form.password, user["hashed_password"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            token = create_access_token({
                "sub": user["username"],
                "user_type": user["user_type"],
                "role": user["user_type"]   # frontend reads .role from JWT
            })
            return {
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "id": user["id"],
                    "full_name": user["full_name"],
                    "username": user["username"],
                    "email": user["email"],
                    "user_type": user["user_type"]
                }
            }
    finally:
        conn.close()
