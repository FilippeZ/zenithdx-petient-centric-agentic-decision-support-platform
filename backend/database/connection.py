# backend/database/connection.py
from __future__ import annotations

import psycopg2
from psycopg2.extras import RealDictCursor
from config import settings

def get_db_connection():
    """Returns a PostgreSQL connection with RealDictCursor factory."""
    return psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
