# db.py

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = "dbname=postgres user=postgres password=postgres host=localhost port=5432"

def get_db_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
