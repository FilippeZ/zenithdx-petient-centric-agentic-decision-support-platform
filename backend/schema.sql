-- Drop dependent tables first (patients references users)
DROP TABLE IF EXISTS patients CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Users table (create first)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,   -- In real systems, store hashed
    user_type VARCHAR(50) NOT NULL,          -- 'doctor' or 'patient'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patients table (references users)
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    symptoms TEXT NOT NULL,
    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'Pending',
    diagnosis TEXT,
    treatment TEXT,
    class_name TEXT,
    confidence FLOAT,
    filepath_1 TEXT,
    filepath_2 TEXT,
    filepath_3 TEXT
);

-- Insert demo users
INSERT INTO users (full_name, username, email, hashed_password, user_type)
VALUES
  ('Philip',    'philip',    'philip@example.com',    '123456', 'patient'),
  ('Doctor 10', 'doctor10',  'doctor10@example.com',  '123456', 'doctor');

-- List all users
SELECT id, full_name, username, user_type, email FROM users;

-- (OPTIONAL) List all patients (will be empty initially)
SELECT * FROM patients;
