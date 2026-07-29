-- ZenithDx PostgreSQL Initialisation
-- Enables pgvector extension and creates core tables

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username    VARCHAR(100) UNIQUE NOT NULL,
    email       VARCHAR(255) UNIQUE NOT NULL,
    hashed_pw   TEXT NOT NULL,
    role        VARCHAR(20) NOT NULL DEFAULT ''patient'',
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    full_name   VARCHAR(255) NOT NULL,
    dob         DATE,
    gender      VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Diagnostic reports table
CREATE TABLE IF NOT EXISTS diagnostic_reports (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id       UUID REFERENCES users(id),
    report_text     TEXT,
    diagnosis       TEXT,
    confidence      FLOAT,
    embedding       vector(768),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Vector similarity index for RAG
CREATE INDEX IF NOT EXISTS idx_reports_embedding
    ON diagnostic_reports USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
