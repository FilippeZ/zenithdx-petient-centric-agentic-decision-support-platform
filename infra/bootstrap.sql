-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       VARCHAR(255) NOT NULL,
    username        VARCHAR(100) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    user_type       VARCHAR(20) NOT NULL DEFAULT 'patient',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Patients
CREATE TABLE IF NOT EXISTS patients (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    full_name   VARCHAR(255) NOT NULL,
    dob         DATE,
    gender      VARCHAR(10),
    created_at  TIMESTAMP DEFAULT NOW()
);

-- Reports
CREATE TABLE IF NOT EXISTS reports (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id   UUID REFERENCES users(id) ON DELETE CASCADE,
    doctor_id    UUID REFERENCES users(id),
    report_text  TEXT,
    diagnosis    TEXT,
    confidence   FLOAT,
    image_path   TEXT,
    status       VARCHAR(20) DEFAULT 'pending',
    created_at   TIMESTAMP DEFAULT NOW()
);

-- Grants
GRANT ALL ON ALL TABLES IN SCHEMA public TO zenithdx;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO zenithdx;
