-- =============================================================================
--  ZenithDx Database Migration
--  Run this ONCE on an existing database to add missing columns.
--  Safe to run even if some columns already exist (uses IF NOT EXISTS).
-- =============================================================================

-- Add all missing columns to patients table
ALTER TABLE patients
    ADD COLUMN IF NOT EXISTS xai_report          TEXT,
    ADD COLUMN IF NOT EXISTS xai_structured      TEXT,
    ADD COLUMN IF NOT EXISTS original_xray       TEXT,
    ADD COLUMN IF NOT EXISTS gradcam_overlay     TEXT,
    ADD COLUMN IF NOT EXISTS captum_image        TEXT,
    ADD COLUMN IF NOT EXISTS classification_results TEXT,
    ADD COLUMN IF NOT EXISTS doctor_message      TEXT,
    ADD COLUMN IF NOT EXISTS report_type         TEXT DEFAULT 'Chest X-Ray';

-- Update existing rows to have a report_type
UPDATE patients SET report_type = 'Chest X-Ray' WHERE report_type IS NULL;

-- Create index for faster patient lookup
CREATE INDEX IF NOT EXISTS idx_patients_user_id ON patients(user_id);
CREATE INDEX IF NOT EXISTS idx_patients_status  ON patients(status);

-- Verify columns
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'patients'
ORDER BY ordinal_position;
