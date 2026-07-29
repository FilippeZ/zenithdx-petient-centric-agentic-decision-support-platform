import psycopg2, sys

DSN = "dbname=zenithdx_db user=zenithdx password=zenithdxsecret host=localhost port=5432"

MIGRATION = """
ALTER TABLE patients ADD COLUMN IF NOT EXISTS xai_report TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS xai_structured TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS original_xray TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS gradcam_overlay TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS captum_image TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS classification_results TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS doctor_message TEXT;
ALTER TABLE patients ADD COLUMN IF NOT EXISTS report_type TEXT DEFAULT 'Chest X-Ray';
"""

try:
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    for stmt in MIGRATION.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt)
                print(f"  OK: {stmt[:60]}")
            except Exception as e:
                print(f"  Skip ({e}): {stmt[:60]}")
    conn.commit()
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='patients' ORDER BY ordinal_position")
    cols = [r[0] for r in cur.fetchall()]
    print(f"\nPatients table columns ({len(cols)}):", cols)
    conn.close()
    print("\nMigration complete!")
except Exception as e:
    print(f"DB connection error: {e}", file=sys.stderr)
    sys.exit(1)
