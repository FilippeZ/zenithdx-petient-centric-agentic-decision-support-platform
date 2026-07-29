from typing import Optional
from fastapi import HTTPException
from db import get_db_connection

TABLE = "patients"

def _exists(cur, report_id: int) -> bool:
    cur.execute(f"SELECT 1 FROM {TABLE} WHERE id = %s", (report_id,))
    return cur.fetchone() is not None

def _set_status(cur, report_id: int, status: str, reason: Optional[str] = None):
    status = status.capitalize()
    valid = {"Pending", "Approved", "Rejected", "Edited", "Saved"}
    if status not in valid:
        raise HTTPException(422, f"Unsupported status: {status}")
    cur.execute(
        f"UPDATE {TABLE} SET status = %s WHERE id = %s",
        (status, report_id),
    )

def approve_report(report_id: int):
    with get_db_connection() as conn, conn.cursor() as cur:
        if not _exists(cur, report_id):
            raise HTTPException(404, "Report not found")
        _set_status(cur, report_id, "Approved")
        conn.commit()
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (report_id,))
        updated = cur.fetchone()
        if updated and updated.get("submission_date"):
            updated["submission_date"] = str(updated["submission_date"])
        return {"message": "Report approved successfully.", "data": updated}

def reject_report(report_id: int, reason: str = ""):
    with get_db_connection() as conn, conn.cursor() as cur:
        if not _exists(cur, report_id):
            raise HTTPException(404, "Report not found")
        _set_status(cur, report_id, "Rejected", reason)
        conn.commit()
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (report_id,))
        updated = cur.fetchone()
        if updated and updated.get("submission_date"):
            updated["submission_date"] = str(updated["submission_date"])
        return {"message": "Report rejected.", "data": updated, "reason": reason}

def edit_report(report_id: int, new_diagnosis: str):
    with get_db_connection() as conn, conn.cursor() as cur:
        if not _exists(cur, report_id):
            raise HTTPException(404, "Report not found")
        cur.execute(
            f"""UPDATE {TABLE}
                    SET diagnosis = %s,
                        status    = 'Edited'
                  WHERE id = %s""",
            (new_diagnosis, report_id),
        )
        conn.commit()
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (report_id,))
        updated = cur.fetchone()
        if updated and updated.get("submission_date"):
            updated["submission_date"] = str(updated["submission_date"])
        return {"message": "Diagnosis edited.", "data": updated}

def save_report_changes(
    report_id: int,
    diagnosis: Optional[str] = None,
    symptoms: Optional[str] = None,
    doctor_message: Optional[str] = None,
):
    if diagnosis is None and symptoms is None and doctor_message is None:
        raise HTTPException(400, "Nothing to update")
    fields, values = [], []
    if diagnosis is not None:
        fields.append("diagnosis = %s")
        values.append(diagnosis)
    if symptoms is not None:
        fields.append("symptoms  = %s")
        values.append(symptoms)
    if doctor_message is not None:
        fields.append("doctor_message = %s")
        values.append(doctor_message)
    fields.append("status = 'Saved'")
    set_clause = ", ".join(fields)
    values.append(report_id)
    with get_db_connection() as conn, conn.cursor() as cur:
        if not _exists(cur, report_id):
            raise HTTPException(404, "Report not found")
        cur.execute(
            f"UPDATE {TABLE} SET {set_clause} WHERE id = %s",
            tuple(values),
        )
        conn.commit()
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (report_id,))
        updated = cur.fetchone()
        if updated is None:
            raise HTTPException(404, "Report not found (after update)")
        if updated.get("submission_date"):
            updated["submission_date"] = str(updated["submission_date"])
        return {"message": "Report changes saved.", "data": updated}

def update_report_status(
    report_id: int,
    status: str,
    reason: Optional[str] = None,
):
    with get_db_connection() as conn, conn.cursor() as cur:
        if not _exists(cur, report_id):
            raise HTTPException(404, "Report not found")
        _set_status(cur, report_id, status, reason)
        conn.commit()
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (report_id,))
        updated = cur.fetchone()
        if updated and updated.get("submission_date"):
            updated["submission_date"] = str(updated["submission_date"])
        return {"message": f"Status updated to {status}.", "data": updated}
