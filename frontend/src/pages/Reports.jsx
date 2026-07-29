import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { useParams, useNavigate } from "react-router-dom";
import {
  FaFilePdf, FaCheckCircle, FaTimesCircle, FaEdit, FaCog,
  FaNotesMedical, FaUser, FaSave, FaCommentMedical, FaInfoCircle,
  FaImage, FaBrain, FaTable, FaSearchMinus, FaSearchPlus,
  FaArrowsAlt, FaSyncAlt, FaArrowLeft,
} from "react-icons/fa";
import Navbar from "../components/Navbar";

const BASE = "http://localhost:8000/";

/* ─── Status colour map ─────────────────────────────────────────── */
const STATUS_STYLES = {
  approved: { bg: "rgba(34,197,94,0.14)",  text: "#4ade80", border: "rgba(34,197,94,0.35)",  dot: "#4ade80"  },
  rejected: { bg: "rgba(239,68,68,0.14)",  text: "#f87171", border: "rgba(239,68,68,0.35)",  dot: "#f87171"  },
  edited:   { bg: "rgba(139,92,246,0.14)", text: "#a78bfa", border: "rgba(139,92,246,0.35)", dot: "#a78bfa"  },
  pending:  { bg: "rgba(245,158,11,0.14)", text: "#fcd34d", border: "rgba(245,158,11,0.35)", dot: "#fcd34d"  },
};
const statusStyle = (s = "") => STATUS_STYLES[(s||"pending").toLowerCase()] || STATUS_STYLES.pending;

/* ─── Dark Glass Card ───────────────────────────────────────────── */
const Card = ({ children, style = {} }) => (
  <div style={{
    background: "rgba(8,15,32,0.88)",
    border: "1px solid rgba(56,189,248,0.15)",
    borderRadius: 24,
    padding: "2.2rem",
    marginBottom: "1.8rem",
    backdropFilter: "blur(20px)",
    boxShadow: "0 20px 50px rgba(0,0,0,0.45)",
    position: "relative",
    overflow: "hidden",
    ...style,
  }}>
    <div style={{ position: "absolute", top: -80, right: -80, width: 250, height: 250, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%)", pointerEvents: "none" }} />
    <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
  </div>
);

/* ─── Section Title ─────────────────────────────────────────────── */
const SectionTitle = ({ icon, children, hint }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.6rem", paddingBottom: "1rem", borderBottom: "1px solid rgba(56,189,248,0.1)" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "0.7rem" }}>
      <span style={{ fontSize: "1.3rem" }}>{icon}</span>
      <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f0f6ff", letterSpacing: "-0.02em", margin: 0 }}>{children}</h2>
    </div>
    {hint && <span style={{ fontSize: "0.78rem", color: "#475569" }}>{hint}</span>}
  </div>
);

/* ─── Status Pill ───────────────────────────────────────────────── */
const StatusPill = ({ status }) => {
  const s = statusStyle(status);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.35rem 0.9rem", borderRadius: 100, fontSize: "0.82rem", fontWeight: 700, background: s.bg, color: s.text, border: `1px solid ${s.border}` }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: s.dot, display: "inline-block", boxShadow: `0 0 7px ${s.dot}` }} />
      {status || "Pending"}
    </span>
  );
};

/* ─── Info Row ──────────────────────────────────────────────────── */
const InfoRow = ({ label, children }) => (
  <div style={{ marginBottom: "1.1rem" }}>
    <div style={{ fontSize: "0.73rem", color: "#475569", fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: "0.3rem" }}>{label}</div>
    <div style={{ fontSize: "0.97rem", color: "#e2e8f0", fontWeight: 500 }}>{children}</div>
  </div>
);

/* ─── Image Viewer with Magnifier & Zoom ────────────────────────── */
const ImageViewer = ({ src, alt, scrollable = false, viewportHeight = 320, enableMagnifier = true }) => {
  const containerRef = useRef(null);
  const [[x, y], setXY] = useState([0, 0]);
  const [[boxW, boxH], setBoxSize] = useState([0, 0]);
  const [[natW, natH], setNatSize] = useState([0, 0]);
  const [showMag, setShowMag] = useState(false);
  const [scale, setScale] = useState(1);
  const isPanning = useRef(false);
  const startRef  = useRef({ x: 0, y: 0, sl: 0, st: 0 });

  const MZW = 150, MZH = 150, ZOOM = 2.2;

  if (!src) return (
    <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 16, background: "rgba(15,23,42,0.6)", border: "1px dashed rgba(56,189,248,0.2)", color: "#475569", fontSize: "0.9rem" }}>
      No image available
    </div>
  );

  const bgSize = `${boxW * ZOOM}px ${boxH * ZOOM}px`;
  const bgPosX = `-${x * ZOOM - MZW / 2}px`;
  const bgPosY = `-${y * ZOOM - MZH / 2}px`;

  return (
    <div style={{ width: "100%" }}>
      {scrollable && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.6rem" }}>
          <span style={{ fontSize: "0.78rem", color: "#475569" }}>
            <FaArrowsAlt style={{ marginRight: 4 }} />Drag · Shift+Wheel
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            {[FaSearchMinus, FaSyncAlt, FaSearchPlus].map((Icon, i) => (
              <button key={i} onClick={() => i === 0 ? setScale(s => Math.max(0.5, s/1.25)) : i === 1 ? setScale(1) : setScale(s => Math.min(6, s*1.25))}
                style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(56,189,248,0.2)", color: "#94a3b8", borderRadius: 8, width: 32, height: 32, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: "0.85rem" }}>
                <Icon />
              </button>
            ))}
          </div>
        </div>
      )}
      <div
        ref={containerRef}
        style={{
          position: "relative",
          overflow: scrollable ? "auto" : "visible",
          width: "100%",
          ...(scrollable ? { height: viewportHeight, cursor: "grab", background: "rgba(10,17,34,0.7)", borderRadius: 16, border: "1px solid rgba(56,189,248,0.12)" } : {}),
        }}
        onMouseDown={scrollable ? e => { isPanning.current = true; startRef.current = { x: e.clientX, y: e.clientY, sl: containerRef.current.scrollLeft, st: containerRef.current.scrollTop }; containerRef.current.style.cursor = "grabbing"; e.preventDefault(); } : undefined}
        onMouseMove={scrollable ? e => { if (!isPanning.current) return; containerRef.current.scrollLeft = startRef.current.sl - (e.clientX - startRef.current.x); containerRef.current.scrollTop = startRef.current.st - (e.clientY - startRef.current.y); } : undefined}
        onMouseUp={scrollable ? () => { isPanning.current = false; if(containerRef.current) containerRef.current.style.cursor = "grab"; } : undefined}
        onMouseLeave={scrollable ? () => { isPanning.current = false; if(containerRef.current) containerRef.current.style.cursor = "grab"; } : undefined}
        onWheel={scrollable ? e => { if(e.shiftKey) { containerRef.current.scrollLeft += e.deltaY; e.preventDefault(); } } : undefined}
      >
        <img
          src={src} alt={alt}
          draggable={false}
          style={{
            display: "block",
            width: scrollable ? (natW ? `${natW}px` : "auto") : "100%",
            maxWidth: scrollable ? "none" : "100%",
            height: "auto",
            objectFit: "contain",
            borderRadius: scrollable ? 0 : 14,
            transform: `scale(${scale})`,
            transformOrigin: "top left",
            boxShadow: scrollable ? "none" : "0 8px 30px rgba(0,0,0,0.4)",
            border: scrollable ? "none" : "1px solid rgba(56,189,248,0.12)",
          }}
          onLoad={e => {
            const { naturalWidth: nw, naturalHeight: nh } = e.currentTarget;
            setNatSize([nw, nh]);
            const { width, height } = e.currentTarget.getBoundingClientRect();
            setBoxSize([width, height]);
          }}
          onMouseEnter={e => { if (!enableMagnifier) return; const { width, height } = e.currentTarget.getBoundingClientRect(); setBoxSize([width, height]); setShowMag(true); }}
          onMouseMove={e => { if (!enableMagnifier) return; const r = e.currentTarget.getBoundingClientRect(); setXY([e.clientX - r.left, e.clientY - r.top]); }}
          onMouseLeave={() => enableMagnifier && setShowMag(false)}
        />
        {enableMagnifier && showMag && (
          <div style={{
            pointerEvents: "none", position: "absolute", zIndex: 10,
            top: y - MZH / 2, left: x - MZW / 2,
            width: MZW, height: MZH,
            border: "2px solid #38bdf8", borderRadius: "50%",
            boxShadow: "0 0 20px rgba(56,189,248,0.4)",
            backgroundImage: `url('${src}')`,
            backgroundRepeat: "no-repeat",
            backgroundSize: bgSize,
            backgroundPositionX: bgPosX,
            backgroundPositionY: bgPosY,
          }} />
        )}
      </div>
    </div>
  );
};

/* ─── Main Report Page ──────────────────────────────────────────── */
export default function Reports() {
  const { reportId } = useParams();
  const navigate = useNavigate();
  const [report, setReport]           = useState(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState("");
  const [isEditing, setIsEditing]     = useState(false);
  const [editFields, setEditFields]   = useState({});
  const [saving, setSaving]           = useState(false);
  const [statusBusy, setStatusBusy]   = useState(false);
  const [xaiStructured, setXaiStructured] = useState(null);
  const [successMsg, setSuccessMsg]   = useState("");

  /* ─── Fetch ─────────────────────────────────────────────────────── */
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const token = localStorage.getItem("token");
        const res   = await fetch(`${BASE}doctor/reports/${reportId}`, {
          headers: { Authorization: `Bearer ${token}` },
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Failed to fetch report (${res.status})`);
        const data = await res.json();
        setReport(data);
        setEditFields({ diagnosis: data.diagnosis_report || "", doctor_message: data.patient_overview?.doctor_message || "" });
        if (data.xai_structured) {
          try { setXaiStructured(typeof data.xai_structured === "string" ? JSON.parse(data.xai_structured) : data.xai_structured); }
          catch { setXaiStructured(null); }
        }
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message);
      } finally { setLoading(false); }
    })();
    return () => controller.abort();
  }, [reportId]);

  /* ─── Save edits ─────────────────────────────────────────────────── */
  const handleSave = async () => {
    setSaving(true);
    try {
      const token = localStorage.getItem("token");
      const res   = await fetch(`${BASE}doctor/reports/${reportId}`, {
        method: "PUT",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ diagnosis: editFields.diagnosis, doctor_message: editFields.doctor_message }),
      });
      if (!res.ok) throw new Error("Save failed");
      setReport(prev => ({
        ...prev,
        diagnosis_report: editFields.diagnosis,
        patient_overview: { ...prev.patient_overview, doctor_message: editFields.doctor_message },
      }));
      setIsEditing(false);
      setSuccessMsg("Report updated successfully!");
      setTimeout(() => setSuccessMsg(""), 3500);
    } catch (e) { setError(e.message); }
    finally { setSaving(false); }
  };

  /* ─── Status update ──────────────────────────────────────────────── */
  const updateStatus = async (action) => {
    setStatusBusy(true);
    try {
      const token = localStorage.getItem("token");
      const res   = await fetch(`${BASE}doctor/reports/${reportId}/${action}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`Status update failed (${res.status})`);
      const newStatus = action === "approve" ? "Approved" : "Rejected";
      setReport(prev => ({ ...prev, patient_overview: { ...prev.patient_overview, status: newStatus } }));
      setSuccessMsg(`Case ${newStatus}!`);
      setTimeout(() => setSuccessMsg(""), 3500);
    } catch (e) { setError(e.message); }
    finally { setStatusBusy(false); }
  };

  /* ─── PDF download ───────────────────────────────────────────────── */
  const downloadPDF = async () => {
    try {
      const token = localStorage.getItem("token");
      const res   = await fetch(`${BASE}doctor/reports/${reportId}/pdf`, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error("PDF unavailable");
      const blob = await res.blob();
      const url  = window.URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url; a.download = `report_${reportId}.pdf`; document.body.appendChild(a); a.click(); a.remove();
    } catch (e) { alert("PDF download: " + e.message); }
  };

  /* ─── Loading / Error ────────────────────────────────────────────── */
  const Shell = ({ children }) => (
    <div style={{ background: "#020818", minHeight: "100vh", color: "#fff", fontFamily: "'Inter', sans-serif" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'); *{box-sizing:border-box;margin:0;padding:0;}`}</style>
      <Navbar />
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem" }}>{children}</div>
    </div>
  );

  if (loading) return (
    <Shell>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", paddingTop: "6rem", color: "#64748b" }}>
        <div style={{ width: 40, height: 40, border: "3px solid rgba(56,189,248,0.2)", borderTopColor: "#38bdf8", borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        Loading report details…
      </div>
    </Shell>
  );

  if (error && !report) return (
    <Shell>
      <div style={{ paddingTop: "4rem", textAlign: "center", color: "#fca5a5" }}>
        <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>⚠️</div>
        <p style={{ fontSize: "1rem", marginBottom: "1.5rem" }}>{error}</p>
        <button onClick={() => navigate(-1)} style={{ background: "rgba(56,189,248,0.1)", border: "1px solid rgba(56,189,248,0.25)", color: "#38bdf8", borderRadius: 10, padding: "0.6rem 1.4rem", cursor: "pointer", fontWeight: 700 }}>
          ← Back to Dashboard
        </button>
      </div>
    </Shell>
  );

  if (!report?.patient_overview) return <Shell><p style={{ color: "#64748b", paddingTop: "4rem", textAlign: "center" }}>No report found.</p></Shell>;

  /* ─── Image URL resolver ─────────────────────────────────────────── */
  const resolveUrl = (path) => {
    if (!path) return null;
    if (path.startsWith("http")) return path;
    if (path.includes("outputs/")) return BASE + "uploads/" + path.substring(path.indexOf("outputs/") + 8).replace(/^\/+/, "");
    if (path.startsWith("/uploads/")) return BASE + path.replace(/^\//, "");
    if (path.startsWith("uploads/")) return BASE + path;
    return BASE + "uploads/" + path.replace(/^\/+/, "");
  };

  const { patient_overview, diagnosis_report, xai_report, original_xray, gradcam_overlay, classification_results = [] } = report;
  const structured = xaiStructured || {};
  const status = patient_overview.status || "Pending";
  const sc = statusStyle(status);

  const imagesToShow = [];
  const seen = new Set();
  [original_xray || structured.original_xray, gradcam_overlay || structured.gradcam_overlay].forEach((p, i) => {
    const url = resolveUrl(p);
    if (url && !seen.has(url)) { imagesToShow.push({ label: i === 0 ? "Original Chest X-ray" : "Grad-CAM Overlay", src: url }); seen.add(url); }
  });

  const classification = (classification_results?.length ? classification_results : structured.classification_results) || [];
  const diagnosisText  = (diagnosis_report || "").trim() || null;
  const xaiText        = (xai_report || "").trim() || null;

  const isApproved = status.toLowerCase() === "approved";
  const isRejected = status.toLowerCase() === "rejected";

  return (
    <div style={{ background: "#020818", minHeight: "100vh", color: "#fff", fontFamily: "'Inter', sans-serif", position: "relative" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; background: #0a1122; }
        ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.25); border-radius: 6px; }
        textarea { background: rgba(10,17,34,0.9) !important; border: 1px solid rgba(56,189,248,0.25) !important; color: #f8fafc !important; border-radius: 12px !important; padding: 0.8rem 1rem !important; font-family: inherit !important; font-size: 0.95rem !important; width: 100% !important; resize: vertical !important; outline: none !important; transition: border-color 0.2s; }
        textarea:focus { border-color: rgba(56,189,248,0.5) !important; }
        .prose-dark p { color: #cbd5e1; line-height: 1.75; margin-bottom: 0.8rem; }
        .prose-dark h1,.prose-dark h2,.prose-dark h3 { color: #f0f6ff; margin-top: 1.2rem; margin-bottom: 0.5rem; }
        .prose-dark code { background: rgba(56,189,248,0.1); color: #7dd3fc; border-radius: 6px; padding: 0.15rem 0.4rem; font-size: 0.87em; }
        .action-btn { border: none; border-radius: 12px; padding: 0.65rem 1.5rem; font-size: 0.9rem; font-weight: 700; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.5rem; }
        .action-btn:hover { transform: translateY(-2px); }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; transform: translateY(-8px); } to { opacity:1; transform:none; } }
      `}</style>

      {/* Ambient glows */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "5%", left: "10%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle,rgba(56,189,248,0.06) 0%,transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "10%", right: "5%", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle,rgba(99,102,241,0.06) 0%,transparent 65%)" }} />
      </div>

      <div style={{ position: "relative", zIndex: 10 }}><Navbar /></div>

      <main style={{ maxWidth: 960, margin: "0 auto", padding: "2rem 1.5rem 5rem", position: "relative", zIndex: 1 }}>

        {/* Back button */}
        <button onClick={() => navigate(-1)} style={{
          background: "rgba(15,23,42,0.7)", border: "1px solid rgba(56,189,248,0.2)", color: "#94a3b8",
          borderRadius: 10, padding: "0.55rem 1.1rem", fontSize: "0.88rem", fontWeight: 600, cursor: "pointer",
          marginBottom: "1.5rem", display: "inline-flex", alignItems: "center", gap: "0.5rem", transition: "all 0.2s",
        }}>
          <FaArrowLeft /> Back to Dashboard
        </button>

        {/* Success / Error toast */}
        {successMsg && (
          <div style={{ animation: "fadeIn 0.3s ease", background: "rgba(34,197,94,0.14)", border: "1px solid rgba(34,197,94,0.35)", borderRadius: 14, padding: "0.85rem 1.4rem", color: "#4ade80", fontWeight: 700, fontSize: "0.92rem", marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <FaCheckCircle /> {successMsg}
          </div>
        )}
        {error && (
          <div style={{ animation: "fadeIn 0.3s ease", background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 14, padding: "0.85rem 1.4rem", color: "#f87171", fontWeight: 600, fontSize: "0.9rem", marginBottom: "1.5rem" }}>
            ⚠️ {error}
          </div>
        )}

        {/* ─── Hero Header ──────────────────────────────────────────── */}
        <Card>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "1.5rem" }}>
            <div>
              <div style={{ fontSize: "0.72rem", color: "#38bdf8", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.5rem" }}>
                Clinical Report Review
              </div>
              <h1 style={{ fontSize: "1.9rem", fontWeight: 900, color: "#f0f6ff", letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: "0.6rem" }}>
                {patient_overview.patient_name || "Patient Report"}
              </h1>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                <StatusPill status={status} />
                {patient_overview.submission_date && (
                  <span style={{ fontSize: "0.84rem", color: "#64748b" }}>
                    📅 {new Date(patient_overview.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
                  </span>
                )}
                <span style={{ fontSize: "0.78rem", color: "#475569", fontFamily: "monospace" }}>
                  #{String(reportId).slice(0, 12)}…
                </span>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.65rem" }}>
              {isEditing ? (
                <>
                  <button className="action-btn" disabled={saving} onClick={handleSave}
                    style={{ background: "linear-gradient(135deg,#059669,#10b981)", color: "#fff", boxShadow: "0 4px 16px rgba(16,185,129,0.3)" }}>
                    {saving ? <span style={{ display: "inline-block", width: 14, height: 14, border: "2px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} /> : <FaSave />}
                    {saving ? "Saving…" : "Save Changes"}
                  </button>
                  <button className="action-btn" onClick={() => setIsEditing(false)}
                    style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(56,189,248,0.2)", color: "#94a3b8" }}>
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button className="action-btn" onClick={() => setIsEditing(true)}
                    style={{ background: "rgba(56,189,248,0.12)", border: "1px solid rgba(56,189,248,0.3)", color: "#38bdf8" }}>
                    <FaEdit /> Edit Report
                  </button>
                  {!isApproved && (
                    <button className="action-btn" disabled={statusBusy} onClick={() => updateStatus("approve")}
                      style={{ background: "linear-gradient(135deg,#059669,#10b981)", color: "#fff", boxShadow: "0 4px 14px rgba(16,185,129,0.3)" }}>
                      <FaCheckCircle />{statusBusy ? "…" : "Approve"}
                    </button>
                  )}
                  {!isRejected && (
                    <button className="action-btn" disabled={statusBusy} onClick={() => updateStatus("reject")}
                      style={{ background: "linear-gradient(135deg,#dc2626,#ef4444)", color: "#fff", boxShadow: "0 4px 14px rgba(239,68,68,0.25)" }}>
                      <FaTimesCircle />{statusBusy ? "…" : "Reject"}
                    </button>
                  )}
                  <button className="action-btn" onClick={downloadPDF}
                    style={{ background: "rgba(15,23,42,0.7)", border: "1px solid rgba(56,189,248,0.2)", color: "#94a3b8" }}>
                    <FaFilePdf /> PDF
                  </button>
                </>
              )}
            </div>
          </div>
        </Card>

        {/* ─── Patient Overview ─────────────────────────────────────── */}
        <Card>
          <SectionTitle icon={<FaUser style={{ color: "#38bdf8" }} />} hint="Patient metadata">
            Patient Overview
          </SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1.5rem" }}>
            <InfoRow label="Patient Name">{patient_overview.patient_name || "—"}</InfoRow>
            <InfoRow label="Submission Date">
              {patient_overview.submission_date
                ? new Date(patient_overview.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })
                : "—"}
            </InfoRow>
            <InfoRow label="Clinical Symptoms">{patient_overview.symptoms || "Not recorded"}</InfoRow>
            <InfoRow label="Current Status"><StatusPill status={status} /></InfoRow>
          </div>

          {/* Doctor message */}
          <div style={{ marginTop: "1.5rem", paddingTop: "1.5rem", borderTop: "1px solid rgba(56,189,248,0.1)" }}>
            <div style={{ fontSize: "0.78rem", color: "#38bdf8", fontWeight: 700, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: "0.6rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <FaCommentMedical /> Doctor's Message to Patient
            </div>
            {isEditing ? (
              <textarea rows={3} value={editFields.doctor_message} onChange={e => setEditFields(p => ({ ...p, doctor_message: e.target.value }))} placeholder="Add a message for the patient..." />
            ) : (
              <div style={{ color: patient_overview.doctor_message ? "#cbd5e1" : "#475569", fontSize: "0.93rem", lineHeight: 1.65, fontStyle: patient_overview.doctor_message ? "normal" : "italic" }}>
                {patient_overview.doctor_message || "No message recorded yet."}
              </div>
            )}
          </div>
        </Card>

        {/* ─── Diagnosis Report ─────────────────────────────────────── */}
        <Card>
          <SectionTitle icon={<FaNotesMedical style={{ color: "#a78bfa" }} />} hint="Primary AI conclusion">
            Diagnosis Report
          </SectionTitle>
          {isEditing ? (
            <textarea rows={7} value={editFields.diagnosis} onChange={e => setEditFields(p => ({ ...p, diagnosis: e.target.value }))} placeholder="Edit diagnosis report..." />
          ) : diagnosisText ? (
            <div className="prose-dark">
              <ReactMarkdown>{diagnosisText}</ReactMarkdown>
            </div>
          ) : (
            <div style={{ color: "#475569", fontStyle: "italic" }}>No diagnosis report generated yet.</div>
          )}
        </Card>

        {/* ─── XAI Imaging ─────────────────────────────────────────── */}
        {imagesToShow.length > 0 && (
          <Card>
            <SectionTitle icon={<FaImage style={{ color: "#f472b6" }} />} hint="Radiograph & AI attention maps">
              Imaging Results
            </SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "1.5rem" }}>
              {imagesToShow.map((img, i) => (
                <div key={i} style={{ background: "rgba(10,17,34,0.6)", border: "1px solid rgba(56,189,248,0.12)", borderRadius: 18, padding: "1.2rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  <div style={{ fontSize: "0.82rem", color: "#94a3b8", fontWeight: 700, textAlign: "center" }}>{img.label}</div>
                  <ImageViewer src={img.src} alt={img.label} enableMagnifier />
                  <div style={{ fontSize: "0.76rem", color: "#475569", textAlign: "center" }}>
                    {i === 0 ? "Raw radiographic scan for clinical review." : "Grad-CAM heatmap highlighting model attention areas."}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* ─── Classification Table ─────────────────────────────────── */}
        {classification.length > 0 && (
          <Card>
            <SectionTitle icon={<FaTable style={{ color: "#34d399" }} />} hint="ResNet-50 pathology scores">
              Model Prediction Scores
            </SectionTitle>
            <div style={{ overflowX: "auto", borderRadius: 14, overflow: "hidden", border: "1px solid rgba(56,189,248,0.1)" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                <thead>
                  <tr style={{ background: "rgba(10,17,34,0.8)", borderBottom: "1px solid rgba(56,189,248,0.12)" }}>
                    <th style={{ padding: "0.9rem 1.3rem", fontSize: "0.75rem", fontWeight: 700, color: "#38bdf8", letterSpacing: "0.07em", textTransform: "uppercase" }}>Pathology</th>
                    <th style={{ padding: "0.9rem 1.3rem", fontSize: "0.75rem", fontWeight: 700, color: "#38bdf8", letterSpacing: "0.07em", textTransform: "uppercase" }}>Probability</th>
                    <th style={{ padding: "0.9rem 1.3rem", width: "40%" }}></th>
                  </tr>
                </thead>
                <tbody>
                  {[...classification].sort((a, b) => b[1] - a[1]).map(([label, prob], i) => {
                    const pct = Math.round((prob || 0) * 100);
                    const barColor = pct >= 70 ? "#34d399" : pct >= 40 ? "#fbbf24" : "#f87171";
                    return (
                      <tr key={label} style={{ background: i % 2 === 0 ? "transparent" : "rgba(15,23,42,0.3)", borderBottom: "1px solid rgba(56,189,248,0.06)" }}>
                        <td style={{ padding: "0.85rem 1.3rem", color: i === 0 ? "#f0f6ff" : "#cbd5e1", fontWeight: i === 0 ? 700 : 500, fontSize: "0.9rem" }}>{label}</td>
                        <td style={{ padding: "0.85rem 1.3rem", fontWeight: 700, color: barColor, fontSize: "0.95rem" }}>{pct}%</td>
                        <td style={{ padding: "0.85rem 1.3rem" }}>
                          <div style={{ height: 8, borderRadius: 8, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${Math.max(5, pct)}%`, background: barColor, borderRadius: 8, transition: "width 0.6s ease" }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div style={{ fontSize: "0.77rem", color: "#475569", textAlign: "center", marginTop: "0.75rem" }}>
              <FaInfoCircle style={{ marginRight: 4 }} />Green ≥70% · Yellow 40–69% · Red &lt;40%
            </div>
          </Card>
        )}

        {/* ─── XAI Report Text ─────────────────────────────────────── */}
        {xaiText && (
          <Card>
            <SectionTitle icon={<FaCog style={{ color: "#38bdf8" }} />} hint="Explainability analysis">
              Explainable AI (XAI) Report
            </SectionTitle>
            <div className="prose-dark">
              <ReactMarkdown>{xaiText}</ReactMarkdown>
            </div>
          </Card>
        )}

      </main>
    </div>
  );
}
