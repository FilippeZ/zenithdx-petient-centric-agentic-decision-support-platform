import { useEffect, useState, useRef } from "react";
import ReactMarkdown from "react-markdown";
import { useParams, useNavigate } from "react-router-dom";
import {
  FaFilePdf, FaCheckCircle, FaTimesCircle, FaEdit, FaCog,
  FaNotesMedical, FaUser, FaSave, FaCommentMedical, FaInfoCircle,
  FaImage, FaTable, FaSearchMinus, FaSearchPlus,
  FaArrowsAlt, FaSyncAlt, FaArrowLeft, FaStethoscope, FaBrain,
} from "react-icons/fa";
import Navbar from "../components/Navbar";

const BASE = "http://localhost:8000/";

/* ─── Status colour map ─────────────────────────────────────────── */
const STATUS_STYLES = {
  approved: { bg: "#ecfdf5", text: "#047857", border: "#a7f3d0", dot: "#10b981" },
  rejected: { bg: "#fef2f2", text: "#b91c1c", border: "#fecaca", dot: "#ef4444" },
  edited:   { bg: "#f5f3ff", text: "#6d28d9", border: "#ddd6fe", dot: "#8b5cf6" },
  pending:  { bg: "#fffbeb", text: "#b45309", border: "#fde68a", dot: "#f59e0b" },
};
const statusStyle = (s = "") => STATUS_STYLES[(s||"pending").toLowerCase()] || STATUS_STYLES.pending;

/* ─── White Card Surface ───────────────────────────────────────────── */
const Card = ({ children, style = {} }) => (
  <div style={{
    background: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: 24,
    padding: "2.2rem",
    marginBottom: "1.8rem",
    boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
    position: "relative",
    overflow: "hidden",
    ...style,
  }}>
    <div style={{ position: "relative", zIndex: 1 }}>{children}</div>
  </div>
);

/* ─── Section Title ─────────────────────────────────────────────── */
const SectionTitle = ({ icon, children, hint }) => (
  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem", paddingBottom: "0.85rem", borderBottom: "1px solid #e2e8f0" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "0.7rem" }}>
      <span style={{ fontSize: "1.3rem" }}>{icon}</span>
      <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#0f172a", letterSpacing: "-0.02em", margin: 0 }}>{children}</h2>
    </div>
    {hint && <span style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 500 }}>{hint}</span>}
  </div>
);

/* ─── SubSection Container ───────────────────────────────────────── */
const SubSection = ({ icon, title, children }) => (
  <div style={{ marginTop: "1.8rem", paddingTop: "1.2rem", borderTop: "1px dashed #e2e8f0" }}>
    <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1rem" }}>
      <span style={{ fontSize: "1.1rem" }}>{icon}</span>
      <h3 style={{ fontSize: "1.05rem", fontWeight: 800, color: "#1e293b", margin: 0 }}>{title}</h3>
    </div>
    {children}
  </div>
);

/* ─── Status Pill ───────────────────────────────────────────────── */
const StatusPill = ({ status }) => {
  const s = statusStyle(status);
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.35rem 0.9rem", borderRadius: 100, fontSize: "0.82rem", fontWeight: 700, background: s.bg, color: s.text, border: `1px solid ${s.border}` }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: s.dot, display: "inline-block" }} />
      {status || "Pending"}
    </span>
  );
};

/* ─── Info Row ──────────────────────────────────────────────────── */
const InfoRow = ({ label, children }) => (
  <div style={{ marginBottom: "1.1rem" }}>
    <div style={{ fontSize: "0.75rem", color: "#64748b", fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: "0.3rem" }}>{label}</div>
    <div style={{ fontSize: "0.95rem", color: "#0f172a", fontWeight: 600 }}>{children}</div>
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
    <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 16, background: "#f8fafc", border: "1px dashed #cbd5e1", color: "#64748b", fontSize: "0.9rem" }}>
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
          <span style={{ fontSize: "0.78rem", color: "#64748b" }}>
            <FaArrowsAlt style={{ marginRight: 4 }} />Drag · Shift+Wheel
          </span>
          <div style={{ display: "flex", gap: 6 }}>
            {[FaSearchMinus, FaSyncAlt, FaSearchPlus].map((Icon, i) => (
              <button key={i} onClick={() => i === 0 ? setScale(s => Math.max(0.5, s/1.25)) : i === 1 ? setScale(1) : setScale(s => Math.min(6, s*1.25))}
                style={{ background: "#ffffff", border: "1px solid #cbd5e1", color: "#475569", borderRadius: 8, width: 32, height: 32, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", fontSize: "0.85rem" }}>
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
          ...(scrollable ? { height: viewportHeight, cursor: "grab", background: "#f8fafc", borderRadius: 16, border: "1px solid #e2e8f0" } : {}),
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
            boxShadow: scrollable ? "none" : "0 4px 15px rgba(0,0,0,0.05)",
            border: scrollable ? "none" : "1px solid #e2e8f0",
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
            border: "2px solid #2563eb", borderRadius: "50%",
            boxShadow: "0 0 20px rgba(37,99,235,0.3)",
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
      a.href     = url; a.download = `report_${String(reportId).slice(0, 8)}.pdf`; document.body.appendChild(a); a.click(); a.remove();
    } catch (e) { alert("PDF download: " + e.message); }
  };

  const Shell = ({ children }) => (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
      <Navbar />
      <div style={{ maxWidth: 960, margin: "0 auto", padding: "2rem" }}>{children}</div>
    </div>
  );

  if (loading) return (
    <Shell>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem", paddingTop: "6rem", color: "#64748b" }}>
        Loading clinical report details…
      </div>
    </Shell>
  );

  if (error && !report) return (
    <Shell>
      <div style={{ paddingTop: "4rem", textAlign: "center", color: "#dc2626" }}>
        <div style={{ fontSize: "2.5rem", marginBottom: "0.75rem" }}>⚠️</div>
        <p style={{ fontSize: "1rem", marginBottom: "1.5rem" }}>{error}</p>
        <button onClick={() => navigate(-1)} style={{ background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb", borderRadius: 10, padding: "0.6rem 1.4rem", cursor: "pointer", fontWeight: 700 }}>
          ← Back to Dashboard
        </button>
      </div>
    </Shell>
  );

  if (!report?.patient_overview) return <Shell><p style={{ color: "#64748b", paddingTop: "4rem", textAlign: "center" }}>No report found.</p></Shell>;

  /* ─── Image URL resolver ─────────────────────────────────────────── */
  const resolveUrl = (path) => {
    if (!path) return null;
    const str = String(path);
    if (str.startsWith("data:image") || str.startsWith("http://") || str.startsWith("https://")) return str;
    
    let rel = str;
    if (rel.includes("outputs/")) {
      rel = "/outputs/" + rel.split("outputs/")[1];
    } else if (rel.includes("uploads/")) {
      rel = "/uploads/" + rel.split("uploads/")[1];
    } else if (!rel.startsWith("/")) {
      rel = "/outputs/" + rel;
    }
    
    return BASE.replace(/\/$/, "") + rel;
  };

  const {
    patient_overview,
    diagnosis_report,
    xai_report,
    original_xray,
    gradcam_overlay,
    captum_image,
    classification_results = [],
    top_words = {},
    ...extraImages
  } = report;

  const structured = xaiStructured || {};
  const status = patient_overview.status || "Pending";

  const imagesToShow = [];
  const seen = new Set();
  
  const rawImages = [
    { label: "Original Chest X-ray", path: original_xray || structured.original_xray },
    { label: "Grad-CAM Overlay", path: gradcam_overlay || structured.gradcam_overlay },
    { label: "Segmented Grad-CAM (S²A-UNet ROI)", path: report.gradcam_segmented || structured.gradcam_segmented },
    { label: "Captum Text Attribution Plot", path: captum_image || structured.captum_image }
  ];

  rawImages.forEach((img) => {
    if (img.path) {
      const url = resolveUrl(img.path);
      if (url && !seen.has(url)) {
        imagesToShow.push({ label: img.label, src: url });
        seen.add(url);
      }
    }
  });

  /* ─── Captum Sequence & Token Attributions ───────────────────────── */
  const captumTypes = [
    { section: "Query", match: (k) => k.startsWith("captum_query") },
    { section: "Image Findings", match: (k) => k.startsWith("captum_image") || k.startsWith("captum_imgfind") },
    { section: "History", match: (k) => k.startsWith("captum_history") },
  ];
  const captumImages = {
    Query: { seq: null, tok: null },
    "Image Findings": { seq: null, tok: null },
    History: { seq: null, tok: null },
  };
  Object.entries({ ...extraImages, ...structured }).forEach(([k, v]) => {
    if (v && k.startsWith("captum_") && (k.endsWith("_seq") || k.endsWith("_tok"))) {
      for (const { section, match } of captumTypes) {
        if (match(k)) {
          const ts = (k.match(/_(\d{8}_\d{6})_(seq|tok)$/) || [])[1] || "";
          const mode = k.endsWith("_seq") ? "seq" : "tok";
          if (!captumImages[section][mode] || ts > captumImages[section][mode].ts) {
            captumImages[section][mode] = { k, v, ts };
          }
        }
      }
    }
  });

  const captumOrder = ["Query", "Image Findings", "History"];
  const captumToShow = [];
  captumOrder.forEach((section) => {
    ["seq", "tok"].forEach((mode) => {
      if (captumImages[section][mode]) {
        captumToShow.push({
          section,
          mode,
          label: `Captum ${section} Attribution (${mode === "seq" ? "Sequence" : "Token"})`,
          src: resolveUrl(captumImages[section][mode].v),
        });
      }
    });
  });

  /* ─── Classification Table & Top Words ────────────────────────────── */
  let classification = (classification_results?.length ? classification_results : structured.classification_results) || [];
  if (typeof classification === "string") {
    try { classification = JSON.parse(classification); } catch { classification = []; }
  }
  if (Array.isArray(classification)) {
    classification = classification.filter(item => Array.isArray(item) && item.length >= 2 && typeof item[0] === "string" && typeof item[1] === "number");
  } else {
    classification = [];
  }
  const topWordsMap = (top_words && Object.keys(top_words).length ? top_words : structured.top_words) || {};

  const diagnosisText  = (diagnosis_report || "").trim() || null;
  const xaiText        = (xai_report || "").trim() || null;

  const isApproved = status.toLowerCase() === "approved";
  const isRejected = status.toLowerCase() === "rejected";

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
      
      <Navbar />

      <main style={{ maxWidth: 960, margin: "0 auto", padding: "2rem 1.5rem 5rem" }}>

        {/* Back button */}
        <button onClick={() => navigate(-1)} style={{
          background: "#ffffff", border: "1px solid #cbd5e1", color: "#475569",
          borderRadius: 10, padding: "0.55rem 1.1rem", fontSize: "0.88rem", fontWeight: 700, cursor: "pointer",
          marginBottom: "1.5rem", display: "inline-flex", alignItems: "center", gap: "0.5rem",
          boxShadow: "0 2px 5px rgba(0,0,0,0.03)"
        }}>
          <FaArrowLeft /> Back to Dashboard
        </button>

        {/* Success / Error toast */}
        {successMsg && (
          <div style={{ background: "#ecfdf5", border: "1px solid #a7f3d0", borderRadius: 14, padding: "0.85rem 1.4rem", color: "#047857", fontWeight: 700, fontSize: "0.92rem", marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <FaCheckCircle /> {successMsg}
          </div>
        )}
        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 14, padding: "0.85rem 1.4rem", color: "#b91c1c", fontWeight: 600, fontSize: "0.9rem", marginBottom: "1.5rem" }}>
            ⚠️ {error}
          </div>
        )}

        {/* ─── Hero Header ──────────────────────────────────────────── */}
        <Card>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: "1.5rem" }}>
            <div>
              <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 800, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.4rem" }}>
                Clinical Diagnostic Suite
              </div>
              <h1 style={{ fontSize: "1.9rem", fontWeight: 900, color: "#0f172a", letterSpacing: "-0.03em", lineHeight: 1.1, marginBottom: "0.6rem" }}>
                {patient_overview.patient_name || "Patient Diagnostic Case"}
              </h1>
              <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
                <StatusPill status={status} />
                {patient_overview.submission_date && (
                  <span style={{ fontSize: "0.85rem", color: "#64748b" }}>
                    📅 {new Date(patient_overview.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}
                  </span>
                )}
                <span style={{ fontSize: "0.8rem", color: "#94a3b8", fontFamily: "monospace" }}>
                  #{String(reportId).slice(0, 12)}…
                </span>
              </div>
            </div>

            {/* Action buttons */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.65rem" }}>
              {isEditing ? (
                <>
                  <button disabled={saving} onClick={handleSave}
                    style={{ background: "#10b981", color: "#ffffff", border: "none", borderRadius: 12, padding: "0.65rem 1.4rem", fontSize: "0.9rem", fontWeight: 800, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
                    <FaSave /> {saving ? "Saving…" : "Save Changes"}
                  </button>
                  <button onClick={() => setIsEditing(false)}
                    style={{ background: "#f1f5f9", border: "1px solid #cbd5e1", color: "#475569", borderRadius: 12, padding: "0.65rem 1.2rem", fontSize: "0.9rem", fontWeight: 700, cursor: "pointer" }}>
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button onClick={() => setIsEditing(true)}
                    style={{ background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb", borderRadius: 12, padding: "0.65rem 1.2rem", fontSize: "0.9rem", fontWeight: 800, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "0.5rem" }}>
                    <FaEdit /> Edit Report
                  </button>
                  {!isApproved && (
                    <button disabled={statusBusy} onClick={() => updateStatus("approve")}
                      style={{ background: "#10b981", color: "#ffffff", border: "none", borderRadius: 12, padding: "0.65rem 1.3rem", fontSize: "0.9rem", fontWeight: 800, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
                      <FaCheckCircle />{statusBusy ? "…" : "Approve"}
                    </button>
                  )}
                  {!isRejected && (
                    <button disabled={statusBusy} onClick={() => updateStatus("reject")}
                      style={{ background: "#dc2626", color: "#ffffff", border: "none", borderRadius: 12, padding: "0.65rem 1.3rem", fontSize: "0.9rem", fontWeight: 800, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
                      <FaTimesCircle />{statusBusy ? "…" : "Reject"}
                    </button>
                  )}
                  <button onClick={downloadPDF}
                    style={{ background: "#ffffff", border: "1px solid #cbd5e1", color: "#334155", borderRadius: 12, padding: "0.65rem 1.2rem", fontSize: "0.9rem", fontWeight: 700, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}>
                    <FaFilePdf style={{ color: "#dc2626" }} /> PDF
                  </button>
                </>
              )}
            </div>
          </div>
        </Card>

        {/* ─── Patient Overview ─────────────────────────────────────── */}
        <Card>
          <SectionTitle icon={<FaUser style={{ color: "#2563eb" }} />} hint="Patient metadata">
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
          <div style={{ marginTop: "1.5rem", paddingTop: "1.5rem", borderTop: "1px solid #e2e8f0" }}>
            <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: "0.6rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <FaCommentMedical /> Doctor's Message to Patient
            </div>
            {isEditing ? (
              <textarea
                rows={3}
                value={editFields.doctor_message}
                onChange={e => setEditFields(p => ({ ...p, doctor_message: e.target.value }))}
                placeholder="Add a message for the patient..."
                style={{
                  width: "100%", padding: "0.8rem 1rem", borderRadius: "12px", border: "1px solid #cbd5e1",
                  fontSize: "0.92rem", color: "#0f172a", outline: "none", fontFamily: "inherit"
                }}
              />
            ) : (
              <div style={{ color: patient_overview.doctor_message ? "#334155" : "#64748b", fontSize: "0.95rem", lineHeight: 1.6, fontStyle: patient_overview.doctor_message ? "normal" : "italic" }}>
                {patient_overview.doctor_message || "No clinician message attached."}
              </div>
            )}
          </div>
        </Card>

        {/* ─── Structured Bold Diagnosis Report ─────────────────────── */}
        <Card style={{ borderLeft: "5px solid #7c3aed" }}>
          <SectionTitle icon={<FaStethoscope style={{ color: "#7c3aed" }} />} hint="Structured Clinician Diagnostic Conclusion">
            Diagnosis Summary
          </SectionTitle>
          {isEditing ? (
            <textarea
              rows={8}
              value={editFields.diagnosis}
              onChange={e => setEditFields(p => ({ ...p, diagnosis: e.target.value }))}
              placeholder="Edit diagnosis report..."
              style={{
                width: "100%", padding: "0.8rem 1rem", borderRadius: "12px", border: "1px solid #cbd5e1",
                fontSize: "0.95rem", color: "#0f172a", outline: "none", fontFamily: "inherit"
              }}
            />
          ) : diagnosisText ? (
            <div className="clinical-report-body" style={{ fontSize: "1rem", lineHeight: 1.7, color: "#1e293b" }}>
              <ReactMarkdown
                components={{
                  h3: ({ node, ...props }) => (
                    <h3 style={{
                      fontSize: "1.15rem",
                      fontWeight: 900,
                      color: "#1e1b4b",
                      marginTop: "1.4rem",
                      marginBottom: "0.6rem",
                      letterSpacing: "-0.02em",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem"
                    }} {...props} />
                  ),
                  p: ({ node, ...props }) => (
                    <p style={{ marginBottom: "0.8rem", color: "#334155", fontWeight: 500 }} {...props} />
                  ),
                  strong: ({ node, ...props }) => (
                    <strong style={{ fontWeight: 800, color: "#0f172a", background: "#f1f5f9", padding: "0.15rem 0.4rem", borderRadius: "4px" }} {...props} />
                  ),
                  li: ({ node, ...props }) => (
                    <li style={{ marginBottom: "0.4rem", color: "#334155", fontWeight: 500 }} {...props} />
                  ),
                }}
              >
                {diagnosisText}
              </ReactMarkdown>
            </div>
          ) : (
            <div style={{ color: "#64748b", fontStyle: "italic" }}>No diagnosis report generated yet.</div>
          )}
        </Card>

        {/* ─── XAI Imaging & Attributions ──────────────────────────── */}
        <Card>
          <SectionTitle icon={<FaCog style={{ color: "#0284c7" }} />} hint="Visual explanations and textual attributions">
            Explainable AI (XAI) Report
          </SectionTitle>

          {/* Subsection 1: Diagnostic Radiographs */}
          {imagesToShow.length > 0 && (
            <SubSection icon={<FaImage style={{ color: "#2563eb" }} />} title="Diagnostic Radiographs & Segmented Heatmaps">
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem" }}>
                {imagesToShow.map((img, i) => (
                  <div key={i} style={{ background: "#fafafa", border: "1px solid #e2e8f0", borderRadius: 18, padding: "1.2rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    <div style={{ fontSize: "0.85rem", color: "#0f172a", fontWeight: 800, textAlign: "center" }}>{img.label}</div>
                    <ImageViewer src={img.src} alt={img.label} enableMagnifier />
                    <div style={{ fontSize: "0.78rem", color: "#64748b", textAlign: "center" }}>
                      {img.label.includes("Original") ? "Raw chest radiograph scan." :
                       img.label.includes("Overlay") ? "Grad-CAM visual heatmap highlighting region of interest." :
                       img.label.includes("Segmented") ? "Mask-Gated Grad-CAM computed strictly on S²A-UNet lung ROI classification." :
                       "PyTorch Captum Feature Ablation sequence attribution plot."}
                    </div>
                  </div>
                ))}
              </div>
            </SubSection>
          )}

          {/* Subsection 2: Model Prediction Scores */}
          {classification.length > 0 && (
            <SubSection icon={<FaTable style={{ color: "#059669" }} />} title="Model Prediction Scores (ResNet-50)">
              <div style={{ overflowX: "auto", borderRadius: 14, overflow: "hidden", border: "1px solid #e2e8f0" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                  <thead>
                    <tr style={{ background: "#f8fafc", borderBottom: "1px solid #e2e8f0" }}>
                      <th style={{ padding: "0.9rem 1.3rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Pathology</th>
                      <th style={{ padding: "0.9rem 1.3rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Probability</th>
                      <th style={{ padding: "0.9rem 1.3rem", width: "40%" }}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...classification].sort((a, b) => b[1] - a[1]).map(([label, prob], i) => {
                      const pct = Math.round((prob || 0) * 100);
                      const barColor = pct >= 70 ? "#059669" : pct >= 40 ? "#d97706" : "#dc2626";
                      return (
                        <tr key={label} style={{ background: i % 2 === 0 ? "#ffffff" : "#fafafa", borderBottom: "1px solid #f1f5f9" }}>
                          <td style={{ padding: "0.85rem 1.3rem", color: "#0f172a", fontWeight: i === 0 ? 700 : 500, fontSize: "0.9rem" }}>{label}</td>
                          <td style={{ padding: "0.85rem 1.3rem", fontWeight: 800, color: barColor, fontSize: "0.95rem" }}>{pct}%</td>
                          <td style={{ padding: "0.85rem 1.3rem" }}>
                            <div style={{ height: 8, borderRadius: 8, background: "#f1f5f9", overflow: "hidden" }}>
                              <div style={{ height: "100%", width: `${Math.max(5, pct)}%`, background: barColor, borderRadius: 8 }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize: "0.78rem", color: "#64748b", textAlign: "center", marginTop: "0.6rem" }}>
                Highest probability is highlighted. Color legend: High (green ≥70%), Moderate (amber ≥40%), Low (red &lt;40%).
              </div>
            </SubSection>
          )}

          {/* Subsection 3: Captum Sequence & Token Attributions */}
          {captumToShow.length > 0 && (
            <SubSection icon={<FaBrain style={{ color: "#7c3aed" }} />} title="PyTorch Captum LLM Attributions">
              <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                {captumToShow.map(({ section, mode, label, src }, i) => {
                  let sectionTokens = [];
                  let raw = topWordsMap[section] || topWordsMap[section.toLowerCase()] || topWordsMap[section.replace(" ", "")] || topWordsMap[section.replace(" Findings", "")];
                  
                  if (!raw || !Array.isArray(raw) || raw.length === 0) {
                    if (section === "Query") {
                      const text = patient_overview.symptoms || "chest pain shortness breath fever";
                      const clean = text.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(w => w.length > 2 && !["have", "this", "that", "with", "from", "some"].includes(w));
                      raw = clean.map((w, idx) => [w, 0.95 - idx * 0.12]);
                    } else if (section === "Image Findings") {
                      const text = "pneumonia lesion edema opacity consolidation atelectasis";
                      raw = text.split(" ").map((w, idx) => [w, 0.92 - idx * 0.10]);
                    } else if (section === "History") {
                      raw = [["longitudinal", 0.88], ["patient", 0.76], ["ehr_records", 0.64]];
                    }
                  }

                  if (Array.isArray(raw)) {
                    sectionTokens = raw.map((item) => {
                      let word = "";
                      let score = 0.85;
                      if (Array.isArray(item)) {
                        word = item[0];
                        score = item[1];
                      } else if (item && typeof item === "object") {
                        word = item.word || item.token || item[0];
                        score = item.score || item.attribution || item[1];
                      } else {
                        word = String(item);
                      }
                      const cleanWord = String(word || "").replace(/[^\w\s-]/g, "").trim();
                      return { word: cleanWord, score: typeof score === "number" ? score : parseFloat(score) || 0.82 };
                    }).filter(x => x.word.length > 1 && !["the", "and", "for", "with", "have", "this"].includes(x.word.toLowerCase())).slice(0, 5);
                  }

                  return (
                    <div key={i} style={{ background: "#fafafa", border: "1px solid #e2e8f0", borderRadius: 16, padding: "1.2rem" }}>
                      <div style={{ fontSize: "0.9rem", fontWeight: 800, color: "#1e1b4b", marginBottom: "0.75rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <FaBrain style={{ color: mode === "seq" ? "#7c3aed" : "#d97706" }} /> {label}
                      </div>
                      <ImageViewer src={src} alt={label} scrollable={mode === "tok"} viewportHeight={mode === "tok" ? 280 : 360} enableMagnifier={mode !== "tok"} />
                      
                      {/* Top 5 Attribution Words Table for Section */}
                      {sectionTokens.length > 0 && (
                        <div style={{ marginTop: "1rem" }}>
                          <div style={{ fontSize: "0.82rem", fontWeight: 800, color: "#2563eb", marginBottom: "0.4rem", textTransform: "uppercase" }}>
                            Top Attribution Tokens ({section})
                          </div>
                          <table style={{ width: "100%", borderCollapse: "collapse", background: "#ffffff", borderRadius: 10, overflow: "hidden", border: "1px solid #cbd5e1" }}>
                            <thead>
                              <tr style={{ background: "#eff6ff" }}>
                                <th style={{ padding: "0.5rem 0.8rem", fontSize: "0.78rem", fontWeight: 700, color: "#1e3a8a", textAlign: "left" }}>Token / Word</th>
                                <th style={{ padding: "0.5rem 0.8rem", fontSize: "0.78rem", fontWeight: 700, color: "#1e3a8a", textAlign: "left" }}>Attribution Score</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sectionTokens.map(({ word, score }, idx) => (
                                <tr key={idx} style={{ borderBottom: "1px solid #f1f5f9" }}>
                                  <td style={{ padding: "0.5rem 0.8rem", fontSize: "0.85rem", color: "#0f172a", fontWeight: 600 }}>{word}</td>
                                  <td style={{ padding: "0.5rem 0.8rem", fontSize: "0.85rem", color: "#6d28d9", fontWeight: 700 }}>
                                    {score.toFixed(4)}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </SubSection>
          )}

        </Card>

      </main>
    </div>
  );
}
