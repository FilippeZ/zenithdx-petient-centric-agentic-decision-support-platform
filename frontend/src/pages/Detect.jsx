import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";

/* ── Animated top progress bar ───────────────────────────────── */
const ProgressBar = ({ active }) => {
  const [pct, setPct] = useState(0);
  const timer = useRef(null);

  useEffect(() => {
    if (active) {
      setPct(0);
      timer.current = setInterval(() => {
        setPct(p => (p < 88 ? p + Math.random() * 4 + 1.5 : p));
      }, 220);
    } else if (pct > 0) {
      clearInterval(timer.current);
      setPct(100);
      setTimeout(() => setPct(0), 600);
    }
    return () => clearInterval(timer.current);
    // eslint-disable-next-line
  }, [active]);

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, width: "100vw", zIndex: 9999,
      pointerEvents: "none", opacity: pct > 0 ? 1 : 0, transition: "opacity 0.4s",
    }}>
      <div style={{ width: "100%", height: 3, background: "rgba(129,140,248,0.15)" }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: "linear-gradient(90deg, #6366f1, #818cf8, #38bdf8)",
          boxShadow: "0 0 12px rgba(129,140,248,0.6)",
          transition: "width 0.35s cubic-bezier(.4,0,.2,1)",
          borderRadius: "0 4px 4px 0",
        }} />
      </div>
    </div>
  );
};

/* ── AI Processing animation steps ──────────────────────────── */
const AI_STEPS = [
  { icon: "🧬", label: "Preprocessing X-ray image…" },
  { icon: "🔬", label: "Running S²A-UNet lung segmentation…" },
  { icon: "🤖", label: "ResNet-50 multilabel classification…" },
  { icon: "📚", label: "Advanced RAG clinical knowledge search…" },
  { icon: "🧠", label: "LLM ReAct reasoning loop…" },
  { icon: "✨", label: "Self-Refine quality validation…" },
  { icon: "📊", label: "Generating Grad-CAM heatmaps…" },
  { icon: "📝", label: "Compiling structured diagnostic report…" },
];

const ProcessingOverlay = ({ active, stepIndex }) => {
  if (!active) return null;
  const step = AI_STEPS[stepIndex % AI_STEPS.length];
  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 1000,
      background: "rgba(2,8,24,0.85)",
      backdropFilter: "blur(12px)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", gap: "1.5rem",
    }}>
      <div style={{
        width: 80, height: 80, borderRadius: "50%",
        background: "rgba(129,140,248,0.1)",
        border: "2px solid rgba(129,140,248,0.3)",
        display: "flex", alignItems: "center", justifyContent: "center", fontSize: "2rem",
        animation: "spin-slow 3s linear infinite",
        boxShadow: "0 0 40px rgba(129,140,248,0.3)",
      }}>
        {step.icon}
      </div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f0f6ff", marginBottom: "0.5rem" }}>
          AI Analysis In Progress
        </div>
        <div style={{ fontSize: "0.9rem", color: "#818cf8", fontWeight: 600, animation: "fadeInUp 0.4s ease" }}>
          {step.label}
        </div>
      </div>
      <div style={{ display: "flex", gap: "0.4rem" }}>
        {AI_STEPS.map((_, i) => (
          <div key={i} style={{
            width: i === stepIndex % AI_STEPS.length ? 24 : 8,
            height: 8, borderRadius: 4,
            background: i === stepIndex % AI_STEPS.length ? "#818cf8" : "rgba(129,140,248,0.2)",
            transition: "all 0.4s ease",
          }} />
        ))}
      </div>
      <div style={{ color: "#475569", fontSize: "0.82rem" }}>
        This may take 30–90 seconds — please wait
      </div>
    </div>
  );
};

/* ── Drop zone ───────────────────────────────────────────────── */
const DropZone = ({ file, onFile }) => {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const previewUrl = file ? URL.createObjectURL(file) : null;

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) onFile(f);
  }, [onFile]);

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !file && inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? "#818cf8" : file ? "rgba(129,140,248,0.5)" : "rgba(129,140,248,0.2)"}`,
        borderRadius: 20,
        height: 320,
        display: "flex", alignItems: "center", justifyContent: "center",
        cursor: file ? "default" : "pointer",
        background: dragging
          ? "rgba(129,140,248,0.08)"
          : file
          ? "rgba(10,17,34,0.5)"
          : "rgba(10,17,34,0.3)",
        transition: "all 0.25s ease",
        position: "relative",
        overflow: "hidden",
        boxShadow: dragging ? "0 0 30px rgba(129,140,248,0.2)" : "none",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={e => { const f = e.target.files[0]; if (f) onFile(f); }}
      />

      {file ? (
        <>
          <img
            src={previewUrl}
            alt="X-ray preview"
            style={{ width: "100%", height: "100%", objectFit: "contain", borderRadius: 18 }}
          />
          {/* Replace button */}
          <button
            type="button"
            onClick={e => { e.stopPropagation(); inputRef.current?.click(); }}
            style={{
              position: "absolute", bottom: 12, right: 12,
              background: "rgba(129,140,248,0.9)",
              border: "none", color: "#fff", borderRadius: 8,
              padding: "0.4rem 0.85rem", fontSize: "0.78rem", fontWeight: 700,
              cursor: "pointer", backdropFilter: "blur(8px)",
            }}
          >
            Replace Image
          </button>
          <button
            type="button"
            onClick={e => { e.stopPropagation(); onFile(null); }}
            style={{
              position: "absolute", bottom: 12, left: 12,
              background: "rgba(239,68,68,0.85)",
              border: "none", color: "#fff", borderRadius: 8,
              padding: "0.4rem 0.75rem", fontSize: "0.78rem", fontWeight: 700,
              cursor: "pointer", backdropFilter: "blur(8px)",
            }}
          >
            ✕ Remove
          </button>
        </>
      ) : (
        <div style={{ textAlign: "center", color: "#475569" }}>
          <div style={{ fontSize: "3.5rem", marginBottom: "0.75rem", filter: "grayscale(0.3)" }}>🫁</div>
          <div style={{ fontSize: "1rem", fontWeight: 700, color: "#64748b", marginBottom: "0.4rem" }}>
            {dragging ? "Drop your X-ray here" : "Drag & drop chest X-ray"}
          </div>
          <div style={{ fontSize: "0.82rem", color: "#334155", marginBottom: "1rem" }}>
            or click to browse files
          </div>
          <div style={{
            display: "inline-block",
            padding: "0.45rem 1.1rem", borderRadius: 10,
            background: "rgba(129,140,248,0.12)", border: "1px solid rgba(129,140,248,0.25)",
            color: "#818cf8", fontSize: "0.8rem", fontWeight: 600,
          }}>
            PNG, JPG, DICOM supported
          </div>
        </div>
      )}
    </div>
  );
};

/* ── Main Page ───────────────────────────────────────────────── */
export default function Detect() {
  const [file, setFile]           = useState(null);
  const [symptoms, setSymptoms]   = useState("");
  const [patientId, setPatientId] = useState("");
  const [loading, setLoading]     = useState(false);
  const [aiStep, setAiStep]       = useState(0);
  const [message, setMessage]     = useState(null);
  const aiTimer = useRef(null);
  const navigate = useNavigate();

  /* Cycle through AI steps while loading */
  useEffect(() => {
    if (loading) {
      setAiStep(0);
      aiTimer.current = setInterval(() => setAiStep(s => s + 1), 2500);
    } else {
      clearInterval(aiTimer.current);
    }
    return () => clearInterval(aiTimer.current);
  }, [loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    if (!symptoms.trim()) {
      setMessage({ type: "error", text: "Please describe your symptoms — this field is required." });
      return;
    }
    setLoading(true);
    const fd = new FormData();
    if (file) fd.append("file", file);
    fd.append("symptoms", symptoms);
    if (patientId.trim()) fd.append("patient_id", patientId.trim());

    try {
      const token = localStorage.getItem("token");
      const res   = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: fd,
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (res.ok) {
        setMessage({ type: "success", text: "✅ Diagnosis request submitted! Your report is being processed." });
        setSymptoms(""); setFile(null); setPatientId("");
        setTimeout(() => navigate("/patient-dashboard"), 2800);
      } else {
        setMessage({ type: "error", text: data.detail || data.error || "Submission failed. Please try again." });
      }
    } catch {
      setMessage({ type: "error", text: "Network error. Please check your connection and try again." });
    } finally {
      setLoading(false);
    }
  };

  const symptomsLeft = 225 - symptoms.length;

  return (
    <div style={{
      background: "#020818",
      minHeight: "100vh",
      color: "#fff",
      fontFamily: "'Inter', -apple-system, sans-serif",
      overflowX: "hidden",
      position: "relative",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; background: #0a1122; }
        ::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.25); border-radius: 6px; }
        textarea, input { font-family: inherit; }
        @keyframes spin-slow { to { transform: rotate(360deg); } }
        @keyframes fadeInUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
        @keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
        @keyframes fadeIn { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:none; } }
      `}</style>

      {/* Ambient glows */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "5%", left: "5%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(129,140,248,0.07) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "10%", right: "5%", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%)" }} />
      </div>

      <ProgressBar active={loading} />
      <ProcessingOverlay active={loading} stepIndex={aiStep} />

      <div style={{ position: "relative", zIndex: 10 }}><Navbar /></div>

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "3rem 2rem 6rem", position: "relative", zIndex: 1 }}>

        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: "3rem" }}>
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "0.5rem",
            background: "rgba(129,140,248,0.09)", border: "1px solid rgba(129,140,248,0.22)",
            borderRadius: 100, padding: "0.35rem 1.1rem",
            fontSize: "0.72rem", color: "#818cf8", fontWeight: 700,
            letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "1.2rem",
          }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#818cf8", display: "inline-block", boxShadow: "0 0 8px #818cf8", animation: "pulse-dot 2s ease-in-out infinite" }} />
            AI Diagnostic Submission
          </div>
          <h1 style={{
            fontSize: "clamp(2rem, 4vw, 3rem)", fontWeight: 900,
            letterSpacing: "-0.04em", lineHeight: 1.1, marginBottom: "0.9rem",
          }}>
            <span style={{
              background: "linear-gradient(135deg, #ffffff 30%, #a5b4fc 70%, #38bdf8 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              Submit Diagnosis Request
            </span>
          </h1>
          <p style={{ color: "#64748b", fontSize: "1rem", maxWidth: 540, margin: "0 auto", lineHeight: 1.75 }}>
            <strong style={{ color: "#818cf8" }}>Symptoms</strong> are required. You may also upload a chest X-ray
            and provide a Patient ID for a more complete AI evaluation.
          </p>
        </div>

        {/* AI Pipeline badge row */}
        <div style={{ display: "flex", justifyContent: "center", gap: "0.65rem", flexWrap: "wrap", marginBottom: "2.5rem" }}>
          {["ResNet-50", "S²A-UNet", "Advanced RAG", "LLM ReAct", "Grad-CAM XAI", "Self-Refine"].map(tag => (
            <span key={tag} style={{
              fontSize: "0.74rem", fontWeight: 700, padding: "0.3rem 0.85rem", borderRadius: 100,
              background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.2)",
              color: "#818cf8", letterSpacing: "0.03em",
            }}>{tag}</span>
          ))}
        </div>

        {/* Toast messages */}
        {message && (
          <div style={{
            animation: "fadeIn 0.3s ease",
            background: message.type === "success" ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.1)",
            border: `1px solid ${message.type === "success" ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
            borderRadius: 14, padding: "1rem 1.5rem",
            color: message.type === "success" ? "#4ade80" : "#f87171",
            fontWeight: 700, fontSize: "0.95rem",
            marginBottom: "1.5rem", textAlign: "center",
          }}>
            {message.text}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} autoComplete="off">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.8rem" }}>

            {/* ── Left: X-ray Upload ──────────────────────────────── */}
            <div style={{
              background: "rgba(8,14,30,0.88)",
              border: "1px solid rgba(129,140,248,0.18)",
              borderRadius: 28, padding: "2rem",
              backdropFilter: "blur(20px)",
              boxShadow: "0 20px 50px rgba(0,0,0,0.4)",
            }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.2rem" }}>
                <div>
                  <div style={{ fontSize: "0.73rem", color: "#818cf8", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.25rem" }}>
                    Step 1
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f0f6ff", letterSpacing: "-0.02em" }}>
                    Chest X-Ray Image
                  </h2>
                </div>
                <span style={{
                  fontSize: "0.73rem", fontWeight: 600, padding: "0.25rem 0.7rem",
                  borderRadius: 100, background: "rgba(71,85,105,0.2)",
                  border: "1px solid rgba(71,85,105,0.3)", color: "#64748b",
                }}>
                  Optional
                </span>
              </div>

              <DropZone file={file} onFile={setFile} />

              {file && (
                <div style={{
                  marginTop: "0.85rem",
                  display: "flex", alignItems: "center", gap: "0.6rem",
                  padding: "0.6rem 1rem", borderRadius: 10,
                  background: "rgba(129,140,248,0.08)", border: "1px solid rgba(129,140,248,0.2)",
                }}>
                  <span style={{ fontSize: "0.85rem", color: "#818cf8" }}>🖼️</span>
                  <span style={{ fontSize: "0.82rem", color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {file.name}
                  </span>
                  <span style={{ fontSize: "0.76rem", color: "#475569", flexShrink: 0 }}>
                    ({(file.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
              )}

              {/* Info note */}
              <div style={{ marginTop: "1rem", padding: "0.8rem 1rem", borderRadius: 12, background: "rgba(56,189,248,0.06)", border: "1px solid rgba(56,189,248,0.12)" }}>
                <div style={{ fontSize: "0.78rem", color: "#64748b", lineHeight: 1.65 }}>
                  <span style={{ color: "#38bdf8", fontWeight: 700 }}>AI Vision Pipeline:</span> Your X-ray undergoes automatic lung segmentation (S²A-UNet) followed by multilabel disease classification (ResNet-50) across 6 pathologies.
                </div>
              </div>
            </div>

            {/* ── Right: Patient Information ──────────────────────── */}
            <div style={{
              background: "rgba(8,14,30,0.88)",
              border: "1px solid rgba(129,140,248,0.18)",
              borderRadius: 28, padding: "2rem",
              backdropFilter: "blur(20px)",
              boxShadow: "0 20px 50px rgba(0,0,0,0.4)",
              display: "flex", flexDirection: "column",
            }}>
              <div style={{ marginBottom: "1.4rem" }}>
                <div style={{ fontSize: "0.73rem", color: "#818cf8", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "0.25rem" }}>
                  Step 2
                </div>
                <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#f0f6ff", letterSpacing: "-0.02em" }}>
                  Patient Information
                </h2>
              </div>

              {/* Symptoms textarea */}
              <div style={{ marginBottom: "1.4rem", flex: 1 }}>
                <label style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.83rem", fontWeight: 700, color: "#94a3b8", marginBottom: "0.55rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>
                  Symptoms
                  <span style={{ color: "#f87171", fontSize: "0.9rem" }}>*</span>
                  <span style={{ marginLeft: "auto", color: symptomsLeft < 30 ? "#f87171" : "#475569", fontSize: "0.75rem", fontWeight: 600, textTransform: "none", letterSpacing: 0 }}>
                    {symptoms.length} / 225
                  </span>
                </label>
                <textarea
                  value={symptoms}
                  onChange={e => setSymptoms(e.target.value)}
                  maxLength={225}
                  required
                  placeholder="Describe your symptoms in your own words — e.g., persistent cough, shortness of breath, chest pain, fever…"
                  style={{
                    width: "100%", minHeight: 150,
                    background: "rgba(15,23,42,0.85)",
                    border: `1.5px solid ${symptoms.length > 0 ? "rgba(129,140,248,0.4)" : "rgba(129,140,248,0.18)"}`,
                    borderRadius: 14, padding: "0.9rem 1rem",
                    color: "#f0f6ff", fontSize: "0.95rem",
                    resize: "vertical", outline: "none",
                    transition: "border-color 0.2s",
                    lineHeight: 1.65,
                  }}
                  onFocus={e => { e.target.style.borderColor = "rgba(129,140,248,0.6)"; e.target.style.boxShadow = "0 0 0 3px rgba(129,140,248,0.08)"; }}
                  onBlur={e => { e.target.style.borderColor = symptoms.length > 0 ? "rgba(129,140,248,0.4)" : "rgba(129,140,248,0.18)"; e.target.style.boxShadow = "none"; }}
                />
                <div style={{ fontSize: "0.77rem", color: "#334155", marginTop: "0.4rem", lineHeight: 1.5 }}>
                  💡 The more detail you provide, the more accurate the AI clinical knowledge retrieval.
                </div>
              </div>

              {/* Patient ID input */}
              <div style={{ marginBottom: "1.6rem" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.83rem", fontWeight: 700, color: "#94a3b8", marginBottom: "0.55rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>
                  Patient ID
                  <span style={{ color: "#475569", fontSize: "0.72rem", fontWeight: 600, textTransform: "none", letterSpacing: 0 }}>— optional</span>
                </label>
                <div style={{ position: "relative" }}>
                  <span style={{ position: "absolute", left: "1rem", top: "50%", transform: "translateY(-50%)", fontSize: "1rem", pointerEvents: "none" }}>🆔</span>
                  <input
                    type="text"
                    value={patientId}
                    onChange={e => setPatientId(e.target.value)}
                    placeholder="Enter your Patient ID to include medical history"
                    style={{
                      width: "100%",
                      background: "rgba(15,23,42,0.85)",
                      border: `1.5px solid ${patientId ? "rgba(129,140,248,0.4)" : "rgba(129,140,248,0.18)"}`,
                      borderRadius: 14, padding: "0.85rem 1rem 0.85rem 2.8rem",
                      color: "#f0f6ff", fontSize: "0.95rem",
                      outline: "none", transition: "border-color 0.2s",
                    }}
                    onFocus={e => { e.target.style.borderColor = "rgba(129,140,248,0.6)"; e.target.style.boxShadow = "0 0 0 3px rgba(129,140,248,0.08)"; }}
                    onBlur={e => { e.target.style.borderColor = patientId ? "rgba(129,140,248,0.4)" : "rgba(129,140,248,0.18)"; e.target.style.boxShadow = "none"; }}
                  />
                </div>
                <div style={{ fontSize: "0.77rem", color: "#334155", marginTop: "0.4rem" }}>
                  Enables Graph ML analysis of your longitudinal medical history.
                </div>
              </div>

              {/* Submit button */}
              <button
                type="submit"
                disabled={loading}
                style={{
                  background: symptoms.trim()
                    ? "linear-gradient(135deg, #6366f1, #818cf8, #38bdf8)"
                    : "rgba(71,85,105,0.3)",
                  border: "none", color: "#fff",
                  borderRadius: 14, padding: "0.95rem 2rem",
                  fontSize: "1rem", fontWeight: 800,
                  cursor: symptoms.trim() ? "pointer" : "not-allowed",
                  boxShadow: symptoms.trim() ? "0 8px 30px rgba(99,102,241,0.4)" : "none",
                  transition: "all 0.25s ease",
                  display: "flex", alignItems: "center", justifyContent: "center", gap: "0.6rem",
                  letterSpacing: "0.02em",
                  opacity: loading ? 0.7 : 1,
                }}
              >
                {loading ? (
                  <>
                    <span style={{ width: 18, height: 18, border: "2.5px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", display: "inline-block", animation: "spin-slow 0.8s linear infinite" }} />
                    Processing…
                  </>
                ) : (
                  <>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 2L11 13M22 2L15 22 11 13 2 9l20-7z" />
                    </svg>
                    Submit Diagnosis Request
                  </>
                )}
              </button>
            </div>
          </div>
        </form>

        {/* What happens next */}
        <div style={{
          marginTop: "2.5rem",
          background: "rgba(8,14,30,0.7)",
          border: "1px solid rgba(129,140,248,0.12)",
          borderRadius: 24, padding: "1.8rem 2rem",
          backdropFilter: "blur(16px)",
        }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 800, color: "#94a3b8", marginBottom: "1.2rem", letterSpacing: "0.03em" }}>
            What happens after submission?
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
            {[
              { icon: "🤖", title: "AI Analysis", desc: "ReAct agent processes your data across 3 parallel pipelines" },
              { icon: "🎯", title: "Grad-CAM XAI", desc: "Heatmap overlays generated to explain the model's attention" },
              { icon: "👨‍⚕️", title: "Doctor Review", desc: "A clinician reviews and verifies the AI-generated report" },
              { icon: "📋", title: "Report Ready", desc: "You receive the approved diagnostic report with PDF download" },
            ].map(item => (
              <div key={item.title} style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
                <div style={{ fontSize: "1.4rem", flexShrink: 0 }}>{item.icon}</div>
                <div>
                  <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#818cf8", marginBottom: "0.2rem" }}>{item.title}</div>
                  <div style={{ fontSize: "0.78rem", color: "#475569", lineHeight: 1.55 }}>{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

      </main>
    </div>
  );
}
