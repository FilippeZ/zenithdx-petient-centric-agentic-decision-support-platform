import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import { motion, AnimatePresence } from "framer-motion";

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
  }, [active]);

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, width: "100vw", zIndex: 9999,
      pointerEvents: "none", opacity: pct > 0 ? 1 : 0, transition: "opacity 0.4s",
    }}>
      <div style={{ width: "100%", height: 4, background: "#e2e8f0" }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: "linear-gradient(90deg, #2563eb, #0284c7, #10b981)",
          boxShadow: "0 0 10px rgba(37, 99, 235, 0.4)",
          transition: "width 0.35s cubic-bezier(.4,0,.2,1)",
        }} />
      </div>
    </div>
  );
};

/* ── AI Processing animation steps ──────────────────────────── */
const AI_STEPS = [
  { icon: "🧬", label: "Preprocessing chest X-ray image…" },
  { icon: "🔬", label: "Running S²A-UNet lung segmentation…" },
  { icon: "🤖", label: "ResNet-50 multi-label classification…" },
  { icon: "📚", label: "Clinical RAG medical knowledge search…" },
  { icon: "🧠", label: "ReAct agent reasoning loop…" },
  { icon: "📊", label: "Generating Grad-CAM visual heatmaps…" },
  { icon: "📝", label: "Compiling structured patient summary…" },
];

const ProcessingOverlay = ({ active, stepIndex }) => {
  if (!active) return null;
  const step = AI_STEPS[stepIndex % AI_STEPS.length];
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        background: "rgba(255, 255, 255, 0.94)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: "1.5rem",
      }}
    >
      <motion.div
        animate={{ rotate: 360, scale: [1, 1.05, 1] }}
        transition={{ rotate: { duration: 6, repeat: Infinity, ease: "linear" }, scale: { duration: 2, repeat: Infinity } }}
        style={{
          width: 84, height: 84, borderRadius: "50%",
          background: "#eff6ff",
          border: "2px solid #bfdbfe",
          display: "flex", alignItems: "center", justifyContent: "center", fontSize: "2.2rem",
          boxShadow: "0 8px 25px rgba(37, 99, 235, 0.2)",
        }}
      >
        {step.icon}
      </motion.div>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: "1.3rem", fontWeight: 900, color: "#0f172a", marginBottom: "0.4rem" }}>
          AI Diagnostic Processing
        </div>
        <div style={{ fontSize: "0.95rem", color: "#2563eb", fontWeight: 700 }}>
          {step.label}
        </div>
      </div>
      <div style={{ display: "flex", gap: "0.4rem" }}>
        {AI_STEPS.map((_, i) => (
          <motion.div
            key={i}
            animate={{
              width: i === stepIndex % AI_STEPS.length ? 26 : 8,
              backgroundColor: i === stepIndex % AI_STEPS.length ? "#2563eb" : "#cbd5e1"
            }}
            transition={{ duration: 0.4 }}
            style={{
              height: 8, borderRadius: 4,
            }}
          />
        ))}
      </div>
      <div style={{ color: "#64748b", fontSize: "0.85rem" }}>
        Please stay on this page — taking ~30–60 seconds
      </div>
    </motion.div>
  );
};

/* ── Drop zone with Framer Motion ───────────────────────────── */
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
    <motion.div
      whileHover={{ scale: file ? 1 : 1.01 }}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !file && inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragging ? "#2563eb" : file ? "#93c5fd" : "#cbd5e1"}`,
        borderRadius: 20,
        height: 300,
        display: "flex", alignItems: "center", justifyContent: "center",
        cursor: file ? "default" : "pointer",
        background: dragging ? "#eff6ff" : file ? "#fafafa" : "#f8fafc",
        transition: "border 0.25s ease, background 0.25s ease",
        position: "relative",
        overflow: "hidden",
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
          <motion.button
            whileHover={{ scale: 1.05 }}
            type="button"
            onClick={e => { e.stopPropagation(); inputRef.current?.click(); }}
            style={{
              position: "absolute", bottom: 12, right: 12,
              background: "#2563eb", border: "none", color: "#fff", borderRadius: 8,
              padding: "0.45rem 0.9rem", fontSize: "0.78rem", fontWeight: 700, cursor: "pointer",
            }}
          >
            Change Image
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.05 }}
            type="button"
            onClick={e => { e.stopPropagation(); onFile(null); }}
            style={{
              position: "absolute", bottom: 12, left: 12,
              background: "#dc2626", border: "none", color: "#fff", borderRadius: 8,
              padding: "0.45rem 0.85rem", fontSize: "0.78rem", fontWeight: 700, cursor: "pointer",
            }}
          >
            ✕ Remove
          </motion.button>
        </>
      ) : (
        <div style={{ textAlign: "center", color: "#475569" }}>
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
            style={{ fontSize: "3.4rem", marginBottom: "0.5rem", display: "inline-block" }}
          >
            🫁
          </motion.div>
          <div style={{ fontSize: "1.05rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.3rem" }}>
            {dragging ? "Release image here" : "Upload Chest X-Ray Image"}
          </div>
          <div style={{ fontSize: "0.85rem", color: "#64748b", marginBottom: "1rem" }}>
            Drag & drop your file or click to browse
          </div>
          <span style={{
            display: "inline-block",
            padding: "0.4rem 1rem", borderRadius: 100,
            background: "#eff6ff", border: "1px solid #bfdbfe",
            color: "#2563eb", fontSize: "0.78rem", fontWeight: 700,
          }}>
            DICOM, PNG, JPG Supported
          </span>
        </div>
      )}
    </motion.div>
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
        setMessage({ type: "success", text: "✅ Diagnosis request submitted successfully! Redirecting to your dashboard..." });
        setSymptoms(""); setFile(null); setPatientId("");
        setTimeout(() => navigate("/patient-dashboard"), 2500);
      } else {
        setMessage({ type: "error", text: data.detail || data.error || "Submission failed. Please try again." });
      }
    } catch {
      setMessage({ type: "error", text: "Network connection issue. Please make sure the server is reachable." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
      
      <ProgressBar active={loading} />
      <AnimatePresence>
        {loading && <ProcessingOverlay active={loading} stepIndex={aiStep} />}
      </AnimatePresence>
      <Navbar />

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "3rem 2rem 6rem" }}>
        
        {/* Animated Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          style={{ textAlign: "center", marginBottom: "3rem" }}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "#eff6ff", border: "1px solid #bfdbfe",
              borderRadius: 100, padding: "0.35rem 1.1rem",
              fontSize: "0.75rem", color: "#2563eb", fontWeight: 700,
              letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "1rem",
            }}
          >
            <motion.span
              animate={{ scale: [1, 1.3, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              style={{ width: 6, height: 6, borderRadius: "50%", background: "#2563eb", display: "inline-block" }}
            />
            STEP-BY-STEP DIAGNOSIS WIZARD
          </motion.div>

          <h1 style={{ fontSize: "clamp(2rem, 3.8vw, 2.8rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "#0f172a", marginBottom: "0.6rem" }}>
            Submit X-Ray Scan for AI Analysis
          </h1>
          <p style={{ color: "#64748b", fontSize: "1rem", maxWidth: 550, margin: "0 auto", lineHeight: 1.6 }}>
            Follow the 3 simple steps below. Describe your symptoms and optionally include your X-ray image for instant diagnostic feedback.
          </p>
        </motion.div>

        {/* Toast alert */}
        <AnimatePresence>
          {message && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              style={{
                background: message.type === "success" ? "#ecfdf5" : "#fef2f2",
                border: `1px solid ${message.type === "success" ? "#a7f3d0" : "#fecaca"}`,
                borderRadius: 14, padding: "1rem 1.5rem",
                color: message.type === "success" ? "#047857" : "#b91c1c",
                fontWeight: 700, fontSize: "0.95rem",
                marginBottom: "2rem", textAlign: "center",
              }}
            >
              {message.text}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "2rem" }}>
            
            {/* Step 1: Upload */}
            <motion.div
              initial={{ opacity: 0, x: -25 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              whileHover={{ y: -5, boxShadow: "0 14px 35px -5px rgba(37, 99, 235, 0.12)" }}
              style={{
                background: "#ffffff", border: "1px solid #e2e8f0",
                borderRadius: "24px", padding: "2rem",
                boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
                transition: "box-shadow 0.3s ease, border-color 0.3s ease"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.2rem" }}>
                <div>
                  <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.2rem" }}>
                    STEP 1
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#0f172a" }}>
                    Chest X-Ray Image
                  </h2>
                </div>
                <span style={{
                  fontSize: "0.75rem", fontWeight: 700, padding: "0.25rem 0.75rem",
                  borderRadius: 100, background: "#f1f5f9", color: "#64748b",
                }}>
                  Optional
                </span>
              </div>

              <DropZone file={file} onFile={setFile} />

              <div style={{ marginTop: "1rem", padding: "0.85rem 1rem", borderRadius: 12, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                <div style={{ fontSize: "0.82rem", color: "#475569", lineHeight: 1.5 }}>
                  <strong>🔒 Privacy Protected:</strong> Uploaded DICOM/image scans are encrypted and strictly analyzed for clinical diagnostic support.
                </div>
              </div>
            </motion.div>

            {/* Step 2 & 3: Symptoms & Patient ID */}
            <motion.div
              initial={{ opacity: 0, x: 25 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              whileHover={{ y: -5, boxShadow: "0 14px 35px -5px rgba(37, 99, 235, 0.12)" }}
              style={{
                background: "#ffffff", border: "1px solid #e2e8f0",
                borderRadius: "24px", padding: "2rem",
                boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
                display: "flex", flexDirection: "column", justifyContent: "space-between",
                transition: "box-shadow 0.3s ease, border-color 0.3s ease"
              }}
            >
              <div>
                <div style={{ marginBottom: "1.4rem" }}>
                  <div style={{ fontSize: "0.75rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.2rem" }}>
                    STEP 2 & 3
                  </div>
                  <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#0f172a" }}>
                    Describe Your Symptoms
                  </h2>
                </div>

                <div style={{ marginBottom: "1.5rem" }}>
                  <label style={{ display: "block", fontSize: "0.83rem", fontWeight: 700, color: "#334155", marginBottom: "0.4rem" }}>
                    Clinical Symptoms <span style={{ color: "#dc2626" }}>*</span>
                  </label>
                  <textarea
                    value={symptoms}
                    onChange={e => setSymptoms(e.target.value)}
                    maxLength={225}
                    required
                    placeholder="Describe how you are feeling (e.g., shortness of breath, persistent cough for 3 days, chest tightness)..."
                    style={{
                      width: "100%", minHeight: 140,
                      background: "#ffffff", border: "1px solid #cbd5e1",
                      borderRadius: 14, padding: "0.9rem 1rem",
                      color: "#0f172a", fontSize: "0.92rem",
                      outline: "none", transition: "all 0.2s",
                      fontFamily: "inherit", lineHeight: 1.6,
                    }}
                    onFocus={e => { e.target.style.borderColor = "#2563eb"; e.target.style.boxShadow = "0 0 0 3px rgba(37, 99, 235, 0.12)"; }}
                    onBlur={e => { e.target.style.borderColor = "#cbd5e1"; e.target.style.boxShadow = "none"; }}
                  />
                </div>

                <div style={{ marginBottom: "1.8rem" }}>
                  <label style={{ display: "block", fontSize: "0.83rem", fontWeight: 700, color: "#334155", marginBottom: "0.4rem" }}>
                    Patient ID <span style={{ color: "#64748b", fontWeight: 500 }}>(Optional)</span>
                  </label>
                  <input
                    type="text"
                    value={patientId}
                    onChange={e => setPatientId(e.target.value)}
                    placeholder="e.g. PAT-90412"
                    style={{
                      width: "100%",
                      background: "#ffffff", border: "1px solid #cbd5e1",
                      borderRadius: 12, padding: "0.8rem 1rem",
                      color: "#0f172a", fontSize: "0.92rem",
                      outline: "none", transition: "all 0.2s",
                    }}
                    onFocus={e => { e.target.style.borderColor = "#2563eb"; e.target.style.boxShadow = "0 0 0 3px rgba(37, 99, 235, 0.12)"; }}
                    onBlur={e => { e.target.style.borderColor = "#cbd5e1"; e.target.style.boxShadow = "none"; }}
                  />
                </div>
              </div>

              <motion.button
                whileHover={{ scale: symptoms.trim() ? 1.02 : 1, boxShadow: symptoms.trim() ? "0 8px 25px rgba(37, 99, 235, 0.35)" : "none" }}
                whileTap={{ scale: symptoms.trim() ? 0.98 : 1 }}
                type="submit"
                disabled={loading}
                style={{
                  width: "100%", padding: "0.95rem",
                  borderRadius: "14px", border: "none",
                  background: symptoms.trim() ? "linear-gradient(135deg, #2563eb, #0284c7)" : "#cbd5e1",
                  color: "#ffffff", fontSize: "1rem", fontWeight: 800,
                  cursor: symptoms.trim() ? "pointer" : "not-allowed",
                  boxShadow: symptoms.trim() ? "0 4px 14px rgba(37,99,235,0.25)" : "none",
                  transition: "background 0.2s ease",
                }}
              >
                {loading ? "Analyzing..." : "Submit Scan for AI Diagnostic Review →"}
              </motion.button>
            </motion.div>

          </div>
        </form>

      </main>
    </div>
  );
}
