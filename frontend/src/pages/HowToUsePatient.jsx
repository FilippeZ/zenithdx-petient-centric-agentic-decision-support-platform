import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import Navbar from "../components/Navbar";
import howusepatient from "../assets/howusepatient.png";
import xrayImage from "../assets/xray-image.png";
import patientid from "../assets/patientid.png";

/* ── Step data — original 3 steps, new design ─────────────────── */
const STEPS = [
  {
    step: "01",
    title: "Upload Chest X-Ray Image",
    subtitle: "Radiographic Input",
    description:
      "Upload your chest X-ray to begin the AI-assisted diagnostic process. Our pipeline performs automatic lung segmentation and multilabel classification, delivering fast, reliable results with transparent, explainable AI.",
    tags: ["S²A-UNet Segmentation", "ResNet-50 Classification", "6 Pathologies"],
    image: xrayImage,
    imageAlt: "X-ray Analysis",
    accent: "#38bdf8",
    accentAlt: "#0ea5e9",
    glow: "rgba(56,189,248,0.18)",
    direction: "left",
  },
  {
    step: "02",
    title: "Describe Your Symptoms",
    subtitle: "Clinical Knowledge Retrieval",
    description:
      "Describe your symptoms in your own words. The AI agent intelligently matches your case with the most relevant clinical knowledge, delivering accurate and truly personalised diagnostics you can trust.",
    tags: ["Advanced RAG", "SciBERT + ColBERT", "Medical Literature"],
    image: howusepatient,
    imageAlt: "Describe Symptoms",
    accent: "#818cf8",
    accentAlt: "#6366f1",
    glow: "rgba(129,140,248,0.18)",
    direction: "right",
  },
  {
    step: "03",
    title: "Provide Your Patient ID",
    subtitle: "Longitudinal History Analysis",
    description:
      "Your Patient ID enables the system to retrieve your personal medical history and prior diagnoses. We use this context to enhance the diagnostic process, ensuring accurate and personalised care without starting from scratch.",
    tags: ["Graph ML (HGT)", "Medical History", "Personalised Care"],
    image: patientid,
    imageAlt: "Patient ID",
    accent: "#34d399",
    accentAlt: "#059669",
    glow: "rgba(52,211,153,0.18)",
    direction: "left",
  },
];

/* ── Main Page ───────────────────────────────────────────────── */
export default function HowToUsePatient() {
  const heroRef    = useRef(null);
  const heroInView = useInView(heroRef, { once: true });

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
        ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.25); border-radius: 6px; }
        @keyframes glow { 0%,100%{opacity:1;box-shadow:0 0 8px #38bdf8;}50%{opacity:0.5;box-shadow:0 0 4px #38bdf8;} }
        @media (max-width: 768px) { .step-grid { grid-template-columns: 1fr !important; direction: ltr !important; } .step-grid > * { direction: ltr !important; } }
      `}</style>

      {/* Ambient glows */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "5%",  left: "5%",  width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", top: "45%", right: "0",  width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, rgba(129,140,248,0.05) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "8%", left: "20%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(52,211,153,0.04) 0%, transparent 65%)" }} />
      </div>
      {/* Grid overlay */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, opacity: 0.25, backgroundImage: "linear-gradient(rgba(56,189,248,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,0.04) 1px,transparent 1px)", backgroundSize: "80px 80px" }} />

      <div style={{ position: "relative", zIndex: 10 }}><Navbar /></div>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "4rem 2rem 8rem", position: "relative", zIndex: 1 }}>

        {/* ── Hero ─────────────────────────────────────────────── */}
        <motion.div
          ref={heroRef}
          initial={{ opacity: 0, y: 40 }}
          animate={heroInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          style={{ textAlign: "center", marginBottom: "6rem" }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={heroInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "rgba(56,189,248,0.09)", border: "1px solid rgba(56,189,248,0.22)",
              borderRadius: 100, padding: "0.38rem 1.1rem",
              fontSize: "0.72rem", fontWeight: 700, color: "#38bdf8",
              letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: "1.5rem",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#38bdf8", display: "inline-block", animation: "glow 2s ease-in-out infinite" }} />
            Patient Diagnostic Guide
          </motion.div>

          <h1 style={{ fontSize: "clamp(2.5rem, 5vw, 3.8rem)", fontWeight: 900, letterSpacing: "-0.04em", lineHeight: 1.08, marginBottom: "1.2rem" }}>
            <span style={{ background: "linear-gradient(135deg, #ffffff 30%, #7dd3fc 70%, #818cf8 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              How to Use
            </span>
            <br />
            <span style={{ color: "#475569", fontSize: "0.55em", fontWeight: 600, letterSpacing: "-0.02em" }}>
              for ZenithDx Patients
            </span>
          </h1>

          <p style={{ fontSize: "1.1rem", color: "#64748b", maxWidth: 540, margin: "0 auto 2rem", lineHeight: 1.75 }}>
            Follow these three simple steps for a seamless, personalised, and explainable AI diagnostic experience.
          </p>

          {/* Pills */}
          <div style={{ display: "flex", justifyContent: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            {[
              { label: "3 Easy Steps", color: "#38bdf8" },
              { label: "AI-Powered Analysis", color: "#818cf8" },
              { label: "Doctor Approved Reports", color: "#34d399" },
            ].map(p => (
              <div key={p.label} style={{
                padding: "0.4rem 1rem", borderRadius: 100,
                background: `${p.color}10`, border: `1px solid ${p.color}28`,
                fontSize: "0.8rem", fontWeight: 700, color: p.color, letterSpacing: "0.03em",
              }}>
                {p.label}
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Steps ────────────────────────────────────────────── */}
        {STEPS.map((step, i) => (
          <div key={step.step}>
            <div
              className="step-grid"
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "3rem",
                alignItems: "center",
                marginBottom: "2rem",
                direction: step.direction === "right" ? "rtl" : "ltr",
              }}
            >
              {/* Image Panel */}
              <motion.div
                initial={{ opacity: 0, x: step.direction === "left" ? -70 : 70 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.75, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                style={{ direction: "ltr", position: "relative" }}
              >
                <div style={{ position: "absolute", inset: "-15%", background: `radial-gradient(circle, ${step.glow} 0%, transparent 70%)`, borderRadius: "50%", pointerEvents: "none", filter: "blur(24px)" }} />
                <div style={{
                  position: "relative",
                  background: "linear-gradient(135deg, rgba(12,20,44,0.9) 0%, rgba(18,28,56,0.7) 100%)",
                  border: `1px solid ${step.accent}22`,
                  borderRadius: 32, padding: "3rem 2.5rem",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  backdropFilter: "blur(20px)",
                  boxShadow: `0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04)`,
                  overflow: "hidden", minHeight: 280,
                }}>
                  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "50%", background: `radial-gradient(ellipse at 30% 0%, ${step.accent}10 0%, transparent 60%)`, pointerEvents: "none" }} />
                  <motion.img
                    src={step.image}
                    alt={step.imageAlt}
                    whileHover={{ scale: 1.06, y: -4 }}
                    transition={{ type: "spring", stiffness: 280, damping: 22 }}
                    style={{
                      width: "100%", maxWidth: 260, height: "auto",
                      objectFit: "contain", position: "relative", zIndex: 1,
                      filter: `drop-shadow(0 16px 36px ${step.glow})`,
                    }}
                  />
                </div>
              </motion.div>

              {/* Text Panel */}
              <motion.div
                initial={{ opacity: 0, x: step.direction === "left" ? 70 : -70 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.75, delay: 0.22, ease: [0.22, 1, 0.36, 1] }}
                style={{ direction: "ltr" }}
              >
                {/* Step badge + subtitle */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.3rem" }}>
                  <div style={{
                    width: 44, height: 44, borderRadius: 13,
                    background: `linear-gradient(135deg, ${step.accent}1a, ${step.accentAlt}2a)`,
                    border: `1px solid ${step.accent}35`,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontWeight: 900, fontSize: "0.8rem", color: step.accent,
                    boxShadow: `0 4px 20px ${step.glow}`,
                  }}>
                    {step.step}
                  </div>
                  <div style={{
                    fontSize: "0.7rem", fontWeight: 700, color: step.accent,
                    letterSpacing: "0.1em", textTransform: "uppercase",
                    padding: "0.28rem 0.85rem", borderRadius: 100,
                    background: `${step.accent}0e`, border: `1px solid ${step.accent}22`,
                  }}>
                    {step.subtitle}
                  </div>
                </div>

                {/* Title */}
                <h2 style={{
                  fontSize: "clamp(1.5rem, 2.4vw, 1.9rem)",
                  fontWeight: 900, lineHeight: 1.15, letterSpacing: "-0.03em",
                  marginBottom: "1rem",
                  background: `linear-gradient(130deg, #f0f6ff 35%, ${step.accent} 110%)`,
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                }}>
                  {step.title}
                </h2>

                {/* Description */}
                <p style={{ fontSize: "0.98rem", color: "#94a3b8", lineHeight: 1.8, marginBottom: "1.8rem", maxWidth: 430 }}>
                  {step.description}
                </p>

                {/* Tags */}
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                  {step.tags.map(tag => (
                    <span key={tag} style={{
                      fontSize: "0.75rem", fontWeight: 700,
                      padding: "0.32rem 0.85rem", borderRadius: 100,
                      background: `${step.accent}0d`, border: `1px solid ${step.accent}25`,
                      color: step.accent, letterSpacing: "0.03em",
                    }}>
                      {tag}
                    </span>
                  ))}
                </div>
              </motion.div>
            </div>

            {/* Step connector */}
            {i < STEPS.length - 1 && (
              <motion.div
                initial={{ scaleY: 0, opacity: 0 }}
                whileInView={{ scaleY: 1, opacity: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 }}
                style={{ display: "flex", flexDirection: "column", alignItems: "center", margin: "1.5rem 0 3rem", gap: "0.25rem" }}
              >
                <div style={{ width: 1, height: 48, background: `linear-gradient(to bottom, ${STEPS[i].accent}50, ${STEPS[i + 1].accent}50)` }} />
                <div style={{ width: 10, height: 10, borderRadius: "50%", background: STEPS[i + 1].accent, boxShadow: `0 0 12px ${STEPS[i + 1].glow}` }} />
                <div style={{ width: 1, height: 48, background: `linear-gradient(to bottom, ${STEPS[i + 1].accent}50, transparent)` }} />
              </motion.div>
            )}
          </div>
        ))}

        {/* ── Bottom CTA ─────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          style={{
            textAlign: "center", marginTop: "5rem", padding: "3.5rem",
            background: "linear-gradient(135deg, rgba(10,20,44,0.9) 0%, rgba(15,28,60,0.7) 100%)",
            border: "1px solid rgba(56,189,248,0.16)",
            borderRadius: 32, backdropFilter: "blur(20px)",
            boxShadow: "0 30px 80px rgba(0,0,0,0.35)",
            position: "relative", overflow: "hidden",
          }}
        >
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "50%", background: "radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.08) 0%, transparent 60%)", pointerEvents: "none" }} />
          <div style={{ position: "relative", zIndex: 1 }}>
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>🩺</div>
            <h3 style={{
              fontSize: "clamp(1.4rem, 2.5vw, 1.85rem)", fontWeight: 900,
              letterSpacing: "-0.03em", marginBottom: "0.75rem",
              background: "linear-gradient(135deg, #fff 40%, #7dd3fc 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            }}>
              Ready to submit your diagnosis request?
            </h3>
            <p style={{ color: "#64748b", fontSize: "0.96rem", maxWidth: 400, margin: "0 auto 2rem", lineHeight: 1.7 }}>
              Upload your X-ray, describe your symptoms, and receive an AI-assisted clinical report.
            </p>
            <a href="/detect" style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              padding: "0.78rem 2rem", borderRadius: 14,
              background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
              color: "#fff", fontWeight: 800, fontSize: "0.95rem",
              textDecoration: "none",
              boxShadow: "0 8px 30px rgba(14,165,233,0.35)",
            }}>
              Submit X-Ray →
            </a>
          </div>
        </motion.div>

      </main>
    </div>
  );
}
