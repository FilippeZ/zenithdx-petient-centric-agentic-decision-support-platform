import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import Navbar from "../components/Navbar";
import howusedoctor from "../assets/howusedoctor.png";
import meds from "../assets/meds.png";
import Ai from "../assets/Ai.png";

/* ── Step data ────────────────────────────────────────────────── */
const STEPS = [
  {
    step: "01",
    title: "Clinical Diagnosis Summary",
    subtitle: "Diagnosis Report",
    description:
      "AI-assisted summary of patient data and radiographic findings, providing clinicians with a provisional diagnosis and confidence scores for expert review.",
    image: howusedoctor,
    imageAlt: "Diagnosis Report",
    accent: "#38bdf8",
    accentAlt: "#0ea5e9",
    glowColor: "rgba(56,189,248,0.18)",
    tags: ["ResNet-50", "Provisional Diagnosis", "Confidence Score"],
    direction: "left",
  },
  {
    step: "02",
    title: "Actions & Documentation",
    subtitle: "Clinical Decision Hub",
    description:
      "Provide personalised advice to the patient, edit the AI-generated findings, approve or reject the diagnostic report, and export the complete summary as a signed PDF.",
    image: meds,
    imageAlt: "Actions & Documentation",
    accent: "#818cf8",
    accentAlt: "#6366f1",
    glowColor: "rgba(129,140,248,0.18)",
    tags: ["Approve / Reject", "PDF Export", "Doctor Message"],
    direction: "right",
  },
  {
    step: "03",
    title: "AI Explainability & Validation",
    subtitle: "Explainable AI (XAI)",
    description:
      "Review Grad-CAM attention heatmaps, Captum attribution plots, and classification scores to fully understand and validate the model's clinical reasoning.",
    image: Ai,
    imageAlt: "Explainable AI",
    accent: "#34d399",
    accentAlt: "#059669",
    glowColor: "rgba(52,211,153,0.18)",
    tags: ["Grad-CAM", "Captum XAI", "Attribution Analysis"],
    direction: "left",
  },
];

/* ── Animated step card ───────────────────────────────────────── */
const StepCard = ({ step, index }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const isLeft = step.direction === "left";

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 60 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.15, ease: [0.22, 1, 0.36, 1] }}
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: "3rem",
        alignItems: "center",
        marginBottom: "5rem",
        direction: isLeft ? "ltr" : "rtl",
      }}
    >
      {/* Image Column */}
      <motion.div
        initial={{ opacity: 0, x: isLeft ? -60 : 60 }}
        animate={inView ? { opacity: 1, x: 0 } : {}}
        transition={{ duration: 0.8, delay: index * 0.15 + 0.15, ease: [0.22, 1, 0.36, 1] }}
        style={{ direction: "ltr", position: "relative" }}
      >
        {/* Glow blob behind image */}
        <div style={{
          position: "absolute",
          inset: "-20%",
          background: `radial-gradient(circle, ${step.glowColor} 0%, transparent 70%)`,
          borderRadius: "50%",
          pointerEvents: "none",
          filter: "blur(20px)",
        }} />

        <div style={{
          position: "relative",
          background: "rgba(10,17,34,0.6)",
          border: `1px solid ${step.accent}28`,
          borderRadius: 28,
          padding: "2.5rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backdropFilter: "blur(16px)",
          boxShadow: `0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px ${step.accent}15`,
          overflow: "hidden",
        }}>
          {/* Subtle corner shimmer */}
          <div style={{
            position: "absolute", top: 0, left: 0,
            width: "60%", height: "60%",
            background: `radial-gradient(ellipse at top left, ${step.accent}12 0%, transparent 60%)`,
            pointerEvents: "none",
          }} />

          <motion.img
            src={step.image}
            alt={step.imageAlt}
            whileHover={{ scale: 1.04, rotate: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            style={{
              width: "100%",
              maxWidth: 280,
              height: "auto",
              objectFit: "contain",
              position: "relative",
              zIndex: 1,
              filter: `drop-shadow(0 12px 28px ${step.glowColor})`,
            }}
          />
        </div>
      </motion.div>

      {/* Text Column */}
      <motion.div
        initial={{ opacity: 0, x: isLeft ? 60 : -60 }}
        animate={inView ? { opacity: 1, x: 0 } : {}}
        transition={{ duration: 0.8, delay: index * 0.15 + 0.25, ease: [0.22, 1, 0.36, 1] }}
        style={{ direction: "ltr" }}
      >
        {/* Step counter */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.2rem" }}>
          <div style={{
            width: 42, height: 42,
            borderRadius: 12,
            background: `linear-gradient(135deg, ${step.accent}22, ${step.accentAlt}33)`,
            border: `1px solid ${step.accent}40`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.75rem", fontWeight: 900, color: step.accent,
            letterSpacing: "0.04em",
            boxShadow: `0 4px 16px ${step.glowColor}`,
          }}>
            {step.step}
          </div>
          <div style={{
            fontSize: "0.72rem", fontWeight: 700,
            color: step.accent,
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            padding: "0.28rem 0.8rem",
            borderRadius: 100,
            background: `${step.accent}12`,
            border: `1px solid ${step.accent}25`,
          }}>
            {step.subtitle}
          </div>
        </div>

        {/* Title */}
        <h2 style={{
          fontSize: "clamp(1.5rem, 2.5vw, 2rem)",
          fontWeight: 900,
          lineHeight: 1.15,
          letterSpacing: "-0.03em",
          marginBottom: "1rem",
          background: `linear-gradient(135deg, #ffffff 40%, ${step.accent} 110%)`,
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          {step.title}
        </h2>

        {/* Description */}
        <p style={{
          fontSize: "1rem",
          color: "#94a3b8",
          lineHeight: 1.8,
          marginBottom: "1.75rem",
          maxWidth: 440,
        }}>
          {step.description}
        </p>

        {/* Tags */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
          {step.tags.map(tag => (
            <span key={tag} style={{
              fontSize: "0.76rem",
              fontWeight: 700,
              padding: "0.32rem 0.85rem",
              borderRadius: 100,
              background: `${step.accent}10`,
              border: `1px solid ${step.accent}28`,
              color: step.accent,
              letterSpacing: "0.03em",
            }}>
              {tag}
            </span>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
};

/* ── Connector line between steps ─────────────────────────────── */
const Connector = ({ color }) => (
  <div style={{
    width: 1,
    height: 60,
    background: `linear-gradient(to bottom, ${color}40, transparent)`,
    margin: "-2rem auto 0",
    position: "relative",
    zIndex: 0,
  }}>
    <div style={{
      position: "absolute",
      bottom: 0,
      left: "50%",
      transform: "translateX(-50%)",
      width: 8, height: 8,
      borderRadius: "50%",
      background: color,
      boxShadow: `0 0 10px ${color}`,
    }} />
  </div>
);

/* ── Main Page ────────────────────────────────────────────────── */
const HowToUseDoctor = () => {
  const heroRef = useRef(null);
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
        @media (max-width: 768px) {
          .step-grid { grid-template-columns: 1fr !important; direction: ltr !important; }
          .step-grid > * { direction: ltr !important; }
        }
      `}</style>

      {/* Ambient background glows */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "5%",  left: "5%",  width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", top: "40%", right: "0%", width: 500, height: 500, borderRadius: "50%", background: "radial-gradient(circle, rgba(129,140,248,0.05) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "10%", left: "20%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(52,211,153,0.04) 0%, transparent 65%)" }} />
      </div>

      {/* Subtle grid overlay */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, opacity: 0.3,
        backgroundImage: "linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px)",
        backgroundSize: "80px 80px",
      }} />

      <div style={{ position: "relative", zIndex: 10 }}>
        <Navbar />
      </div>

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "4rem 2rem 8rem", position: "relative", zIndex: 1 }}>

        {/* ── Hero Header ────────────────────────────────────────── */}
        <motion.div
          ref={heroRef}
          initial={{ opacity: 0, y: 40 }}
          animate={heroInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          style={{ textAlign: "center", marginBottom: "6rem" }}
        >
          {/* Eyebrow */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={heroInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.6, delay: 0.1 }}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "rgba(56,189,248,0.09)",
              border: "1px solid rgba(56,189,248,0.22)",
              borderRadius: 100, padding: "0.38rem 1.1rem",
              fontSize: "0.72rem", fontWeight: 700,
              color: "#38bdf8", letterSpacing: "0.1em",
              textTransform: "uppercase", marginBottom: "1.5rem",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#38bdf8", display: "inline-block", boxShadow: "0 0 8px #38bdf8", animation: "glow 2s ease-in-out infinite" }} />
            Clinical Workflow Documentation
          </motion.div>

          <h1 style={{
            fontSize: "clamp(2.5rem, 5vw, 3.8rem)",
            fontWeight: 900,
            letterSpacing: "-0.04em",
            lineHeight: 1.08,
            marginBottom: "1.2rem",
          }}>
            <span style={{
              background: "linear-gradient(135deg, #ffffff 30%, #7dd3fc 70%, #818cf8 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}>
              Clinical Guide
            </span>
            <br />
            <span style={{ color: "#475569", fontSize: "0.55em", fontWeight: 600, letterSpacing: "-0.02em" }}>
              for ZenithDx Clinicians
            </span>
          </h1>

          <p style={{
            fontSize: "1.1rem",
            color: "#64748b",
            maxWidth: 560,
            margin: "0 auto 2rem",
            lineHeight: 1.75,
          }}>
            Follow these three steps for an efficient, explainable, and transparent AI-assisted diagnostic process.
          </p>

          {/* Step count pills */}
          <div style={{ display: "flex", justifyContent: "center", gap: "1rem", flexWrap: "wrap" }}>
            {[
              { label: "3 Clinical Steps", color: "#38bdf8" },
              { label: "Grad-CAM XAI", color: "#818cf8" },
              { label: "PDF Export", color: "#34d399" },
            ].map(pill => (
              <div key={pill.label} style={{
                padding: "0.4rem 1rem", borderRadius: 100,
                background: `${pill.color}10`,
                border: `1px solid ${pill.color}28`,
                fontSize: "0.8rem", fontWeight: 700,
                color: pill.color, letterSpacing: "0.03em",
              }}>
                {pill.label}
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Step Cards ─────────────────────────────────────────── */}
        <div>
          {STEPS.map((step, i) => (
            <div key={step.step}>
              <div className="step-grid" style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "3rem",
                alignItems: "center",
                marginBottom: "2rem",
                direction: step.direction === "right" ? "rtl" : "ltr",
              }}>
                {/* Image Panel */}
                <motion.div
                  initial={{ opacity: 0, x: step.direction === "left" ? -70 : 70 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true, margin: "-80px" }}
                  transition={{ duration: 0.75, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                  style={{ direction: "ltr", position: "relative" }}
                >
                  <div style={{
                    position: "absolute", inset: "-15%",
                    background: `radial-gradient(circle, ${step.glowColor} 0%, transparent 70%)`,
                    borderRadius: "50%", pointerEvents: "none", filter: "blur(24px)",
                  }} />
                  <div style={{
                    position: "relative",
                    background: "linear-gradient(135deg, rgba(12,20,44,0.9) 0%, rgba(18,28,56,0.7) 100%)",
                    border: `1px solid ${step.accent}22`,
                    borderRadius: 32,
                    padding: "3rem 2.5rem",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    backdropFilter: "blur(20px)",
                    boxShadow: `0 30px 80px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.04)`,
                    overflow: "hidden",
                    minHeight: 280,
                  }}>
                    {/* Inner corner glow */}
                    <div style={{
                      position: "absolute", top: 0, left: 0, right: 0,
                      height: "50%",
                      background: `radial-gradient(ellipse at 30% 0%, ${step.accent}10 0%, transparent 60%)`,
                      pointerEvents: "none",
                    }} />
                    <motion.img
                      src={step.image}
                      alt={step.imageAlt}
                      whileHover={{ scale: 1.06, y: -4 }}
                      transition={{ type: "spring", stiffness: 280, damping: 22 }}
                      style={{
                        width: "100%", maxWidth: 260, height: "auto",
                        objectFit: "contain", position: "relative", zIndex: 1,
                        filter: `drop-shadow(0 16px 36px ${step.glowColor})`,
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
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.3rem" }}>
                    <div style={{
                      width: 44, height: 44, borderRadius: 13,
                      background: `linear-gradient(135deg, ${step.accent}1a, ${step.accentAlt}2a)`,
                      border: `1px solid ${step.accent}35`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontWeight: 900, fontSize: "0.8rem", color: step.accent,
                      letterSpacing: "0.03em",
                      boxShadow: `0 4px 20px ${step.glowColor}`,
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

                  <h2 style={{
                    fontSize: "clamp(1.5rem, 2.4vw, 1.9rem)",
                    fontWeight: 900, lineHeight: 1.15,
                    letterSpacing: "-0.03em", marginBottom: "1rem",
                    background: `linear-gradient(130deg, #f0f6ff 35%, ${step.accent} 110%)`,
                    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                  }}>
                    {step.title}
                  </h2>

                  <p style={{
                    fontSize: "0.98rem", color: "#94a3b8",
                    lineHeight: 1.8, marginBottom: "1.8rem", maxWidth: 430,
                  }}>
                    {step.description}
                  </p>

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

              {/* Step connector (skip after last) */}
              {i < STEPS.length - 1 && (
                <motion.div
                  initial={{ scaleY: 0, opacity: 0 }}
                  whileInView={{ scaleY: 1, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  style={{
                    display: "flex", flexDirection: "column", alignItems: "center",
                    margin: "1.5rem 0 3rem", gap: "0.25rem",
                  }}
                >
                  <div style={{ width: 1, height: 48, background: `linear-gradient(to bottom, ${STEPS[i].accent}50, ${STEPS[i+1].accent}50)` }} />
                  <div style={{
                    width: 10, height: 10, borderRadius: "50%",
                    background: STEPS[i+1].accent,
                    boxShadow: `0 0 12px ${STEPS[i+1].glowColor}`,
                  }} />
                  <div style={{ width: 1, height: 48, background: `linear-gradient(to bottom, ${STEPS[i+1].accent}50, transparent)` }} />
                </motion.div>
              )}
            </div>
          ))}
        </div>

        {/* ── Bottom CTA ─────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          style={{
            textAlign: "center",
            marginTop: "5rem",
            padding: "3.5rem",
            background: "linear-gradient(135deg, rgba(10,20,44,0.9) 0%, rgba(15,28,60,0.7) 100%)",
            border: "1px solid rgba(56,189,248,0.16)",
            borderRadius: 32,
            backdropFilter: "blur(20px)",
            boxShadow: "0 30px 80px rgba(0,0,0,0.35)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "50%", background: "radial-gradient(ellipse at 50% 0%, rgba(56,189,248,0.08) 0%, transparent 60%)", pointerEvents: "none" }} />
          <div style={{ position: "relative", zIndex: 1 }}>
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>🩺</div>
            <h3 style={{
              fontSize: "clamp(1.4rem, 2.5vw, 1.85rem)",
              fontWeight: 900, letterSpacing: "-0.03em",
              background: "linear-gradient(135deg, #fff 40%, #7dd3fc 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              marginBottom: "0.75rem",
            }}>
              Ready to review your clinical queue?
            </h3>
            <p style={{ color: "#64748b", fontSize: "0.96rem", marginBottom: "2rem", maxWidth: 420, margin: "0 auto 2rem" }}>
              Access your patient case manager, verify AI diagnostics, and issue clinical approvals.
            </p>
            <a href="/homedoctor" style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              padding: "0.75rem 2rem", borderRadius: 14,
              background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
              color: "#fff", fontWeight: 800, fontSize: "0.95rem",
              textDecoration: "none",
              boxShadow: "0 8px 30px rgba(14,165,233,0.35)",
              letterSpacing: "0.01em",
              transition: "all 0.2s",
            }}>
              Go to Case Manager →
            </a>
          </div>
        </motion.div>

      </main>

      <style>{`
        @keyframes glow {
          0%,100% { opacity:1; box-shadow: 0 0 8px #38bdf8; }
          50% { opacity:0.5; box-shadow: 0 0 4px #38bdf8; }
        }
        @media (max-width: 768px) {
          .step-grid { grid-template-columns: 1fr !important; direction: ltr !important; gap: 2rem !important; }
        }
      `}</style>
    </div>
  );
};

export default HowToUseDoctor;
