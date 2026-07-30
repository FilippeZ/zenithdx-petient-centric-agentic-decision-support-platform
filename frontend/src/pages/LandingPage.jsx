import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "../components/Navbar";

const FRAME_COUNT = 80;
const FRAMES = Array.from({ length: FRAME_COUNT }, (_, i) => {
  const num = String(i).padStart(3, "0");
  return `/landing_frames/Kanto_professional_gia_202602241405_ob1cp_${num}.jpg`;
});

const FPS = 14;

const FEATURES = [
  {
    icon: "🩺",
    badge: "VISION AI",
    title: "Instant Diagnostic Screening",
    desc: "Advanced neural networks analyze chest X-rays in seconds to detect pathologies, highlighting exact regions of concern with pinpoint precision.",
  },
  {
    icon: "🔍",
    badge: "TRANSPARENCY",
    title: "Visual Heatmaps (XAI)",
    desc: "Grad-CAM visual overlays explain every single AI finding, so clinicians and patients can clearly see what the algorithm evaluated.",
  },
  {
    icon: "📚",
    badge: "EVIDENCE",
    title: "RAG Medical Knowledge",
    desc: "Real-time search across peer-reviewed clinical guidelines and literature to substantiate every diagnostic insight with scientific proof.",
  },
  {
    icon: "🕸️",
    badge: "INTELLIGENCE",
    title: "Longitudinal Graph Insights",
    desc: "Connects past medical histories, lab results, and patient symptoms into a comprehensive longitudinal health timeline.",
  },
  {
    icon: "⚡",
    badge: "AGENTIC REASONING",
    title: "Multi-Agent AI Specialist",
    desc: "Autonomous ReAct agents cross-examine multi-modal data, self-correct observations, and construct plain-language diagnostic summaries.",
  },
  {
    icon: "🔒",
    badge: "SECURITY",
    title: "Role-Based Patient Security",
    desc: "Bank-grade JWT encryption with dedicated simplified views for patients and deep analytical toolkits for clinicians.",
  },
];

const OVERLAY_CARDS = [
  {
    icon: "🫁",
    title: "Multi-Modal Vision Pipeline",
    subtitle: "S²A-UNet Segmentation + ResNet-50 Classification",
    pill: "🟢 99.2% Accuracy",
    pillBg: "#ecfdf5", pillColor: "#047857", pillBorder: "#a7f3d0"
  },
  {
    icon: "🔥",
    title: "Grad-CAM Explainability (XAI)",
    subtitle: "PyTorch Spatial Attention Heatmap Overlay",
    pill: "🔵 Active XAI",
    pillBg: "#eff6ff", pillColor: "#1d4ed8", pillBorder: "#bfdbfe"
  },
  {
    icon: "📚",
    title: "RAG Literature & EHR Fusion",
    subtitle: "SciBERT Knowledge Search & MIMIC-IV History",
    pill: "🟣 Clinical Evidence",
    pillBg: "#f5f3ff", pillColor: "#6d28d9", pillBorder: "#ddd6fe"
  }
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [currentFrame, setCurrentFrame] = useState(0);
  const [overlayIndex, setOverlayIndex] = useState(0);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(null);
  const frameInterval = 1000 / FPS;

  useEffect(() => {
    const timer = setInterval(() => {
      setOverlayIndex(i => (i + 1) % OVERLAY_CARDS.length);
    }, 3500);
    return () => clearInterval(timer);
  }, []);

  const currentOverlayCard = OVERLAY_CARDS[overlayIndex];

  // Preload video frames
  useEffect(() => {
    FRAMES.forEach((src) => {
      const img = new Image();
      img.src = src;
    });
  }, []);

  // Frame animation loop
  useEffect(() => {
    const animate = (timestamp) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      const elapsed = timestamp - lastTimeRef.current;
      if (elapsed >= frameInterval) {
        setCurrentFrame((prev) => (prev + 1) % FRAMES.length);
        lastTimeRef.current = timestamp;
      }
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [frameInterval]);

  return (
    <div style={{ background: "#f8fafc", color: "#0f172a", fontFamily: "'Inter', sans-serif", overflowX: "hidden", minHeight: "100vh" }}>
      
      {/* Navbar */}
      <Navbar />

      {/* ═══════════════════════════════════════════════
          HERO SECTION — Crisp White & Vibrant Soft Glow
      ══════════════════════════════════════════════ */}
      <section style={{ position: "relative", width: "100%", minHeight: "90vh", display: "flex", alignItems: "center", justifyContent: "center", padding: "4rem 1.5rem", overflow: "hidden" }}>
        
        {/* Soft Ambient Background Meshes */}
        <motion.div
          animate={{ scale: [1, 1.1, 1], opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          style={{
            position: "absolute", top: "-10%", left: "50%", transform: "translateX(-50%)",
            width: "1000px", height: "600px",
            background: "radial-gradient(circle at center, rgba(14, 165, 233, 0.2) 0%, rgba(99, 102, 241, 0.1) 45%, transparent 70%)",
            filter: "blur(60px)", pointerEvents: "none", zIndex: 0,
          }}
        />

        <div style={{ maxWidth: 1240, width: "100%", margin: "0 auto", position: "relative", zIndex: 1 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "3.5rem", alignItems: "center" }}>
            
            {/* Left Hero Text Column */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            >
              {/* Pill Badge */}
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.1, duration: 0.5 }}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.6rem",
                  background: "#ffffff", border: "1px solid #e2e8f0",
                  borderRadius: "100px", padding: "0.45rem 1.25rem", marginBottom: "1.8rem",
                  fontSize: "0.82rem", color: "#0284c7", letterSpacing: "0.04em", fontWeight: 700,
                  boxShadow: "0 4px 14px rgba(0,0,0,0.04)",
                }}
              >
                <motion.span
                  animate={{ scale: [1, 1.4, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  style={{
                    width: 8, height: 8, borderRadius: "50%", background: "#10b981",
                    display: "inline-block", boxShadow: "0 0 8px #10b981"
                  }}
                />
                PATIENT & DOCTOR CLINICAL PLATFORM
              </motion.div>

              <h1 style={{
                fontSize: "clamp(2.6rem, 5vw, 4.2rem)",
                fontWeight: 900, lineHeight: 1.12, marginBottom: "1.5rem",
                letterSpacing: "-0.035em", color: "#0f172a",
              }}>
                Intelligent Diagnostic Insights{" "}
                <span style={{
                  background: "linear-gradient(135deg, #2563eb, #0284c7)",
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                }}>
                  Made Transparent & Simple.
                </span>
              </h1>

              <p style={{
                fontSize: "clamp(1.05rem, 1.6vw, 1.2rem)",
                color: "#475569", maxWidth: "580px", marginBottom: "2.4rem",
                lineHeight: 1.7, fontWeight: 400,
              }}>
                ZenithDx pairs advanced medical vision AI with clear explainable visual heatmaps and autonomous clinical reasoning—delivering reassuring summaries for patients and precise decision support for healthcare clinicians.
              </p>

              <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                <motion.button
                  whileHover={{ scale: 1.04, boxShadow: "0 12px 28px -4px rgba(37, 99, 235, 0.45)" }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate("/auth")}
                  style={{
                    background: "linear-gradient(135deg, #2563eb, #0284c7)",
                    color: "#ffffff", border: "none", borderRadius: "12px",
                    padding: "0.95rem 2.2rem", fontSize: "1.02rem", fontWeight: 700,
                    cursor: "pointer", letterSpacing: "0.01em",
                    boxShadow: "0 8px 24px -4px rgba(37, 99, 235, 0.35)",
                    transition: "all 0.2s ease",
                  }}
                >
                  Get Started Now →
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.04, backgroundColor: "#f8fafc", borderColor: "#94a3b8" }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => document.getElementById("portals")?.scrollIntoView({ behavior: "smooth" })}
                  style={{
                    background: "#ffffff",
                    color: "#334155", border: "1px solid #cbd5e1",
                    borderRadius: "12px", padding: "0.95rem 2rem",
                    fontSize: "1.02rem", fontWeight: 600, cursor: "pointer",
                    boxShadow: "0 2px 6px rgba(0,0,0,0.03)",
                    transition: "all 0.2s ease",
                  }}
                >
                  Explore Portals
                </motion.button>
              </div>

              {/* Trust Indicators */}
              <div style={{ display: "flex", alignItems: "center", gap: "2rem", marginTop: "3rem", borderTop: "1px solid #e2e8f0", paddingTop: "1.8rem" }}>
                <motion.div whileHover={{ scale: 1.06 }}>
                  <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#0f172a" }}>99.2%</div>
                  <div style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: 600 }}>Diagnostic Precision</div>
                </motion.div>
                <div style={{ width: 1, height: 32, background: "#cbd5e1" }} />
                <motion.div whileHover={{ scale: 1.06 }}>
                  <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#0f172a" }}>Sub-second</div>
                  <div style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: 600 }}>Heatmap Generation</div>
                </motion.div>
                <div style={{ width: 1, height: 32, background: "#cbd5e1" }} />
                <motion.div whileHover={{ scale: 1.06 }}>
                  <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "#10b981" }}>100%</div>
                  <div style={{ fontSize: "0.8rem", color: "#64748b", fontWeight: 600 }}>Explainable AI</div>
                </motion.div>
              </div>
            </motion.div>

            {/* Right Interactive Animated Card Showcase */}
            <motion.div
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
              style={{ position: "relative" }}
            >
              <div style={{
                position: "relative",
                borderRadius: "28px",
                overflow: "hidden",
                boxShadow: "0 25px 60px -12px rgba(15, 23, 42, 0.15), 0 0 0 1px rgba(226, 232, 240, 0.8)",
                background: "#ffffff",
              }}>
                {/* Animated Scanning Line */}
                <motion.div
                  animate={{ y: [0, 440, 0] }}
                  transition={{ duration: 3.5, repeat: Infinity, ease: "linear" }}
                  style={{
                    position: "absolute", left: 0, right: 0, height: "2px",
                    background: "linear-gradient(90deg, transparent, #38bdf8, transparent)",
                    boxShadow: "0 0 12px #38bdf8", zIndex: 3, pointerEvents: "none"
                  }}
                />

                <img
                  src={FRAMES[currentFrame]}
                  alt="ZenithDx Clinical AI Frame"
                  style={{
                    width: "100%",
                    height: "440px",
                    objectFit: "cover",
                    display: "block",
                  }}
                />

                {/* Overlaid Animated Floating Card with Dynamic Cycling */}
                <motion.div
                  animate={{ y: [0, -8, 0] }}
                  transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                  style={{
                    position: "absolute", bottom: "20px", left: "20px", right: "20px",
                    background: "rgba(255, 255, 255, 0.94)",
                    backdropFilter: "blur(16px)",
                    WebkitBackdropFilter: "blur(16px)",
                    borderRadius: "18px",
                    padding: "1.2rem",
                    border: "1px solid rgba(255, 255, 255, 0.9)",
                    boxShadow: "0 12px 30px rgba(0,0,0,0.1)",
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    zIndex: 4, minHeight: "80px"
                  }}
                >
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={currentOverlayCard.title}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.4 }}
                      style={{ display: "flex", alignItems: "center", justifyContent: "space-between", width: "100%" }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: "0.9rem" }}>
                        <div style={{
                          width: 44, height: 44, borderRadius: "12px",
                          background: "#eff6ff", border: "1px solid #bfdbfe",
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: "1.4rem",
                        }}>
                          {currentOverlayCard.icon}
                        </div>
                        <div>
                          <div style={{ fontSize: "0.92rem", fontWeight: 800, color: "#0f172a" }}>{currentOverlayCard.title}</div>
                          <div style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 500 }}>{currentOverlayCard.subtitle}</div>
                        </div>
                      </div>
                      <span style={{
                        padding: "0.35rem 0.85rem", borderRadius: "100px",
                        background: currentOverlayCard.pillBg, color: currentOverlayCard.pillColor,
                        fontSize: "0.78rem", fontWeight: 800, border: `1px solid ${currentOverlayCard.pillBorder}`,
                        whiteSpace: "nowrap"
                      }}>
                        {currentOverlayCard.pill}
                      </span>
                    </motion.div>
                  </AnimatePresence>
                </motion.div>
              </div>
            </motion.div>

          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          ROLE SELECTION PORTALS
      ══════════════════════════════════════════════ */}
      <section id="portals" style={{ padding: "5rem 2rem", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: "3.5rem" }}>
          <div style={sectionBadge}>TAILORED WORKFLOWS</div>
          <h2 style={sectionTitle}>Designed for Healthcare Humans</h2>
          <p style={sectionSubtitle}>Select your path to access personalized tools engineered for your specific needs.</p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "2rem" }}>
          
          {/* Patient Card */}
          <motion.div
            whileHover={{ y: -8, boxShadow: "0 16px 35px -5px rgba(37, 99, 235, 0.1)" }}
            transition={{ duration: 0.3 }}
            style={{
              background: "#ffffff", borderRadius: "24px", padding: "2.5rem",
              border: "1px solid #e2e8f0",
              boxShadow: "0 10px 30px -5px rgba(0,0,0,0.05)",
              display: "flex", flexDirection: "column", justifyContent: "space-between",
              transition: "all 0.3s ease"
            }}
          >
            <div>
              <div style={{
                width: 56, height: 56, borderRadius: "16px", background: "#eff6ff",
                color: "#2563eb", display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "1.8rem", marginBottom: "1.5rem", border: "1px solid #bfdbfe",
              }}>
                🧑‍⚕️
              </div>
              <div style={{ fontSize: "0.75rem", fontWeight: 800, color: "#2563eb", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
                For Patients & Families
              </div>
              <h3 style={{ fontSize: "1.4rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.8rem" }}>
                Patient Health Portal
              </h3>
              <p style={{ color: "#475569", fontSize: "0.95rem", lineHeight: 1.6, marginBottom: "1.8rem" }}>
                Upload your chest X-rays, describe symptoms in plain terms, and receive easy-to-understand AI explanations with zero confusing medical jargon.
              </p>
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 2rem 0", display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                {["Simple 3-step scan submission", "Clear plain-language findings", "Direct PDF download for your doctor"].map((item, idx) => (
                  <li key={idx} style={{ display: "flex", alignItems: "center", gap: "0.6rem", fontSize: "0.88rem", color: "#334155", fontWeight: 500 }}>
                    <span style={{ color: "#10b981", fontWeight: 800 }}>✓</span> {item}
                  </li>
                ))}
              </ul>
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                localStorage.setItem("user_role", "patient");
                navigate("/auth");
              }}
              style={{
                width: "100%", padding: "0.85rem", borderRadius: "12px",
                background: "#f1f5f9", color: "#0f172a", border: "1px solid #cbd5e1",
                fontSize: "0.95rem", fontWeight: 700, cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              Enter Patient Portal →
            </motion.button>
          </motion.div>

          {/* Clinician Card */}
          <motion.div
            whileHover={{ y: -8, boxShadow: "0 16px 35px -5px rgba(37, 99, 235, 0.15)", borderColor: "#93c5fd" }}
            transition={{ duration: 0.3 }}
            style={{
              background: "#ffffff", borderRadius: "24px", padding: "2.5rem",
              border: "1px solid #bfdbfe",
              boxShadow: "0 12px 35px -5px rgba(37, 99, 235, 0.08)",
              display: "flex", flexDirection: "column", justifyContent: "space-between",
              position: "relative", overflow: "hidden",
              transition: "all 0.3s ease"
            }}
          >
            <div style={{
              position: "absolute", top: "18px", right: "18px",
              background: "linear-gradient(135deg, #2563eb, #0284c7)", color: "#fff",
              fontSize: "0.7rem", fontWeight: 800, padding: "0.3rem 0.8rem", borderRadius: "100px",
              letterSpacing: "0.04em"
            }}>
              CLINICAL SUITE
            </div>
            <div>
              <div style={{
                width: 56, height: 56, borderRadius: "16px", background: "#ecfdf5",
                color: "#059669", display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "1.8rem", marginBottom: "1.5rem", border: "1px solid #a7f3d0",
              }}>
                🩺
              </div>
              <div style={{ fontSize: "0.75rem", fontWeight: 800, color: "#059669", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.5rem" }}>
                For Radiologists & Physicians
              </div>
              <h3 style={{ fontSize: "1.4rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.8rem" }}>
                Clinician Workstation
              </h3>
              <p style={{ color: "#475569", fontSize: "0.95rem", lineHeight: 1.6, marginBottom: "1.8rem" }}>
                Full analytical workstation with interactive Grad-CAM visual heatmaps, multi-label pathology probability scoring, RAG evidence search, and case triage.
              </p>
              <ul style={{ listStyle: "none", padding: 0, margin: "0 0 2rem 0", display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                {["Grad-CAM visual attribution", "Peer-reviewed literature RAG search", "LangGraph multi-agent synthesis"].map((item, idx) => (
                  <li key={idx} style={{ display: "flex", alignItems: "center", gap: "0.6rem", fontSize: "0.88rem", color: "#334155", fontWeight: 500 }}>
                    <span style={{ color: "#2563eb", fontWeight: 800 }}>✓</span> {item}
                  </li>
                ))}
              </ul>
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                localStorage.setItem("user_role", "doctor");
                navigate("/auth");
              }}
              style={{
                width: "100%", padding: "0.85rem", borderRadius: "12px",
                background: "linear-gradient(135deg, #2563eb, #0284c7)", color: "#ffffff", border: "none",
                fontSize: "0.95rem", fontWeight: 700, cursor: "pointer",
                boxShadow: "0 4px 14px rgba(37, 99, 235, 0.25)",
                transition: "all 0.2s ease",
              }}
            >
              Enter Doctor Portal →
            </motion.button>
          </motion.div>

        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          FEATURES GRID
      ══════════════════════════════════════════════ */}
      <section id="features" style={{ padding: "6rem 2rem", maxWidth: 1240, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: "4rem" }}>
          <div style={sectionBadge}>ADVANCED PLATFORM FEATURES</div>
          <h2 style={sectionTitle}>Everything You Need for Diagnostic Clarity</h2>
          <p style={sectionSubtitle}>Combining medical precision with intuitive design for seamless everyday clinical operations.</p>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
          gap: "2rem",
        }}>
          {FEATURES.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 25 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
            >
              <FeatureCard {...f} />
            </motion.div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════ */}
      <footer style={{
        background: "#ffffff",
        borderTop: "1px solid #e2e8f0",
        padding: "3rem 2rem",
        textAlign: "center",
        color: "#64748b",
        fontSize: "0.88rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem", marginBottom: "0.85rem" }}>
          <img src="/logo.png" alt="" style={{ width: 32, height: 32 }} />
          <span style={{ color: "#0f172a", fontWeight: 800, fontSize: "1.1rem" }}>ZenithDx</span>
        </div>
        <p style={{ maxWidth: "500px", margin: "0 auto 1.5rem", color: "#64748b", fontSize: "0.85rem" }}>
          Patient-Centric Agentic Clinical AI Platform. Designed to support healthcare decision-making with explainability and transparency.
        </p>
        <div>
          © {new Date().getFullYear()} ZenithDx — All rights reserved.
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, badge, title, desc }) {
  return (
    <motion.div
      whileHover={{ y: -8, scale: 1.02, boxShadow: "0 16px 35px -5px rgba(37, 99, 235, 0.12)", borderColor: "#93c5fd" }}
      transition={{ duration: 0.3 }}
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: "20px", padding: "2.2rem",
        boxShadow: "0 4px 15px rgba(0,0,0,0.02)",
        height: "100%",
        display: "flex", flexDirection: "column", justifyContent: "space-between",
        cursor: "default", transition: "all 0.3s ease"
      }}
    >
      <div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.2rem" }}>
          <motion.div whileHover={{ rotate: 10, scale: 1.15 }} style={{ fontSize: "2.4rem", display: "inline-block" }}>{icon}</motion.div>
          <span style={{
            fontSize: "0.68rem", fontWeight: 800, color: "#0284c7",
            background: "#f0f9ff", border: "1px solid #bae6fd",
            padding: "0.25rem 0.65rem", borderRadius: "100px", letterSpacing: "0.06em",
          }}>
            {badge}
          </span>
        </div>
        <h3 style={{
          fontSize: "1.15rem", fontWeight: 800, marginBottom: "0.75rem",
          color: "#0f172a",
        }}>
          {title}
        </h3>
        <p style={{ color: "#475569", fontSize: "0.92rem", lineHeight: 1.65, margin: 0 }}>
          {desc}
        </p>
      </div>
    </motion.div>
  );
}

const sectionBadge = {
  display: "inline-block",
  background: "#eff6ff", border: "1px solid #bfdbfe",
  borderRadius: "100px", padding: "0.38rem 1.1rem",
  fontSize: "0.75rem", color: "#2563eb", letterSpacing: "0.08em", fontWeight: 700,
  marginBottom: "1rem",
};

const sectionTitle = {
  fontSize: "clamp(2rem, 3.8vw, 3rem)",
  fontWeight: 900, lineHeight: 1.15, letterSpacing: "-0.03em",
  color: "#0f172a", marginBottom: "0.8rem",
};

const sectionSubtitle = {
  color: "#64748b", fontSize: "1.05rem", lineHeight: 1.65,
  maxWidth: "600px", margin: "0 auto",
};
