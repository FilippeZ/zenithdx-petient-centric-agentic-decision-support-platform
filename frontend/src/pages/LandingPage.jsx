import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

// Generate 80 frame image URLs from public folder
const FRAME_COUNT = 80;
const FRAMES = Array.from({ length: FRAME_COUNT }, (_, i) => {
  const num = String(i).padStart(3, "0");
  return `/landing_frames/Kanto_professional_gia_202602241405_ob1cp_${num}.jpg`;
});

const FPS = 14; // Faster, smooth video animation playback rate

const FEATURES = [
  {
    icon: "🧠",
    title: "AI-Powered Radiology Analysis",
    desc: "S²A-UNet segmentation combined with ResNet-50 multi-label classification for precise pulmonary pathology detection.",
  },
  {
    icon: "🔍",
    title: "Explainable AI (XAI)",
    desc: "Grad-CAM visual heatmaps and Captum feature attribution ensure complete mathematical transparency behind every decision.",
  },
  {
    icon: "📚",
    title: "RAG Clinical Knowledge",
    desc: "FAISS dense vector search and BM25 sparse retrieval query peer-reviewed medical literature in real time.",
  },
  {
    icon: "🕸️",
    title: "Graph EHR Intelligence",
    desc: "Heterogeneous Graph Transformers mine longitudinal patient history to identify clinical patterns and risks.",
  },
  {
    icon: "🤖",
    title: "Agentic LangGraph Orchestrator",
    desc: "ReAct reasoning loop coordinates multi-modal tools, self-refines findings, and formats structured clinical reports.",
  },
  {
    icon: "🔒",
    title: "Enterprise Role Security",
    desc: "JWT-authenticated role-based dashboards tailored for both practicing clinicians and patients.",
  },
];

export default function LandingPage() {
  const navigate = useNavigate();
  const [currentFrame, setCurrentFrame] = useState(0);
  const [scrollY, setScrollY] = useState(0);
  const [heroVisible, setHeroVisible] = useState(false);
  const rafRef = useRef(null);
  const lastTimeRef = useRef(null);
  const frameInterval = 1000 / FPS;

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

  // Parallax scroll tracking
  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Hero text entrance animation
  useEffect(() => {
    const t = setTimeout(() => setHeroVisible(true), 150);
    return () => clearTimeout(t);
  }, []);

  return (
    <div style={{ background: "#020818", color: "#fff", fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", overflowX: "hidden" }}>

      {/* ═══════════════════════════════════════════════
          NAVBAR
      ══════════════════════════════════════════════ */}
      <nav style={{
        position: "fixed", top: 0, left: 0, right: 0, zIndex: 100,
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "1.2rem 3rem",
        background: scrollY > 60
          ? "rgba(2,8,24,0.94)"
          : "linear-gradient(180deg, rgba(2,8,24,0.85) 0%, transparent 100%)",
        backdropFilter: scrollY > 60 ? "blur(20px)" : "none",
        borderBottom: scrollY > 60 ? "1px solid rgba(56,189,248,0.12)" : "none",
        transition: "all 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.85rem", cursor: "pointer" }} onClick={() => navigate("/")}>
          <img src="/logo.png" alt="ZenithDx" style={{ width: 44, height: 44 }} />
          <div>
            <div style={{ fontSize: "1.25rem", fontWeight: 800, letterSpacing: "0.04em",
              background: "linear-gradient(90deg, #38bdf8, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              ZenithDx
            </div>
            <div style={{ fontSize: "0.68rem", color: "#94a3b8", letterSpacing: "0.04em", fontWeight: 500 }}>
              Agentic Clinical Decision Support
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "2.5rem", alignItems: "center" }}>
          <a href="#features" style={navLinkStyle}>Features</a>
          <a href="/about-us" style={navLinkStyle}>About Us</a>
          <button onClick={() => navigate("/auth")} style={ctaBtnSmall}>
            Get Started →
          </button>
        </div>
      </nav>

      {/* ═══════════════════════════════════════════════
          HERO — SMOOTH VIDEO BACKDROP
      ══════════════════════════════════════════════ */}
      <section style={{ position: "relative", width: "100%", height: "100vh", overflow: "hidden" }}>

        {/* Video frame animation background */}
        <img
          src={FRAMES[currentFrame]}
          alt="ZenithDx Clinical AI"
          style={{
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
            objectFit: "cover",
            transform: `translateY(${scrollY * 0.3}px) scale(1.04)`,
            transition: "none",
            willChange: "transform",
          }}
        />

        {/* Dark gradient overlay */}
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(circle at center, rgba(2,8,24,0.45) 0%, rgba(2,8,24,0.88) 85%)",
        }} />

        {/* Grid pattern overlay */}
        <div style={{
          position: "absolute", inset: 0, pointerEvents: "none",
          backgroundImage: "linear-gradient(rgba(56,189,248,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.03) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }} />

        {/* Hero content */}
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          textAlign: "center", padding: "0 1.5rem",
          transform: `translateY(${scrollY * 0.15}px)`,
        }}>
          <div style={{
            opacity: heroVisible ? 1 : 0,
            transform: heroVisible ? "translateY(0)" : "translateY(30px)",
            transition: "all 0.9s cubic-bezier(0.16, 1, 0.3, 1)",
          }}>
            {/* Pill Badge */}
            <div style={{
              display: "inline-flex", alignItems: "center", gap: "0.6rem",
              background: "rgba(15,23,42,0.75)", border: "1px solid rgba(56,189,248,0.3)",
              borderRadius: "100px", padding: "0.45rem 1.25rem", marginBottom: "2rem",
              fontSize: "0.82rem", color: "#38bdf8", letterSpacing: "0.06em", fontWeight: 600,
              boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
            }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#38bdf8",
                animation: "pulse 2s infinite", display: "inline-block" }} />
              PATIENT-CENTRIC CLINICAL AI PLATFORM
            </div>

            <h1 style={{
              fontSize: "clamp(2.8rem, 6.5vw, 5.2rem)",
              fontWeight: 900, lineHeight: 1.1, marginBottom: "1.5rem",
              letterSpacing: "-0.03em",
            }}>
              <span style={{
                background: "linear-gradient(135deg, #ffffff 40%, #7dd3fc 75%, #a5b4fc 100%)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
              }}>
                Agentic Intelligence for<br />Clinical Diagnostic Support
              </span>
            </h1>

            <p style={{
              fontSize: "clamp(1.05rem, 1.8vw, 1.25rem)",
              color: "#94a3b8", maxWidth: "620px", margin: "0 auto 2.8rem",
              lineHeight: 1.7, fontWeight: 400,
            }}>
              Fusing multi-label vision models, explainable Grad-CAM heatmaps, graph EHR mining,
              and an autonomous ReAct loop to empower evidence-based clinical decisions.
            </p>

            <div style={{ display: "flex", gap: "1.2rem", justifyContent: "center", flexWrap: "wrap" }}>
              <button onClick={() => navigate("/auth")} style={ctaBtnPrimary}>
                Begin Healthcare Journey →
              </button>
              <button onClick={() => document.getElementById("features").scrollIntoView({ behavior: "smooth" })}
                style={ctaBtnSecondary}>
                View Capabilities
              </button>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div style={{
          position: "absolute", bottom: "3rem", left: "50%", transform: "translateX(-50%)",
          display: "flex", flexDirection: "column", alignItems: "center", gap: "0.4rem",
          opacity: scrollY > 40 ? 0 : 1, transition: "opacity 0.3s ease",
        }}>
          <div style={{ fontSize: "0.68rem", color: "#64748b", letterSpacing: "0.15em", fontWeight: 600 }}>SCROLL DOWN</div>
          <div style={{
            width: 22, height: 38, border: "1.5px solid rgba(148,163,184,0.35)", borderRadius: 12,
            display: "flex", justifyContent: "center", paddingTop: 6,
          }}>
            <div style={{
              width: 4, height: 7, background: "#38bdf8", borderRadius: 2,
              animation: "scrollDot 1.8s ease infinite",
            }} />
          </div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          FEATURES GRID
      ══════════════════════════════════════════════ */}
      <section id="features" style={{ padding: "8rem 2.5rem", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: "4.5rem" }}>
          <div style={sectionBadge}>MULTI-MODAL CAPABILITIES</div>
          <h2 style={sectionTitle}>
            Integrated Medical AI{" "}
            <span style={{ background: "linear-gradient(135deg, #38bdf8, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Architecture
            </span>
          </h2>
          <p style={sectionSubtitle}>
            Engineered to deliver end-to-end diagnostic transparency from radiograph to structured report.
          </p>
        </div>

        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
          gap: "1.8rem",
        }}>
          {FEATURES.map((f, i) => (
            <FeatureCard key={i} {...f} />
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          CTA SECTION
      ══════════════════════════════════════════════ */}
      <section style={{
        padding: "7rem 2.5rem",
        background: "linear-gradient(180deg, rgba(2,8,24,0) 0%, rgba(14,165,233,0.06) 100%)",
        borderTop: "1px solid rgba(56,189,248,0.12)",
        textAlign: "center",
      }}>
        <div style={sectionBadge}>GET STARTED</div>
        <h2 style={{ ...sectionTitle, marginTop: "1rem" }}>
          Ready to experience<br />
          <span style={{ background: "linear-gradient(135deg, #38bdf8, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            ZenithDx Decision Support?
          </span>
        </h2>
        <p style={{ ...sectionSubtitle, maxWidth: 500, margin: "0 auto 2.8rem" }}>
          Access the platform as a clinician to review AI reports or as a patient to manage your medical history.
        </p>
        <div style={{ display: "flex", gap: "1.2rem", justifyContent: "center", flexWrap: "wrap" }}>
          <button onClick={() => navigate("/auth")} style={ctaBtnPrimary}>
            Access Portal →
          </button>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════
          FOOTER
      ══════════════════════════════════════════════ */}
      <footer style={{
        background: "#020818",
        borderTop: "1px solid rgba(56,189,248,0.08)",
        padding: "2.5rem 2rem",
        textAlign: "center",
        color: "#475569",
        fontSize: "0.85rem",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.65rem", marginBottom: "0.75rem" }}>
          <img src="/logo.png" alt="" style={{ width: 28, height: 28 }} />
          <span style={{ color: "#f8fafc", fontWeight: 700, fontSize: "1rem" }}>ZenithDx</span>
        </div>
        © {new Date().getFullYear()} ZenithDx — Patient-Centric Agentic Decision Support Platform
      </footer>

      {/* Global CSS */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.85); }
        }
        @keyframes scrollDot {
          0% { transform: translateY(0); opacity: 1; }
          80% { transform: translateY(14px); opacity: 0; }
          100% { transform: translateY(0); opacity: 0; }
        }
        html { scroll-behavior: smooth; }
        * { box-sizing: border-box; }
        a { text-decoration: none; }
      `}</style>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered
          ? "linear-gradient(135deg, rgba(30,41,59,0.8), rgba(15,23,42,0.9))"
          : "rgba(15,23,42,0.5)",
        border: `1px solid ${hovered ? "rgba(56,189,248,0.4)" : "rgba(56,189,248,0.12)"}`,
        borderRadius: "16px", padding: "2.2rem",
        transition: "all 0.35s cubic-bezier(0.16, 1, 0.3, 1)",
        transform: hovered ? "translateY(-6px)" : "translateY(0)",
        backdropFilter: "blur(12px)",
        boxShadow: hovered ? "0 12px 30px rgba(0,0,0,0.3)" : "none",
        cursor: "default",
      }}
    >
      <div style={{ fontSize: "2.4rem", marginBottom: "1.2rem" }}>{icon}</div>
      <h3 style={{
        fontSize: "1.15rem", fontWeight: 700, marginBottom: "0.7rem",
        color: hovered ? "#38bdf8" : "#f1f5f9", transition: "color 0.3s",
      }}>{title}</h3>
      <p style={{ color: "#64748b", fontSize: "0.92rem", lineHeight: 1.7 }}>{desc}</p>
    </div>
  );
}

const navLinkStyle = {
  color: "#94a3b8", fontSize: "0.93rem", fontWeight: 500,
  letterSpacing: "0.02em", transition: "color 0.2s",
  cursor: "pointer",
};

const ctaBtnPrimary = {
  background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
  color: "#fff", border: "none", borderRadius: "10px",
  padding: "0.95rem 2.2rem", fontSize: "1.02rem", fontWeight: 700,
  cursor: "pointer", letterSpacing: "0.02em",
  boxShadow: "0 4px 24px rgba(14,165,233,0.35)",
  transition: "all 0.25s ease",
};

const ctaBtnSecondary = {
  background: "rgba(15,23,42,0.6)",
  color: "#cbd5e1", border: "1px solid rgba(148,163,184,0.3)",
  borderRadius: "10px", padding: "0.95rem 2.2rem",
  fontSize: "1.02rem", fontWeight: 600, cursor: "pointer",
  transition: "all 0.25s ease",
};

const ctaBtnSmall = {
  ...ctaBtnPrimary,
  padding: "0.6rem 1.4rem", fontSize: "0.88rem",
};

const sectionBadge = {
  display: "inline-block",
  background: "rgba(56,189,248,0.1)", border: "1px solid rgba(56,189,248,0.25)",
  borderRadius: "100px", padding: "0.38rem 1.1rem",
  fontSize: "0.75rem", color: "#38bdf8", letterSpacing: "0.12em", fontWeight: 600,
  marginBottom: "1.2rem",
};

const sectionTitle = {
  fontSize: "clamp(2rem, 4vw, 3.2rem)",
  fontWeight: 900, lineHeight: 1.15, letterSpacing: "-0.03em",
  color: "#f8fafc", marginBottom: "1rem",
};

const sectionSubtitle = {
  color: "#64748b", fontSize: "1.08rem", lineHeight: 1.7,
  marginBottom: "0",
};
