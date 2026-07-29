import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import homedoctor from "../assets/homedoctor.png";

const FIELDS = {
  register: [
    { id: "full_name", label: "Full Name", type: "text", placeholder: "Dr. John Smith", icon: "👤" },
    { id: "username",  label: "Username",  type: "text", placeholder: "johnsmith",       icon: "🆔" },
    { id: "email",     label: "Email Address", type: "email", placeholder: "john@clinic.com", icon: "✉️" },
    { id: "password",  label: "Password",  type: "password", placeholder: "••••••••••••",  icon: "🔒" },
    { id: "confirm",   label: "Confirm Password", type: "password", placeholder: "••••••••••••", icon: "🔑" },
  ],
  login: [
    { id: "username", label: "Username", type: "text",     placeholder: "johnsmith",  icon: "🆔" },
    { id: "password", label: "Password", type: "password", placeholder: "••••••••••••", icon: "🔒" },
  ],
};

export default function AuthPage() {
  const navigate = useNavigate();
  const [mode, setMode]         = useState("login"); // "login" | "register"
  const [userType, setUserType] = useState("patient");
  const [form, setForm]         = useState({});
  const [error, setError]       = useState("");
  const [success, setSuccess]   = useState("");
  const [loading, setLoading]   = useState(false);
  const [mounted, setMounted]   = useState(false);

  useEffect(() => {
    setTimeout(() => setMounted(true), 50);
  }, []);

  const handleChange = (id, val) => setForm(f => ({ ...f, [id]: val }));

  const switchMode = (m) => {
    setMode(m); setError(""); setSuccess(""); setForm({});
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(""); setSuccess(""); setLoading(true);

    try {
      if (mode === "register") {
        if (!form.full_name || !form.username || !form.email || !form.password || !form.confirm) {
          setError("All fields are required."); setLoading(false); return;
        }
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email)) {
          setError("Please enter a valid email address."); setLoading(false); return;
        }
        if (form.password !== form.confirm) {
          setError("Passwords do not match."); setLoading(false); return;
        }
        if (form.password.length < 8) {
          setError("Password must be at least 8 characters."); setLoading(false); return;
        }

        const body = new URLSearchParams({
          full_name: form.full_name,
          username:  form.username,
          email:     form.email,
          password:  form.password,
          user_type: userType,
        });

        const res  = await fetch("http://localhost:8000/register", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body,
        });
        const data = await res.json();

        if (res.ok) {
          setSuccess("Account created successfully! Signing in…");
          await doLogin(form.username, form.password);
        } else {
          setError(data.detail || data.error || "Registration failed.");
        }

      } else {
        await doLogin(form.username, form.password);
      }
    } catch (err) {
      console.error(err);
      setError("Unable to connect to ZenithDx server.");
    } finally {
      setLoading(false);
    }
  };

  const doLogin = async (username, password) => {
    const res  = await fetch("http://localhost:8000/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password }),
    });
    const data = await res.json();

    if (res.ok) {
      const { access_token } = data;
      localStorage.setItem("token", access_token);
      const decoded = jwtDecode(access_token);
      const role = decoded.role || decoded.user_type;
      localStorage.setItem("user_role", role);
      navigate(role === "doctor" ? "/homedoctor" : "/patient-dashboard");
    } else {
      setError(data.detail || "Invalid credentials. Please try again.");
    }
  };

  const fields = FIELDS[mode];

  return (
    <div style={{
      background: "#020818", minHeight: "100vh", color: "#fff",
      fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      display: "flex", flexDirection: "column", overflowX: "hidden",
      position: "relative",
    }}>

      {/* Ambient Lighting Background */}
      <div style={{
        position: "absolute", top: "-15%", left: "10%", width: "700px", height: "700px",
        background: "radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 65%)",
        pointerEvents: "none", zIndex: 0
      }} />
      <div style={{
        position: "absolute", bottom: "-15%", right: "5%", width: "700px", height: "700px",
        background: "radial-gradient(circle, rgba(129,140,248,0.12) 0%, transparent 65%)",
        pointerEvents: "none", zIndex: 0
      }} />

      {/* Top Navbar Header */}
      <header style={{
        padding: "1.2rem 3rem", display: "flex", alignItems: "center", justifyContent: "space-between",
        borderBottom: "1px solid rgba(56,189,248,0.1)", zIndex: 10, backdropFilter: "blur(16px)",
        background: "rgba(2,8,24,0.85)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.85rem", cursor: "pointer" }} onClick={() => navigate("/")}>
          <img src="/logo.png" alt="ZenithDx" style={{ width: 44, height: 44 }} />
          <div>
            <div style={{
              fontSize: "1.3rem", fontWeight: 800, letterSpacing: "0.04em",
              background: "linear-gradient(90deg, #38bdf8, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
            }}>
              ZenithDx
            </div>
            <div style={{ fontSize: "0.68rem", color: "#64748b", letterSpacing: "0.04em", fontWeight: 500 }}>
              Agentic Clinical Decision Support Platform
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
          <button onClick={() => navigate("/")} style={{
            background: "none", border: "none", color: "#94a3b8", fontSize: "0.93rem",
            fontWeight: 500, cursor: "pointer", transition: "color 0.2s"
          }}>
            Home
          </button>
          <button onClick={() => navigate("/about-us")} style={{
            background: "none", border: "none", color: "#94a3b8", fontSize: "0.93rem",
            fontWeight: 500, cursor: "pointer", transition: "color 0.2s"
          }}>
            About Us
          </button>
        </div>
      </header>

      {/* Main Grid Section */}
      <main style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        padding: "3rem 2.5rem 4rem", maxWidth: "1280px", margin: "0 auto", width: "100%",
        position: "relative", zIndex: 1, gap: "4.5rem",
      }} className="auth-hero-grid">

        {/* Left Column: Heading + Floating Clinician Portrait Showcase */}
        <div style={{
          flex: 1.1, display: "flex", flexDirection: "column",
          opacity: mounted ? 1 : 0, transform: mounted ? "translateY(0)" : "translateY(20px)",
          transition: "all 0.8s cubic-bezier(0.16, 1, 0.3, 1)",
        }}>

          {/* Badge Pill */}
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "0.6rem",
            background: "rgba(56,189,248,0.12)", border: "1px solid rgba(56,189,248,0.3)",
            borderRadius: "100px", padding: "0.45rem 1.25rem", fontSize: "0.78rem",
            color: "#38bdf8", letterSpacing: "0.08em", fontWeight: 600, width: "fit-content",
            marginBottom: "1.5rem"
          }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "#38bdf8", display: "inline-block", boxShadow: "0 0 10px #38bdf8" }} />
            CLINICAL DECISION SUPPORT SYSTEM
          </div>

          <h1 style={{
            fontSize: "clamp(2.6rem, 4.8vw, 4.2rem)", fontWeight: 900,
            lineHeight: 1.1, letterSpacing: "-0.03em", marginBottom: "1.2rem"
          }}>
            <span style={{
              background: "linear-gradient(135deg, #ffffff 40%, #7dd3fc 75%, #a5b4fc 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
            }}>
              {mode === "login" ? "Sign In to" : "Register to"}<br />
              begin your Healthcare Journey
            </span>
          </h1>

          <p style={{ color: "#94a3b8", fontSize: "1.1rem", lineHeight: 1.7, maxWidth: "540px", marginBottom: "2.5rem", fontWeight: 400 }}>
            Access AI-assisted chest radiograph diagnostics, XAI Grad-CAM overlays, and multi-modal patient intelligence.
          </p>

          {/* Clinician Portrait Showcase Card with Floating Metric Badges */}
          <div style={{
            position: "relative", width: "100%", height: "340px",
            background: "linear-gradient(135deg, rgba(15,23,42,0.8) 0%, rgba(30,41,59,0.4) 100%)",
            border: "1px solid rgba(56,189,248,0.2)", borderRadius: "24px",
            overflow: "hidden", display: "flex", alignItems: "flex-end", justifyContent: "center",
            boxShadow: "0 20px 50px rgba(0,0,0,0.5)", backdropFilter: "blur(16px)"
          }}>

            {/* Glowing Backdrop Light */}
            <div style={{
              position: "absolute", top: "20%", left: "50%", transform: "translateX(-50%)",
              width: "250px", height: "250px", borderRadius: "50%",
              background: "radial-gradient(circle, rgba(56,189,248,0.25) 0%, transparent 70%)",
              pointerEvents: "none"
            }} />

            {/* Doctor Image */}
            <img
              src={homedoctor}
              alt="ZenithDx Clinician"
              style={{
                maxHeight: "335px", width: "auto", objectFit: "contain",
                filter: "drop-shadow(0 15px 30px rgba(0,0,0,0.6))",
                position: "relative", zIndex: 1
              }}
            />

            {/* Floating Glass Metric Badge 1 (Top Left) */}
            <div style={{
              position: "absolute", top: "1.2rem", left: "1.2rem", zIndex: 2,
              background: "rgba(10,17,34,0.85)", border: "1px solid rgba(56,189,248,0.3)",
              borderRadius: "14px", padding: "0.6rem 1rem", backdropFilter: "blur(12px)",
              display: "flex", alignItems: "center", gap: "0.6rem",
              boxShadow: "0 8px 20px rgba(0,0,0,0.4)"
            }}>
              <span style={{ fontSize: "1.2rem" }}>🎯</span>
              <div>
                <div style={{ fontSize: "0.75rem", color: "#38bdf8", fontWeight: 700 }}>Grad-CAM XAI</div>
                <div style={{ fontSize: "0.65rem", color: "#94a3b8" }}>Heatmap Explanation</div>
              </div>
            </div>

            {/* Floating Glass Metric Badge 2 (Top Right) */}
            <div style={{
              position: "absolute", top: "1.2rem", right: "1.2rem", zIndex: 2,
              background: "rgba(10,17,34,0.85)", border: "1px solid rgba(129,140,248,0.3)",
              borderRadius: "14px", padding: "0.6rem 1rem", backdropFilter: "blur(12px)",
              display: "flex", alignItems: "center", gap: "0.6rem",
              boxShadow: "0 8px 20px rgba(0,0,0,0.4)"
            }}>
              <span style={{ fontSize: "1.2rem" }}>⚡</span>
              <div>
                <div style={{ fontSize: "0.75rem", color: "#818cf8", fontWeight: 700 }}>95.3% AUC</div>
                <div style={{ fontSize: "0.65rem", color: "#94a3b8" }}>14 Pathologies</div>
              </div>
            </div>

            {/* Bottom Gradient Overlay Fade */}
            <div style={{
              position: "absolute", bottom: 0, left: 0, right: 0, height: "60px",
              background: "linear-gradient(180deg, transparent 0%, #020818 100%)", zIndex: 2, pointerEvents: "none"
            }} />
          </div>
        </div>

        {/* Right Column: Premium Auth Glass Card */}
        <div style={{
          width: "100%", maxWidth: "460px",
          background: "rgba(10, 17, 34, 0.82)",
          border: "1px solid rgba(56, 189, 248, 0.25)",
          borderRadius: "26px", padding: "2.8rem 2.4rem",
          boxShadow: "0 30px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06)",
          backdropFilter: "blur(24px)",
          opacity: mounted ? 1 : 0, transform: mounted ? "translateY(0)" : "translateY(30px)",
          transition: "all 0.8s cubic-bezier(0.16, 1, 0.3, 1) 0.1s",
        }}>

          {/* Mode Switcher Tabs */}
          <div style={{
            display: "flex", background: "rgba(2, 8, 24, 0.85)",
            border: "1px solid rgba(56, 189, 248, 0.18)", borderRadius: "14px",
            padding: "5px", marginBottom: "2rem", gap: "4px"
          }}>
            <button
              type="button"
              onClick={() => switchMode("login")}
              style={{
                flex: 1, padding: "0.72rem", border: "none", borderRadius: "10px",
                fontSize: "0.93rem", fontWeight: 700, cursor: "pointer",
                transition: "all 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
                background: mode === "login"
                  ? "linear-gradient(135deg, #0ea5e9, #6366f1)"
                  : "transparent",
                color: mode === "login" ? "#fff" : "#64748b",
                boxShadow: mode === "login" ? "0 4px 18px rgba(14,165,233,0.4)" : "none",
              }}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => switchMode("register")}
              style={{
                flex: 1, padding: "0.72rem", border: "none", borderRadius: "10px",
                fontSize: "0.93rem", fontWeight: 700, cursor: "pointer",
                transition: "all 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
                background: mode === "register"
                  ? "linear-gradient(135deg, #0ea5e9, #6366f1)"
                  : "transparent",
                color: mode === "register" ? "#fff" : "#64748b",
                boxShadow: mode === "register" ? "0 4px 18px rgba(14,165,233,0.4)" : "none",
              }}
            >
              Register
            </button>
          </div>

          <h3 style={{ fontSize: "1.8rem", fontWeight: 800, color: "#f8fafc", marginBottom: "0.35rem", letterSpacing: "-0.02em" }}>
            {mode === "login" ? "Welcome Back" : "Create Account"}
          </h3>
          <p style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: "1.8rem", lineHeight: 1.5 }}>
            {mode === "login"
              ? "Enter your credentials to access your clinical dashboard"
              : "Select account type and complete your registration"}
          </p>

          {/* Role Selector (Register Mode) */}
          {mode === "register" && (
            <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1.5rem" }}>
              <button
                type="button"
                onClick={() => setUserType("patient")}
                style={{
                  flex: 1, padding: "0.75rem", borderRadius: "12px",
                  border: `1px solid ${userType === "patient" ? "rgba(56,189,248,0.6)" : "rgba(56,189,248,0.15)"}`,
                  background: userType === "patient" ? "rgba(56,189,248,0.14)" : "rgba(2,8,24,0.6)",
                  color: userType === "patient" ? "#38bdf8" : "#64748b",
                  fontSize: "0.88rem", fontWeight: 600, cursor: "pointer", transition: "all 0.2s"
                }}
              >
                🏥 Patient
              </button>
              <button
                type="button"
                onClick={() => setUserType("doctor")}
                style={{
                  flex: 1, padding: "0.75rem", borderRadius: "12px",
                  border: `1px solid ${userType === "doctor" ? "rgba(56,189,248,0.6)" : "rgba(56,189,248,0.15)"}`,
                  background: userType === "doctor" ? "rgba(56,189,248,0.14)" : "rgba(2,8,24,0.6)",
                  color: userType === "doctor" ? "#38bdf8" : "#64748b",
                  fontSize: "0.88rem", fontWeight: 600, cursor: "pointer", transition: "all 0.2s"
                }}
              >
                👨‍⚕️ Clinician
              </button>
            </div>
          )}

          {/* Error / Success Alerts */}
          {error && (
            <div style={{
              padding: "0.85rem 1rem", borderRadius: "12px", marginBottom: "1.3rem",
              background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)",
              color: "#fca5a5", fontSize: "0.88rem", display: "flex", alignItems: "center", gap: "0.6rem"
            }}>
              <span>⚠️</span> {error}
            </div>
          )}
          {success && (
            <div style={{
              padding: "0.85rem 1rem", borderRadius: "12px", marginBottom: "1.3rem",
              background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.3)",
              color: "#86efac", fontSize: "0.88rem", display: "flex", alignItems: "center", gap: "0.6rem"
            }}>
              <span>✅</span> {success}
            </div>
          )}

          {/* Form Fields */}
          <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
            {fields.map(({ id, label, type, placeholder, icon }) => (
              <div key={id}>
                <label style={{ display: "block", fontSize: "0.8rem", color: "#94a3b8", fontWeight: 600, marginBottom: "0.45rem", letterSpacing: "0.03em" }}>
                  {label}
                </label>
                <div style={{ position: "relative" }}>
                  <span style={{ position: "absolute", left: "0.95rem", top: "50%", transform: "translateY(-50%)", fontSize: "1rem" }}>
                    {icon}
                  </span>
                  <input
                    type={type}
                    placeholder={placeholder}
                    value={form[id] || ""}
                    onChange={(e) => handleChange(id, e.target.value)}
                    style={{
                      width: "100%", padding: "0.82rem 1.1rem 0.82rem 2.8rem",
                      background: "rgba(2, 8, 24, 0.85)",
                      border: "1px solid rgba(56, 189, 248, 0.22)",
                      borderRadius: "12px", color: "#f8fafc", fontSize: "0.95rem",
                      outline: "none", transition: "all 0.2s ease", boxSizing: "border-box"
                    }}
                    onFocus={(e) => e.target.style.borderColor = "rgba(56, 189, 248, 0.6)"}
                    onBlur={(e) => e.target.style.borderColor = "rgba(56, 189, 248, 0.22)"}
                    required
                  />
                </div>
              </div>
            ))}

            <button
              type="submit"
              disabled={loading}
              style={{
                marginTop: "0.6rem", width: "100%", padding: "0.95rem",
                background: loading ? "rgba(14,165,233,0.4)" : "linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%)",
                border: "none", borderRadius: "12px", color: "#fff",
                fontSize: "1.02rem", fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
                boxShadow: loading ? "none" : "0 6px 25px rgba(14,165,233,0.4)", transition: "all 0.25s ease",
                letterSpacing: "0.02em",
              }}
            >
              {loading ? "Processing..." : mode === "login" ? "Sign In →" : "Create Account →"}
            </button>
          </form>

          {/* Footer link */}
          <p style={{ textAlign: "center", color: "#64748b", fontSize: "0.88rem", marginTop: "1.6rem", marginBottom: 0 }}>
            {mode === "login" ? "Don't have an account? " : "Already have an account? "}
            <button
              type="button"
              onClick={() => switchMode(mode === "login" ? "register" : "login")}
              style={{ background: "none", border: "none", color: "#38bdf8", cursor: "pointer", fontWeight: 700, fontSize: "0.88rem", padding: 0 }}
            >
              {mode === "login" ? "Register" : "Sign In"}
            </button>
          </p>
        </div>
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; }
        @media (max-width: 900px) {
          .auth-hero-grid { flex-direction: column !important; gap: 2.5rem !important; }
        }
      `}</style>
    </div>
  );
}
