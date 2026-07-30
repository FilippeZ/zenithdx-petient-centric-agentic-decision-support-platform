import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "../components/Navbar";
import doctor from "../assets/doctor.png";
import patient from "../assets/patient.png";

const FIELDS = {
  register: [
    { id: "full_name", label: "Full Name", type: "text", placeholder: "e.g. Sarah Jenkins", icon: "👤" },
    { id: "username",  label: "Username",  type: "text", placeholder: "sarahj",            icon: "🆔" },
    { id: "email",     label: "Email Address", type: "email", placeholder: "sarah@example.com", icon: "✉️" },
    { id: "password",  label: "Password",  type: "password", placeholder: "••••••••••••",   icon: "🔒" },
    { id: "confirm",   label: "Confirm Password", type: "password", placeholder: "••••••••••••", icon: "🔑" },
  ],
  login: [
    { id: "username", label: "Username", type: "text",     placeholder: "Enter your username",  icon: "🆔" },
    { id: "password", label: "Password", type: "password", placeholder: "••••••••••••",        icon: "🔒" },
  ],
};

export default function AuthPage() {
  const navigate = useNavigate();
  const [mode, setMode]         = useState("login"); // "login" | "register"
  const [userType, setUserType] = useState(() => localStorage.getItem("user_role") || "patient");
  const [form, setForm]         = useState({});
  const [error, setError]       = useState("");
  const [success, setSuccess]   = useState("");
  const [loading, setLoading]   = useState(false);

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
      let role = userType;
      try {
        const decoded = jwtDecode(access_token);
        role = decoded.role || decoded.user_type || userType;
      } catch {
        // fallback
      }
      localStorage.setItem("user_role", role);
      navigate(role === "doctor" ? "/homedoctor" : "/patient-dashboard");
    } else {
      setError(data.detail || "Invalid credentials. Please try again.");
    }
  };

  const fields = FIELDS[mode];

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif", display: "flex", flexDirection: "column" }}>
      <Navbar />

      <main style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "3rem 1.5rem 6rem" }}>
        
        {/* Animated Split Container */}
        <div style={{
          width: "100%", maxWidth: "1050px",
          display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))",
          background: "#ffffff",
          borderRadius: "32px",
          overflow: "hidden",
          boxShadow: "0 25px 60px -15px rgba(15, 23, 42, 0.12), 0 0 0 1px rgba(226, 232, 240, 0.8)",
        }}>
          
          {/* Left Column: Animated Medical Illustration Banner */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            style={{
              background: "linear-gradient(135deg, #090d16 0%, #1e293b 50%, #0f172a 100%)",
              padding: "3.5rem 3rem",
              display: "flex", flexDirection: "column", justifyContent: "space-between",
              position: "relative", overflow: "hidden", color: "#ffffff"
            }}
          >
            {/* Ambient Pulse Glow */}
            <motion.div
              animate={{ scale: [1, 1.2, 1], opacity: [0.2, 0.4, 0.2] }}
              transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
              style={{
                position: "absolute", top: "-20%", left: "-20%", width: "140%", height: "140%",
                background: "radial-gradient(circle at center, rgba(37, 99, 235, 0.3), transparent 70%)",
                pointerEvents: "none"
              }}
            />

            <div style={{ position: "relative", zIndex: 2 }}>
              <div style={{
                display: "inline-flex", alignItems: "center", gap: "0.5rem",
                background: "rgba(59, 130, 246, 0.12)", border: "1px solid rgba(59, 130, 246, 0.3)",
                borderRadius: 100, padding: "0.4rem 1.1rem", fontSize: "0.75rem",
                color: "#60a5fa", fontWeight: 800, letterSpacing: "0.08em", textTransform: "uppercase",
                marginBottom: "1.5rem"
              }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#38bdf8", display: "inline-block", boxShadow: "0 0 8px #38bdf8" }} />
                NEXT-GEN DIAGNOSTIC PLATFORM
              </div>

              <h2 style={{ fontSize: "clamp(1.8rem, 3vw, 2.5rem)", fontWeight: 900, lineHeight: 1.2, marginBottom: "1rem" }}>
                Multi-Modal Clinical Intelligence
              </h2>

              <p style={{ color: "#94a3b8", fontSize: "1rem", lineHeight: 1.65, maxWidth: "400px" }}>
                Access transparent, explainable diagnostic insights powered by S²A-UNet segmentation, ResNet-50 classification, and longitudinal EHR integration.
              </p>
            </div>

            {/* Floating Doctor Image */}
            <div style={{ position: "relative", zIndex: 2, display: "flex", justifyContent: "center", marginTop: "2rem", marginBottom: "1rem" }}>
              <motion.img
                src={userType === "doctor" ? doctor : patient}
                alt="ZenithDx Medical Support"
                animate={{ y: [0, -8, 0] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                style={{ maxHeight: "260px", width: "auto", objectFit: "contain", filter: "drop-shadow(0 15px 25px rgba(0,0,0,0.5))" }}
              />
            </div>

            {/* Feature Badges */}
            <div style={{ position: "relative", zIndex: 2, display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>
              {["🔒 ISO 13485 Compliant", "🧠 ReAct Agent", "📊 Grad-CAM XAI"].map(b => (
                <span key={b} style={{
                  fontSize: "0.75rem", fontWeight: 700, background: "rgba(255, 255, 255, 0.06)",
                  border: "1px solid rgba(255, 255, 255, 0.12)", color: "#cbd5e1",
                  padding: "0.35rem 0.8rem", borderRadius: 100
                }}>
                  {b}
                </span>
              ))}
            </div>
          </motion.div>

          {/* Right Column: Animated Form Container */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            style={{ padding: "3.5rem 3rem", display: "flex", flexDirection: "column", justifyContent: "center" }}
          >
            {/* Header text */}
            <div style={{ marginBottom: "2rem" }}>
              <h1 style={{ fontSize: "1.85rem", fontWeight: 900, color: "#0f172a", marginBottom: "0.4rem", letterSpacing: "-0.02em" }}>
                {mode === "login" ? "Welcome Back" : "Create Your Account"}
              </h1>
              <p style={{ fontSize: "0.92rem", color: "#64748b", margin: 0 }}>
                {mode === "login" ? "Sign in to access your diagnostic portal" : "Join ZenithDx for transparent clinical support"}
              </p>
            </div>

            {/* User Type Toggle (Patient vs Clinician) */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem",
              background: "#f1f5f9", padding: "0.35rem", borderRadius: "14px",
              marginBottom: "1.5rem", border: "1px solid #e2e8f0",
            }}>
              <button
                type="button"
                onClick={() => setUserType("patient")}
                style={{
                  padding: "0.65rem", borderRadius: "10px", border: "none",
                  fontSize: "0.85rem", fontWeight: 700, cursor: "pointer",
                  transition: "all 0.2s ease",
                  background: userType === "patient" ? "#ffffff" : "transparent",
                  color: userType === "patient" ? "#2563eb" : "#64748b",
                  boxShadow: userType === "patient" ? "0 2px 8px rgba(37,99,235,0.12)" : "none",
                }}
              >
                🧑‍⚕️ Patient
              </button>
              <button
                type="button"
                onClick={() => setUserType("doctor")}
                style={{
                  padding: "0.65rem", borderRadius: "10px", border: "none",
                  fontSize: "0.85rem", fontWeight: 700, cursor: "pointer",
                  transition: "all 0.2s ease",
                  background: userType === "doctor" ? "#ffffff" : "transparent",
                  color: userType === "doctor" ? "#2563eb" : "#64748b",
                  boxShadow: userType === "doctor" ? "0 2px 8px rgba(37,99,235,0.12)" : "none",
                }}
              >
                🩺 Clinician
              </button>
            </div>

            {/* Mode Switch (Login vs Register) */}
            <div style={{ display: "flex", borderBottom: "1px solid #e2e8f0", marginBottom: "1.8rem" }}>
              <button
                type="button"
                onClick={() => switchMode("login")}
                style={{
                  flex: 1, padding: "0.75rem", border: "none", background: "none",
                  fontSize: "0.92rem", fontWeight: mode === "login" ? 800 : 600,
                  color: mode === "login" ? "#2563eb" : "#64748b",
                  borderBottom: mode === "login" ? "2.5px solid #2563eb" : "2.5px solid transparent",
                  cursor: "pointer", transition: "all 0.2s",
                }}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => switchMode("register")}
                style={{
                  flex: 1, padding: "0.75rem", border: "none", background: "none",
                  fontSize: "0.92rem", fontWeight: mode === "register" ? 800 : 600,
                  color: mode === "register" ? "#2563eb" : "#64748b",
                  borderBottom: mode === "register" ? "2.5px solid #2563eb" : "2.5px solid transparent",
                  cursor: "pointer", transition: "all 0.2s",
                }}
              >
                Register Account
              </button>
            </div>

            {/* Alerts */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  style={{
                    background: "#fef2f2", border: "1px solid #fecaca", color: "#991b1b",
                    borderRadius: "12px", padding: "0.85rem 1rem", fontSize: "0.86rem",
                    marginBottom: "1.2rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem"
                  }}
                >
                  ⚠️ {error}
                </motion.div>
              )}

              {success && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  style={{
                    background: "#ecfdf5", border: "1px solid #a7f3d0", color: "#065f46",
                    borderRadius: "12px", padding: "0.85rem 1rem", fontSize: "0.86rem",
                    marginBottom: "1.2rem", fontWeight: 600, display: "flex", alignItems: "center", gap: "0.5rem"
                  }}
                >
                  ✅ {success}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Form */}
            <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "1.2rem" }}>
              {fields.map((f) => (
                <div key={f.id}>
                  <label style={{ display: "block", fontSize: "0.82rem", fontWeight: 700, color: "#334155", marginBottom: "0.4rem" }}>
                    {f.label}
                  </label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <span style={{ position: "absolute", left: "1rem", fontSize: "1rem", pointerEvents: "none" }}>
                      {f.icon}
                    </span>
                    <input
                      type={f.type}
                      placeholder={f.placeholder}
                      value={form[f.id] || ""}
                      onChange={(e) => handleChange(f.id, e.target.value)}
                      style={{
                        width: "100%",
                        padding: "0.8rem 1rem 0.8rem 2.8rem",
                        borderRadius: "12px",
                        border: "1px solid #cbd5e1",
                        background: "#ffffff",
                        fontSize: "0.92rem",
                        color: "#0f172a",
                        outline: "none",
                        transition: "all 0.2s ease",
                      }}
                      onFocus={e => { e.target.style.borderColor = "#2563eb"; e.target.style.boxShadow = "0 0 0 3px rgba(37, 99, 235, 0.15)"; }}
                      onBlur={e => { e.target.style.borderColor = "#cbd5e1"; e.target.style.boxShadow = "none"; }}
                    />
                  </div>
                </div>
              ))}

              <motion.button
                whileHover={{ scale: loading ? 1 : 1.02, boxShadow: "0 8px 25px rgba(37, 99, 235, 0.35)" }}
                whileTap={{ scale: loading ? 1 : 0.98 }}
                type="submit"
                disabled={loading}
                style={{
                  marginTop: "0.6rem",
                  width: "100%",
                  padding: "0.9rem",
                  borderRadius: "12px",
                  border: "none",
                  background: "linear-gradient(135deg, #2563eb, #0284c7)",
                  color: "#ffffff",
                  fontSize: "0.98rem",
                  fontWeight: 800,
                  cursor: loading ? "wait" : "pointer",
                  boxShadow: "0 4px 14px rgba(37, 99, 235, 0.28)",
                  transition: "all 0.2s ease",
                  opacity: loading ? 0.7 : 1,
                }}
              >
                {loading ? "Processing..." : mode === "login" ? "Sign In →" : "Create Account →"}
              </motion.button>
            </form>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
