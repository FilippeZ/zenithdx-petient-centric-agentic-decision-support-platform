import { useState, useEffect, useRef } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";

/* ── Icon helpers ─────────────────────────────────────────────── */
const Icons = {
  home:       "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6",
  dashboard:  "M4 5a1 1 0 011-1h4a1 1 0 011 1v5a1 1 0 01-1 1H5a1 1 0 01-1-1V5zm10 0a1 1 0 011-1h4a1 1 0 011 1v2a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zm10-4a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1v-7z",
  scan:       "M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18",
  guide:      "M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253",
  about:      "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
  logout:     "M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1",
  menu:       "M4 6h16M4 12h16M4 18h16",
  close:      "M6 18L18 6M6 6l12 12",
  pulse:      "M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z",
};

const SVG = ({ d, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);

/* ── Nav Link ─────────────────────────────────────────────────── */
const NavLink = ({ to, icon, children, active, onClick }) => (
  <Link
    to={to}
    onClick={onClick}
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "0.45rem",
      padding: "0.48rem 0.95rem",
      borderRadius: 10,
      fontSize: "0.86rem",
      fontWeight: 600,
      letterSpacing: "0.01em",
      textDecoration: "none",
      transition: "all 0.2s ease",
      color: active ? "#f0f6ff" : "#94a3b8",
      background: active ? "rgba(56,189,248,0.14)" : "transparent",
      border: active ? "1px solid rgba(56,189,248,0.28)" : "1px solid transparent",
    }}
    onMouseEnter={e => {
      if (!active) {
        e.currentTarget.style.color = "#e2e8f0";
        e.currentTarget.style.background = "rgba(255,255,255,0.05)";
        e.currentTarget.style.border = "1px solid rgba(255,255,255,0.08)";
      }
    }}
    onMouseLeave={e => {
      if (!active) {
        e.currentTarget.style.color = "#94a3b8";
        e.currentTarget.style.background = "transparent";
        e.currentTarget.style.border = "1px solid transparent";
      }
    }}
  >
    {icon && <SVG d={icon} size={15} />}
    {children}
  </Link>
);

/* ── Logout Button ────────────────────────────────────────────── */
const LogoutBtn = ({ onClick, mobile }) => (
  <button
    onClick={onClick}
    style={{
      display: "inline-flex",
      alignItems: "center",
      gap: "0.45rem",
      padding: mobile ? "0.65rem 1.2rem" : "0.48rem 1rem",
      borderRadius: 10,
      fontSize: mobile ? "0.9rem" : "0.86rem",
      fontWeight: 700,
      cursor: "pointer",
      letterSpacing: "0.01em",
      border: "1px solid rgba(239,68,68,0.3)",
      background: "rgba(239,68,68,0.08)",
      color: "#f87171",
      transition: "all 0.2s ease",
      width: mobile ? "100%" : "auto",
      justifyContent: "center",
    }}
    onMouseEnter={e => {
      e.currentTarget.style.background = "rgba(239,68,68,0.16)";
      e.currentTarget.style.borderColor = "rgba(239,68,68,0.5)";
      e.currentTarget.style.color = "#fca5a5";
    }}
    onMouseLeave={e => {
      e.currentTarget.style.background = "rgba(239,68,68,0.08)";
      e.currentTarget.style.borderColor = "rgba(239,68,68,0.3)";
      e.currentTarget.style.color = "#f87171";
    }}
  >
    <SVG d={Icons.logout} size={15} />
    Sign Out
  </button>
);

/* ── Main Navbar ──────────────────────────────────────────────── */
const Navbar = () => {
  const [menuOpen, setMenuOpen]   = useState(false);
  const [userRole, setUserRole]   = useState(null);
  const [scrolled, setScrolled]   = useState(false);
  const navigate  = useNavigate();
  const location  = useLocation();
  const menuRef   = useRef(null);

  useEffect(() => {
    setUserRole(localStorage.getItem("user_role"));
  }, [location]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 16);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setMenuOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user_role");
    setMenuOpen(false);
    navigate("/");
  };

  const handleLogoClick = (e) => {
    e.preventDefault();
    const token = localStorage.getItem("token");
    const role  = localStorage.getItem("user_role");
    if (token && role === "doctor")  navigate("/homedoctor");
    else if (token && role === "patient") navigate("/patient-dashboard");
    else navigate("/");
  };

  const isActive = (path) => location.pathname === path || location.pathname.startsWith(path + "/");
  const close    = () => setMenuOpen(false);

  /* Doctor links */
  const doctorLinks = [
    { to: "/homedoctor",       icon: Icons.dashboard, label: "Case Manager" },
    { to: "/how-to-use-doctor", icon: Icons.guide,    label: "Clinical Guide" },
  ];

  /* Patient links */
  const patientLinks = [
    { to: "/patient-dashboard", icon: Icons.dashboard, label: "My Dashboard" },
    { to: "/detect",            icon: Icons.scan,      label: "Submit X-Ray" },
    { to: "/how-to-use-patient",icon: Icons.guide,     label: "How to Use" },
  ];

  /* Public links */
  const publicLinks = [
    { to: "/", icon: Icons.home, label: "Home" },
  ];

  const roleLinks = userRole === "doctor" ? doctorLinks : userRole === "patient" ? patientLinks : publicLinks;

  return (
    <>
      <style>{`
        @keyframes slideDown { from { opacity:0; transform: translateY(-12px); } to { opacity:1; transform:none; } }
        @keyframes fadeIn    { from { opacity:0; } to { opacity:1; } }
      `}</style>

      <nav
        ref={menuRef}
        style={{
          position: "sticky",
          top: 0,
          zIndex: 1000,
          width: "100%",
          background: scrolled
            ? "rgba(2,8,24,0.96)"
            : "rgba(2,8,24,0.82)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          borderBottom: scrolled
            ? "1px solid rgba(56,189,248,0.18)"
            : "1px solid rgba(255,255,255,0.05)",
          boxShadow: scrolled ? "0 8px 40px rgba(0,0,0,0.5)" : "none",
          transition: "all 0.3s ease",
          fontFamily: "'Inter', -apple-system, sans-serif",
        }}
      >
        <div style={{
          maxWidth: 1400,
          margin: "0 auto",
          padding: "0 2rem",
          height: 70,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1rem",
        }}>

          {/* ── Logo ─────────────────────────────────────────── */}
          <a href="/" onClick={handleLogoClick} style={{ textDecoration: "none", flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div style={{ position: "relative" }}>
                <img
                  src="/logo.png"
                  alt="ZenithDx"
                  style={{ width: 46, height: 46, objectFit: "contain", display: "block" }}
                />
                {/* Live indicator dot */}
                <span style={{
                  position: "absolute", bottom: 0, right: 0,
                  width: 10, height: 10, borderRadius: "50%",
                  background: "#38bdf8",
                  border: "2px solid rgba(2,8,24,0.95)",
                  boxShadow: "0 0 8px #38bdf8",
                  animation: "pulse-dot 2.2s ease-in-out infinite",
                }} />
              </div>
              <div>
                <div style={{
                  fontSize: "1.22rem",
                  fontWeight: 900,
                  letterSpacing: "-0.03em",
                  background: "linear-gradient(135deg, #ffffff 30%, #7dd3fc 90%)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  lineHeight: 1.1,
                }}>
                  ZenithDx
                </div>
                <div style={{
                  fontSize: "0.62rem",
                  fontWeight: 600,
                  color: "#475569",
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                  lineHeight: 1.2,
                }}>
                  Agentic Clinical AI
                </div>
              </div>
            </div>
          </a>

          {/* ── Desktop Links ─────────────────────────────────── */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "0.25rem",
            flex: 1,
            justifyContent: "center",
          }}>
            {/* Separator before role links */}
            {userRole && (
              <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.08)", margin: "0 0.5rem" }} />
            )}

            {roleLinks.map(link => (
              <NavLink key={link.to} to={link.to} icon={link.icon} active={isActive(link.to)}>
                {link.label}
              </NavLink>
            ))}

            {/* About — always visible */}
            <NavLink to="/about-us" icon={Icons.about} active={isActive("/about-us")}>
              About
            </NavLink>
          </div>

          {/* ── Right actions ─────────────────────────────────── */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexShrink: 0 }}>
            {userRole ? (
              <>
                {/* Role badge */}
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.4rem",
                  padding: "0.35rem 0.85rem",
                  borderRadius: 100,
                  background: userRole === "doctor" ? "rgba(56,189,248,0.1)" : "rgba(129,140,248,0.1)",
                  border: userRole === "doctor" ? "1px solid rgba(56,189,248,0.25)" : "1px solid rgba(129,140,248,0.25)",
                  fontSize: "0.75rem",
                  fontWeight: 700,
                  color: userRole === "doctor" ? "#38bdf8" : "#818cf8",
                  letterSpacing: "0.04em",
                  textTransform: "uppercase",
                }}>
                  <span style={{ width: 6, height: 6, borderRadius: "50%", background: userRole === "doctor" ? "#38bdf8" : "#818cf8", display: "inline-block" }} />
                  {userRole === "doctor" ? "Clinician" : "Patient"}
                </div>

                <LogoutBtn onClick={handleLogout} />
              </>
            ) : (
              <Link
                to="/auth"
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.45rem",
                  padding: "0.5rem 1.2rem",
                  borderRadius: 10,
                  fontSize: "0.88rem",
                  fontWeight: 700,
                  textDecoration: "none",
                  background: "linear-gradient(135deg, #0ea5e9, #6366f1)",
                  color: "#fff",
                  boxShadow: "0 4px 16px rgba(14,165,233,0.28)",
                  transition: "all 0.2s ease",
                  letterSpacing: "0.01em",
                }}
                onMouseEnter={e => { e.currentTarget.style.opacity = "0.88"; e.currentTarget.style.transform = "translateY(-1px)"; }}
                onMouseLeave={e => { e.currentTarget.style.opacity = "1"; e.currentTarget.style.transform = "none"; }}
              >
                Sign In →
              </Link>
            )}

            {/* Hamburger (mobile) */}
            <button
              onClick={() => setMenuOpen(o => !o)}
              style={{
                display: "none",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "#94a3b8",
                borderRadius: 10,
                width: 40, height: 40,
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                transition: "all 0.2s",
                flexShrink: 0,
              }}
              id="hamburger-btn"
            >
              <SVG d={menuOpen ? Icons.close : Icons.menu} size={18} />
            </button>
          </div>
        </div>

        {/* ── Mobile Dropdown ──────────────────────────────────── */}
        {menuOpen && (
          <div style={{
            animation: "slideDown 0.22s ease",
            background: "rgba(4,10,28,0.98)",
            borderTop: "1px solid rgba(56,189,248,0.12)",
            padding: "1.2rem 1.5rem 1.5rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.5rem",
          }}>
            {roleLinks.map(link => (
              <Link
                key={link.to}
                to={link.to}
                onClick={close}
                style={{
                  display: "flex", alignItems: "center", gap: "0.65rem",
                  padding: "0.75rem 1rem",
                  borderRadius: 12,
                  fontSize: "0.92rem", fontWeight: 600,
                  color: isActive(link.to) ? "#f0f6ff" : "#94a3b8",
                  background: isActive(link.to) ? "rgba(56,189,248,0.1)" : "transparent",
                  textDecoration: "none",
                  transition: "all 0.18s",
                  border: isActive(link.to) ? "1px solid rgba(56,189,248,0.2)" : "1px solid transparent",
                }}
              >
                <SVG d={link.icon} size={16} />
                {link.label}
              </Link>
            ))}
            <Link
              to="/about-us"
              onClick={close}
              style={{
                display: "flex", alignItems: "center", gap: "0.65rem",
                padding: "0.75rem 1rem", borderRadius: 12, fontSize: "0.92rem",
                fontWeight: 600, color: "#94a3b8", textDecoration: "none",
                border: "1px solid transparent",
              }}
            >
              <SVG d={Icons.about} size={16} />
              About ZenithDx
            </Link>

            <div style={{ height: 1, background: "rgba(56,189,248,0.1)", margin: "0.5rem 0" }} />

            {userRole ? (
              <LogoutBtn onClick={handleLogout} mobile />
            ) : (
              <Link
                to="/auth"
                onClick={close}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem",
                  padding: "0.75rem", borderRadius: 12, fontSize: "0.92rem", fontWeight: 700,
                  background: "linear-gradient(135deg,#0ea5e9,#6366f1)",
                  color: "#fff", textDecoration: "none",
                }}
              >
                Sign In →
              </Link>
            )}
          </div>
        )}
      </nav>

      <style>{`
        @keyframes pulse-dot {
          0%,100% { opacity:1; box-shadow: 0 0 8px #38bdf8; }
          50% { opacity:0.5; box-shadow: 0 0 4px #38bdf8; }
        }
        @media (max-width: 768px) {
          #hamburger-btn { display: flex !important; }
        }
      `}</style>
    </>
  );
};

export default Navbar;
