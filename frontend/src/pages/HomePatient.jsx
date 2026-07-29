import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import ladydoc from "../assets/ladydoc.png";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const BASE = "http://localhost:8000";

/* ── Status colour map ─────────────────────────────────────────── */
const STATUS_COLORS = {
  approved: { bg: "rgba(34,197,94,0.14)",  text: "#4ade80", border: "rgba(34,197,94,0.35)",  dot: "#4ade80"  },
  rejected: { bg: "rgba(239,68,68,0.14)",  text: "#f87171", border: "rgba(239,68,68,0.35)",  dot: "#f87171"  },
  edited:   { bg: "rgba(139,92,246,0.14)", text: "#a78bfa", border: "rgba(139,92,246,0.35)", dot: "#a78bfa"  },
  pending:  { bg: "rgba(245,158,11,0.14)", text: "#fcd34d", border: "rgba(245,158,11,0.35)", dot: "#fcd34d"  },
};
const statusColor = (s = "") => STATUS_COLORS[(s||"pending").toLowerCase()] || STATUS_COLORS.pending;

export default function HomePatient() {
  const [reports, setReports]       = useState([]);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [dateFilter, setDateFilter] = useState("");
  const navigate = useNavigate();

  /* ── Fetch patient reports ─────────────────────────────────────── */
  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const token = localStorage.getItem("token");
      const res   = await fetch(`${BASE}/patient/reports`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const data  = await res.json();
      setReports(Array.isArray(data.data) ? data.data : []);
    } catch (e) {
      setError(e.message || "Failed to load reports");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchReports(); }, [fetchReports]);

  /* ── Derived stats ─────────────────────────────────────────────── */
  const pendingCount  = reports.filter(r => (r.status||"").toLowerCase() === "pending").length;
  const approvedCount = reports.filter(r => (r.status||"").toLowerCase() === "approved").length;
  const rejectedCount = reports.filter(r => (r.status||"").toLowerCase() === "rejected").length;
  const totalCount    = reports.length;

  /* ── Filters ───────────────────────────────────────────────────── */
  const filtered = reports.filter(r => {
    const matchStatus = statusFilter === "All" || (r.status||"").toLowerCase() === statusFilter.toLowerCase();
    const matchDate   = !dateFilter || (r.submission_date &&
      new Date(r.submission_date).toLocaleDateString() === new Date(dateFilter).toLocaleDateString());
    return matchStatus && matchDate;
  });

  /* ── Chart ─────────────────────────────────────────────────────── */
  const chartData = {
    labels: ["Pending", "Approved", "Rejected"],
    datasets: [{
      label: "Reports",
      data: [pendingCount, approvedCount, rejectedCount],
      backgroundColor: ["rgba(245,158,11,0.75)", "rgba(52,211,153,0.75)", "rgba(248,113,113,0.75)"],
      borderColor: ["#fbbf24", "#34d399", "#f87171"],
      borderWidth: 2,
      borderRadius: 10,
    }],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(2,8,24,0.92)",
        titleColor: "#818cf8",
        bodyColor: "#f8fafc",
        borderColor: "rgba(129,140,248,0.28)",
        borderWidth: 1,
        padding: 12,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#94a3b8", font: { family: "Inter", size: 12, weight: "600" } },
      },
      y: {
        beginAtZero: true,
        ticks: { stepSize: 1, color: "#64748b", font: { family: "Inter", size: 11 } },
        grid: { color: "rgba(129,140,248,0.07)" },
      },
    },
  };

  /* ── Handle view report ─────────────────────────────────────────── */
  const handleView = (report) => {
    if ((report.status||"").toLowerCase() !== "approved") {
      return; // button is disabled, but safety guard
    }
    navigate(`/patient/reports/${report.id || report.report_id}`);
  };

  /* ── Render ──────────────────────────────────────────────────────── */
  return (
    <div style={{
      background: "#020818",
      minHeight: "100vh",
      color: "#fff",
      fontFamily: "'Inter', -apple-system, sans-serif",
      display: "flex",
      flexDirection: "column",
      overflowX: "hidden",
      position: "relative",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; background: #0a1122; }
        ::-webkit-scrollbar-thumb { background: rgba(129,140,248,0.25); border-radius: 6px; }

        .stat-card { transition: transform 0.22s, box-shadow 0.22s; }
        .stat-card:hover { transform: translateY(-4px); box-shadow: 0 20px 50px rgba(0,0,0,0.5); }

        .row-tr { transition: background 0.18s; }
        .row-tr:hover { background: rgba(129,140,248,0.05) !important; }

        .view-btn {
          background: linear-gradient(135deg, #818cf8, #6366f1);
          border: none; color: #fff; border-radius: 8px;
          padding: 0.42rem 1.1rem; font-size: 0.8rem; font-weight: 700;
          cursor: pointer; transition: all 0.18s;
          box-shadow: 0 2px 10px rgba(99,102,241,0.3);
          white-space: nowrap;
        }
        .view-btn:hover { opacity: 0.88; transform: translateY(-1px); }

        .view-btn-locked {
          background: rgba(71,85,105,0.18);
          border: 1px solid rgba(71,85,105,0.25);
          color: #475569; border-radius: 8px;
          padding: 0.42rem 1.1rem; font-size: 0.8rem; font-weight: 600;
          cursor: not-allowed; white-space: nowrap;
        }

        .refresh-btn {
          background: rgba(15,23,42,0.9); border: 1px solid rgba(129,140,248,0.22);
          color: #94a3b8; border-radius: 10px; padding: 0.65rem 1.25rem;
          font-size: 0.88rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .refresh-btn:hover { border-color: rgba(129,140,248,0.5); color: #818cf8; }

        .filter-select, .filter-input {
          background: rgba(2,8,24,0.85); border: 1px solid rgba(129,140,248,0.22);
          color: #f8fafc; border-radius: 10px; padding: 0.55rem 0.9rem;
          font-size: 0.85rem; font-weight: 500; outline: none;
          transition: border-color 0.2s;
        }
        .filter-select:focus, .filter-input:focus { border-color: rgba(129,140,248,0.55); }
        .filter-select option { background: #0a1530; }

        .reset-btn {
          background: rgba(129,140,248,0.1); border: 1px solid rgba(129,140,248,0.28);
          color: #818cf8; border-radius: 10px; padding: 0.55rem 1.1rem;
          font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .reset-btn:hover { background: rgba(129,140,248,0.18); }

        .submit-btn {
          background: linear-gradient(135deg, #818cf8, #6366f1);
          border: none; color: #fff; border-radius: 12px;
          padding: 0.72rem 1.8rem; font-size: 0.92rem; font-weight: 700;
          cursor: pointer; transition: all 0.22s;
          box-shadow: 0 4px 18px rgba(99,102,241,0.35);
          display: inline-flex; align-items: center; gap: 0.5rem;
          letter-spacing: 0.01em;
        }
        .submit-btn:hover { opacity: 0.88; transform: translateY(-2px); box-shadow: 0 6px 28px rgba(99,102,241,0.45); }

        @keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>

      {/* Ambient Glows */}
      <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "-8%", right: "5%", width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(129,140,248,0.07) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "5%", left: "3%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.05) 0%, transparent 65%)" }} />
      </div>

      <div style={{ position: "relative", zIndex: 10 }}><Navbar /></div>

      <main style={{ flex: 1, padding: "2.5rem 2.5rem 5rem", maxWidth: 1400, margin: "0 auto", width: "100%", position: "relative", zIndex: 1 }}>

        {/* ── Page Header ──────────────────────────────────────────── */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "2rem", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", background: "rgba(129,140,248,0.09)", border: "1px solid rgba(129,140,248,0.22)", borderRadius: 100, padding: "0.32rem 1rem", fontSize: "0.72rem", color: "#818cf8", letterSpacing: "0.09em", fontWeight: 700, marginBottom: "0.55rem" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#818cf8", display: "inline-block", boxShadow: "0 0 8px #818cf8", animation: "pulse-dot 2s ease-in-out infinite" }} />
              PATIENT HEALTH PORTAL
            </div>
            <h1 style={{ fontSize: "clamp(1.9rem, 3.5vw, 2.7rem)", fontWeight: 900, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
              <span style={{ background: "linear-gradient(135deg, #ffffff 40%, #a5b4fc 80%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                My Health Dashboard
              </span>
            </h1>
            <p style={{ color: "#64748b", fontSize: "0.9rem", marginTop: "0.4rem" }}>
              Track your X-ray submissions · View AI diagnostic reports · Monitor clinical approvals
            </p>
          </div>

          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button className="refresh-btn" onClick={fetchReports}>🔄 Refresh</button>
            <button className="submit-btn" onClick={() => navigate("/detect")}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 5v14M5 12h14" />
              </svg>
              Submit X-Ray
            </button>
          </div>
        </div>

        {/* ── Stat Cards ────────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px,1fr))", gap: "1.2rem", marginBottom: "2.2rem" }}>
          {[
            { label: "Total Submissions",  value: totalCount,    icon: "🗂️", color: "#818cf8", glow: "rgba(129,140,248,0.2)" },
            { label: "Pending Review",     value: pendingCount,  icon: "⏳", color: "#fcd34d", glow: "rgba(245,158,11,0.2)"  },
            { label: "Doctor Approved",    value: approvedCount, icon: "✅", color: "#4ade80", glow: "rgba(34,197,94,0.2)"   },
            { label: "Rejected Cases",     value: rejectedCount, icon: "🚫", color: "#f87171", glow: "rgba(239,68,68,0.2)"   },
          ].map((s, i) => (
            <div key={i} className="stat-card" style={{
              background: "rgba(10,17,34,0.78)",
              border: "1px solid rgba(129,140,248,0.12)",
              borderRadius: 20, padding: "1.5rem 1.6rem",
              backdropFilter: "blur(16px)",
              boxShadow: "0 8px 30px rgba(0,0,0,0.3)",
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <div>
                <div style={{ fontSize: "0.78rem", color: "#64748b", fontWeight: 600, marginBottom: "0.35rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>{s.label}</div>
                <div style={{ fontSize: "2.1rem", fontWeight: 900, color: s.color, letterSpacing: "-0.03em", lineHeight: 1 }}>{s.value}</div>
              </div>
              <div style={{
                width: 50, height: 50, borderRadius: 14,
                background: s.glow, border: `1px solid ${s.glow}`,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.5rem",
                boxShadow: `0 0 20px ${s.glow}`,
              }}>{s.icon}</div>
            </div>
          ))}
        </div>

        {/* ── Hero Banner + Chart ───────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.8rem", marginBottom: "2.5rem" }}>

          {/* Patient Hero Card */}
          <div style={{
            background: "linear-gradient(135deg, rgba(14,22,50,0.95) 0%, rgba(22,18,68,0.85) 60%, rgba(10,17,34,0.9) 100%)",
            border: "1px solid rgba(129,140,248,0.22)",
            borderRadius: 28, padding: "2.5rem 2.5rem 0",
            position: "relative", overflow: "hidden",
            boxShadow: "0 25px 60px rgba(0,0,0,0.5)",
            backdropFilter: "blur(20px)",
            minHeight: 300,
            display: "flex", flexDirection: "column", justifyContent: "flex-start",
          }}>
            {/* Corner glow */}
            <div style={{ position: "absolute", top: -60, left: -60, width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, rgba(129,140,248,0.12) 0%, transparent 70%)", pointerEvents: "none" }} />

            <div style={{ position: "relative", zIndex: 2, maxWidth: "55%" }}>
              <div style={{ fontSize: "0.73rem", color: "#818cf8", fontWeight: 700, letterSpacing: "0.1em", marginBottom: "0.5rem", textTransform: "uppercase" }}>
                Patient Portal
              </div>
              <h2 style={{ fontSize: "1.9rem", fontWeight: 900, color: "#f8fafc", lineHeight: 1.2, marginBottom: "0.75rem", letterSpacing: "-0.02em" }}>
                AI-Assisted<br />Chest Diagnostics
              </h2>
              <p style={{ color: "#94a3b8", fontSize: "0.88rem", lineHeight: 1.65, marginBottom: "1.5rem" }}>
                Submit your chest X-ray for AI analysis, receive Grad-CAM diagnostic reports, and track doctor approvals in real time.
              </p>
              <div style={{ display: "flex", gap: "0.7rem", flexWrap: "wrap" }}>
                <span style={{ background: "rgba(129,140,248,0.13)", border: "1px solid rgba(129,140,248,0.3)", borderRadius: 10, padding: "0.45rem 0.9rem", fontSize: "0.8rem", color: "#818cf8", fontWeight: 700 }}>
                  {approvedCount} Approved
                </span>
                <span style={{ background: "rgba(245,158,11,0.13)", border: "1px solid rgba(245,158,11,0.3)", borderRadius: 10, padding: "0.45rem 0.9rem", fontSize: "0.8rem", color: "#fcd34d", fontWeight: 700 }}>
                  {pendingCount} Pending
                </span>
              </div>
            </div>

            {/* Doctor / Medical illustration */}
            <img
              src={ladydoc}
              alt="Medical AI"
              style={{
                position: "absolute",
                right: 0,
                bottom: 0,
                height: "110%",
                width: "auto",
                objectFit: "contain",
                objectPosition: "bottom right",
                pointerEvents: "none",
                zIndex: 1,
                filter: "drop-shadow(0 0 28px rgba(129,140,248,0.18))",
              }}
            />
          </div>

          {/* Chart Card */}
          <div style={{
            background: "rgba(10,17,34,0.82)",
            border: "1px solid rgba(129,140,248,0.2)",
            borderRadius: 28, padding: "2rem 2.2rem",
            backdropFilter: "blur(16px)",
            display: "flex", flexDirection: "column",
            boxShadow: "0 20px 50px rgba(0,0,0,0.4)",
          }}>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 800, color: "#f8fafc", marginBottom: "0.2rem" }}>Report Status Overview</h3>
            <p style={{ color: "#64748b", fontSize: "0.84rem", marginBottom: "1.4rem" }}>Your submission status breakdown</p>
            <div style={{ flex: 1, minHeight: 200, position: "relative" }}>
              {totalCount === 0 ? (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "0.75rem", color: "#475569" }}>
                  <span style={{ fontSize: "2.5rem" }}>📊</span>
                  <span style={{ fontSize: "0.9rem" }}>No submissions yet</span>
                  <button className="submit-btn" onClick={() => navigate("/detect")} style={{ padding: "0.55rem 1.2rem", fontSize: "0.82rem" }}>
                    Submit First X-Ray →
                  </button>
                </div>
              ) : (
                <Bar data={chartData} options={chartOptions} />
              )}
            </div>
          </div>
        </div>

        {/* ── Reports Table ─────────────────────────────────────────── */}
        <div style={{
          background: "rgba(8,14,30,0.88)",
          border: "1px solid rgba(129,140,248,0.18)",
          borderRadius: 28, overflow: "hidden",
          backdropFilter: "blur(24px)",
          boxShadow: "0 28px 70px rgba(0,0,0,0.55)",
        }}>

          {/* Table Header Bar */}
          <div style={{
            padding: "1.6rem 2rem",
            borderBottom: "1px solid rgba(129,140,248,0.1)",
            background: "rgba(10,18,40,0.7)",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: "1rem",
          }}>
            <div>
              <h3 style={{ fontSize: "1.35rem", fontWeight: 800, color: "#f8fafc", letterSpacing: "-0.02em" }}>
                My Diagnostic Reports
              </h3>
              <div style={{ fontSize: "0.82rem", color: "#64748b", marginTop: "0.2rem" }}>
                Showing <span style={{ color: "#818cf8", fontWeight: 700 }}>{filtered.length}</span> of <span style={{ color: "#94a3b8" }}>{totalCount}</span> submissions
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", flexWrap: "wrap" }}>
              <select className="filter-select" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
                <option value="All">All Statuses</option>
                <option value="Pending">Pending</option>
                <option value="Approved">Approved</option>
                <option value="Rejected">Rejected</option>
              </select>
              <input type="date" className="filter-input" value={dateFilter} onChange={e => setDateFilter(e.target.value)} />
              <button className="reset-btn" onClick={() => { setStatusFilter("All"); setDateFilter(""); }}>Reset</button>
            </div>
          </div>

          {/* Table */}
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
              <thead>
                <tr style={{ background: "rgba(2,8,24,0.9)", borderBottom: "1px solid rgba(129,140,248,0.1)" }}>
                  {["Report ID", "Scan Type", "Submission Date", "Doctor Message", "Status", "Actions"].map((h, i) => (
                    <th key={i} style={{
                      padding: "1rem 1.5rem",
                      fontSize: "0.74rem", fontWeight: 700,
                      color: "#818cf8", letterSpacing: "0.09em", textTransform: "uppercase",
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={6} style={EMPTY}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem" }}>
                      <span style={{ display: "inline-block", width: 18, height: 18, border: "2.5px solid rgba(129,140,248,0.3)", borderTopColor: "#818cf8", borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
                      Loading your diagnostic reports…
                    </div>
                  </td></tr>
                ) : error ? (
                  <tr><td colSpan={6} style={{ ...EMPTY, color: "#fca5a5" }}>⚠️ {error}</td></tr>
                ) : filtered.length === 0 ? (
                  <tr><td colSpan={6} style={EMPTY}>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
                      <span style={{ fontSize: "2.5rem" }}>🩺</span>
                      <span>No reports match the selected filter.</span>
                      {totalCount === 0 && (
                        <button className="submit-btn" onClick={() => navigate("/detect")} style={{ padding: "0.55rem 1.4rem", fontSize: "0.85rem" }}>
                          Submit Your First X-Ray →
                        </button>
                      )}
                    </div>
                  </td></tr>
                ) : (
                  filtered.map((r, idx) => {
                    const rid = r.id || r.report_id;
                    const sc  = statusColor(r.status);
                    const isApproved = (r.status||"").toLowerCase() === "approved";
                    return (
                      <tr key={rid || idx} className="row-tr" style={{
                        borderBottom: "1px solid rgba(129,140,248,0.07)",
                        background: idx % 2 === 0 ? "transparent" : "rgba(14,22,44,0.35)",
                      }}>
                        {/* Report ID */}
                        <td style={{ ...TD, fontFamily: "monospace", color: "#475569", fontSize: "0.78rem" }}>
                          #{String(rid).slice(0, 8)}…
                        </td>

                        {/* Scan Type */}
                        <td style={{ ...TD, color: "#cbd5e1", fontWeight: 600 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <span style={{ fontSize: "1rem" }}>🫁</span>
                            {r.report_type || r.data_type || "Chest X-Ray"}
                          </div>
                        </td>

                        {/* Date */}
                        <td style={{ ...TD, color: "#94a3b8", fontSize: "0.84rem", whiteSpace: "nowrap" }}>
                          {r.submission_date
                            ? new Date(r.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
                            : "—"}
                        </td>

                        {/* Doctor Message */}
                        <td style={{ ...TD, color: "#64748b", fontSize: "0.84rem", maxWidth: 200 }}>
                          {r.doctor_message ? (
                            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.4rem" }}>
                              <span style={{ color: "#818cf8", flexShrink: 0 }}>💬</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#94a3b8" }}>
                                {r.doctor_message}
                              </span>
                            </div>
                          ) : (
                            <span style={{ color: "#334155", fontStyle: "italic", fontSize: "0.8rem" }}>No message yet</span>
                          )}
                        </td>

                        {/* Status Pill */}
                        <td style={{ ...TD }}>
                          <span style={{
                            display: "inline-flex", alignItems: "center", gap: "0.4rem",
                            padding: "0.3rem 0.85rem", borderRadius: 100,
                            fontSize: "0.77rem", fontWeight: 700, letterSpacing: "0.03em",
                            background: sc.bg, color: sc.text, border: `1px solid ${sc.border}`,
                            whiteSpace: "nowrap",
                          }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: sc.dot, display: "inline-block", boxShadow: `0 0 6px ${sc.dot}` }} />
                            {r.status || "Pending"}
                          </span>
                        </td>

                        {/* Actions */}
                        <td style={{ ...TD, textAlign: "center" }}>
                          {isApproved ? (
                            <button className="view-btn" onClick={() => handleView(r)}>
                              View Report →
                            </button>
                          ) : (
                            <span className="view-btn-locked">
                              {(r.status||"pending").toLowerCase() === "rejected" ? "🚫 Rejected" : "⏳ Pending"}
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* ── Quick Actions Footer ─────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px,1fr))", gap: "1.2rem", marginTop: "2rem" }}>
          {[
            { label: "Submit New X-Ray", desc: "Upload a chest radiograph for AI analysis", icon: "📤", to: "/detect", primary: true },
            { label: "Clinical Guide",   desc: "Learn how to interpret your AI reports",  icon: "📖", to: "/how-to-use-patient", primary: false },
          ].map(action => (
            <div
              key={action.label}
              onClick={() => navigate(action.to)}
              style={{
                background: action.primary
                  ? "linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(129,140,248,0.1) 100%)"
                  : "rgba(10,17,34,0.7)",
                border: `1px solid ${action.primary ? "rgba(129,140,248,0.3)" : "rgba(129,140,248,0.12)"}`,
                borderRadius: 20, padding: "1.4rem 1.6rem",
                cursor: "pointer",
                display: "flex", alignItems: "center", gap: "1rem",
                transition: "all 0.22s",
                backdropFilter: "blur(12px)",
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 16px 40px rgba(0,0,0,0.4)"; }}
              onMouseLeave={e => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "none"; }}
            >
              <div style={{
                width: 46, height: 46, borderRadius: 13,
                background: "rgba(129,140,248,0.15)", border: "1px solid rgba(129,140,248,0.25)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: "1.4rem", flexShrink: 0,
              }}>
                {action.icon}
              </div>
              <div>
                <div style={{ fontWeight: 800, color: "#f0f6ff", fontSize: "0.95rem", marginBottom: "0.2rem" }}>{action.label}</div>
                <div style={{ color: "#64748b", fontSize: "0.82rem" }}>{action.desc}</div>
              </div>
              <div style={{ marginLeft: "auto", color: "#475569", fontSize: "1.1rem" }}>→</div>
            </div>
          ))}
        </div>

      </main>
    </div>
  );
}

const TD = {
  padding: "1.05rem 1.5rem",
  fontSize: "0.9rem",
};

const EMPTY = {
  textAlign: "center",
  padding: "4rem",
  color: "#64748b",
  fontSize: "0.95rem",
};
