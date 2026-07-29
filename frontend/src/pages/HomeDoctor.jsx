import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import homedoctor from "../assets/homedoctor.png";
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

/* ─── Status colour map ─────────────────────────────────────────── */
const STATUS_COLORS = {
  approved: { bg: "rgba(34,197,94,0.14)", text: "#4ade80", border: "rgba(34,197,94,0.35)", dot: "#4ade80" },
  rejected: { bg: "rgba(239,68,68,0.14)",  text: "#f87171", border: "rgba(239,68,68,0.35)",  dot: "#f87171" },
  edited:   { bg: "rgba(139,92,246,0.14)", text: "#a78bfa", border: "rgba(139,92,246,0.35)", dot: "#a78bfa" },
  pending:  { bg: "rgba(245,158,11,0.14)", text: "#fcd34d", border: "rgba(245,158,11,0.35)", dot: "#fcd34d" },
};
const statusColor = (s = "pending") => STATUS_COLORS[(s || "pending").toLowerCase()] || STATUS_COLORS.pending;

export default function HomeDoctor() {
  const [reports, setReports]       = useState([]);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [dateFilter, setDateFilter] = useState("");
  const [actionBusy, setActionBusy] = useState({});
  const navigate = useNavigate();

  /* ─── Fetch reports ─────────────────────────────────────────────── */
  const fetchReports = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const token = localStorage.getItem("token");
      const res   = await fetch(`${BASE}/doctor/reports`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error(`Server ${res.status}`);
      const { data } = await res.json();
      setReports(data || []);
    } catch (e) {
      setError(e.message || "Failed to load reports");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchReports(); }, [fetchReports]);

  /* ─── Quick approve / reject ────────────────────────────────────── */
  const quickAction = async (reportId, action) => {
    setActionBusy(p => ({ ...p, [reportId]: action }));
    try {
      const token = localStorage.getItem("token");
      await fetch(`${BASE}/doctor/reports/${reportId}/${action}`, {
        method: "PATCH",
        headers: { Authorization: `Bearer ${token}` },
      });
      setReports(prev =>
        prev.map(r =>
          String(r.report_id) === String(reportId)
            ? { ...r, status: action === "approve" ? "Approved" : "Rejected" }
            : r
        )
      );
    } catch {
      alert(`Failed to ${action} report.`);
    } finally {
      setActionBusy(p => { const n = { ...p }; delete n[reportId]; return n; });
    }
  };

  /* ─── Derived stats ─────────────────────────────────────────────── */
  const pendingCount  = reports.filter(r => (r.status||"").toLowerCase() === "pending").length;
  const approvedCount = reports.filter(r => (r.status||"").toLowerCase() === "approved").length;
  const rejectedCount = reports.filter(r => (r.status||"").toLowerCase() === "rejected").length;

  const patientMap = {};
  reports.forEach(r => {
    if (r.patient_name) patientMap[r.patient_name] = (patientMap[r.patient_name] || 0) + 1;
  });
  const newCount = Object.values(patientMap).filter(c => c === 1).length;
  const returningCount = Object.values(patientMap).filter(c => c > 1).length;

  /* ─── Filters ───────────────────────────────────────────────────── */
  const filtered = reports.filter(r => {
    const matchStatus = statusFilter === "All" || (r.status||"").toLowerCase() === statusFilter.toLowerCase();
    const matchDate   = !dateFilter || (r.submission_date &&
      new Date(r.submission_date).toLocaleDateString() === new Date(dateFilter).toLocaleDateString());
    return matchStatus && matchDate;
  });

  /* ─── Chart ─────────────────────────────────────────────────────── */
  const chartData = {
    labels: ["Pending", "Approved", "Rejected"],
    datasets: [{
      label: "Cases",
      data: [pendingCount, approvedCount, rejectedCount],
      backgroundColor: ["rgba(251,191,36,0.75)", "rgba(52,211,153,0.75)", "rgba(248,113,113,0.75)"],
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
        titleColor: "#38bdf8",
        bodyColor: "#f8fafc",
        borderColor: "rgba(56,189,248,0.28)",
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
        grid: { color: "rgba(56,189,248,0.07)" },
      },
    },
  };

  /* ─── Render ─────────────────────────────────────────────────────── */
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
        ::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.25); border-radius: 6px; }

        .stat-card { transition: transform 0.22s, box-shadow 0.22s; }
        .stat-card:hover { transform: translateY(-4px); box-shadow: 0 20px 50px rgba(0,0,0,0.5); }

        .row-tr { transition: background 0.18s; }
        .row-tr:hover { background: rgba(56,189,248,0.05) !important; }

        .review-btn {
          background: linear-gradient(135deg,#0ea5e9,#6366f1);
          border: none; color: #fff; border-radius: 8px;
          padding: 0.42rem 1rem; font-size: 0.8rem; font-weight: 700;
          cursor: pointer; transition: all 0.18s;
          box-shadow: 0 2px 10px rgba(14,165,233,0.28);
          white-space: nowrap;
        }
        .review-btn:hover { opacity: 0.88; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(14,165,233,0.4); }

        .approve-btn {
          background: rgba(34,197,94,0.13); border: 1px solid rgba(34,197,94,0.35);
          color: #4ade80; border-radius: 8px; padding: 0.38rem 0.8rem;
          font-size: 0.78rem; font-weight: 700; cursor: pointer; transition: all 0.18s;
          white-space: nowrap;
        }
        .approve-btn:hover { background: rgba(34,197,94,0.22); }

        .reject-btn {
          background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3);
          color: #f87171; border-radius: 8px; padding: 0.38rem 0.8rem;
          font-size: 0.78rem; font-weight: 700; cursor: pointer; transition: all 0.18s;
          white-space: nowrap;
        }
        .reject-btn:hover { background: rgba(239,68,68,0.22); }

        .refresh-btn {
          background: rgba(15,23,42,0.9); border: 1px solid rgba(56,189,248,0.22);
          color: #94a3b8; border-radius: 10px; padding: 0.65rem 1.25rem;
          font-size: 0.88rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .refresh-btn:hover { border-color: rgba(56,189,248,0.5); color: #38bdf8; }

        .filter-select, .filter-input {
          background: rgba(2,8,24,0.85); border: 1px solid rgba(56,189,248,0.22);
          color: #f8fafc; border-radius: 10px; padding: 0.55rem 0.9rem;
          font-size: 0.85rem; font-weight: 500; outline: none;
          transition: border-color 0.2s;
        }
        .filter-select:focus, .filter-input:focus { border-color: rgba(56,189,248,0.55); }
        .filter-select option { background: #0a1530; }

        .reset-btn {
          background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.28);
          color: #38bdf8; border-radius: 10px; padding: 0.55rem 1.1rem;
          font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .reset-btn:hover { background: rgba(56,189,248,0.18); }

        @keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
      `}</style>

      {/* Ambient Glows */}
      <div style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "hidden", zIndex: 0 }}>
        <div style={{ position: "absolute", top: "-8%", left: "8%",  width: 700, height: 700, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 65%)" }} />
        <div style={{ position: "absolute", bottom: "5%", right: "3%", width: 600, height: 600, borderRadius: "50%", background: "radial-gradient(circle, rgba(99,102,241,0.07) 0%, transparent 65%)" }} />
      </div>

      {/* Navbar */}
      <div style={{ position: "relative", zIndex: 10 }}><Navbar /></div>

      <main style={{ flex: 1, padding: "2.5rem 2.5rem 5rem", maxWidth: 1400, margin: "0 auto", width: "100%", position: "relative", zIndex: 1 }}>

        {/* ─── Page Header ──────────────────────────────────────────── */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "2rem", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.5rem", background: "rgba(56,189,248,0.09)", border: "1px solid rgba(56,189,248,0.22)", borderRadius: 100, padding: "0.32rem 1rem", fontSize: "0.72rem", color: "#38bdf8", letterSpacing: "0.09em", fontWeight: 700, marginBottom: "0.55rem" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#38bdf8", display: "inline-block", boxShadow: "0 0 8px #38bdf8", animation: "pulse-dot 2s ease-in-out infinite" }} />
              CLINICIAN DECISION SUPPORT SYSTEM
            </div>
            <h1 style={{ fontSize: "clamp(1.9rem, 3.5vw, 2.7rem)", fontWeight: 900, letterSpacing: "-0.03em", lineHeight: 1.1 }}>
              <span style={{ background: "linear-gradient(135deg,#ffffff 40%,#7dd3fc 80%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                Doctor Review Portal
              </span>
            </h1>
            <p style={{ color: "#64748b", fontSize: "0.9rem", marginTop: "0.4rem" }}>
              AI-assisted case management · Grad-CAM verification · Clinical approvals
            </p>
          </div>
          <button className="refresh-btn" onClick={fetchReports}>🔄 Refresh Queue</button>
        </div>

        {/* ─── Stat Cards ───────────────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px,1fr))", gap: "1.2rem", marginBottom: "2.2rem" }}>
          {[
            { label: "Total Reports",    value: reports.length,  icon: "📋", color: "#38bdf8", glow: "rgba(56,189,248,0.2)" },
            { label: "Pending Review",   value: pendingCount,    icon: "⏳", color: "#fcd34d", glow: "rgba(245,158,11,0.2)" },
            { label: "Approved Cases",   value: approvedCount,   icon: "✅", color: "#4ade80", glow: "rgba(34,197,94,0.2)" },
            { label: "Rejected Cases",   value: rejectedCount,   icon: "🚫", color: "#f87171", glow: "rgba(239,68,68,0.2)" },
            { label: "New Patients",     value: newCount,        icon: "👤", color: "#a78bfa", glow: "rgba(139,92,246,0.2)" },
          ].map((s, i) => (
            <div key={i} className="stat-card" style={{
              background: "rgba(10,17,34,0.78)",
              border: `1px solid rgba(56,189,248,0.14)`,
              borderRadius: 20,
              padding: "1.5rem 1.6rem",
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
                background: s.glow,
                border: `1px solid ${s.glow}`,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.5rem",
                boxShadow: `0 0 20px ${s.glow}`,
              }}>{s.icon}</div>
            </div>
          ))}
        </div>

        {/* ─── Doctor Banner + Chart ─────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.8rem", marginBottom: "2.5rem" }}>

          {/* Doctor Hero Card */}
          <div style={{
            background: "linear-gradient(135deg, rgba(14,22,50,0.95) 0%, rgba(15,38,71,0.85) 60%, rgba(10,17,34,0.9) 100%)",
            border: "1px solid rgba(56,189,248,0.22)",
            borderRadius: 28,
            padding: "2.5rem 2.5rem 0",
            position: "relative",
            overflow: "hidden",
            boxShadow: "0 25px 60px rgba(0,0,0,0.5)",
            backdropFilter: "blur(20px)",
            minHeight: 300,
            display: "flex",
            flexDirection: "column",
            justifyContent: "flex-start",
          }}>
            {/* Glow */}
            <div style={{ position: "absolute", top: -60, right: -60, width: 280, height: 280, borderRadius: "50%", background: "radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%)", pointerEvents: "none" }} />

            <div style={{ position: "relative", zIndex: 2, maxWidth: "55%" }}>
              <div style={{ fontSize: "0.73rem", color: "#38bdf8", fontWeight: 700, letterSpacing: "0.1em", marginBottom: "0.5rem", textTransform: "uppercase" }}>
                AI Case Manager
              </div>
              <h2 style={{ fontSize: "1.9rem", fontWeight: 900, color: "#f8fafc", lineHeight: 1.2, marginBottom: "0.75rem", letterSpacing: "-0.02em" }}>
                Clinical<br />Report Queue
              </h2>
              <p style={{ color: "#94a3b8", fontSize: "0.88rem", lineHeight: 1.65, marginBottom: "1.5rem" }}>
                Review ReAct diagnostic reports, verify Grad-CAM attention heatmaps, and issue clinical approvals.
              </p>
              <div style={{ display: "flex", gap: "0.7rem", flexWrap: "wrap" }}>
                <span style={{ background: "rgba(56,189,248,0.13)", border: "1px solid rgba(56,189,248,0.3)", borderRadius: 10, padding: "0.45rem 0.9rem", fontSize: "0.8rem", color: "#38bdf8", fontWeight: 700 }}>
                  New: {newCount}
                </span>
                <span style={{ background: "rgba(129,140,248,0.13)", border: "1px solid rgba(129,140,248,0.3)", borderRadius: 10, padding: "0.45rem 0.9rem", fontSize: "0.8rem", color: "#818cf8", fontWeight: 700 }}>
                  Returning: {returningCount}
                </span>
              </div>
            </div>

            {/* Doctor image - large, bottom-right, no cropping */}
            <img
              src={homedoctor}
              alt="Clinical AI Doctor"
              style={{
                position: "absolute",
                right: 0,
                bottom: 0,
                height: "115%",
                width: "auto",
                objectFit: "contain",
                objectPosition: "bottom right",
                pointerEvents: "none",
                zIndex: 1,
                filter: "drop-shadow(0 0 32px rgba(56,189,248,0.18))",
              }}
            />
          </div>

          {/* Chart Card */}
          <div style={{
            background: "rgba(10,17,34,0.82)",
            border: "1px solid rgba(56,189,248,0.2)",
            borderRadius: 28,
            padding: "2rem 2.2rem",
            backdropFilter: "blur(16px)",
            display: "flex",
            flexDirection: "column",
            boxShadow: "0 20px 50px rgba(0,0,0,0.4)",
          }}>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 800, color: "#f8fafc", marginBottom: "0.2rem" }}>Case Status Overview</h3>
            <p style={{ color: "#64748b", fontSize: "0.84rem", marginBottom: "1.4rem" }}>Real-time report status breakdown</p>
            <div style={{ flex: 1, minHeight: 200, position: "relative" }}>
              <Bar data={chartData} options={chartOptions} />
            </div>
          </div>
        </div>

        {/* ─── Patient Case Queue Table ──────────────────────────────── */}
        <div style={{
          background: "rgba(8,14,30,0.88)",
          border: "1px solid rgba(56,189,248,0.2)",
          borderRadius: 28,
          overflow: "hidden",
          backdropFilter: "blur(24px)",
          boxShadow: "0 28px 70px rgba(0,0,0,0.55)",
        }}>

          {/* Table Header Bar */}
          <div style={{
            padding: "1.6rem 2rem",
            borderBottom: "1px solid rgba(56,189,248,0.1)",
            background: "rgba(10,18,40,0.7)",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: "1rem",
          }}>
            <div>
              <h3 style={{ fontSize: "1.35rem", fontWeight: 800, color: "#f8fafc", letterSpacing: "-0.02em" }}>
                Patient Case Queue
              </h3>
              <div style={{ fontSize: "0.82rem", color: "#64748b", marginTop: "0.2rem" }}>
                Showing <span style={{ color: "#38bdf8", fontWeight: 700 }}>{filtered.length}</span> of <span style={{ color: "#94a3b8" }}>{reports.length}</span> diagnostic records
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
                <tr style={{ background: "rgba(2,8,24,0.9)", borderBottom: "1px solid rgba(56,189,248,0.1)" }}>
                  {["Patient Name", "Clinical Symptoms", "Submission Date", "Status", "Report ID", "Actions"].map((h, i) => (
                    <th key={i} style={TH_STYLE}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={6} style={EMPTY_CELL}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.75rem" }}>
                      <span style={{ display: "inline-block", width: 18, height: 18, border: "2.5px solid rgba(56,189,248,0.3)", borderTopColor: "#38bdf8", borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
                      Loading clinical reports queue…
                    </div>
                  </td></tr>
                ) : error ? (
                  <tr><td colSpan={6} style={{ ...EMPTY_CELL, color: "#fca5a5" }}>⚠️ {error}</td></tr>
                ) : filtered.length === 0 ? (
                  <tr><td colSpan={6} style={EMPTY_CELL}>No diagnostic reports match the selected filter.</td></tr>
                ) : (
                  filtered.map((r, idx) => {
                    const sc  = statusColor(r.status);
                    const rid = r.report_id;
                    const busy = actionBusy[rid];
                    const isApproved = (r.status||"").toLowerCase() === "approved";
                    const isRejected = (r.status||"").toLowerCase() === "rejected";
                    return (
                      <tr key={rid || idx} className="row-tr" style={{
                        borderBottom: "1px solid rgba(56,189,248,0.07)",
                        background: idx % 2 === 0 ? "transparent" : "rgba(14,22,44,0.35)",
                      }}>
                        {/* Patient Name */}
                        <td style={{ ...TD_STYLE, fontWeight: 700, color: "#f0f6ff" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                            <div style={{
                              width: 34, height: 34, borderRadius: "50%",
                              background: "linear-gradient(135deg,rgba(56,189,248,0.2),rgba(99,102,241,0.2))",
                              border: "1px solid rgba(56,189,248,0.25)",
                              display: "flex", alignItems: "center", justifyContent: "center",
                              fontSize: "0.88rem", fontWeight: 800, color: "#38bdf8", flexShrink: 0,
                            }}>
                              {(r.patient_name||"?")[0].toUpperCase()}
                            </div>
                            {r.patient_name || "Anonymous"}
                          </div>
                        </td>

                        {/* Symptoms */}
                        <td style={{ ...TD_STYLE, color: "#cbd5e1", maxWidth: 250 }}>
                          <div style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {r.symptoms || "—"}
                          </div>
                        </td>

                        {/* Date */}
                        <td style={{ ...TD_STYLE, color: "#94a3b8", fontSize: "0.84rem", whiteSpace: "nowrap" }}>
                          {r.submission_date
                            ? new Date(r.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
                            : "N/A"}
                        </td>

                        {/* Status pill */}
                        <td style={{ ...TD_STYLE, textAlign: "center" }}>
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

                        {/* Report ID */}
                        <td style={{ ...TD_STYLE, color: "#475569", fontFamily: "monospace", fontSize: "0.78rem" }}>
                          #{String(rid).slice(0, 8)}…
                        </td>

                        {/* Actions */}
                        <td style={{ ...TD_STYLE, textAlign: "center" }}>
                          <div style={{ display: "flex", gap: "0.45rem", justifyContent: "center", flexWrap: "nowrap" }}>
                            <button className="review-btn" onClick={() => navigate(`/reports/${rid}`)}>
                              Review →
                            </button>
                            {!isApproved && (
                              <button
                                className="approve-btn"
                                disabled={!!busy}
                                onClick={() => quickAction(rid, "approve")}
                              >
                                {busy === "approve" ? "…" : "✓"}
                              </button>
                            )}
                            {!isRejected && (
                              <button
                                className="reject-btn"
                                disabled={!!busy}
                                onClick={() => quickAction(rid, "reject")}
                              >
                                {busy === "reject" ? "…" : "✕"}
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

      </main>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

const TH_STYLE = {
  padding: "1rem 1.5rem",
  fontSize: "0.74rem",
  fontWeight: 700,
  color: "#38bdf8",
  letterSpacing: "0.09em",
  textTransform: "uppercase",
};

const TD_STYLE = {
  padding: "1.05rem 1.5rem",
  fontSize: "0.9rem",
};

const EMPTY_CELL = {
  textAlign: "center",
  padding: "4rem",
  color: "#64748b",
  fontSize: "0.95rem",
};
