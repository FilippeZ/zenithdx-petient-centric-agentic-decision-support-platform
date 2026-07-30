import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import ladydoc from "../assets/ladydoc.png";
import { Bar } from "react-chartjs-2";
import { motion, AnimatePresence } from "framer-motion";
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
  approved: { bg: "#ecfdf5", text: "#047857", border: "#a7f3d0", dot: "#10b981" },
  rejected: { bg: "#fef2f2", text: "#b91c1c", border: "#fecaca", dot: "#ef4444" },
  edited:   { bg: "#f5f3ff", text: "#6d28d9", border: "#ddd6fe", dot: "#8b5cf6" },
  pending:  { bg: "#fffbeb", text: "#b45309", border: "#fde68a", dot: "#f59e0b" },
};
const statusColor = (s = "") => STATUS_COLORS[(s||"pending").toLowerCase()] || STATUS_COLORS.pending;

export default function HomePatient() {
  const [reports, setReports]           = useState([]);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [dateFilter, setDateFilter]     = useState("");
  const [searchTerm, setSearchTerm]     = useState("");
  const [selectedReportNote, setSelectedReportNote] = useState(null);
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
    const matchSearch = !searchTerm || 
      (r.report_type || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (r.symptoms || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (r.report_id || r.id || "").toLowerCase().includes(searchTerm.toLowerCase());
    return matchStatus && matchDate && matchSearch;
  });

  /* ── Chart ─────────────────────────────────────────────────────── */
  const chartData = {
    labels: ["Pending", "Approved", "Rejected"],
    datasets: [{
      label: "Reports",
      data: [pendingCount, approvedCount, rejectedCount],
      backgroundColor: ["#f59e0b", "#10b981", "#ef4444"],
      borderRadius: 8,
    }],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "#0f172a",
        titleColor: "#ffffff",
        bodyColor: "#f8fafc",
        padding: 12,
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { color: "#475569", font: { family: "Inter", size: 12, weight: "600" } },
      },
      y: {
        beginAtZero: true,
        ticks: { stepSize: 1, color: "#64748b", font: { family: "Inter", size: 11 } },
        grid: { color: "#f1f5f9" },
      },
    },
  };

  const handleView = (report) => {
    if ((report.status||"").toLowerCase() !== "approved") return;
    navigate(`/patient/reports/${report.id || report.report_id}`);
  };

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif", display: "flex", flexDirection: "column" }}>
      <Navbar />

      <main style={{ flex: 1, padding: "2.5rem 2rem 5rem", maxWidth: 1400, margin: "0 auto", width: "100%" }}>
        
        {/* ── Page Header with Animations ──────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "2rem", flexWrap: "wrap", gap: "1rem" }}
        >
          <div>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "#ecfdf5", border: "1px solid #a7f3d0", borderRadius: 100,
              padding: "0.35rem 1rem", fontSize: "0.75rem", color: "#047857", fontWeight: 700, marginBottom: "0.5rem"
            }}>
              <motion.span
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block" }}
              />
              PATIENT HEALTH PORTAL
            </div>
            <h1 style={{ fontSize: "clamp(1.8rem, 3vw, 2.5rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "#0f172a" }}>
              My Diagnostic Dashboard
            </h1>
            <p style={{ color: "#64748b", fontSize: "0.92rem", marginTop: "0.2rem" }}>
              Track scan submissions · Review plain-language AI results · Doctor feedback
            </p>
          </div>

          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={fetchReports}
              style={{
                background: "#ffffff", border: "1px solid #cbd5e1", color: "#334155",
                borderRadius: "12px", padding: "0.65rem 1.3rem", fontSize: "0.88rem", fontWeight: 700,
                cursor: "pointer", boxShadow: "0 2px 5px rgba(0,0,0,0.04)", display: "flex", alignItems: "center", gap: "0.4rem"
              }}
            >
              <motion.span animate={{ rotate: loading ? 360 : 0 }} transition={{ repeat: loading ? Infinity : 0, duration: 1 }}>🔄</motion.span> Refresh
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05, boxShadow: "0 6px 20px rgba(37,99,235,0.35)" }}
              whileTap={{ scale: 0.95 }}
              onClick={() => navigate("/detect")}
              style={{
                background: "linear-gradient(135deg, #2563eb, #0284c7)",
                border: "none", color: "#ffffff", borderRadius: "12px",
                padding: "0.65rem 1.5rem", fontSize: "0.9rem", fontWeight: 800,
                cursor: "pointer", boxShadow: "0 4px 14px rgba(37,99,235,0.25)",
                display: "inline-flex", alignItems: "center", gap: "0.5rem",
              }}
            >
              ➕ Submit New Scan
            </motion.button>
          </div>
        </motion.div>

        {/* ── Stat Cards with Motion ────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1.2rem", marginBottom: "2.2rem" }}>
          {[
            { label: "Total Submissions", value: totalCount,    icon: "🗂️", bg: "#eff6ff", color: "#2563eb", border: "#bfdbfe" },
            { label: "Pending Review",    value: pendingCount,  icon: "⏳", bg: "#fffbeb", color: "#d97706", border: "#fde68a" },
            { label: "Doctor Approved",   value: approvedCount, icon: "✅", bg: "#ecfdf5", color: "#059669", border: "#a7f3d0" },
            { label: "Rejected Cases",    value: rejectedCount, icon: "🚫", bg: "#fef2f2", color: "#dc2626", border: "#fecaca" },
          ].map((s, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.5 }}
              whileHover={{ y: -5, boxShadow: "0 10px 25px -5px rgba(2, 132, 199, 0.12)" }}
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "20px",
                padding: "1.4rem 1.6rem",
                boxShadow: "0 4px 15px -3px rgba(0,0,0,0.03)",
                display: "flex", alignItems: "center", justifyContent: "space-between",
                transition: "all 0.2s ease"
              }}
            >
              <div>
                <div style={{ fontSize: "0.75rem", color: "#64748b", fontWeight: 700, marginBottom: "0.3rem", letterSpacing: "0.04em", textTransform: "uppercase" }}>{s.label}</div>
                <div style={{ fontSize: "2rem", fontWeight: 900, color: "#0f172a", letterSpacing: "-0.03em", lineHeight: 1 }}>{s.value}</div>
              </div>
              <div style={{
                width: 48, height: 48, borderRadius: 14,
                background: s.bg, border: `1px solid ${s.border}`,
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: "1.4rem",
              }}>{s.icon}</div>
            </motion.div>
          ))}
        </div>

        {/* ── Hero Banner + Chart ───────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "1.8rem", marginBottom: "2.5rem" }}>
          
          {/* Patient Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            style={{
              background: "linear-gradient(135deg, #0284c7, #2563eb)",
              borderRadius: "24px",
              padding: "2.5rem 2.5rem 0",
              position: "relative",
              overflow: "hidden",
              color: "#ffffff",
              boxShadow: "0 15px 35px -5px rgba(2, 132, 199, 0.3)",
              minHeight: 280,
              display: "flex", flexDirection: "column", justifyContent: "space-between",
            }}
          >
            <div style={{ position: "relative", zIndex: 2, maxWidth: "60%" }}>
              <div style={{ fontSize: "0.75rem", color: "#bae6fd", fontWeight: 800, letterSpacing: "0.08em", marginBottom: "0.4rem", textTransform: "uppercase" }}>
                Patient Care Center
              </div>
              <h2 style={{ fontSize: "1.8rem", fontWeight: 900, lineHeight: 1.2, marginBottom: "0.8rem" }}>
                Your Diagnostic History
              </h2>
              <p style={{ color: "#e0f2fe", fontSize: "0.88rem", lineHeight: 1.6, marginBottom: "1.5rem" }}>
                Submit a new chest X-ray anytime for instant neural net analysis and plain-language summaries reviewed by real medical professionals.
              </p>
            </div>

            <motion.img
              src={ladydoc}
              alt="Medical Care"
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              style={{
                position: "absolute", right: 0, bottom: 0,
                height: "115%", width: "auto", objectFit: "contain",
                pointerEvents: "none", zIndex: 1, opacity: 0.9,
              }}
            />
          </motion.div>

          {/* Chart Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            style={{
              background: "#ffffff",
              border: "1px solid #e2e8f0",
              borderRadius: "24px",
              padding: "2rem 2.2rem",
              boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
              display: "flex", flexDirection: "column",
            }}
          >
            <h3 style={{ fontSize: "1.15rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.2rem" }}>Report Status Overview</h3>
            <p style={{ color: "#64748b", fontSize: "0.84rem", marginBottom: "1.4rem" }}>Status of your diagnostic submissions</p>
            <div style={{ flex: 1, minHeight: 180, position: "relative" }}>
              {totalCount === 0 ? (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "0.75rem", color: "#64748b" }}>
                  <span style={{ fontSize: "2.5rem" }}>📊</span>
                  <span style={{ fontSize: "0.9rem" }}>No submissions yet</span>
                </div>
              ) : (
                <Bar data={chartData} options={chartOptions} />
              )}
            </div>
          </motion.div>
        </div>

        {/* ── Reports Table ─────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          style={{
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: "24px",
            overflow: "hidden",
            boxShadow: "0 10px 30px -5px rgba(0,0,0,0.04)",
          }}
        >

          {/* Table Header Bar with Search & Filter */}
          <div style={{
            padding: "1.5rem 2rem",
            borderBottom: "1px solid #e2e8f0",
            background: "#fafafa",
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: "1rem",
          }}>
            <div>
              <h3 style={{ fontSize: "1.25rem", fontWeight: 800, color: "#0f172a" }}>
                My Diagnostic Reports
              </h3>
              <div style={{ fontSize: "0.82rem", color: "#64748b", marginTop: "0.2rem" }}>
                Showing <span style={{ color: "#2563eb", fontWeight: 700 }}>{filtered.length}</span> of <span style={{ color: "#0f172a" }}>{totalCount}</span> submissions
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "0.7rem", flexWrap: "wrap" }}>
              {/* Search Bar Functionality */}
              <input
                type="text"
                placeholder="🔍 Search scan or ID..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                style={{
                  background: "#ffffff", border: "1px solid #cbd5e1", color: "#0f172a",
                  borderRadius: "10px", padding: "0.5rem 0.9rem", fontSize: "0.85rem", fontWeight: 500,
                  minWidth: "180px"
                }}
              />

              <select
                value={statusFilter}
                onChange={e => setStatusFilter(e.target.value)}
                style={{
                  background: "#ffffff", border: "1px solid #cbd5e1", color: "#0f172a",
                  borderRadius: "10px", padding: "0.5rem 0.9rem", fontSize: "0.85rem", fontWeight: 500,
                }}
              >
                <option value="All">All Statuses</option>
                <option value="Pending">Pending</option>
                <option value="Approved">Approved</option>
                <option value="Rejected">Rejected</option>
              </select>

              <input
                type="date"
                value={dateFilter}
                onChange={e => setDateFilter(e.target.value)}
                style={{
                  background: "#ffffff", border: "1px solid #cbd5e1", color: "#0f172a",
                  borderRadius: "10px", padding: "0.5rem 0.9rem", fontSize: "0.85rem", fontWeight: 500,
                }}
              />

              <button
                onClick={() => { setStatusFilter("All"); setDateFilter(""); setSearchTerm(""); }}
                style={{
                  background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb",
                  borderRadius: "10px", padding: "0.5rem 1rem", fontSize: "0.85rem", fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                Reset
              </button>
            </div>
          </div>

          {/* Table */}
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
              <thead>
                <tr style={{ background: "#f8fafc", borderBottom: "1px solid #e2e8f0" }}>
                  {["Report ID", "Scan Type", "Submission Date", "Doctor Note", "Status", "Actions"].map((h, i) => (
                    <th key={i} style={TH_STYLE}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={6} style={EMPTY}>Loading your diagnostic reports…</td></tr>
                ) : error ? (
                  <tr><td colSpan={6} style={{ ...EMPTY, color: "#dc2626" }}>⚠️ {error}</td></tr>
                ) : filtered.length === 0 ? (
                  <tr><td colSpan={6} style={EMPTY}>
                    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1rem" }}>
                      <span style={{ fontSize: "2.5rem" }}>🩺</span>
                      <span>No reports match the selected filter.</span>
                    </div>
                  </td></tr>
                ) : (
                  filtered.map((r, idx) => {
                    const rid = r.id || r.report_id;
                    const sc  = statusColor(r.status);
                    const isApproved = (r.status||"").toLowerCase() === "approved";
                    return (
                      <motion.tr
                        key={rid || idx}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.04 }}
                        style={{
                          borderBottom: "1px solid #f1f5f9",
                          background: idx % 2 === 0 ? "#ffffff" : "#fafafa",
                        }}
                      >
                        {/* Report ID */}
                        <td style={{ ...TD, fontFamily: "monospace", color: "#94a3b8", fontSize: "0.8rem" }}>
                          #{String(rid).slice(0, 8)}…
                        </td>

                        {/* Scan Type */}
                        <td style={{ ...TD, color: "#0f172a", fontWeight: 600 }}>
                          {(() => {
                            const hasXray = Boolean(r.image_path || r.has_image || r.xray_image || (r.xray_findings && r.xray_findings.length > 0));
                            const scanType = r.report_type || r.data_type || (hasXray ? "Chest Radiograph" : "Clinical Symptoms (Text-Only)");
                            const scanIcon = hasXray ? "🫁" : "📝";
                            return (
                              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                                <span style={{ fontSize: "1rem" }}>{scanIcon}</span>
                                <span style={{ fontSize: "0.88rem" }}>{scanType}</span>
                              </div>
                            );
                          })()}
                        </td>

                        {/* Date */}
                        <td style={{ ...TD, color: "#64748b", fontSize: "0.85rem", whiteSpace: "nowrap" }}>
                          {r.submission_date
                            ? new Date(r.submission_date).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
                            : "—"}
                        </td>

                        {/* Doctor Message / Interactive Note Click Functionality */}
                        <td style={{ ...TD, color: "#475569", fontSize: "0.85rem", maxWidth: 220 }}>
                          {r.doctor_message ? (
                            <motion.button
                              whileHover={{ scale: 1.02 }}
                              onClick={() => setSelectedReportNote(r)}
                              style={{
                                background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb",
                                borderRadius: "8px", padding: "0.3rem 0.6rem", fontSize: "0.8rem",
                                cursor: "pointer", display: "flex", alignItems: "center", gap: "0.4rem",
                                textAlign: "left", width: "100%"
                              }}
                            >
                              <span>💬</span>
                              <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontWeight: 600 }}>
                                {r.doctor_message}
                              </span>
                            </motion.button>
                          ) : (
                            <span style={{ color: "#94a3b8", fontStyle: "italic", fontSize: "0.8rem" }}>Awaiting clinician note</span>
                          )}
                        </td>

                        {/* Status Pill */}
                        <td style={{ ...TD }}>
                          <span style={{
                            display: "inline-flex", alignItems: "center", gap: "0.4rem",
                            padding: "0.3rem 0.85rem", borderRadius: 100,
                            fontSize: "0.78rem", fontWeight: 700,
                            background: sc.bg, color: sc.text, border: `1px solid ${sc.border}`,
                            whiteSpace: "nowrap",
                          }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: sc.dot, display: "inline-block" }} />
                            {r.status || "Pending"}
                          </span>
                        </td>

                        {/* Actions */}
                        <td style={{ ...TD, textAlign: "center" }}>
                          {isApproved ? (
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => handleView(r)}
                              style={{
                                background: "linear-gradient(135deg, #2563eb, #0284c7)",
                                border: "none", color: "#ffffff", borderRadius: 8,
                                padding: "0.42rem 1.1rem", fontSize: "0.8rem", fontWeight: 700,
                                cursor: "pointer", boxShadow: "0 2px 8px rgba(37,99,235,0.2)",
                              }}
                            >
                              View Report →
                            </motion.button>
                          ) : (
                            <span style={{
                              background: "#f1f5f9", border: "1px solid #e2e8f0", color: "#64748b",
                              borderRadius: 8, padding: "0.4rem 0.9rem", fontSize: "0.78rem", fontWeight: 600,
                            }}>
                              {(r.status||"pending").toLowerCase() === "rejected" ? "🚫 Rejected" : "⏳ Pending"}
                            </span>
                          )}
                        </td>
                      </motion.tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* ── New Functionality: Doctor Feedback Modal ────────────────── */}
        <AnimatePresence>
          {selectedReportNote && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelectedReportNote(null)}
              style={{
                position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
                background: "rgba(15, 23, 42, 0.6)", backdropFilter: "blur(6px)",
                display: "flex", alignItems: "center", justifyContent: "center",
                zIndex: 9999, padding: "1.5rem"
              }}
            >
              <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.9, y: 20 }}
                onClick={e => e.stopPropagation()}
                style={{
                  background: "#ffffff", borderRadius: "24px", padding: "2rem",
                  maxWidth: "500px", width: "100%", boxShadow: "0 20px 40px rgba(0,0,0,0.2)",
                  border: "1px solid #e2e8f0"
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                  <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "#0f172a", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span>🩺</span> Doctor Clinician Note
                  </div>
                  <button onClick={() => setSelectedReportNote(null)} style={{ background: "none", border: "none", fontSize: "1.2rem", cursor: "pointer", color: "#64748b" }}>✕</button>
                </div>
                <div style={{ background: "#f8fafc", padding: "1.2rem", borderRadius: "14px", border: "1px solid #e2e8f0", color: "#334155", fontSize: "0.95rem", lineHeight: 1.6, marginBottom: "1.5rem" }}>
                  "{selectedReportNote.doctor_message}"
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", color: "#64748b" }}>
                  <span>Report ID: #{String(selectedReportNote.id || selectedReportNote.report_id).slice(0, 8)}</span>
                  <span>Status: <strong style={{ color: "#059669" }}>{selectedReportNote.status}</strong></span>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>
    </div>
  );
}

const TH_STYLE = {
  padding: "1rem 1.5rem",
  fontSize: "0.75rem",
  fontWeight: 700,
  color: "#475569",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
};

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
