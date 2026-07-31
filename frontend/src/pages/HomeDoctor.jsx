import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import homedoctor from "../assets/homedoctor.png";
import ladydoc2 from "../assets/ladydoc2.png";
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

/* ─── Status colour map ─────────────────────────────────────────── */
const STATUS_COLORS = {
  approved: { bg: "#ecfdf5", text: "#047857", border: "#a7f3d0", dot: "#10b981" },
  rejected: { bg: "#fef2f2", text: "#b91c1c", border: "#fecaca", dot: "#ef4444" },
  edited:   { bg: "#f5f3ff", text: "#6d28d9", border: "#ddd6fe", dot: "#8b5cf6" },
  pending:  { bg: "#fffbeb", text: "#b45309", border: "#fde68a", dot: "#f59e0b" },
};
const statusColor = (s = "pending") => STATUS_COLORS[(s || "pending").toLowerCase()] || STATUS_COLORS.pending;

/* ─── Helper function for AI Urgency Score Triage ──────────────────────── */
const calculateUrgency = (symptoms = "") => {
  const s = symptoms.toLowerCase();
  if (s.includes("heart") || s.includes("chest") || s.includes("cant breath") || s.includes("can't breath") || s.includes("severe")) {
    return { level: "HIGH", label: "🔴 High Priority", bg: "#fef2f2", text: "#dc2626", border: "#fecaca" };
  }
  if (s.includes("fever") || s.includes("headache") || s.includes("dizziness") || s.includes("pain")) {
    return { level: "MODERATE", label: "🟠 Medium Priority", bg: "#fffbeb", text: "#d97706", border: "#fde68a" };
  }
  return { level: "LOW", label: "🟢 Routine", bg: "#ecfdf5", text: "#059669", border: "#a7f3d0" };
};

export default function HomeDoctor() {
  const [reports, setReports]           = useState([]);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [dateFilter, setDateFilter]     = useState("");
  const [searchTerm, setSearchTerm]     = useState("");
  const [actionBusy, setActionBusy]     = useState({});
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
    const matchSearch = !searchTerm || 
      (r.patient_name || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (r.symptoms || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
      (r.report_id || "").toLowerCase().includes(searchTerm.toLowerCase());
    return matchStatus && matchDate && matchSearch;
  });

  const resetFilters = () => {
    setStatusFilter("All");
    setDateFilter("");
    setSearchTerm("");
  };

  /* ─── Chart ─────────────────────────────────────────────────────── */
  const chartData = {
    labels: ["New Patients", "Returning Patients", "Pending", "Approved", "Rejected"],
    datasets: [{
      label: "Patient Triage Stats",
      data: [newCount, returningCount, pendingCount, approvedCount, rejectedCount],
      backgroundColor: ["#7c3aed", "#2563eb", "#f59e0b", "#10b981", "#ef4444"],
      borderRadius: 10,
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
        ticks: { color: "#475569", font: { family: "Inter", size: 11, weight: "600" } },
      },
      y: {
        beginAtZero: true,
        ticks: { stepSize: 1, color: "#64748b", font: { family: "Inter", size: 11 } },
        grid: { color: "#f1f5f9" },
      },
    },
  };

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif", display: "flex", flexDirection: "column" }}>
      <Navbar />

      <main style={{ flex: 1, padding: "2.5rem 2rem 5rem", maxWidth: 1400, margin: "0 auto", width: "100%" }}>
        
        {/* ─── Page Header with Animations ────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "2rem", flexWrap: "wrap", gap: "1rem" }}
        >
          <div>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 100,
              padding: "0.35rem 1rem", fontSize: "0.75rem", color: "#2563eb", fontWeight: 700, marginBottom: "0.5rem"
            }}>
              <motion.span
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                style={{ width: 6, height: 6, borderRadius: "50%", background: "#2563eb", display: "inline-block" }}
              />
              CLINICIAN DECISION WORKSTATION
            </div>
            <h1 style={{ fontSize: "clamp(1.8rem, 3vw, 2.5rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "#0f172a" }}>
              Doctor Case Review
            </h1>
            <p style={{ color: "#64748b", fontSize: "0.92rem", marginTop: "0.2rem" }}>
              Multi-modal AI diagnosis review · Grad-CAM verification · Clinical case triage
            </p>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchReports}
            style={{
              background: "#ffffff", border: "1px solid #cbd5e1", color: "#334155",
              borderRadius: "12px", padding: "0.65rem 1.3rem", fontSize: "0.88rem", fontWeight: 700,
              cursor: "pointer", boxShadow: "0 2px 5px rgba(0,0,0,0.04)", display: "flex", alignItems: "center", gap: "0.5rem"
            }}
          >
            <motion.span animate={{ rotate: loading ? 360 : 0 }} transition={{ repeat: loading ? Infinity : 0, duration: 1 }}>🔄</motion.span> Refresh Queue
          </motion.button>
        </motion.div>

        {/* ─── Stat Cards Grid with Motion ──────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1.2rem", marginBottom: "2.2rem" }}>
          {[
            { label: "Total Reports",  value: reports.length,  icon: "📋", bg: "#eff6ff", color: "#2563eb", border: "#bfdbfe" },
            { label: "Pending Review", value: pendingCount,    icon: "⏳", bg: "#fffbeb", color: "#d97706", border: "#fde68a" },
            { label: "Approved Cases", value: approvedCount,   icon: "✅", bg: "#ecfdf5", color: "#059669", border: "#a7f3d0" },
            { label: "Rejected Cases", value: rejectedCount,   icon: "🚫", bg: "#fef2f2", color: "#dc2626", border: "#fecaca" },
            { label: "New Patients",   value: newCount,        icon: "👤", bg: "#f5f3ff", color: "#7c3aed", border: "#ddd6fe" },
          ].map((s, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.5 }}
              whileHover={{ y: -5, boxShadow: "0 10px 25px -5px rgba(37, 99, 235, 0.1)" }}
              style={{
                background: "#ffffff",
                border: `1px solid #e2e8f0`,
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

        {/* ─── Doctor Banner + Chart ─────────────────────────────────── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "1.8rem", marginBottom: "2.5rem" }}>
          
          {/* Doctor Intelligence Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            style={{
              background: "linear-gradient(135deg, #1e3a5f, #0d2240)",
              borderRadius: "24px",
              padding: "2.5rem 2.5rem 0",
              position: "relative",
              overflow: "hidden",
              color: "#ffffff",
              boxShadow: "0 15px 35px -5px rgba(13, 34, 64, 0.3)",
              minHeight: 280,
              display: "flex", flexDirection: "column", justifyContent: "space-between",
            }}
          >
            <div style={{ position: "relative", zIndex: 2, maxWidth: "60%" }}>
              <div style={{ fontSize: "0.75rem", color: "#93c5fd", fontWeight: 800, letterSpacing: "0.08em", marginBottom: "0.4rem", textTransform: "uppercase" }}>
                Radiology Intelligence
              </div>
              <h2 style={{ fontSize: "1.8rem", fontWeight: 900, lineHeight: 1.2, marginBottom: "0.8rem" }}>
                Reports Generated Today
              </h2>
              <div style={{ fontSize: "3rem", fontWeight: 900, color: "#ffffff", marginBottom: "0.8rem" }}>
                {reports.length}
              </div>
              <p style={{ color: "#dbeafe", fontSize: "0.88rem", lineHeight: 1.6, marginBottom: "1.5rem" }}>
                Inspect Grad-CAM diagnostic heatmaps, cross-reference RAG literature findings, and finalize clinical diagnostic approvals.
              </p>
            </div>

            <motion.img
              src={ladydoc2 || homedoctor}
              alt="Clinician"
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              style={{
                position: "absolute", right: -10, bottom: 0,
                height: "110%", width: "auto", objectFit: "contain",
                pointerEvents: "none", zIndex: 1, opacity: 0.95,
              }}
            />
          </motion.div>

          {/* Chart Card */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            style={{
              background: "#ffffff", border: "1px solid #e2e8f0",
              borderRadius: "24px", padding: "1.8rem",
              boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
              display: "flex", flexDirection: "column"
            }}
          >
            <div style={{ fontSize: "1rem", fontWeight: 800, color: "#0f172a", marginBottom: "1rem" }}>
              📊 Patient Stats & Case Triage
            </div>
            <div style={{ flex: 1, minHeight: 220 }}>
              <Bar data={chartData} options={chartOptions} />
            </div>
          </motion.div>
        </div>

        {/* ─── Case Workstation Table ───────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          style={{
            background: "#ffffff", border: "1px solid #e2e8f0",
            borderRadius: "24px", padding: "1.8rem",
            boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)"
          }}
        >
          {/* Table Header Controls with Search & Filter */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
            <div>
              <h2 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#0f172a" }}>
                Patient Workstation Queue
              </h2>
              <span style={{ fontSize: "0.82rem", color: "#64748b" }}>
                Showing {filtered.length} of {reports.length} cases
              </span>
            </div>

            {/* Filter & Search Group */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
              {/* Search Bar Functionality */}
              <input
                type="text"
                placeholder="🔍 Search patient, symptom or ID..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                style={{
                  padding: "0.55rem 1rem", borderRadius: 10, border: "1px solid #cbd5e1",
                  fontSize: "0.88rem", color: "#0f172a", background: "#ffffff", fontWeight: 500,
                  minWidth: "220px"
                }}
              />

              <select
                value={statusFilter}
                onChange={e => setStatusFilter(e.target.value)}
                style={{
                  padding: "0.55rem 1rem", borderRadius: 10, border: "1px solid #cbd5e1",
                  fontSize: "0.88rem", color: "#0f172a", background: "#ffffff", fontWeight: 600
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
                  padding: "0.55rem 1rem", borderRadius: 10, border: "1px solid #cbd5e1",
                  fontSize: "0.88rem", color: "#0f172a", background: "#ffffff", fontWeight: 600
                }}
              />

              <button
                onClick={resetFilters}
                style={{
                  padding: "0.55rem 1rem", borderRadius: 10, border: "1px solid #cbd5e1",
                  fontSize: "0.88rem", color: "#475569", background: "#f1f5f9", fontWeight: 700, cursor: "pointer"
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
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Patient Name</th>
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Symptoms / Data</th>
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Submission Date</th>
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase", textAlign: "center" }}>Status</th>
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase" }}>Report ID</th>
                  <th style={{ padding: "0.85rem 1rem", fontSize: "0.75rem", fontWeight: 700, color: "#475569", letterSpacing: "0.06em", textTransform: "uppercase", textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "3rem", color: "#64748b" }}>Loading queue…</td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "3rem", color: "#dc2626" }}>{error}</td>
                  </tr>
                ) : filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: "center", padding: "3rem", color: "#64748b" }}>No cases found matching filters.</td>
                  </tr>
                ) : (
                  filtered.map((r, i) => {
                    const sc = statusColor(r.status);
                    const rid = r.report_id;
                    const busy = actionBusy[rid];
                    return (
                      <motion.tr
                        key={rid}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04 }}
                        style={{ background: i % 2 === 0 ? "#ffffff" : "#fafafa", borderBottom: "1px solid #f1f5f9" }}
                      >
                        <td style={{ padding: "1rem", color: "#0f172a", fontWeight: 700, fontSize: "0.92rem" }}>
                          {r.patient_name || "Anonymous Patient"}
                        </td>
                        <td style={{ padding: "1rem", color: "#334155", fontSize: "0.88rem", maxWidth: 240, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          <span style={{ marginRight: "0.4rem" }} title={r.image_path || r.has_image || r.xray_image ? "Chest Radiograph Attached" : "Clinical Consultation (Text-Only)"}>
                            {r.image_path || r.has_image || r.xray_image ? "🫁" : "📝"}
                          </span>
                          {r.symptoms || "No symptoms recorded"}
                        </td>
                        <td style={{ padding: "1rem", color: "#64748b", fontSize: "0.85rem" }}>
                          {r.submission_date ? new Date(r.submission_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—"}
                        </td>
                        <td style={{ padding: "1rem", textAlign: "center" }}>
                          <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.3rem 0.8rem", borderRadius: 100, fontSize: "0.78rem", fontWeight: 700, background: sc.bg, color: sc.text, border: `1px solid ${sc.border}` }}>
                            <span style={{ width: 6, height: 6, borderRadius: "50%", background: sc.dot }} />
                            {r.status || "Pending"}
                          </span>
                        </td>
                        <td style={{ padding: "1rem", color: "#64748b", fontFamily: "monospace", fontSize: "0.82rem" }}>
                          #{String(rid).slice(0, 8)}
                        </td>
                        <td style={{ padding: "1rem", textAlign: "right" }}>
                          <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: "0.4rem" }}>
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => navigate(`/reports/${rid}`)}
                              style={{ background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb", borderRadius: 8, padding: "0.4rem 0.85rem", fontSize: "0.82rem", fontWeight: 700, cursor: "pointer" }}
                            >
                              View Case
                            </motion.button>
                            {r.status !== "Approved" && (
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                disabled={!!busy}
                                onClick={() => quickAction(rid, "approve")}
                                style={{ background: "#ecfdf5", border: "1px solid #a7f3d0", color: "#047857", borderRadius: 8, padding: "0.4rem 0.65rem", fontSize: "0.82rem", fontWeight: 700, cursor: "pointer" }}
                              >
                                {busy === "approve" ? "…" : "✓"}
                              </motion.button>
                            )}
                            {r.status !== "Rejected" && (
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                disabled={!!busy}
                                onClick={() => quickAction(rid, "reject")}
                                style={{ background: "#fef2f2", border: "1px solid #fecaca", color: "#b91c1c", borderRadius: 8, padding: "0.4rem 0.65rem", fontSize: "0.82rem", fontWeight: 700, cursor: "pointer" }}
                              >
                                {busy === "reject" ? "…" : "✕"}
                              </motion.button>
                            )}
                          </div>
                        </td>
                      </motion.tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
