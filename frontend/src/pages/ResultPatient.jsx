import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useParams, useNavigate } from "react-router-dom";
import { FaFilePdf, FaStar, FaNotesMedical, FaArrowLeft } from "react-icons/fa";
import Navbar from "../components/Navbar";
import ladydoc from "../assets/ladydoc.png";
import { motion } from "framer-motion";

const ResultPatient = () => {
  const { reportId } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [rating, setRating] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const token = localStorage.getItem("token");
        const response = await fetch(
          `http://localhost:8000/patient/reports/${reportId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!response.ok) throw new Error("Failed to fetch report");
        const data = await response.json();
        if (!data || !data.status) {
          throw new Error("No report data found or missing status.");
        }
        if (data.status.trim().toLowerCase() !== "approved") {
          throw new Error("This report has not been approved by the doctor.");
        }
        setReport(data);

        if (data.rating && data.rating >= 1 && data.rating <= 5) {
          setRating(data.rating);
          setSubmitted(true);
        } else {
          setRating(0);
          setSubmitted(false);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, [reportId]);

  const downloadPDF = async () => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch(
        `http://localhost:8000/patient/reports/${reportId}/pdf`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!response.ok) throw new Error("Failed to download PDF");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `report_${reportId}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError(err.message);
    }
  };

  const submitFeedback = async (stars) => {
    try {
      const token = localStorage.getItem("token");
      const response = await fetch(
        `http://localhost:8000/patient/reports/${reportId}/feedback`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ rating: stars }),
        }
      );
      if (!response.ok) throw new Error("Failed to submit feedback");
      setSubmitted(true);
    } catch (err) {
      setError("Failed to submit feedback");
    }
  };

  const handleStarClick = (index) => {
    if (submitted) return;
    setRating(index + 1);
    submitFeedback(index + 1);
  };

  if (loading) {
    return (
      <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
        <Navbar />
        <div style={{ maxWidth: 800, margin: "4rem auto", textAlign: "center", color: "#64748b" }}>
          Loading your approved diagnostic report...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
        <Navbar />
        <div style={{ maxWidth: 600, margin: "4rem auto", textAlign: "center", padding: "2rem", background: "#ffffff", borderRadius: "20px", border: "1px solid #fecaca", boxShadow: "0 4px 20px rgba(0,0,0,0.04)" }}>
          <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>⚠️</div>
          <p style={{ color: "#dc2626", fontWeight: 700, marginBottom: "1.5rem" }}>{error}</p>
          <button
            onClick={() => navigate(-1)}
            style={{
              background: "#eff6ff", border: "1px solid #bfdbfe", color: "#2563eb",
              borderRadius: "10px", padding: "0.6rem 1.4rem", fontWeight: 700, cursor: "pointer",
            }}
          >
            ← Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif", display: "flex", flexDirection: "column" }}>
      <Navbar />

      <main style={{ maxWidth: 900, margin: "0 auto", padding: "3rem 1.5rem 6rem", width: "100%" }}>
        
        {/* Back button */}
        <button
          onClick={() => navigate(-1)}
          style={{
            display: "inline-flex", alignItems: "center", gap: "0.5rem",
            background: "#ffffff", border: "1px solid #cbd5e1", borderRadius: "10px",
            padding: "0.5rem 1rem", fontSize: "0.85rem", fontWeight: 700, color: "#475569",
            cursor: "pointer", marginBottom: "1.5rem", boxShadow: "0 2px 5px rgba(0,0,0,0.03)"
          }}
        >
          <FaArrowLeft /> Back to Dashboard
        </button>

        {/* Diagnosis Report Card */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: "24px",
            padding: "2.5rem",
            boxShadow: "0 10px 30px -5px rgba(0,0,0,0.05)",
            marginBottom: "2rem",
            position: "relative",
          }}
        >
          <div style={{
            display: "inline-flex", alignItems: "center", gap: "0.5rem",
            background: "#eff6ff", border: "1px solid #bfdbfe",
            borderRadius: "100px", padding: "0.4rem 1.2rem",
            color: "#2563eb", fontWeight: 800, fontSize: "0.85rem",
            marginBottom: "1.5rem"
          }}>
            <FaNotesMedical /> Approved AI Diagnostic Report
          </div>

          <div style={{ fontSize: "1.05rem", lineHeight: 1.8, color: "#334155" }}>
            <ReactMarkdown>{report.diagnosis?.trim()}</ReactMarkdown>
          </div>
        </motion.div>

        {/* Doctor's Note */}
        <div style={{
          background: "linear-gradient(135deg, #eff6ff 0%, #ffffff 100%)",
          border: "1px solid #bfdbfe",
          borderRadius: "24px",
          padding: "2rem",
          boxShadow: "0 4px 20px -3px rgba(37,99,235,0.08)",
          marginBottom: "2rem",
          display: "flex", alignItems: "center", gap: "1.5rem", flexWrap: "wrap",
        }}>
          <img
            src={ladydoc}
            alt="Doctor"
            style={{ width: 80, height: 80, borderRadius: "50%", objectFit: "cover", objectPosition: "top", border: "3px solid #ffffff", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }}
          />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "0.75rem", fontWeight: 800, color: "#2563eb", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: "0.3rem" }}>
              Attending Clinician Note
            </div>
            <div style={{ fontSize: "0.95rem", color: "#0f172a", fontWeight: 600, lineHeight: 1.6 }}>
              {report.doctor_message ? (
                <ReactMarkdown>{report.doctor_message.trim()}</ReactMarkdown>
              ) : (
                <span style={{ color: "#64748b", fontStyle: "italic" }}>No specific clinician notes added.</span>
              )}
            </div>
          </div>
        </div>

        {/* Review & Rating Section */}
        <div style={{
          background: "#ffffff", border: "1px solid #e2e8f0",
          borderRadius: "20px", padding: "1.8rem", textAlign: "center",
          boxShadow: "0 4px 15px rgba(0,0,0,0.02)", marginBottom: "2.5rem",
        }}>
          <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.8rem" }}>
            Rate Your Report Experience
          </div>
          {submitted ? (
            <div>
              <div style={{ color: "#059669", fontWeight: 700, marginBottom: "0.5rem" }}>
                Thank you for your feedback!
              </div>
              <div style={{ display: "flex", gap: "0.4rem", justifyContent: "center" }}>
                {[...Array(5)].map((_, idx) => (
                  <FaStar
                    key={idx}
                    style={{ fontSize: "1.5rem", color: idx < rating ? "#f59e0b" : "#cbd5e1" }}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div>
              <div style={{ display: "flex", gap: "0.5rem", justifyContent: "center", cursor: "pointer" }}>
                {[...Array(5)].map((_, index) => (
                  <FaStar
                    key={index}
                    onClick={() => handleStarClick(index)}
                    style={{
                      fontSize: "1.8rem",
                      color: index < rating ? "#f59e0b" : "#cbd5e1",
                      transition: "transform 0.15s",
                    }}
                  />
                ))}
              </div>
              <div style={{ fontSize: "0.82rem", color: "#64748b", marginTop: "0.6rem" }}>
                Click a star to submit your review
              </div>
            </div>
          )}
        </div>

        {/* Download PDF Button */}
        <div style={{ textAlign: "center" }}>
          <button
            onClick={downloadPDF}
            style={{
              background: "#dc2626", color: "#ffffff", border: "none",
              borderRadius: "14px", padding: "0.9rem 2.2rem",
              fontSize: "1.05rem", fontWeight: 800, cursor: "pointer",
              boxShadow: "0 4px 16px rgba(220, 38, 38, 0.28)",
              display: "inline-flex", alignItems: "center", gap: "0.6rem",
              transition: "all 0.2s ease",
            }}
            onMouseEnter={e => e.currentTarget.style.opacity = "0.9"}
            onMouseLeave={e => e.currentTarget.style.opacity = "1"}
          >
            <FaFilePdf style={{ fontSize: "1.3rem" }} /> Download Official PDF Report
          </button>
        </div>

      </main>
    </div>
  );
};

export default ResultPatient;
