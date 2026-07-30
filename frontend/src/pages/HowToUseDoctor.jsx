import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import howusedoctor from "../assets/howusedoctor.png";
import meds from "../assets/meds.png";
import Ai from "../assets/Ai.png";

const STEPS = [
  {
    step: "01",
    title: "Clinical Diagnosis Summary",
    subtitle: "Diagnosis Report",
    description:
      "AI-assisted summary of patient data and radiographic findings, providing clinicians with a provisional diagnosis and confidence scores for expert review.",
    image: howusedoctor,
    imageAlt: "Diagnosis Report",
    tags: ["ResNet-50", "Provisional Diagnosis", "Confidence Score"],
  },
  {
    step: "02",
    title: "Actions & Documentation",
    subtitle: "Clinical Decision Hub",
    description:
      "Provide personalised advice to the patient, edit the AI-generated findings, approve or reject the diagnostic report, and export the complete summary as a signed PDF.",
    image: meds,
    imageAlt: "Actions & Documentation",
    tags: ["Approve / Reject", "PDF Export", "Doctor Message"],
  },
  {
    step: "03",
    title: "AI Explainability & Validation",
    subtitle: "Explainable AI (XAI)",
    description:
      "Review Grad-CAM attention heatmaps, Captum attribution plots, and classification scores to fully understand and validate the model's clinical reasoning.",
    image: Ai,
    imageAlt: "Explainable AI",
    tags: ["Grad-CAM", "Captum XAI", "Attribution Analysis"],
  },
];

export default function HowToUseDoctor() {
  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
      <Navbar />

      <main style={{ maxWidth: 1100, margin: "0 auto", padding: "3rem 1.5rem 6rem" }}>
        
        {/* Animated Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          style={{ textAlign: "center", marginBottom: "4rem" }}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            style={{
              display: "inline-flex", alignItems: "center", gap: "0.5rem",
              background: "#eff6ff", border: "1px solid #bfdbfe",
              borderRadius: 100, padding: "0.35rem 1.1rem",
              fontSize: "0.75rem", color: "#2563eb", fontWeight: 700,
              letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "1rem",
            }}
          >
            CLINICIAN USER GUIDE
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            style={{ fontSize: "clamp(2.2rem, 4vw, 3.2rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "#0f172a", marginBottom: "1rem" }}
          >
            How ZenithDx Supports Clinicians
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            style={{ color: "#64748b", fontSize: "1.1rem", maxWidth: "620px", margin: "0 auto", lineHeight: 1.7 }}
          >
            A step-by-step walkthrough of the clinician workstation, from initial X-ray review to Grad-CAM heatmap validation and patient communication.
          </motion.p>
        </motion.div>

        {/* Steps with Motion Animations */}
        <div style={{ display: "flex", flexDirection: "column", gap: "3rem" }}>
          {STEPS.map((step, idx) => (
            <motion.div
              key={step.step}
              initial={{ opacity: 0, y: 35 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.6, delay: idx * 0.15, ease: [0.16, 1, 0.3, 1] }}
              whileHover={{ y: -6, boxShadow: "0 14px 35px -5px rgba(37, 99, 235, 0.12)" }}
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "24px",
                padding: "2.5rem",
                boxShadow: "0 4px 20px -3px rgba(0,0,0,0.03)",
                display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
                gap: "2.5rem", alignItems: "center",
                transition: "box-shadow 0.3s ease, border-color 0.3s ease"
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "1rem" }}>
                  <motion.span
                    whileHover={{ scale: 1.15 }}
                    style={{ fontSize: "1.8rem", fontWeight: 900, color: "#2563eb", display: "inline-block" }}
                  >
                    {step.step}
                  </motion.span>

                  <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "#0284c7", background: "#f0f9ff", border: "1px solid #bae6fd", padding: "0.25rem 0.75rem", borderRadius: 100, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                    {step.subtitle}
                  </span>
                </div>

                <h3 style={{ fontSize: "1.4rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.8rem" }}>
                  {step.title}
                </h3>

                <p style={{ color: "#475569", fontSize: "0.98rem", lineHeight: 1.7, marginBottom: "1.5rem" }}>
                  {step.description}
                </p>

                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                  {step.tags.map(tag => (
                    <motion.span
                      key={tag}
                      whileHover={{ scale: 1.06, backgroundColor: "#e0f2fe", color: "#0284c7" }}
                      style={{
                        fontSize: "0.75rem", fontWeight: 700,
                        background: "#f1f5f9", color: "#475569",
                        border: "1px solid #e2e8f0", padding: "0.3rem 0.75rem",
                        borderRadius: 100, cursor: "default",
                        transition: "all 0.2s ease"
                      }}
                    >
                      {tag}
                    </motion.span>
                  ))}
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "center" }}>
                <motion.img
                  src={step.image}
                  alt={step.imageAlt}
                  whileHover={{ scale: 1.05 }}
                  transition={{ duration: 0.3 }}
                  style={{ maxHeight: 260, width: "auto", objectFit: "contain", borderRadius: 16 }}
                />
              </div>
            </motion.div>
          ))}
        </div>

      </main>
    </div>
  );
}
