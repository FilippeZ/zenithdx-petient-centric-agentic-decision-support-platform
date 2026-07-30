import React from "react";
import Navbar from "../components/Navbar";
import time from "../assets/time.png";
import settings from "../assets/setting.png";
import money from "../assets/money.png";
import multimodel from "../assets/multimodel.png";
import Ai from "../assets/Ai.png";
import human from "../assets/human.png";
import { motion } from "framer-motion";

const features = [
  {
    img: time,
    title: "Accelerate Diagnosis Time",
    description: "Rapid multi-modal analysis reduces diagnostic turnaround time from hours to seconds.",
    badge: "Speed & Agility"
  },
  {
    img: settings,
    title: "Prevent Errors and Ensure Accuracy",
    description: "Deep learning models cross-verify radiological & clinical findings to minimize diagnostic missteps.",
    badge: "Precision AI"
  },
  {
    img: money,
    title: "Save Costs, Boost Efficiency",
    description: "Automated workflow triage reduces hospital resource strain and streamlines clinical decision-making.",
    badge: "Resource Optimization"
  },
  {
    img: multimodel,
    title: "Multimodal Healthcare Insights",
    description: "Seamless fusion of X-ray vision analysis, longitudinal EHR records, and literature RAG retrieval.",
    badge: "360° Data Fusion"
  },
  {
    img: Ai,
    title: "Explainable AI Transparency",
    description: "GradCAM visual heatmaps and text saliency attributions ensure full clinician transparency.",
    badge: "Trust & XAI"
  },
  {
    img: human,
    title: "Adaptable and Scalable Solutions",
    description: "Interoperable architecture designed to seamlessly integrate across clinical enterprise workflows.",
    badge: "Enterprise Ready"
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12,
      delayChildren: 0.2
    },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 35, scale: 0.96 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
  },
};

const AboutUs = () => {
  return (
    <div style={{ background: "#f8fafc", minHeight: "100vh", color: "#0f172a", fontFamily: "'Inter', sans-serif" }}>
      <Navbar />

      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "3rem 1.5rem 6rem" }}>
        
        {/* Animated Hero Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          style={{ textAlign: "center", marginBottom: "4rem" }}
        >
          {/* Animated Badge */}
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
              boxShadow: "0 2px 10px rgba(37, 99, 235, 0.08)"
            }}
          >
            <motion.span
              animate={{ scale: [1, 1.3, 1], opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 2, repeat: Infinity }}
              style={{ width: 7, height: 7, borderRadius: "50%", background: "#2563eb", display: "inline-block" }}
            />
            NEXT-GEN CLINICAL DECISION SUPPORT
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.6 }}
            style={{ fontSize: "clamp(2.2rem, 4vw, 3.2rem)", fontWeight: 900, letterSpacing: "-0.03em", color: "#0f172a", marginBottom: "1rem" }}
          >
            About ZenithDx
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            style={{ color: "#64748b", fontSize: "1.1rem", maxWidth: "680px", margin: "0 auto", lineHeight: 1.7 }}
          >
            Empowering healthcare clinicians and patients through transparent, explainable, multi-modal AI intelligence designed to accelerate diagnostic decision-making.
          </motion.p>
        </motion.div>

        {/* Feature Cards Grid with Stagger & Motion */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: "2rem",
          }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              whileHover={{ y: -8, scale: 1.02, boxShadow: "0 16px 35px -5px rgba(37, 99, 235, 0.12)", borderColor: "#93c5fd" }}
              transition={{ duration: 0.3 }}
              style={{
                background: "#ffffff",
                border: "1px solid #e2e8f0",
                borderRadius: "20px",
                padding: "2.2rem",
                boxShadow: "0 4px 15px -3px rgba(0,0,0,0.03)",
                display: "flex", flexDirection: "column", justifyContent: "space-between",
                cursor: "default",
                transition: "box-shadow 0.3s ease, border-color 0.3s ease"
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }}>
                  {/* Animated Icon Container */}
                  <motion.div
                    whileHover={{ rotate: 10, scale: 1.15 }}
                    transition={{ duration: 0.3 }}
                    style={{
                      width: 54, height: 54, borderRadius: "14px",
                      background: "#eff6ff", border: "1px solid #bfdbfe",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      boxShadow: "0 4px 12px rgba(37, 99, 235, 0.1)"
                    }}
                  >
                    <img src={feature.img} alt={feature.title} style={{ width: 32, height: 32, objectFit: "contain" }} />
                  </motion.div>

                  {/* Badge */}
                  <motion.span
                    whileHover={{ scale: 1.08, backgroundColor: "#dbeafe", color: "#1d4ed8" }}
                    style={{
                      fontSize: "0.7rem", fontWeight: 800, color: "#2563eb",
                      background: "#f0f9ff", border: "1px solid #bae6fd",
                      padding: "0.3rem 0.75rem", borderRadius: "100px", letterSpacing: "0.04em",
                      transition: "all 0.2s ease"
                    }}
                  >
                    {feature.badge}
                  </motion.span>
                </div>

                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, color: "#0f172a", marginBottom: "0.6rem" }}>
                  {feature.title}
                </h3>

                <p style={{ color: "#475569", fontSize: "0.92rem", lineHeight: 1.65, margin: 0 }}>
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </motion.div>

      </main>
    </div>
  );
};

export default AboutUs;
