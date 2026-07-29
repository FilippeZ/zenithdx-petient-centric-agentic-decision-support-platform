import Navbar from "../components/Navbar";
import time from "../assets/time.png";
import settings from "../assets/setting.png";
import money from "../assets/money.png";
import multimodel from "../assets/multimodel.png";
import Ai from "../assets/Ai.png";
import human from "../assets/human.png";
import { motion } from "framer-motion";
import "./sphere.css";

const AboutUs = () => {
  return (
    <div className="bg-[#0d2240] w-full min-h-screen flex flex-col items-center p-5 text-white">
      <Navbar />
      <div className="text-center mt-12 px-4 md:px-10">
        <h1 className="text-3xl md:text-4xl font-bold">About Us</h1>
        <p className="mt-4 text-lg md:text-xl max-w-3xl mx-auto">
          Our AI-powered clinical decision support system enhances the accuracy of diagnoses and personalizes patient care.
        </p>
      </div>

      {/* Features Section with Interactive Spheres */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 mt-12 px-4">
        {[
          { img: time, title: "Accelerate Diagnosis Time" },
          { img: settings, title: "Prevent Errors and Ensure Accuracy" },
          { img: money, title: "Save Costs, Boost Efficiency" },
          { img: multimodel, title: "Multimodal Healthcare Insights" },
          { img: Ai, title: "Explainable AI Transparency" },
          { img: human, title: "Adaptable and Scalable Solutions" },
        ].map((feature, index) => (
          <motion.div
            key={index}
            className="flex flex-col items-center"
            whileHover={{ scale: 1.1, rotate: [0, 5, -5, 0] }}
            whileTap={{ scale: 0.9 }}
            transition={{ duration: 0.4, ease: "easeInOut" }}
          >
            <img src={feature.img} alt={feature.title} className="sphere-image" />
            <h3 className="mt-4 text-lg font-bold">{feature.title}</h3>
          </motion.div>
        ))}
      </div>

      {/* Contact Icons and Info below each icon */}
      <div className="flex flex-row items-center mt-12 space-x-16">
        {/* Phone */}
        <div className="flex flex-col items-center">
          <a
            href="tel:+302105876293"
            className="text-3xl hover:text-gray-400 transition-transform transform hover:scale-125 duration-300"
            title="Call us"
          >
            📞
          </a>
          <div className="text-center text-lg mt-2">
            <span className="font-semibold">Phone:</span>{" "}
            <a href="tel:+302105876293" className="hover:underline">
              +30 210 5876293
            </a>
          </div>
        </div>
        {/* Email */}
        <div className="flex flex-col items-center">
          <a
            href="mailto:info@medicalreview.com"
            className="text-4xl hover:text-gray-400 transition-transform transform hover:scale-125 duration-300"
            title="Email us"
          >
            ✉️
          </a>
          <div className="text-center text-lg mt-2">
            <span className="font-semibold">Email:</span>{" "}
            <a href="mailto:info@medicalreview.com" className="hover:underline">
              info@medicalreview.com
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutUs;
