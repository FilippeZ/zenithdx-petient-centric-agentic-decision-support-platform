import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { useParams, useNavigate } from "react-router-dom";
import { FaFilePdf, FaStar, FaNotesMedical } from "react-icons/fa";
import Navbar from "../components/Navbar";
import ladydoc from "../assets/ladydoc.png";

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

        // Εμφάνισε το αποθηκευμένο rating αν υπάρχει
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
    if (submitted) return; // Αποτρέπει αλλαγή rating αν υπάρχει ήδη
    setRating(index + 1);
    submitFeedback(index + 1);
  };

  if (loading) {
    return (
      <div className="bg-[#0D2240] w-full min-h-screen flex flex-col items-center p-5 text-white">
        <Navbar />
        <div className="mt-6 w-full max-w-6xl text-center">
          <p>Loading report details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-[#0D2240] w-full min-h-screen flex flex-col items-center p-5 text-white">
        <Navbar />
        <div className="mt-6 w-full max-w-6xl text-center text-red-300">
          <p>{error}</p>
          <button
            onClick={() => navigate(-1)}
            className="mt-4 bg-blue-500/20 text-blue-300 px-4 py-2 rounded-md hover:bg-blue-500/30"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="bg-[#0D2240] w-full min-h-screen flex flex-col items-center p-5 text-white">
        <Navbar />
        <div className="mt-6 w-full max-w-6xl text-center">
          <p>No report found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#0D2240] w-full min-h-screen flex flex-col items-center p-5 text-white">
      <Navbar />
      <div className="mt-6 w-full max-w-4xl relative">
        {/* Diagnosis Report */}
        <div className="relative bg-gradient-to-br from-white via-gray-100 to-purple-100 text-black w-full p-12 rounded-3xl shadow-xl min-h-[420px]">
          <span className="absolute top-[-12px] left-6 bg-purple-700 text-white px-6 py-2 text-lg rounded-full shadow-lg flex items-center gap-3 font-bold">
            <FaNotesMedical /> Diagnosis Report
          </span>
          <div className="mt-14 text-2xl font-semibold overflow-auto max-h-[600px] prose prose-lg text-center">
            <ReactMarkdown>{report.diagnosis?.trim()}</ReactMarkdown>
          </div>
        </div>

        {/* Review & Rating Section */}
        <div className="relative bg-white/90 text-black w-full p-6 rounded-3xl min-h-[180px] flex flex-col justify-center items-center shadow-xl border-2 border-blue-100 mt-10">
          <span className="absolute top-[-12px] left-4 bg-blue-900 text-white px-4 py-1 text-sm rounded-full shadow-lg font-bold">
            Rate the Report
          </span>
          <div className="flex flex-col items-center justify-center w-full mt-4">
            {submitted ? (
              <div className="text-green-600 font-bold text-lg text-center">
                Thank you for your feedback!
                <div className="flex gap-1 justify-center mt-2">
                  {[...Array(5)].map((_, idx) => (
                    <FaStar
                      key={idx}
                      className={
                        idx < rating
                          ? "text-yellow-400 text-3xl"
                          : "text-gray-300 text-3xl"
                      }
                    />
                  ))}
                </div>
                <p className="text-gray-700 mt-2 text-sm">
                  You rated this report with {rating} star{rating !== 1 && "s"}.
                </p>
              </div>
            ) : (
              <>
                <div className="flex gap-1 text-yellow-400 text-3xl mt-2">
                  {[...Array(5)].map((_, index) => (
                    <FaStar
                      key={index}
                      onClick={() => handleStarClick(index)}
                      className={`cursor-pointer transition-transform ${
                        index < rating ? "text-yellow-400 scale-125" : "text-gray-300"
                      }`}
                    />
                  ))}
                </div>
                <p className="text-gray-700 mt-2 text-sm">
                  Click to rate your experience!
                </p>
              </>
            )}
          </div>
        </div>

        {/* Doctor's Message & Image */}
        <div className="relative flex flex-row items-center gap-8 w-full min-h-60 mt-10 md:mt-0">
          <div className="bg-white rounded-3xl shadow-2xl flex items-center justify-center w-[180px] h-[230px] border-4 border-white z-20">
            <img
              src={ladydoc}
              alt="Doctor"
              className="w-[145px] h-[200px] object-cover rounded-2xl"
              style={{ objectPosition: "top" }}
            />
          </div>
          <div
            className="relative bg-gradient-to-tr from-blue-50 via-white to-[#f3f7ff] text-black rounded-3xl px-10 py-8 shadow-2xl text-base font-medium min-w-[200px] max-w-[430px] border border-blue-100 flex flex-col justify-center"
          >
            <span className="absolute -top-4 left-8 bg-purple-700 text-white px-4 py-1 text-xs rounded-full shadow-lg font-bold flex items-center gap-2 select-none"
              style={{ letterSpacing: ".5px" }}>
              Doctor's Message
            </span>
            <div className="mt-7 text-base leading-relaxed min-h-[36px]">
              {report.doctor_message ? (
                <ReactMarkdown>{report.doctor_message.trim()}</ReactMarkdown>
              ) : (
                <span className="italic text-gray-400">No message provided.</span>
              )}
            </div>
            <div
              className="absolute left-[-23px] top-1/2 -translate-y-1/2"
              style={{
                width: "0",
                height: "0",
                borderTop: "22px solid transparent",
                borderBottom: "22px solid transparent",
                borderRight: "23px solid #f3f7ff",
                filter: "drop-shadow(0 0 3px #d1d5db)",
              }}
            ></div>
          </div>
        </div>
      </div>

      {/* Download Button */}
      <div className="flex justify-center mt-10">
        <button
          onClick={downloadPDF}
          className="flex items-center gap-2 bg-red-600 text-white px-8 py-3 rounded-xl shadow-lg text-lg font-semibold hover:bg-red-700 transition-all"
        >
          <FaFilePdf className="text-2xl" /> Download PDF
        </button>
      </div>
    </div>
  );
};

export default ResultPatient;
