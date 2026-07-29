import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import homedoctor from "../assets/homedoctor.png";
import { jwtDecode } from "jwt-decode";
import { FaUser, FaEnvelope, FaLock } from "react-icons/fa";

const Home = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const response = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ username, password }),
      });

      const data = await response.json();

      if (response.ok) {
        const { access_token } = data;
        localStorage.setItem("token", access_token);

        // Decode JWT to get user role
        const decodedToken = jwtDecode(access_token);
        const userRole = decodedToken.role; // Backend sends role inside JWT

        // Store user role in localStorage
        localStorage.setItem("user_role", userRole);

        // Redirect based on role
        if (userRole === "doctor") {
          navigate("/homedoctor");
        } else {
          navigate("/patient-dashboard");
        }
      } else {
        alert(data.detail || "Login failed");
      }
    } catch (error) {
      console.error("Login error:", error);
    }
  };

  const handleRegister = async () => {
    setError("");

    if (!username || !email || !password || !confirmPassword) {
      setError("All fields are required.");
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }

    try {
      const response = await fetch("http://localhost:8000/register", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          full_name: username,
          username,
          email,
          password,
          user_type: "patient",
        }),
      });

      const data = await response.json();

      if (response.ok) {
        alert("Registration successful! Please log in.");
        setIsLogin(true);
        // Μένει στη login φόρμα
      } else {
        setError(data.error || "Registration failed.");
      }
    } catch (error) {
      console.error("Registration error:", error);
      setError("An error occurred during registration.");
    }
  };

  return (
    <div className="bg-[#0d2240] w-full min-h-screen flex flex-col items-center p-5 overflow-hidden">
      <Navbar />
      <div className="flex flex-col lg:flex-row items-center justify-between w-full flex-1 mt-12 px-4 md:px-10">
        <div className="flex flex-col lg:flex-row items-center lg:items-start justify-between min-h-screen">
          {/* Text Section */}
          <div className="text-white text-center lg:text-left max-w-lg mb-10 lg:mb-0 lg:mr-10">
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold">
              {isLogin ? "Sign In to" : "Register to"}
            </h1>
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold whitespace-nowrap">
              begin your Healthcare Journey
            </h2>
          </div>

          {/* Image Section */}
          <div className="w-full lg:w-auto flex justify-center lg:justify-end absolute bottom-0 lg:bottom-0 left-1/2 lg:left-auto transform -translate-x-1/2 lg:translate-x-0">
            <img
              src={homedoctor}
              alt="Healthcare Journey"
              className="w-auto max-w-md lg:max-w-4xl h-auto lg:h-[64vh] object-cover"
            />
          </div>
        </div>

        <div className="p-8 md:p-10 rounded-lg shadow-lg w-full max-w-md">
          <h3 className="text-white text-2xl md:text-3xl font-bold mb-6">
            {isLogin ? "Welcome Back" : "Create Account"}
          </h3>

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-100 text-red-700 rounded-lg">
              {error}
            </div>
          )}

          {/* Username Input */}
          <div className="mb-6 relative">
            <input
              type="text"
              placeholder="Enter Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-white text-black p-4 pl-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
            <span className="absolute left-4 top-5 text-gray-400">
              <FaUser className="h-5 w-5" />
            </span>
          </div>

          {/* Email Input (only for register) */}
          {!isLogin && (
            <div className="mb-6 relative">
              <input
                type="email"
                placeholder="Enter Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full bg-white text-black p-4 pl-12 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
              <span className="absolute left-4 top-5 text-gray-400">
                <FaEnvelope className="h-5 w-5" />
              </span>
            </div>
          )}

          {/* Password Input */}
          <div className="mb-6 relative">
            <input
              type="password"
              placeholder="Enter Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-white p-4 pl-12 text-black border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
            <span className="absolute left-4 top-5 text-gray-400">
              <FaLock className="h-5 w-5" />
            </span>
          </div>

          {/* Confirm Password Input (only for register) */}
          {!isLogin && (
            <div className="mb-6 relative">
              <input
                type="password"
                placeholder="Confirm Password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full bg-white p-4 pl-12 text-black border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
              <span className="absolute left-4 top-5 text-gray-400">
                <FaLock className="h-5 w-5" />
              </span>
            </div>
          )}

          <button
            className="w-full bg-blue-600 text-white p-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition duration-300"
            onClick={isLogin ? handleLogin : handleRegister}
          >
            {isLogin ? "Sign In" : "Register"}
          </button>

          <div className="mt-4 text-center">
            <span className="text-white text-sm">
              {isLogin
                ? "Don't have an account? "
                : "Already have an account? "}
              <button
                onClick={() => {
                  setIsLogin(!isLogin);
                  setError(""); // Clear error when switching forms
                }}
                className="text-blue-300 hover:underline focus:outline-none"
              >
                {isLogin ? "Register" : "Sign In"}
              </button>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
