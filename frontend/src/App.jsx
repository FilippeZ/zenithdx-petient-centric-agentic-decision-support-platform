import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingPage   from "./pages/LandingPage";
import AuthPage      from "./pages/AuthPage";
import HomeDoctor    from "./pages/HomeDoctor";
import AboutPage     from "./pages/AboutPage";
import HowToUseDoctor  from "./pages/HowToUseDoctor";
import Detect        from "./pages/Detect";
import Reports       from "./pages/Reports";
import ResultPatient from "./pages/ResultPatient";
import HomePatient   from "./pages/HomePatient";
import ProtectedRoute from "./components/ProtectedRoute";
import HowToUsePatient from "./pages/HowToUsePatient";

const App = () => {
  return (
    <Router>
      <Routes>
        {/* Public */}
        <Route path="/"         element={<LandingPage />} />
        <Route path="/auth"     element={<AuthPage />} />
        <Route path="/about-us" element={<AboutPage />} />

        {/* Protected — Doctor */}
        <Route path="/homedoctor"       element={<ProtectedRoute element={<HomeDoctor />}      roleRequired="doctor" />} />
        <Route path="/how-to-use-doctor" element={<ProtectedRoute element={<HowToUseDoctor />} roleRequired="doctor" />} />
        <Route path="/reports/:reportId" element={<ProtectedRoute element={<Reports />}         roleRequired="doctor" />} />

        {/* Protected — Patient */}
        <Route path="/patient-dashboard"          element={<ProtectedRoute element={<HomePatient />}   roleRequired="patient" />} />
        <Route path="/how-to-use-patient"          element={<ProtectedRoute element={<HowToUsePatient />} roleRequired="patient" />} />
        <Route path="/detect"                      element={<ProtectedRoute element={<Detect />}        roleRequired="patient" />} />
        <Route path="/patient/reports/:reportId"   element={<ProtectedRoute element={<ResultPatient />} roleRequired="patient" />} />
      </Routes>
    </Router>
  );
};

export default App;
