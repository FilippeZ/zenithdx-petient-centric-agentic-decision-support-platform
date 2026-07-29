import { Navigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";

const ProtectedRoute = ({ element, roleRequired }) => {
    const token = localStorage.getItem("token");

    if (!token) {
        // Redirect to home if not logged in
        return <Navigate to="/" replace />;
    }

    try {
        const decodedToken = jwtDecode(token);
        const userRole = decodedToken.role;

        // Ensure users can only access their designated routes
        if (roleRequired && userRole !== roleRequired) {
            console.warn(`Unauthorized access attempt by ${userRole}`);
            return <Navigate to="/" replace />;
        }

        return element;
    } catch (error) {
        console.error("Invalid token:", error);
        localStorage.removeItem("token");
        // Redirect to home if token is invalid
        return <Navigate to="/" replace />;
    }
};

export default ProtectedRoute;
