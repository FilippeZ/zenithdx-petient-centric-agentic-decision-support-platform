import os
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

out_dir = os.path.join(os.getcwd(), "docs", "images")
os.makedirs(out_dir, exist_ok=True)

def get_driver():
    try:
        options = EdgeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Edge(options=options)
        print("[Driver] Initialized 1920x1080 Headless Edge Driver OK.")
        return driver
    except Exception as e:
        print(f"[Driver] Edge failed ({e}), falling back to Chrome...")
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        return driver

def capture_pages():
    driver = get_driver()
    try:
        # 1. Landing Page
        print("[Capturing] 1. Landing Page (http://localhost:5173/)...")
        driver.get("http://localhost:5173/")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "landing_page.png"))

        # 2. Auth / Login Portal (/auth)
        print("[Capturing] 2. Auth Portal (http://localhost:5173/auth)...")
        driver.get("http://localhost:5173/auth")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "auth_portal.png"))

        # 3. About Page (/about-us)
        print("[Capturing] 3. About Page (http://localhost:5173/about-us)...")
        driver.get("http://localhost:5173/about-us")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "about_zenithdx.png"))

        # 4. Doctor User Guide (/how-to-use-doctor with doctor role)
        print("[Capturing] 4. Doctor Guide (http://localhost:5173/how-to-use-doctor)...")
        driver.get("http://localhost:5173/how-to-use-doctor")
        driver.execute_script("localStorage.setItem('user_role', 'doctor'); localStorage.setItem('token', 'mock_token');")
        driver.get("http://localhost:5173/how-to-use-doctor")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "doctor_guide.png"))

        # 5. Doctor Workstation Queue (/homedoctor with doctor role)
        print("[Capturing] 5. Doctor Workstation (http://localhost:5173/homedoctor)...")
        driver.get("http://localhost:5173/homedoctor")
        driver.execute_script("localStorage.setItem('user_role', 'doctor'); localStorage.setItem('token', 'mock_token');")
        driver.get("http://localhost:5173/homedoctor")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "doctor_workstation.png"))

        # 6. Patient User Guide (/how-to-use-patient with patient role)
        print("[Capturing] 6. Patient Guide (http://localhost:5173/how-to-use-patient)...")
        driver.get("http://localhost:5173/how-to-use-patient")
        driver.execute_script("localStorage.setItem('user_role', 'patient'); localStorage.setItem('token', 'mock_token');")
        driver.get("http://localhost:5173/how-to-use-patient")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "patient_guide.png"))

        # 7. Diagnosis Wizard (/detect with patient role)
        print("[Capturing] 7. Diagnosis Wizard (http://localhost:5173/detect)...")
        driver.get("http://localhost:5173/detect")
        driver.execute_script("localStorage.setItem('user_role', 'patient'); localStorage.setItem('token', 'mock_token');")
        driver.get("http://localhost:5173/detect")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "diagnosis_wizard.png"))

        # 8. Patient Dashboard (/patient-dashboard with patient role)
        print("[Capturing] 8. Patient Dashboard (http://localhost:5173/patient-dashboard)...")
        driver.get("http://localhost:5173/patient-dashboard")
        driver.execute_script("localStorage.setItem('user_role', 'patient'); localStorage.setItem('token', 'mock_token');")
        driver.get("http://localhost:5173/patient-dashboard")
        time.sleep(3)
        driver.save_screenshot(os.path.join(out_dir, "patient_dashboard.png"))

        # Verify sizes
        print("\n--- FINAL SCREENSHOT SIZES ---")
        for fname in ["landing_page.png", "auth_portal.png", "diagnosis_wizard.png", "doctor_guide.png", "patient_guide.png", "about_zenithdx.png", "doctor_workstation.png", "patient_dashboard.png"]:
            path = os.path.join(out_dir, fname)
            sz = os.path.getsize(path)
            print(f"  -> {fname}: {sz} bytes")

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_pages()
