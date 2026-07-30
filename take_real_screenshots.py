import os
import time
from selenium import webdriver
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions

out_dir = os.path.join(os.getcwd(), "docs", "images")
os.makedirs(out_dir, exist_ok=True)

PAGES = [
    ("http://localhost:5173/", "landing_page.png"),
    ("http://localhost:5173/login", "auth_portal.png"),
    ("http://localhost:5173/detect", "diagnosis_wizard.png"),
    ("http://localhost:5173/how-to-use-doctor", "doctor_guide.png"),
    ("http://localhost:5173/how-to-use-patient", "patient_guide.png"),
    ("http://localhost:5173/about", "about_zenithdx.png"),
    ("http://localhost:5173/homedoctor", "doctor_workstation.png"),
    ("http://localhost:5173/homepatient", "patient_dashboard.png"),
]

def get_driver():
    try:
        options = EdgeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,900")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Edge(options=options)
        print("[Driver] Initialized headless Edge Driver successfully!")
        return driver
    except Exception as e:
        print(f"[Driver] Edge driver failed: {e}. Trying Chrome driver...")
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,900")
        driver = webdriver.Chrome(options=options)
        return driver

def capture_all():
    driver = get_driver()
    try:
        for url, fname in PAGES:
            print(f"[Capturing] {url} -> {fname}...")
            driver.get(url)
            time.sleep(2.5)  # Allow Framer Motion animations and HMR to settle
            filepath = os.path.join(out_dir, fname)
            driver.save_screenshot(filepath)
            size = os.path.getsize(filepath)
            print(f"  [OK] Saved {fname} ({size} bytes)")
    finally:
        driver.quit()
        print("[Finished] Captured all real page screenshots!")

if __name__ == "__main__":
    capture_all()
