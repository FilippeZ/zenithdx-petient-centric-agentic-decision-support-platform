import sys
import urllib.request
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

for attempt in range(15):
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        data = json.loads(resp.read().decode('utf-8'))
        print(f"[OK] Success on attempt {attempt+1}:", data)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} waiting for server initialization: {e}")
        time.sleep(2)
