# backend/test_usecase.py
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from run_all_usecases import main

if __name__ == "__main__":
    main()
