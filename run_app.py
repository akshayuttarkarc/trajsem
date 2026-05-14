import subprocess
import sys
from pathlib import Path

APP = Path(__file__).parent / "app.py"

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP)]))
