import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path

# ================================================================
#  ComfyUI-T2V Launcher
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
META_PATH = BASE_DIR / "metadata.json"

# -------------------------------
# Load metadata.json
# -------------------------------
def load_metadata():
    if not META_PATH.exists():
        sys.exit("❌ metadata.json not found! Make sure you're in the project root.")
    with open(META_PATH, "r") as f:
        return json.load(f)

meta = load_metadata()
BACKEND = BASE_DIR / meta["backend"]["path"]
FRONTEND = BASE_DIR / meta["frontend"]["path"]

# -------------------------------
# Ensure Python version
# -------------------------------
def check_python_version():
    import sys
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 10:
        sys.exit("❌ Python 3.10+ required. Please upgrade your environment.")
    print(f"✅ Python {major}.{minor} detected.")

# -------------------------------
# Install dependencies
# -------------------------------
def ensure_dependencies():
    print("🔍 Checking Python dependencies...")
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        required = set(meta["dependencies"])
        missing = [pkg for pkg in required if pkg not in installed]

        if missing:
            print(f"⬇️ Installing missing dependencies: {missing}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        else:
            print("✅ All dependencies already installed.")
    except Exception as e:
        print(f"⚠️ Dependency check failed: {e}")

# -------------------------------
# Import backend and initialize
# -------------------------------
def init_backend():
    print("🧠 Initializing backend...")
    spec = importlib.util.spec_from_file_location("backend_comfyui", BACKEND)
    backend = importlib.util.module_from_spec(spec)
    sys.modules["backend_comfyui"] = backend
    spec.loader.exec_module(backend)
    backend.setup_models(use_q6=False)
    print("✅ Backend initialized.")
    return backend

# -------------------------------
# Launch frontend
# -------------------------------
def launch_frontend():
    print("🚀 Launching Gradio frontend...")
    subprocess.run([sys.executable, str(FRONTEND)], check=True)

# ================================================================
#  Main
# ================================================================

if __name__ == "__main__":
    print(f"🎬 Starting {meta['project_name']} v{meta['version']}")
    check_python_version()
    ensure_dependencies()
    backend = init_backend()
    launch_frontend()
