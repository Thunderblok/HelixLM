"""
HF Job wrapper for HelixLM 5M optimal smoke test.
Runs the optimal config from regression fix ablations.
"""
import os, subprocess, sys, time

# Install git
print("Installing git...")
subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["apt-get", "install", "-y", "-qq", "git"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Clone repo
REPO_URL = "https://github.com/david-thrower/HelixLM.git"
REPO_DIR = "/tmp/HelixLM"
print(f"Cloning {REPO_URL}...")
subprocess.run(["git", "clone", "-b", "agent-2026-05-14-final-prod-prep", REPO_URL, REPO_DIR], check=True)

# Install deps
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "datasets", "accelerate", "tqdm"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Run optimal 5M smoke test
print("Starting 5M optimal smoke test...")
os.chdir(REPO_DIR)
start = time.time()
result = subprocess.run(
    [sys.executable, "launch_helixlm_5m_smoke_optimal.py",
     "--push_to_hub",
     "--hub_model_id", "david-thrower/helixlm-5m-optimal",
     "--output_dir", "./checkpoints_5m_optimal"],
    capture_output=False
)
elapsed = time.time() - start
print(f"\nCompleted in {elapsed/60:.1f} minutes. Exit code: {result.returncode}")
sys.exit(result.returncode)
