"""
HF Job wrapper for HelixLM 5M optimal smoke test.
Clones repo, installs deps, runs optimal config.
"""
import os, subprocess, sys

# Install git if not present
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(["apt-get", "install", "-y", "-qq", "git"], check=False)

# Clone repo
REPO_URL = "https://github.com/david-thrower/HelixLM.git"
REPO_DIR = "/tmp/HelixLM"
subprocess.run(["git", "clone", "-b", "agent-2026-05-14-final-prod-prep", REPO_URL, REPO_DIR], check=True)

# Install deps
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "datasets", "accelerate", "tqdm"], check=False)

# Run optimal 5M smoke test
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "launch_helixlm_5m_smoke_optimal.py",
                "--push_to_hub",
                "--hub_model_id", "david-thrower/helixlm-5m-optimal",
                "--output_dir", "./checkpoints_5m_optimal"], check=True)
