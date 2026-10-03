import subprocess
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
subprocess.run([sys.executable, str(root / 'scripts/audit_research_history.py'), '--claim', 'C04'], cwd=root, check=True)
