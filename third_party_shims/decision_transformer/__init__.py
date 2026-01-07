import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "submodules" / "decision-transformer"
sys.path.insert(0, str(SUB))
