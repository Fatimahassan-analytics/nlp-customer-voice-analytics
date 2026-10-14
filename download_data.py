"""Build PDFs directory + seed the placeholder document."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data import ensure_dirs, seed_default_pdfs

ensure_dirs()
seed_default_pdfs()
print(f"PDF directory: {PROJECT_ROOT / 'data' / 'raw' / 'pdfs'}")
