import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The experiment scripts import from src/ by bare module name and the
# downloader lives at the repository root; make both importable here.
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
