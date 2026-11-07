import sys
from pathlib import Path

# Make `import sim` work when pytest is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
