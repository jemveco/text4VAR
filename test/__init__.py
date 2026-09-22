import sys
from pathlib import Path


# Navigate up one directory level from the current file
parent_dir = Path(__file__).resolve().parent.parent
# Add the parent directory to the search path
sys.path.append(str(parent_dir))

