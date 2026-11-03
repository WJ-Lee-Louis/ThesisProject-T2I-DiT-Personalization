"""Project wrapper plus unchanged official UNO namespace."""
from pathlib import Path
__path__.append(str(Path(__file__).parent / "upstream" / "uno"))
