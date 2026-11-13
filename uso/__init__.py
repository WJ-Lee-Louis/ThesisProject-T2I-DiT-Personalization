"""Project wrapper and unchanged official USO namespace."""
from pathlib import Path
__path__.append(str(Path(__file__).parent / 'upstream' / 'uso'))
