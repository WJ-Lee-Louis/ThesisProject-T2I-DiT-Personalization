"""Benchmark integration; official modules remain byte-identical in upstream/."""
from pathlib import Path
__path__.append(str(Path(__file__).parent / "upstream/dreamo"))
