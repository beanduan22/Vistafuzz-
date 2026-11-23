from __future__ import annotations

import io
import statistics
import tokenize
from pathlib import Path

from .native import LIBRARIES


def source_lines(path):
    source = Path(path).read_text()
    ignored = {tokenize.ENCODING, tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
               tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    occupied = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in ignored:
            occupied.update(range(token.start[0], token.end[0] + 1))
    return sum(index in occupied and bool(line.strip())
               for index, line in enumerate(source.splitlines(), 1))


def measure():
    directory = Path(__file__).parent
    counts = {name: source_lines(directory / ("native_" + name + ".py"))
              for name in LIBRARIES if name != "cv2"}
    return {"definition": "nonblank, non-comment physical source lines",
            "scope": "dedicated modules for the eleven libraries added beyond OpenCV",
            "libraries": counts, "median": statistics.median(counts.values()),
            "min": min(counts.values()), "max": max(counts.values()),
            "shared_excluded": {name: source_lines(directory / name)
                                for name in ("native_base.py", "native.py")}}
