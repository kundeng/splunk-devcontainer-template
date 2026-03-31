#!/usr/bin/env python3
"""Patch CDK imports that break after stripping heavy deps.

CDK 6.x eagerly imports pandas, numpy, serpyco_rs, etc. at init time.
After stripping those from lib/, the import chain breaks. This script
rewrites the specific import sites to stub them out or make them lazy.

Usage: python3 patch_imports.py <lib_dir>
"""

import re
import sys
from pathlib import Path


def patch_file(path: Path, old: str, new: str) -> bool:
    """Replace exact string in file. Returns True if patched."""
    text = path.read_text()
    if old not in text:
        return False
    path.write_text(text.replace(old, new))
    return True


def patch(lib: Path):
    patched = []

    # 1. response_to_file_extractor.py — unconditional pandas + numpy imports
    f = lib / "airbyte_cdk/sources/declarative/extractors/response_to_file_extractor.py"
    if f.exists():
        changed = False
        changed |= patch_file(f,
            "import pandas as pd",
            "try:\n    import pandas as pd\nexcept ImportError:\n    pd = None  # stripped")
        changed |= patch_file(f,
            "from numpy import nan",
            "try:\n    from numpy import nan\nexcept ImportError:\n    nan = float('nan')  # stripped")
        if changed:
            patched.append(str(f.relative_to(lib)))

    # 2. call_rate.py — imports pyrate_limiter which imports sqlite3
    #    and imports requests_cache which imports sqlite3
    f = lib / "airbyte_cdk/sources/streams/call_rate.py"
    if f.exists():
        text = f.read_text()
        replacements = {
            "import requests_cache":
                "try:\n    import requests_cache\nexcept ImportError:\n    requests_cache = None  # stripped",
            "from pyrate_limiter import InMemoryBucket, Limiter, RateItem, TimeClock":
                "try:\n    from pyrate_limiter import InMemoryBucket, Limiter, RateItem, TimeClock\nexcept ImportError:\n    InMemoryBucket = Limiter = RateItem = TimeClock = None  # stripped",
            "from pyrate_limiter import Rate as PyRateRate":
                "try:\n    from pyrate_limiter import Rate as PyRateRate\nexcept ImportError:\n    PyRateRate = None  # stripped",
            "from pyrate_limiter.exceptions import BucketFullException":
                "try:\n    from pyrate_limiter.exceptions import BucketFullException\nexcept ImportError:\n    BucketFullException = Exception  # stripped",
        }
        changed = False
        for old, new in replacements.items():
            if old in text:
                text = text.replace(old, new)
                changed = True
        if changed:
            f.write_text(text)
            patched.append(str(f.relative_to(lib)))

    # 3. jwt/utils.py — already has try/except ModuleNotFoundError but
    #    Splunk's broken cryptography raises ImportError, not ModuleNotFoundError
    f = lib / "jwt/utils.py"
    if f.exists() and patch_file(f,
        "except ModuleNotFoundError:",
        "except (ModuleNotFoundError, ImportError, RuntimeError):"):
        patched.append(str(f.relative_to(lib)))

    # jwt/algorithms.py — same ModuleNotFoundError vs ImportError issue
    f = lib / "jwt/algorithms.py"
    if f.exists() and patch_file(f,
        "except ModuleNotFoundError:",
        "except (ModuleNotFoundError, ImportError, RuntimeError):"):
        patched.append(str(f.relative_to(lib)))

    # 4. serpyco_rs — used by airbyte protocol models
    #    Alias is a metadata annotation: Annotated[T, Alias("name")]
    f = lib / "airbyte_cdk/models/airbyte_protocol.py"
    if f.exists() and patch_file(f,
        "from serpyco_rs.metadata import Alias",
        "try:\n    from serpyco_rs.metadata import Alias\n"
        "except ImportError:\n"
        "    class Alias:\n"
        "        def __init__(self, name): self.name = name"):
        patched.append(str(f.relative_to(lib)))

    # 5. serpyco_rs in airbyte_protocol_serializers — needs CustomType + Serializer stubs
    f = lib / "airbyte_cdk/models/airbyte_protocol_serializers.py"
    if f.exists() and patch_file(f,
        "from serpyco_rs import CustomType, Serializer",
        "try:\n"
        "    from serpyco_rs import CustomType, Serializer\n"
        "except ImportError:\n"
        "    import typing as _t\n"
        "    _T1 = _t.TypeVar('_T1'); _T2 = _t.TypeVar('_T2')\n"
        "    class CustomType(_t.Generic[_T1, _T2]):\n"
        "        pass\n"
        "    class Serializer:\n"
        "        def __init__(self, *a, **kw): pass\n"
        "        def dump(self, obj): return obj\n"
        "        def load(self, data): return data"):
        patched.append(str(f.relative_to(lib)))

    # 6. google.protobuf — used by CDK models
    f = lib / "airbyte_cdk/models/__init__.py"
    if f.exists():
        text = f.read_text()
        if "from .airbyte_protocol import" in text and "try:" not in text[:100]:
            # The top-level model imports should already work since we keep
            # the pydantic-based models. Only patch if protobuf is missing.
            pass

    for p in patched:
        print(f"  patched: {p}")

    if not patched:
        print("  (no patches needed)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <lib_dir>")
        sys.exit(1)
    lib = Path(sys.argv[1])
    if not lib.exists():
        print(f"ERROR: {lib} does not exist")
        sys.exit(1)
    patch(lib)
