"""Check that every third-party package the repository imports is in requirements.txt.

Owner: Muhammad Abdullah Saeed (plan Section 6.5: "Update requirements.txt if any
dependency was missed").

It reads every .py file in src/, scripts/, tests/ and data_scripts/, collects the imported
packages, ignores the standard library and the repository's own modules, and compares the
rest with requirements.txt. Exit code 1 means a package is imported but not listed.

Usage (repository root):
    python scripts/check_requirements.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable

SOURCE_DIRS = ("src", "scripts", "tests", "data_scripts")
# import name -> pip name, for packages whose two names differ
IMPORT_TO_PIP = {
    "yaml": "PyYAML", "sklearn": "scikit-learn", "PIL": "Pillow", "cv2": "opencv-python",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "dateutil": "python-dateutil",
}


def normalise(name: str) -> str:
    """pip's name comparison: case-insensitive, '-', '_' and '.' are equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "git+", "http")):
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        if match:
            names.add(normalise(match.group(0)))
    return names


def local_modules(root: Path) -> set[str]:
    """Names that resolve to the repository's own code (packages, folders and script files)."""
    names = set(SOURCE_DIRS)
    for folder in SOURCE_DIRS:
        base = root / folder
        if not base.is_dir():
            continue
        for item in base.rglob("*"):
            if item.suffix == ".py":
                names.add(item.stem)
            elif item.is_dir() and (item / "__init__.py").exists():
                names.add(item.name)
    return names


def imported_packages(path: Path) -> set[str]:
    """Top-level names imported anywhere in one Python file (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def check(root: Path) -> tuple[dict[str, list[str]], set[str]]:
    """Return ({missing pip name: [files that import it]}, {listed but never imported})."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is None:
        raise RuntimeError("Python 3.10 or newer is required for this check")
    requirements = parse_requirements(root / "requirements.txt")
    local = local_modules(root)

    used: dict[str, list[str]] = {}
    for folder in SOURCE_DIRS:
        for path in sorted((root / folder).rglob("*.py")) if (root / folder).is_dir() else []:
            for name in imported_packages(path):
                if name in stdlib or name in local or name == "__future__":
                    continue
                used.setdefault(name, []).append(path.relative_to(root).as_posix())

    missing = {}
    for name, files in used.items():
        pip_name = IMPORT_TO_PIP.get(name, name)
        if normalise(pip_name) not in requirements:
            missing[pip_name] = files
    imported_pip = {normalise(IMPORT_TO_PIP.get(n, n)) for n in used}
    unused = requirements - imported_pip
    return missing, unused


def main(argv: Iterable[str] = ()) -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        missing, unused = check(root)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if missing:
        print("Imported but NOT in requirements.txt:")
        for name, files in sorted(missing.items()):
            shown = ", ".join(files[:3]) + (" ..." if len(files) > 3 else "")
            print(f"  [FAIL] {name}  (used in {shown})")
    else:
        print("[PASS] every imported third-party package is listed in requirements.txt")
    if unused:
        print("Listed but never imported by .py files (fine for command-line tools): "
              + ", ".join(sorted(unused)))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
