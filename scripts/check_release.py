"""Release readiness checks for cfad v0.1.0."""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    """Single check result row."""

    idx: int
    name: str
    passed: bool
    detail: str


def _check_license() -> tuple[bool, str]:
    path = ROOT / "LICENSE"
    if not path.exists():
        return False, "LICENSE missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    return ("MIT License" in text, "MIT header found" if "MIT License" in text else "MIT header missing")


def _check_citation_orcid() -> tuple[bool, str]:
    path = ROOT / "CITATION.cff"
    if not path.exists():
        return False, "CITATION.cff missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    has_orcid = re.search(r"^\s*orcid\s*:\s*.+$", text, flags=re.MULTILINE) is not None
    return has_orcid, "orcid field found" if has_orcid else "orcid field missing"


def _check_zenodo_json() -> tuple[bool, str]:
    path = ROOT / ".zenodo.json"
    if not path.exists():
        return False, ".zenodo.json missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid JSON: {exc}"
    title = data.get("title")
    return isinstance(title, str) and len(title) > 0, "valid JSON" if isinstance(title, str) and len(title) > 0 else "title missing/invalid"


def _check_readme_installation() -> tuple[bool, str]:
    path = ROOT / "README.md"
    if not path.exists():
        return False, "README.md missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    ok = re.search(r"^##\s+Installation\b", text, flags=re.MULTILINE) is not None
    return ok, "Installation section found" if ok else "Installation section missing"


def _check_paper_front_matter() -> tuple[bool, str]:
    path = ROOT / "paper" / "paper.md"
    if not path.exists():
        return False, "paper/paper.md missing"

    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return False, "YAML front matter not found"
    front = match.group(1)
    has_title = re.search(r"^title\s*:\s*.+$", front, flags=re.MULTILINE) is not None
    return has_title, "title found in YAML" if has_title else "title missing in YAML"


def _check_references_bib_entries() -> tuple[bool, str]:
    path = ROOT / "paper" / "references.bib"
    if not path.exists():
        return False, "paper/references.bib missing"
    text = path.read_text(encoding="utf-8", errors="ignore")
    entries = len(re.findall(r"^\s*@\w+\{", text, flags=re.MULTILINE))
    return entries >= 5, f"{entries} BibTeX entries"


def _check_import_all_modules() -> tuple[bool, str]:
    package_dir = ROOT / "cfad"
    if not package_dir.exists():
        return False, "cfad package directory missing"

    py_files = sorted(
        p
        for p in package_dir.rglob("*.py")
        if "__pycache__" not in p.parts
    )

    failures: list[str] = []
    for path in py_files:
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {exc}")

    if failures:
        preview = "; ".join(failures[:3])
        return False, f"{len(failures)} import failures ({preview})"
    return True, f"{len(py_files)} modules importable"


def _check_pytest_passes() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, "pytest tests/ passed"

    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-6:]
    msg = " | ".join(tail) if tail else "pytest failed"
    return False, msg


def _extract_citation_version() -> str | None:
    path = ROOT / "CITATION.cff"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^\s*version\s*:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return None
    val = m.group(1).strip()
    return val.strip('"\'')


def _check_version_match() -> tuple[bool, str]:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        return False, "pyproject.toml missing"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    py_ver = str(data.get("project", {}).get("version", "")).strip()
    cff_ver = _extract_citation_version()
    if not py_ver or not cff_ver:
        return False, f"version missing (pyproject={py_ver!r}, citation={cff_ver!r})"
    ok = py_ver == cff_ver
    return ok, f"pyproject={py_ver}, CITATION={cff_ver}"


def _check_changelog_contains_version() -> tuple[bool, str]:
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        return False, "CHANGELOG.md missing"

    pyproject = ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    version = str(data.get("project", {}).get("version", "")).strip()
    if not version:
        return False, "project.version missing in pyproject.toml"

    text = changelog.read_text(encoding="utf-8", errors="ignore")
    ok = version in text
    return ok, f"version {version} {'found' if ok else 'not found'} in CHANGELOG.md"


def _print_summary(results: list[CheckResult]) -> None:
    header = f"{'#':<3} {'Check':<62} {'Status':<6} Detail"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        name = (r.name[:59] + "...") if len(r.name) > 62 else r.name
        print(f"{r.idx:<3} {name:<62} {status:<6} {r.detail}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\nSummary:", f"{passed}/{total} checks passed")


def main() -> int:
    checks = [
        ("LICENSE exists", _check_license),
        ("CITATION.cff exists and has orcid field", _check_citation_orcid),
        (".zenodo.json exists and is valid JSON", _check_zenodo_json),
        ("README.md has Installation section", _check_readme_installation),
        ("paper/paper.md has title in YAML front matter", _check_paper_front_matter),
        ("paper/references.bib has at least 5 entries", _check_references_bib_entries),
        ("All cfad Python modules are importable", _check_import_all_modules),
        ("python -m pytest tests/ succeeds", _check_pytest_passes),
        ("pyproject.toml version matches CITATION.cff", _check_version_match),
        ("CHANGELOG.md contains current version", _check_changelog_contains_version),
    ]

    results: list[CheckResult] = []
    for idx, (name, fn) in enumerate(checks, start=1):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"unexpected error: {exc}"
        results.append(CheckResult(idx=idx, name=name, passed=ok, detail=detail))

    _print_summary(results)

    all_ok = all(r.passed for r in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
