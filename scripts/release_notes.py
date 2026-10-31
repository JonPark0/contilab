"""Print GitHub release notes for one version from CHANGELOG.md (used by .github/workflows/release.yml).

Usage: python scripts/release_notes.py 0.2.0 [CHANGELOG.md]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = "https://github.com/JonPark0/contilab"


def section(changelog: str, version: str) -> str:
    """Body of the `## <version> …` section, with `###` headings lifted to `##`."""
    m = re.search(rf"^## {re.escape(version)}\b.*?$\n(.*?)(?=^## |\Z)", changelog, flags=re.M | re.S)
    if m is None:
        raise SystemExit(f"CHANGELOG has no section for {version}")
    return re.sub(r"^### ", "## ", m.group(1).strip(), flags=re.M)


def notes(changelog: str, version: str) -> str:
    return "\n".join([
        section(changelog, version),
        "",
        f"Results and documentation: [wiki]({REPO}/wiki) · [Results]({REPO}/wiki/Results)",
        "",
        "## Install",
        "",
        "```bash",
        f"pip install {REPO}/releases/download/v{version}/contilab-{version}-py3-none-any.whl",
        f"# or from source: git clone --branch v{version} {REPO} && pip install -e ./contilab",
        "```",
        "",
    ])


if __name__ == "__main__":
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    print(notes(path.read_text(), sys.argv[1]))
