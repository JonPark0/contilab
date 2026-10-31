import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notes_hold_only_that_versions_changelog_section():
    rn = _module()
    text = (ROOT / "CHANGELOG.md").read_text()
    notes = rn.notes(text, "0.2.0")
    assert "## Added" in notes and "## Fixed" in notes and "### " not in notes
    assert "First public release" not in notes  # the 0.1.0 section
    assert "releases/download/v0.2.0/contilab-0.2.0-py3-none-any.whl" in notes


def test_unknown_version_is_an_error():
    with pytest.raises(SystemExit):
        _module().section("# Changelog\n\n## 0.1.0 — x\n\n- a\n", "9.9.9")


def test_package_version_has_a_changelog_section():
    import tomllib

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert _module().section((ROOT / "CHANGELOG.md").read_text(), version)
