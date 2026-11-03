import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _module():
    spec = importlib.util.spec_from_file_location("hf_dataset", ROOT / "scripts" / "hf_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_card_metadata_lists_every_csv(tmp_path):
    mod = _module()
    mod.build(str(tmp_path))
    text = (tmp_path / "README.md").read_text()
    meta = yaml.safe_load(text.split("---\n")[1])
    names = [c["config_name"] for c in meta["configs"]]
    assert names == [n for n, _, _ in mod.FILES] and meta["license"] == "mit"
    for n in names:
        assert (tmp_path / f"{n}.csv").read_bytes() == (ROOT / "docs/assets/results/data" / f"{n}.csv").read_bytes()
        assert f"| `{n}.csv` | `{n}` | {len(mod.read(n))} |" in text


def test_card_table_follows_the_data():
    mod = _module()
    table = mod.overview(mod.read("cpu_main"))
    assert table.count("\n") == 1 + len({r["method"] for r in mod.read("cpu_main")})
    assert "| Naive |" in table and "| DER++ |" in table
