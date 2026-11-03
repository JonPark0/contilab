"""Builds the Hugging Face dataset of the CPU study (docs/assets/results/data/*.csv plus a dataset card) and,
with --upload, publishes it as jonpark0/contilab-cpu-study. Run by .github/workflows/hf-dataset.yml.

    python3 scripts/hf_dataset.py --out hf             # write hf/README.md and copy the CSVs, for a look
    python3 scripts/hf_dataset.py --out hf --upload    # and publish them (HF_TOKEN with write access)

The card is generated from the CSVs, pyproject.toml and CITATION.cff, so its numbers follow the data. Building it
needs only the standard library; --upload needs huggingface_hub.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from statistics import mean

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "docs", "assets", "results", "data")
REPO_ID = "jonpark0/contilab-cpu-study"
GITHUB = "https://github.com/JonPark0/contilab"
SITE = "https://jonpark0.github.io/contilab/"

FILES = [  # (config name, description, experiment configs)
    ("cpu_main", "Main study: every method with default hyperparameters on 6 benchmarks",
     "`cpu_main.yaml` + `cpu_main_seeds.yaml`"),
    ("cpu_tuned", "Tuned comparison: per-method grids on the two MNIST benchmarks, variant chosen on validation "
     "accuracy (`selected`)", "`cpu_tuned.yaml`"),
    ("cpu_student", "Teacher → smaller student: distillation into a 2×100 MLP", "`cpu_student.yaml`"),
]
PRESETS = {"split_mnist_class": "Split MNIST (Class-IL)", "split_mnist_task": "Split MNIST (Task-IL)",
           "permuted_mnist": "Permuted MNIST (Domain-IL)", "split_cifar10_class": "Split CIFAR-10 (Class-IL)",
           "split_cifar100_class": "Split CIFAR-100 (Class-IL)", "text5_class": "text5 + Pythia-70M (Class-IL)"}
NAMES = {"naive": "Naive", "joint": "Joint (upper bound)", "ewc": "EWC", "lwf": "LwF", "er": "Experience Replay",
         "derpp": "DER++", "agem": "A-GEM", "lora_per_task": "LoRA-per-task", "consolidate:ta": "Task Arithmetic",
         "consolidate:ties": "TIES-Merging", "consolidate:distill": "Multi-teacher distillation",
         "consolidate:continual_distill": "Continual distillation"}


def read(name: str) -> list[dict]:
    with open(os.path.join(DATA, f"{name}.csv"), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def field(path: str, pattern: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        m = re.search(pattern, f.read(), re.M)
    return m.group(1) if m else ""


def overview(rows: list[dict]) -> str:
    """Mean final accuracy per method and benchmark; a benchmark where every seed diverged reads "diverged"."""
    presets = [p for p in PRESETS if any(r["preset"] == p for r in rows)]
    methods = [m for m in NAMES if any(r["method"] == m for r in rows)]
    out = ["| Method | " + " | ".join(PRESETS[p] for p in presets) + " |", "|---|" + "---|" * len(presets)]
    for m in methods:
        cells = []
        for p in presets:
            sel = [r for r in rows if r["method"] == m and r["preset"] == p]
            acc = [float(r["acc"]) for r in sel if r["status"] == "done" and r["acc"]]
            cells.append(f"{mean(acc):.3f}" if acc else ("diverged" if sel else "—"))
        out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def card() -> str:
    version = field("pyproject.toml", r'^version = "([^"]+)"')
    doi = field("CITATION.cff", r'^doi:\s*"?([^"\s]+)"?\s*$')
    main = read("cpu_main")
    status = {s: sum(r["status"] == s for r in main) for s in ("done", "diverged", "failed")}
    seeds = {p: len({r["seed"] for r in main if r["preset"] == p}) for p in PRESETS}
    files = "\n".join(f"| `{n}.csv` | `{n}` | {len(read(n))} | {d} | {c} |" for n, d, c in FILES)
    configs = "\n".join(f"- config_name: {n}\n  data_files: {n}.csv" + ("\n  default: true" if i == 0 else "")
                        for i, (n, _, _) in enumerate(FILES))
    cite_doi = f"  doi     = {{{doi}}},\n" if doi else ""
    doi_line = f" · DOI [{doi}](https://doi.org/{doi})" if doi else ""
    return f"""---
license: mit
pretty_name: "contilab CPU study: continual learning results"
tags:
- continual-learning
- lifelong-learning
- catastrophic-forgetting
- class-incremental-learning
- benchmark
- model-merging
- knowledge-distillation
size_categories:
- n<1K
configs:
{configs}
---

# contilab CPU study: continual learning results

Per-run results of an empirical comparison of **12 continual-learning methods on 6 benchmarks**, produced with
[contilab]({GITHUB}) {version}, a PyTorch framework for continual learning research. One row per run, with every
scalar metric: final accuracy, forgetting, backward and forward transfer, the stability gap, task-recency bias,
calibration, representation metrics, memory and wall time.

[Results page]({SITE}wiki/Results.html) (rankings, significance tests, findings) · [Docs]({SITE}) ·
[GitHub]({GITHUB}){doi_line}

## Files

| File | Config | Rows | Contents | Experiment config |
|---|---|---|---|---|
{files}

The main study has {len(main)} runs: {status['done']} done, {status['diverged']} diverged, {status['failed']} failed.
Seeds per benchmark: {", ".join(f"{PRESETS[p]} {n}" for p, n in seeds.items() if n)}.

```python
from datasets import load_dataset

main = load_dataset("{REPO_ID}", "cpu_main", split="train")
```

## Columns

- **Identifiers:** `run_id`, `preset` (benchmark and scenario), `method`, `variant` (swept hyperparameters),
  `seed`, `status` (`done`, `diverged` or `failed`). `cpu_tuned` adds `selected` (the variant chosen on validation
  accuracy) and `cell` (the comparison-axis part of the variant).
- **Metrics:** every other column. Definitions and references are on the [Metrics]({SITE}wiki/Metrics.html)
  page. An empty cell means the metric is undefined for that run, for example forward transfer in Class-IL, stream
  metrics for consolidation methods, or anything for a diverged run.

## Final accuracy (cpu_main, mean over seeds)

{overview(main)}

"diverged" marks a benchmark where every seed of the method diverged (its loss stopped being finite); such runs
are kept with `status = diverged`.

## How the data were produced

- Hardware: 4 CPU cores (Intel Xeon @ 2.1 GHz), no GPU.
- Vision: MLP 2×400 on the MNIST benchmarks (1 epoch per task), reduced ResNet-18 on Split CIFAR-10/100
  (2 epochs). Language: Pythia-70M on text5 (AG News → Yelp → Amazon → DBpedia → Yahoo Answers), 1,000 training
  texts per task.
- Replay and distillation methods keep 500 samples (vision) or 200 (text).
- Reproduce with `contilab run configs/experiments/<config>.yaml`; the setup and every deviation from the design
  are listed on the [Results]({SITE}wiki/Results.html) page.

## Citation

```bibtex
@software{{contilab2026,
  author  = {{Park, Yeochoon}},
  title   = {{contilab: a performance-first framework for continual learning benchmarking, distillation and model merging}},
  year    = {{2026}},
  version = {{{version}}},
{cite_doi}  url     = {{{GITHUB}}}
}}
```

## License

MIT, like contilab. The benchmarks' source datasets keep their own licenses.
"""


def build(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for name, _, _ in FILES:
        shutil.copy2(os.path.join(DATA, f"{name}.csv"), os.path.join(out, f"{name}.csv"))
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(card())


def upload(out: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    sha = os.environ.get("GITHUB_SHA", "")[:7]
    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=out,
                      commit_message=f"Sync from {GITHUB}" + (f" ({sha})" if sha else ""))
    print(f"published https://huggingface.co/datasets/{REPO_ID}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory for README.md and the CSVs")
    ap.add_argument("--upload", action="store_true", help="publish the directory to the Hugging Face Hub")
    a = ap.parse_args()
    build(a.out)
    print(f"wrote {a.out}/README.md and {len(FILES)} CSV files")
    if a.upload:
        upload(a.out)


if __name__ == "__main__":
    main()
