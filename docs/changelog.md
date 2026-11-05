---
title: "Changelog: what changed in each contilab release"
description: "Release notes of contilab: new continual-learning methods, metrics, benchmarks, results, speed-ups and fixes in each version."
nav_order: 16
nav_title: "Changelog"
---
<!-- Generated from CHANGELOG.md by scripts/sync_docs.py; edit CHANGELOG.md instead. -->

# Changelog

## Unreleased

### Changed
- EWC's default λ on vision is 30, the value selected on validation accuracy in the tuned comparison, instead of
  100. With λ = 100 every seed diverged on Permuted MNIST; with λ = 30 it reaches 0.832 (Naive 0.805).
- The Split CIFAR-10 and CIFAR-100 presets of the `cpu` profile train in bf16 without `torch.compile`: compile added
  no throughput at steady state and took about 50 s per graph.
- The main study's CIFAR runs and MNIST EWC runs were re-run with these defaults, and the results, tables and
  figures were updated.
- The tuned comparison (`cpu_tuned`) has 6 seeds, like the MNIST benchmarks of the main study. Its config pins the
  consolidation defaults it was run with, so its run ids stay those of the published results.

### Fixed
- EWC no longer diverges on long task streams under SGD. A step multiplies a parameter's distance from its anchor
  by (1 − lr·λ·F), and the online Fisher only grows, so once lr·λ·F exceeded 2 the run diverged: with λ = 30, late
  in the stream on Split CIFAR-10 (1 of 3 seeds) and CIFAR-100 (3 of 3). The stored Fisher is now clamped at
  `max_stiffness / (lr·λ)` (`method.params.max_stiffness`, default 1, `null` turns it off) when the optimiser is
  SGD; Adam-type optimisers are not clamped. No run of the main study or the tuned EWC sweep diverges any more; the
  MNIST results are unchanged, and on CIFAR EWC is stable but below Naive. The EWC runs of both studies were
  re-run.

### Added
- Four methods:
  - **GEM** (`gem`): one reference gradient per previous task and the dual-QP projection of the paper, solved
    exactly per member. It uses A-GEM's memory and reference-batch size, so the two differ only in the projection.
    Strategies can now ask for one reference gradient per group of the replay batch (`Strategy.ref_groups`).
  - **ER-ACE** (`er_ace`): ER with an asymmetric cross-entropy. The incoming batch's softmax covers only the
    classes present in it.
  - **O-LoRA** (`olora`): a new LoRA adapter per task with an orthogonality penalty against the earlier ones; all
    adapters stay active at inference, so it needs no task identity.
  - **MagMax** (`consolidate:magmax`): experts fine-tuned sequentially, merged by keeping each parameter's
    largest-magnitude task-vector value. Sequential experts (`train_experts(..., sequential=True)`) have their
    own cache entry.
- The smoke matrix covers the new methods (90 runs, about 40 s on 4 cores).
- Experiment config `cpu_text5_full`: text5 at the design scale (2,000 training and 500 test texts per task). Its
  36 runs are in `cpu_text5_full.csv` and on the Results page: the method order of the main study holds (rank
  correlation 0.97), and most methods gain 0.01–0.05.
- Zenodo DOI `10.5281/zenodo.23906803` (all versions): a badge in the README, `doi` in `CITATION.cff` and the BibTeX entries,
  and the site's structured data. Version 0.3.1 is `10.5281/zenodo.23906804`.
- The per-run results are a Hugging Face dataset, [jonpark0/contilab-cpu-study](https://huggingface.co/datasets/jonpark0/contilab-cpu-study), republished by the
  `hf-dataset` workflow when the CSVs change; the site, README and `llms.txt` link to it.

## 0.3.1 — 2026-11-03

### Added
- Docs site (<https://jonpark0.github.io/contilab/>) on Just the Docs:
  - search, dark mode, navigation, and a home page with a results chart generated from the CSVs
  - a Korean home page and a changelog page
  - a FAQ written around common questions
- Discoverability:
  - per-page titles and descriptions, a social preview image, a favicon, `llms.txt`
  - schema.org metadata for the software and for the results CSVs (as a dataset)
  - IndexNow submission of changed pages after each Pages build
- `scripts/sync_docs.py` writes the chart and the changelog page; the `docs` workflow checks that they are current
  and that the site builds.
- The release workflow adds each release's Zenodo DOI to its notes once `CITATION.cff` has the concept DOI.

### Changed
- `CITATION.cff`, the package metadata and the release notes link to the docs site.

## 0.3.0 — 2026-11-02

### Added
- Release workflow: pushing a tag `vX.Y.Z` (or running the workflow with a tag) builds the sdist and wheel and
  publishes a GitHub release with that version's CHANGELOG section as notes. Its PyPI job uses trusted publishing
  and stays off until the `PUBLISH_PYPI` repository variable is set (see `RELEASING.md`).
- `contilab bench --precision` (default `fp32,auto`): throughput at the precision the profiles actually use.
- A group that runs out of memory is re-run as two smaller groups (recorded as `memory_split` in `env.json`).
  A strategy that cannot be vmapped runs member by member, recorded as `vmap_fallback`.
- Split CIFAR-100 (Class-IL, 10 × 10 classes) in the `cpu` profile and in `cpu_main`. `cpu_main_seeds` adds
  seeds 3–5 to the three MNIST benchmarks, so exact paired tests there can reach p < 0.05.

### Fixed
- `torch.compile` no longer recompiles for every batch size. The batch axis is marked dynamic, so A-GEM's
  reference batches, filling replay buffers and the last batch of a task share one graph. On CPU, each ResNet
  recompilation took about 25–50 s, and A-GEM runs hit Dynamo's limit of 8 graphs and then trained eagerly.
- LoRA-per-task stays compiled beyond 8 tasks: the graph limit per compiled function is now the number of
  tasks + 8.

### Changed
- Less repeated work:
  - the class table is cached per stream and device
  - LwF runs one student forward per step instead of two
  - distillation reuses the run's data pool
  - probe-based metrics fit the probe matrix once
- Results (wiki and README) cover 6 benchmarks and 324 runs, with exact paired tests over 6 seeds on MNIST.
  DER++ now has the best mean on Split MNIST Class-IL, not significantly different from multi-teacher
  distillation.

## 0.2.0 — 2026-10-31

### Added
- `consolidate:continual_distill`: teacher → student continual distillation. Teachers arrive one per task and
  are dropped after use; the student combines knowledge distillation with self-distillation on a replay buffer.
- Smaller students for both distillation methods (`method.params.student`).
- Validation-based selection of sweep variants (`val_acc`, `selected` column, ✓ in reports, ranking over
  selected variants).
- `method_sweep`: one hyperparameter grid per method.
- `state_bytes` metric. `buffer_bytes` now counts stored samples rather than index bytes.
- Experiment configs `cpu_tuned` (validation-tuned comparison) and `cpu_student` (teacher → smaller student).
- GitHub Actions test workflow.

### Fixed
- `cka_final` no longer averages in the trivial comparison of the last task with itself.
- Joint runs record continual-evaluation accuracy on every task, not only the first.
- Few-shot recovery fine-tunes with the run's optimizer instead of SGD.
- A resumed run keeps the divergence threshold of the uninterrupted run.
- `torch.compile` errors fall back to eager at any recompilation, not only at the first call.
- Checkpoints, caches, raw logs and run files are written atomically. An expert cache without its metadata file is
  rebuilt.
- Expert training and distillation are checked for divergence, and affected seeds are marked as diverged.
- `contilab run exp.yaml --root runs key=value` works on Python < 3.12. Before, argparse left the override
  unparsed and the command exited.

### Changed
- Multi-teacher distillation defaults (`consolidate.init: ties`, `routing: routed`, `epochs: 10`), chosen on
  validation accuracy in `cpu_tuned`. The v0.1 defaults (`base`, `entropy`, 1 epoch) under-trained the student.
- Run ids hash the full resolved configuration, so runs made with v0.1 defaults are not reused when defaults
  change.

## 0.1.0 — 2026-10-23

First public release: 8 continual-learning strategies, Task Arithmetic, TIES and multi-teacher distillation;
30+ metrics; seed-parallel `vmap` ensembles; CPU study of 11 methods on 5 benchmarks.
