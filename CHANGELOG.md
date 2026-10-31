# Changelog

## Unreleased

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

### Changed
- Run ids hash the full resolved configuration, so runs made with v0.1 defaults are not reused when defaults
  change.

## 0.1.0 — 2026-10-23

First public release: 8 continual-learning strategies, Task Arithmetic, TIES and multi-teacher distillation;
30+ metrics; seed-parallel `vmap` ensembles; CPU study of 11 methods on 5 benchmarks.
