# contilab wiki

**contilab** is a PyTorch framework for empirical **continual learning** research (lifelong / incremental
learning). It benchmarks methods that fight **catastrophic forgetting** on vision and language models,
compares sequential training with **model merging** and **knowledge distillation**, and reports 30+
evaluation metrics with proper statistics.

## Start here

| Page | What it covers |
|---|---|
| [Getting Started](Getting-Started.md) | Install, first run, reading the report |
| [Architecture](Architecture.md) | Modules, data flow, the ensemble axis, raw-log-first metrics |
| [Methods](Methods.md) | Scenarios and the CL strategies: Naive, Joint, EWC, LwF, ER, DER++, A-GEM, LoRA-per-task |
| [Consolidation](Consolidation.md) | Train-then-consolidate: Task Arithmetic, TIES-Merging, multi-teacher distillation |
| [Metrics](Metrics.md) | Every metric with its formula and reference, statistics, figures |
| [Benchmarks](Benchmarks.md) | Split/Permuted MNIST, Split CIFAR, text5, synthetic streams |
| [Performance](Performance.md) | vmap ensembles, torch.compile, bf16, measured throughput |
| [Results](Results.md) | CPU study: 11 methods × 5 benchmarks × 3 seeds, rankings, findings |
| [Configuration](Configuration.md) | Profiles, presets, experiment matrices, sweeps, overrides |
| [Running Experiments](Running-Experiments.md) | CLI, run directories, resume, reports, post-hoc metrics |
| [Extending](Extending.md) | Adding a strategy, metric, benchmark or consolidator |
| [Reproducibility](Reproducibility.md) | Seeds, determinism, caching and what is recorded |
| [FAQ](FAQ.md) | Common questions |
| [Citing](Citing.md) | BibTeX and CITATION.cff |

## Why contilab

- **Fair comparisons.** Every method runs through the same training loop, optimiser, epochs, batch size,
  memory budget and evaluation protocol.
- **Fast seeds.** Seeds that differ only by their random seed train together as one stacked-parameter
  ensemble, so three seeds of a small model cost little more than one.
- **Metrics you can add later.** Training stores raw data (accuracy matrices, continual-evaluation curves,
  predictions, features, activations, gradients); metrics are pure functions of those records.
- **Vision and LLMs behind one interface.** The same strategies run on an MLP, a reduced ResNet-18 or a
  Hugging Face causal LM with a classification head.
