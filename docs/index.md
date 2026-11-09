---
title: contilab — continual learning benchmark, model merging and distillation framework for PyTorch
description: PyTorch framework for continual learning research. Benchmark EWC, LwF, experience replay, ER-ACE, DER++, A-GEM, GEM, LoRA and O-LoRA against catastrophic forgetting on vision and language models, compare them with Task Arithmetic, TIES-Merging, MagMax and knowledge distillation, and report 30+ metrics with proper statistics.
nav_order: 1
nav_title: "Home"
---

# contilab
{: .fs-9 }

A PyTorch framework for continual learning research: benchmark methods against catastrophic forgetting, and
compare them with model merging and distillation, on vision and language models.
{: .fs-6 .fw-300 }

[Get started](wiki/Getting-Started.md){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[See the results](wiki/Results.md){: .btn .fs-5 .mb-4 .mb-md-0 .mr-2 }
[GitHub](https://github.com/JonPark0/contilab){: .btn .fs-5 .mb-4 .mb-md-0 }

---

**contilab** trains a model on a sequence of tasks and measures what it forgets. This setting is called
**continual learning**, and also *lifelong* or *incremental learning*. contilab is built for empirical studies:
fair, reproducible comparisons of methods across seeds, benchmarks and model sizes.

- **16 methods, one evaluation.**
  - Sequential methods: Naive fine-tuning, Joint training, EWC, LwF, Experience Replay, ER-ACE, DER++, A-GEM,
    GEM, LoRA-per-task and O-LoRA.
  - Train-then-consolidate pipelines: Task Arithmetic, TIES-Merging, MagMax, multi-teacher distillation and
    teacher → student continual distillation.
  - Every method is scored with the same metrics.
- **30+ metrics, computed after training.** Forgetting, backward and forward transfer, the stability gap,
  linear-probe and CKA representation metrics, plasticity loss, task-recency bias, calibration, few-shot
  recovery and LLM general-ability retention. Raw data are stored, so a new metric needs no retraining.
- **Statistics included.**
  - Mean ± std and bootstrap confidence intervals over seeds
  - Exact paired tests, and Friedman and Nemenyi critical-difference diagrams across benchmarks
  - Hyperparameter sweeps resolved on validation accuracy
- **Fast on a CPU.** Several seeds train at once as one `vmap` ensemble: 3.8× the throughput for an MLP
  ([Performance](wiki/Performance.md)).
- **Vision and language.** Split MNIST, Permuted MNIST, Split CIFAR-10/100, and text5 with Hugging Face language
  models such as Pythia and Qwen2.5.

## Results at a glance

![Final accuracy on Split MNIST (class-incremental), mean over 6 seeds: Joint 0.950, GEM 0.882, ER-ACE 0.875, DER++ 0.871, multi-teacher distillation 0.863, Experience Replay 0.858, continual distillation 0.586, MagMax 0.572, TIES-Merging 0.561, Task Arithmetic 0.531, A-GEM 0.335, LwF 0.228, EWC 0.196, Naive 0.196, O-LoRA 0.195, LoRA-per-task 0.184](assets/results/split-mnist-class-acc.svg)

A CPU study ran 16 methods on 6 benchmarks: 432 runs, with 3 to 6 seeds each.

- **Replay leads.** After Joint training, DER++, ER-ACE and GEM have the best average ranks across the 6
  benchmarks: 3.3, 3.5 and 4.2 (Friedman p = 1.5·10⁻⁸). ER-ACE is the best continual method on both CIFAR
  benchmarks.
- **Multi-teacher distillation needed tuning.** Settings chosen on validation accuracy lifted it from 0.223 to
  0.871 on Split MNIST (class-incremental). With them as defaults it reaches 0.863 over 6 seeds, not
  significantly different from DER++ (0.871).
- **Regularisation alone stays at the Naive level** in class-incremental learning, even after tuning: EWC
  0.196 and LwF 0.228 against 0.196. Without stored samples, only MagMax merging beats Naive on every class- and
  domain-incremental benchmark.
- **Continual distillation stores no experts.** Its 2×100 student keeps 98 % of the full-size accuracy on
  Permuted MNIST in 0.34 MB.

Every number, the tuned comparison and the per-run CSV files are on the [Results](wiki/Results.md) page.

## Quick start

```bash
git clone https://github.com/JonPark0/contilab && cd contilab
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# every method on tiny synthetic streams (under a minute on 4 CPU cores)
.venv/bin/contilab run configs/experiments/smoke_all.yaml
.venv/bin/contilab report runs/smoke_all          # runs/smoke_all/report.md and report.html

# one real benchmark: Split MNIST, class-incremental, all 16 methods
.venv/bin/contilab run configs/experiments/cpu_main.yaml "matrix.preset=[split_mnist_class]"
```

Or install a [release](https://github.com/JonPark0/contilab/releases) wheel with `pip`. The
[Getting started](wiki/Getting-Started.md) page walks through a first run and the report; the
[FAQ](wiki/FAQ.md) answers the common questions.

## What is included

| Area | Contents |
|---|---|
| Scenarios | Task-incremental, class-incremental, domain-incremental |
| [Methods](wiki/Methods.md) | Naive, Joint, EWC (online), LwF, Experience Replay, ER-ACE, DER++, A-GEM, GEM, LoRA-per-task, O-LoRA |
| [Merging and distillation](wiki/Consolidation.md) | Task Arithmetic, TIES-Merging, multi-teacher distillation, teacher → student continual distillation, smaller students |
| [Benchmarks](wiki/Benchmarks.md) | Split MNIST, Permuted MNIST, Split CIFAR-10, Split CIFAR-100, text5 (AG News → Yelp → Amazon → DBpedia → Yahoo Answers) |
| [Metrics](wiki/Metrics.md) | 30+ metrics, figures (accuracy heatmaps, stability-gap curves, CKA, t-SNE, loss barriers, critical-difference diagrams) |
| [Engineering](wiki/Performance.md) | `vmap` seed ensembles, `torch.compile`, bf16, resumable experiment matrices, failure isolation, Markdown and HTML reports |

## Citing

```bibtex
@software{contilab2026,
  author  = {Park, Yeochoon},
  title   = {contilab: a performance-first framework for continual learning benchmarking, distillation and model merging},
  year    = {2026},
  version = {0.4.0},
  doi     = {10.5281/zenodo.23906803},
  url     = {https://github.com/JonPark0/contilab}
}
```

Every release is archived on [Zenodo](https://doi.org/10.5281/zenodo.23906803). The DOI `10.5281/zenodo.23906803` stands for all versions and
resolves to the latest one; each release also has its own DOI, listed on the Zenodo page.

Source code: <https://github.com/JonPark0/contilab> · License: MIT
