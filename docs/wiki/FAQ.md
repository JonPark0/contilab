---
title: "FAQ: continual learning, catastrophic forgetting and contilab"
description: "Questions and answers about contilab and continual learning: which method to use against catastrophic forgetting, how it compares with Avalanche and Mammoth, language models, GPUs, EWC divergence and missing metrics."
nav_order: 4
nav_title: "FAQ"
---

# FAQ

## What is continual learning, and what is catastrophic forgetting?

**Continual learning** (also *lifelong* or *incremental learning*) trains one model on a sequence of tasks, one
after another, without keeping all earlier data. **Catastrophic forgetting** is what usually happens: training on
the new task overwrites what the model learned before. On Split MNIST (class-incremental), plain fine-tuning
ends at 0.196 accuracy over the 10 digits, against 0.950 when all tasks are trained together
([Results](Results.md)).

## What is contilab?

contilab is a PyTorch framework for empirical continual-learning research. It runs methods on benchmarks across
seeds, stores the raw outcomes, computes 30+ metrics afterwards, and reports means, confidence intervals and
significance tests. It covers the usual sequential methods (EWC, LwF, Experience Replay, DER++, A-GEM,
LoRA-per-task) and train-then-consolidate pipelines: Task Arithmetic, TIES-Merging, multi-teacher distillation
and teacher → student continual distillation ([Consolidation](Consolidation.md)).

## What is the difference between task-, class- and domain-incremental learning?

- **Task-incremental (Task-IL):** the task identity is given at test time, so the model only picks among that
  task's classes.
- **Class-incremental (Class-IL):** no task identity; the model picks among all classes seen so far. This is
  the hard case, and where most methods fail.
- **Domain-incremental (Domain-IL):** the classes stay the same and the input distribution changes, as in
  Permuted MNIST.

contilab expresses each scenario as a logit mask on one shared head ([Methods](Methods.md#scenarios)).

## Which continual-learning method should I use?

It depends on what you may store. In contilab's [CPU study](Results.md) of 12 methods on 6 benchmarks:

- **If you may keep a small replay buffer**, use DER++ or Experience Replay. After Joint training they have the
  best average ranks (2.7 and 4.2), with 500 stored samples on the vision benchmarks.
- **If you can train one expert per task**, multi-teacher distillation into one student is not significantly
  different from DER++ on Split MNIST (class-incremental) once tuned. It costs the expert training.
- **If you may store no data at all**, regularisation (EWC, LwF) does not solve class-incremental learning: it
  stays at the Naive level even after tuning. In the domain-incremental setting the gains are small: on
  Permuted MNIST, LwF reaches 0.817 and tuned EWC 0.821, against about 0.81 for Naive.

The [Results](Results.md) page has every number, the tuned comparison and the limitations of a CPU-scale study.

## How is contilab different from other continual-learning libraries?

Libraries such as [Avalanche](https://github.com/ContinualAI/avalanche) and
[Mammoth](https://github.com/aimagelab/mammoth) offer many more methods and benchmarks. contilab is narrower and
focuses on the comparison itself:

- Seeds train in parallel as one `vmap` ensemble, so 3 to 8 seeds cost little more than one on small models.
- Metrics are computed after training from stored raw data, including continual-evaluation, representation,
  plasticity-loss, calibration and recovery metrics; a new metric needs no retraining.
- Statistics are built in: bootstrap confidence intervals, exact paired tests, and Friedman and Nemenyi tests
  across benchmarks.
- Model merging and distillation are evaluated with the same metrics as sequential methods.

## What is the stability gap?

The **stability gap** is a temporary drop in the accuracy of earlier tasks right after training on a new task
starts, which then partly recovers. It is invisible if accuracy is measured only at task boundaries. contilab
evaluates during training (every ρ steps) and reports it as `stability_gap`, with min-ACC and WC-ACC (De Lange
et al. 2023; [Metrics](Metrics.md#continual-evaluation-extended)).

## Does it support large language models?

Yes. Any Hugging Face causal LM can be used through `model.name: hf`; LoRA targets the attention projections. The
`gpu` profile uses Qwen2.5-1.5B on text5; the `cpu` profile uses Pythia-70M. For language models contilab also
measures general-ability retention: the change in WikiText perplexity of the original LM head.

## Do I need a GPU?

No. Every result on the [Results](Results.md) page was produced on a 4-core CPU, and the `cpu` profile picks the
fastest setting per model (seed ensembles for MLPs, bf16 for convolutional networks). For a GPU, use the `gpu`
profile or set `engine.device: cuda`; mixed precision and compilation are then chosen automatically.

## How do I add a method, metric or benchmark?

Strategies, metrics, benchmarks and consolidation methods are registered plugins. [Extending](Extending.md) shows
the interface of each and the tests it needs.

## Why is EWC marked `diverged` in some runs?

With the literature default λ = 100 and an accumulated (online) Fisher, the penalty becomes too stiff for the
learning rate on long or hard sequences, and the loss explodes. Diverged runs are kept and flagged rather than
dropped; sweep `method.params.lam` to tune it. λ = 30, chosen on validation accuracy, trains stably on Permuted
MNIST.

## Why is FWT empty in class-incremental learning?

Before task j is learned, its classes are masked out of the Class-IL prediction, so its accuracy is zero by
construction and forward transfer is undefined.

## Why is windowed forgetting empty?

It is undefined when the window Δ (10 % of a task's length) is shorter than the evaluation spacing ρ; lower
`metrics.rho`.

## How do I cite it?

See [Citing](Citing.md).
