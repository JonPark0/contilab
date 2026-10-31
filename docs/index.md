---
title: contilab — continual learning benchmark, distillation & model-merging framework
description: PyTorch framework for empirical continual (lifelong, incremental) learning research on vision and language models, with 30+ metrics and model merging and distillation baselines.
---

# contilab

**contilab** is an open-source PyTorch framework for **continual learning** research — also known as
**lifelong learning** or **incremental learning**. It benchmarks methods against **catastrophic forgetting** on
vision and language models and compares sequential training with **model merging** and **knowledge
distillation**.

- **Methods:** EWC, LwF, experience replay, DER++, A-GEM, LoRA-per-task, Joint and Naive baselines
- **Consolidation:** Task Arithmetic, TIES-Merging, multi-teacher distillation, teacher → student continual distillation
- **Scenarios:** task-incremental, class-incremental, domain-incremental
- **Benchmarks:** Split MNIST, Permuted MNIST, Split CIFAR-10/100, text5 (AG News, Yelp, Amazon, DBpedia,
  Yahoo Answers) with Pythia and Qwen language models
- **Metrics:** accuracy matrix, forgetting, backward/forward transfer, stability gap, worst-case accuracy,
  linear probes, CKA, dormant neurons, effective rank, task-recency bias, calibration, few-shot recovery, loss
  barriers, LLM general-ability retention — with bootstrap CIs, paired tests and critical-difference diagrams
- **Speed:** seed ensembles with `torch.func.vmap`, `torch.compile`, bf16

## Documentation

[Getting Started](wiki/Getting-Started.md) · [Architecture](wiki/Architecture.md) · [Methods](wiki/Methods.md) ·
[Consolidation](wiki/Consolidation.md) · [Metrics](wiki/Metrics.md) · [Benchmarks](wiki/Benchmarks.md) ·
[Performance](wiki/Performance.md) · [Results](wiki/Results.md) · [Configuration](wiki/Configuration.md) ·
[Running Experiments](wiki/Running-Experiments.md) · [Extending](wiki/Extending.md) ·
[Reproducibility](wiki/Reproducibility.md) · [FAQ](wiki/FAQ.md) · [Citing](wiki/Citing.md)

Source code: <https://github.com/JonPark0/contilab> · License: MIT
