---
title: "Continual learning benchmarks: Split MNIST, Permuted MNIST, Split CIFAR-10/100 and text5"
description: "The benchmarks and models in contilab: Split MNIST, Permuted MNIST, Split CIFAR-10, Split CIFAR-100, the text5 language benchmark with Hugging Face language models, synthetic streams, and the CPU and GPU profiles."
nav_order: 8
nav_title: "Benchmarks"
---

# Benchmarks

| Name | Tasks | Scenarios | Source |
|---|---|---|---|
| `split_mnist` | 5 × 2 classes | Task-IL, Class-IL | MNIST (Keras mirror on GCS) |
| `permuted_mnist` | 10 pixel permutations, 10 shared classes | Domain-IL | MNIST |
| `split_cifar10` | 5 × 2 classes | Task-IL, Class-IL | CIFAR-10 (fast.ai mirror) |
| `split_cifar100` | 10 × 10 classes | Task-IL, Class-IL | CIFAR-100 (fast.ai mirror) |
| `text5` | AG News (4) → Yelp (5) → Amazon (5) → DBpedia (14) → Yahoo Answers (10) | Class-IL, Task-IL | Zhang et al. 2015 CSVs (fast.ai mirror) |
| `synthetic`, `synthetic_text` | Gaussian clusters / marker-token sequences | all | generated (smoke tests) |

- **text5** follows de Masson d'Autume et al. (2019): Yelp and Amazon share the five rating labels, giving 33
  classes. Texts are tokenised once with the model's tokenizer and cached; language-model retention is measured on
  WikiText-2.
- 10 % of each task's training data is held out as a validation split (used for merge coefficients and sweeps).
- `task_order_seed` shuffles task order.

## Profiles

| Preset | `cpu` profile | `gpu` profile |
|---|---|---|
| Split MNIST | full data, MLP 2×400, 1 epoch | same |
| Permuted MNIST | 10 tasks, 10k train / 2k test per task | full data |
| Split CIFAR-10 | full data, reduced ResNet-18, 2 epochs | 3 epochs |
| Split CIFAR-100 | full data, reduced ResNet-18, 2 epochs | 3 epochs |
| text5 | Pythia-70M, 1,000 train / 200 test per task | Qwen2.5-1.5B, 115k train / 7.6k test per task |
| Continual evaluation spacing ρ | 20 steps on MNIST, 50 on CIFAR and text5 | 20 steps on vision, 50 on text5 |
| Seeds trained together (`max_E`) | 8 on MNIST, 1 on CIFAR and text5 | 8 on vision, 1 on text5 |

Windowed forgetting needs ρ to be at most its window, 10 % of a task's length. In the `cpu` profile it is
undefined on Split CIFAR-100 and text5, where tasks are short: text5 tasks are 57 steps long, so continual
evaluation sees 2–3 points per task. The `gpu` profile's longer tasks and ρ = 20 on vision keep it defined on every
benchmark.

## Models

- `mlp` — MLP (default 2 × 400 ReLU)
- `resnet18r` — reduced ResNet-18 (nf = 20, Lopez-Paz & Ranzato 2017) with GroupNorm (vmap-safe)
- `hf` — any Hugging Face causal LM; classification uses the last non-padding token; the original LM head is kept
  frozen for perplexity measurements
