# Results: CPU study

An empirical comparison of 11 continual-learning methods on 5 benchmarks, run end to end on a 4-core CPU with
contilab (`configs/experiments/cpu_main.yaml`, profile `cpu`).

## Setup

| | |
|---|---|
| Hardware | 4 CPU cores (Intel Xeon @ 2.1 GHz, AMX/AVX512-BF16), 15 GB RAM, no GPU |
| Matrix | 5 benchmarks × 11 methods × 3 seeds = **165 runs**: 159 done, 6 diverged (EWC), 0 failed |
| Vision | Split MNIST (Class-IL and Task-IL) and Permuted MNIST (10 tasks, 10k/2k samples per task) with an MLP 2×400, 1 epoch, SGD lr 0.03; Split CIFAR-10 (Class-IL) with a reduced ResNet-18, **2 epochs**, bf16 + `torch.compile` |
| Language | text5 (AG News → Yelp → Amazon → DBpedia → Yahoo, 33 classes) with **Pythia-70M**, **1,000 train / 200 test texts per task**, 1 epoch, AdamW lr 1e-4, batch 16 |
| Memory budget | 500 samples (vision) / 200 (text) for ER, DER++, A-GEM and buffer distillation |
| Metrics tier | `full` on Split MNIST Class-IL, `extended` elsewhere; ρ = 20 (MNIST) / 50 (CIFAR, text) |

Deviations from the original design, all made to fit a CPU budget:

- CIFAR-10 was trained for 2 epochs instead of 3.
- text5 used 1,000 training and 200 test texts per task, instead of 2,000 and 500.
- A-GEM on text used a 64-sample reference batch instead of 256.

## Accuracy overview

Final average accuracy (ACC), mean ± std over 3 seeds. † = number of seeds that diverged.

| Method | Split MNIST (Class-IL) | Split MNIST (Task-IL) | Permuted MNIST (Domain-IL) | Split CIFAR-10 (Class-IL) | text5 + Pythia-70M (Class-IL) |
|---|---|---|---|---|---|
| naive | 0.196 ± 0.000 | 0.980 ± 0.002 | 0.808 ± 0.005 | 0.181 ± 0.004 | 0.097 ± 0.016 |
| joint | 0.951 ± 0.002 | 0.992 ± 0.000 | 0.911 ± 0.002 | 0.398 ± 0.099 | 0.529 ± 0.015 |
| ewc | 0.198 ± 0.002 | 0.983 ± 0.002 | 0.468 ± 0.197 †3 | 0.124 ± 0.036 †3 | 0.105 ± 0.015 |
| lwf | 0.225 ± 0.004 | 0.990 ± 0.001 | 0.809 ± 0.004 | 0.173 ± 0.003 | 0.105 ± 0.006 |
| er | 0.856 ± 0.011 | 0.983 ± 0.001 | 0.866 ± 0.002 | 0.302 ± 0.017 | 0.329 ± 0.038 |
| derpp | 0.861 ± 0.012 | 0.990 ± 0.000 | 0.901 ± 0.001 | 0.256 ± 0.054 | 0.397 ± 0.068 |
| agem | 0.344 ± 0.013 | 0.984 ± 0.000 | 0.861 ± 0.009 | 0.181 ± 0.003 | 0.100 ± 0.008 |
| lora_per_task | 0.184 ± 0.004 | 0.987 ± 0.001 | 0.625 ± 0.013 | 0.137 ± 0.006 | 0.024 ± 0.002 |
| consolidate:ta | 0.525 ± 0.025 | 0.979 ± 0.004 | 0.565 ± 0.026 | 0.158 ± 0.027 | 0.034 ± 0.003 |
| consolidate:ties | 0.552 ± 0.027 | 0.983 ± 0.003 | 0.675 ± 0.012 | 0.166 ± 0.015 | 0.033 ± 0.004 |
| consolidate:distill | 0.204 ± 0.020 | 0.874 ± 0.057 | 0.091 ± 0.005 | 0.114 ± 0.024 | 0.081 ± 0.020 |

![Critical-difference diagram](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/cd_diagram.png)

**Ranking across the 5 benchmarks (ACC):**

| Rank | Method |
|---|---|
| 1.0 | Joint |
| 2.4 | DER++ |
| 3.4 | ER |
| 5.0 | LwF |
| 5.2 | A-GEM |
| 7.2 | Naive, TIES |
| 8.0 | EWC |
| 8.2 | TA |
| 8.6 | LoRA-per-task |
| 9.8 | distillation |

The Friedman test gives p = 8.3·10⁻⁵, and the Nemenyi critical difference is 6.75 (α = 0.05). With only 5
benchmarks the CD is wide, so only the extremes differ significantly:

- Joint is better than EWC, TA, LoRA-per-task and distillation.
- DER++ is better than distillation.

## Findings

1. **Replay is the strongest family.** DER++ is the best non-Joint method on 3 of 5 benchmarks:
   - Split MNIST Class-IL 0.861 vs 0.196 for Naive
   - Permuted MNIST 0.901
   - text5 0.397

   ER is best on Split CIFAR-10 with 0.302, and on Split MNIST Task-IL DER++ (0.990) is within 0.001 of LwF. Replay also has the smallest stability gap: 0.151 vs 0.995 for Naive
   on Split MNIST Class-IL, and 0.068–0.126 vs 0.498 on text5.
2. **Regularisation alone does not solve Class-IL.** EWC (0.198) and LwF (0.225) stay at the Naive level (0.196)
   on Split MNIST Class-IL. Their task-recency bias is ≈ 1.0: almost every old-task sample is predicted as a
   newest-task class. The same holds on text5 (0.105 vs 0.097).
3. **EWC is sensitive to λ.** With the literature default λ = 100 and an online (accumulated) Fisher, all seeds
   diverged on Permuted MNIST and Split CIFAR-10. A supplementary sweep on Permuted MNIST
   (`configs/experiments/cpu_ewc_sweep.yaml`) shows that smaller λ trains stably and beats Naive:

   | λ | ACC | BWT | Learning accuracy |
   |---|---|---|---|
   | 1 | 0.812 ± 0.004 | −0.086 | 0.889 |
   | 10 | 0.821 ± 0.002 | −0.063 | 0.878 |
   | 30 | 0.821 ± 0.009 | −0.046 | 0.862 |
   | 100 (default) | diverged | — | — |
   | Naive | 0.808 ± 0.005 | −0.093 | 0.892 |

4. **A-GEM helps in Domain-IL but not in Class-IL.** It reaches 0.861 on Permuted MNIST but only 0.344 on Split
   MNIST Class-IL, and stays at the Naive level on CIFAR-10 and text5. This matches the known weakness of
   gradient projection under class-incremental output heads.
5. **Merging beats Naive in Class-IL and Domain-IL but trails replay.**
   - TIES is consistently above Task Arithmetic: Split MNIST Class-IL 0.552 vs 0.525, Permuted MNIST 0.675 vs 0.565.
   - Merged models show almost no recency bias (0.12–0.13), but learning accuracy drops to 0.77–0.80 because
     experts interfere.
   - Multi-teacher distillation with default settings (student from θ₀, 500-sample buffer, 1 epoch) is
     under-trained: in Class-IL and Domain-IL its learning accuracy is only 0.10–0.46. It needs more transfer data or epochs, or TA/TIES
     initialisation (`consolidate.init`).
6. **Task-IL is easy at this scale.** Every method except distillation reaches ≥ 0.979 on Split MNIST Task-IL.
7. **Language models trade task learning against general ability.** WikiText perplexity of the original LM head
   rises by +4,500 to +14,000 for full fine-tuning methods, but only by **+186 for LoRA-per-task** and +1,709 for
   TIES. Those methods barely learn the text tasks (learning accuracy 0.21–0.25), so this is a
   stability–plasticity trade-off rather than a free win.
8. **LoRA-per-task on vision uses a frozen randomly initialised backbone** (no pretraining), which limits it to
   0.184 on Split MNIST Class-IL and 0.625 on Permuted MNIST. It is a meaningful baseline only with pretrained
   backbones.

![Continual evaluation on Split MNIST Class-IL](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/split_mnist_class_stream_curves.png)

*Continual evaluation on Split MNIST Class-IL: accuracy of each task over training steps. Dashed lines are task
boundaries; Naive, EWC, LwF and LoRA collapse to zero on old tasks immediately, while ER and DER++ keep them.*

![Accuracy matrices on Split MNIST Class-IL](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/split_mnist_class_acc_heatmap.png)

## Per-benchmark details

Means over 3 seeds. `wall_time` is training time per run, including expert training for consolidation methods
and Fisher computation for EWC.

### Split MNIST (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | recency_bias | wall_time |
|---|---|---|---|---|---|---|---|
| naive | 0.196 | -0.992 | 0.990 | 0.000 | 0.995 | 1.000 | 11 s |
| joint | 0.951 | — | — | — | — | 0.013 | 10 s |
| ewc | 0.198 | -0.990 | 0.990 | 0.000 | 0.995 | 0.998 | 38 s |
| lwf | 0.225 | -0.957 | 0.991 | 0.021 | 0.973 | 0.960 | 18 s |
| er | 0.856 | -0.151 | 0.977 | 0.762 | 0.151 | 0.130 | 14 s |
| derpp | 0.861 | -0.156 | 0.986 | 0.753 | 0.151 | 0.130 | 17 s |
| agem | 0.344 | -0.807 | 0.990 | 0.103 | 0.854 | 0.807 | 36 s |
| lora_per_task | 0.184 | -0.987 | 0.974 | 0.000 | 0.988 | 1.000 | 17 s |
| consolidate:ta | 0.525 | -0.306 | 0.770 | — | — | 0.130 | 15 s |
| consolidate:ties | 0.552 | -0.305 | 0.796 | — | — | 0.124 | 16 s |
| consolidate:distill | 0.204 | -0.316 | 0.456 | — | — | 0.321 | 13 s |

### Split MNIST (Task-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | wall_time |
|---|---|---|---|---|---|---|
| naive | 0.980 | -0.013 | 0.990 | 0.969 | 0.019 | 10 s |
| joint | 0.992 | — | — | — | — | 10 s |
| ewc | 0.983 | -0.008 | 0.990 | 0.973 | 0.015 | 39 s |
| lwf | 0.990 | 0.000 | 0.990 | 0.992 | 0.003 | 18 s |
| er | 0.983 | -0.005 | 0.988 | 0.983 | 0.008 | 14 s |
| derpp | 0.990 | -0.002 | 0.992 | 0.985 | 0.008 | 18 s |
| agem | 0.984 | -0.007 | 0.990 | 0.980 | 0.010 | 35 s |
| lora_per_task | 0.987 | 0.000 | 0.987 | 0.993 | 0.000 | 10 s |
| consolidate:ta | 0.979 | -0.005 | 0.983 | — | — | 15 s |
| consolidate:ties | 0.983 | -0.006 | 0.988 | — | — | 15 s |
| consolidate:distill | 0.874 | -0.017 | 0.888 | — | — | 13 s |

### Permuted MNIST (Domain-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | wall_time |
|---|---|---|---|---|---|---|
| naive | 0.808 | -0.093 | 0.892 | 0.772 | 0.024 | 20 s |
| joint | 0.911 | — | — | — | — | 16 s |
| ewc | 0.468 | -0.034 | 0.498 | 0.418 | 0.028 | 85 s |
| lwf | 0.809 | -0.088 | 0.888 | 0.770 | 0.022 | 33 s |
| er | 0.866 | -0.047 | 0.909 | 0.841 | 0.019 | 27 s |
| derpp | 0.901 | -0.002 | 0.902 | 0.882 | -0.002 | 34 s |
| agem | 0.861 | -0.034 | 0.892 | 0.831 | 0.003 | 67 s |
| lora_per_task | 0.625 | -0.094 | 0.710 | 0.590 | 0.034 | 52 s |
| consolidate:ta | 0.565 | -0.150 | 0.701 | — | — | 36 s |
| consolidate:ties | 0.675 | -0.107 | 0.771 | — | — | 39 s |
| consolidate:distill | 0.091 | -0.052 | 0.138 | — | — | 30 s |

### Split CIFAR-10 (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | recency_bias | wall_time |
|---|---|---|---|---|---|---|---|
| naive | 0.181 | -0.799 | 0.820 | 0.000 | 0.795 | 1.000 | 238 s |
| joint | 0.398 | — | — | — | — | 0.052 | 201 s |
| ewc | 0.124 | -0.701 | 0.685 | 0.069 | 0.686 | 0.667 | 365 s |
| lwf | 0.173 | -0.793 | 0.807 | 0.000 | 0.787 | 1.000 | 439 s |
| er | 0.302 | -0.670 | 0.838 | 0.063 | 0.686 | 0.647 | 415 s |
| derpp | 0.256 | -0.665 | 0.788 | 0.066 | 0.659 | 0.763 | 930 s |
| agem | 0.181 | -0.817 | 0.834 | 0.000 | 0.815 | 1.000 | 3465 s |
| lora_per_task | 0.137 | -0.722 | 0.714 | 0.000 | 0.732 | 1.000 | 1635 s |
| consolidate:ta | 0.158 | -0.240 | 0.350 | — | — | 0.008 | 430 s |
| consolidate:ties | 0.166 | -0.238 | 0.357 | — | — | 0.017 | 431 s |
| consolidate:distill | 0.114 | -0.108 | 0.200 | — | — | 0.333 | 284 s |

### text5 + Pythia-70M (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | general_ppl_delta | wall_time |
|---|---|---|---|---|---|---|---|
| naive | 0.097 | -0.589 | 0.568 | 0.002 | 0.498 | +7481 | 461 s |
| joint | 0.529 | — | — | — | — | +4488 | 917 s |
| ewc | 0.105 | -0.588 | 0.575 | 0.003 | 0.494 | +7541 | 1199 s |
| lwf | 0.105 | -0.606 | 0.590 | 0.000 | 0.480 | +8488 | 816 s |
| er | 0.329 | -0.294 | 0.564 | 0.292 | 0.068 | +14037 | 733 s |
| derpp | 0.397 | -0.231 | 0.582 | 0.284 | 0.126 | +6036 | 1044 s |
| agem | 0.100 | -0.602 | 0.582 | 0.001 | 0.505 | +7056 | 1563 s |
| lora_per_task | 0.024 | -0.233 | 0.210 | 0.000 | 0.187 | +186 | 755 s |
| consolidate:ta | 0.034 | -0.268 | 0.249 | — | — | +5743 | 628 s |
| consolidate:ties | 0.033 | -0.277 | 0.254 | — | — | +1709 | 674 s |
| consolidate:distill | 0.081 | -0.031 | 0.105 | — | — | +6652 | 486 s |


## Throughput

![Training throughput](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/bench_bars.png)

See [Performance](Performance.md). For the MLP, training 8 seeds in one `vmap` ensemble gives 3.8× the
throughput of one seed (40.7k vs 10.8k samples/s). For the reduced ResNet-18, bf16 + `torch.compile` gives 2.3×.

## Success criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Smoke profile: every method and consolidation path finishes within 60 s; all tests pass | Met | 66 smoke runs completed within the 60 s test on an idle 4-core CPU; 146 tests pass (5 network tests skipped by default) |
| 2 | CPU profile matrix completes in the session and produces a report | Met, with the budget deviations listed above | 165/165 runs finished (159 done, 6 diverged, 0 failed); `report.md` / `report.html` generated |
| 3 | Split MNIST Class-IL ordering Naive < {EWC, LwF} < {ER, DER++} < Joint (mean ACC) | Met on means; EWC ≈ Naive | 0.196 < 0.198 / 0.225 < 0.856 / 0.861 < 0.951. The EWC–Naive gap (0.002) is within seed noise, and 3 seeds cannot give p < 0.25 in an exact paired test |
| 4 | `contilab bench` reports vmap / compile speed-ups | Met | MLP vmap E=8: 3.8×; ResNet bf16 + compile: 2.3×; ResNet vmap: no gain (reported) |
| 5 | Equivalence and consistency tests pass | Met | vmap members = single runs, compile = eager, EWC(λ=0) / ER(M=0) / DER++(0,0) = Naive, merge identities, resume = uninterrupted run |

## Limitations

- **Statistical power:** 3 seeds per configuration are enough for means and spreads but not for per-benchmark
  significance tests.
- **Scale:** this is a CPU-scale study. CIFAR-10 at 2 epochs and text5 at 1,000 texts per task are far below
  their converged regimes; read absolute numbers as relative comparisons.
- **Default hyperparameters:** methods use literature defaults without tuning (except the EWC λ sweep). The
  distillation defaults in particular are under-trained.
- **Head assembly in merging:** Class-IL merging assembles classifier rows from different experts, leaving
  logit-scale mismatch between tasks.

## Reproduce

```bash
contilab run configs/experiments/cpu_main.yaml       # ~16 h of training on 4 CPU cores (resumable)
contilab run configs/experiments/cpu_ewc_sweep.yaml  # ~5 min
contilab report runs/cpu_main
```
