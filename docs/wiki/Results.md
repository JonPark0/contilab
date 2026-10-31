# Results: CPU study

An empirical comparison of 12 continual-learning methods on 5 benchmarks, run end to end on a 4-core CPU with
contilab. The page has three experiments:

| Experiment | Config | What it answers |
|---|---|---|
| Main study | `configs/experiments/cpu_main.yaml` | All methods with default hyperparameters on 5 benchmarks |
| Tuned comparison | `configs/experiments/cpu_tuned.yaml` | The same methods when each one's hyperparameters are chosen on validation accuracy (two MNIST benchmarks) |
| Smaller student | `configs/experiments/cpu_student.yaml` | Distilling full-size experts into a student with 5.3× fewer parameters |

> **v0.2 update.** The results were refreshed after these changes:
> - Metric fixes: `cka_final`, `buffer_bytes`, and the continual-evaluation curves of Joint.
> - New distillation defaults, chosen on validation accuracy in the tuned comparison. The v0.1 defaults
>   under-trained the student.
> - A new method: teacher → student continual distillation.
>
> Accuracy numbers of the other methods are unchanged.

## Setup

| | |
|---|---|
| Hardware | 4 CPU cores (Intel Xeon @ 2.1 GHz, AMX/AVX512-BF16), 15 GB RAM, no GPU |
| Main study | 5 benchmarks × 12 methods × 3 seeds = **180 runs**: 174 done, 6 diverged (EWC λ = 100), 0 failed |
| Vision | Split MNIST (Class-IL and Task-IL) and Permuted MNIST (10 tasks, 10k/2k samples per task) with an MLP 2×400, 1 epoch, SGD lr 0.03; Split CIFAR-10 (Class-IL) with a reduced ResNet-18, **2 epochs**, bf16 + `torch.compile` |
| Language | text5 (AG News → Yelp → Amazon → DBpedia → Yahoo, 33 classes) with **Pythia-70M**, **1,000 train / 200 test texts per task**, 1 epoch, AdamW lr 1e-4, batch 16 |
| Memory budget | 500 samples (vision) / 200 (text) for ER, DER++, A-GEM and both distillation methods |
| Distillation | multi-teacher: TIES-merged student, targets routed to each sample's teacher, 10 passes over the buffer; continual: one pass (`train.epochs`) over each task plus the buffer |
| Metrics tier | `full` on Split MNIST Class-IL, `extended` elsewhere; ρ = 20 (MNIST) / 50 (CIFAR, text) |

Deviations from the original design, all made to fit a CPU budget:

- CIFAR-10 was trained for 2 epochs instead of 3.
- text5 used 1,000 training and 200 test texts per task, instead of 2,000 and 500.
- A-GEM on text used a 64-sample reference batch instead of 256.

## Accuracy overview

Final average accuracy (ACC), mean ± std over 3 seeds. † = number of seeds that diverged.

| Method | Split MNIST (Class-IL) | Split MNIST (Task-IL) | Permuted MNIST (Domain-IL) | Split CIFAR-10 (Class-IL) | text5 + Pythia-70M (Class-IL) |
|---|---|---|---|---|---|
| Naive | 0.196 ± 0.000 | 0.980 ± 0.002 | 0.808 ± 0.005 | 0.181 ± 0.004 | 0.097 ± 0.016 |
| Joint | 0.951 ± 0.002 | 0.992 ± 0.000 | 0.911 ± 0.002 | 0.398 ± 0.099 | 0.529 ± 0.015 |
| EWC | 0.198 ± 0.002 | 0.983 ± 0.002 | diverged (3/3) | diverged (3/3) | 0.105 ± 0.015 |
| LwF | 0.225 ± 0.004 | 0.990 ± 0.001 | 0.809 ± 0.004 | 0.173 ± 0.003 | 0.105 ± 0.006 |
| ER | 0.856 ± 0.011 | 0.983 ± 0.001 | 0.866 ± 0.002 | 0.302 ± 0.017 | 0.329 ± 0.038 |
| DER++ | 0.861 ± 0.012 | 0.990 ± 0.000 | 0.901 ± 0.001 | 0.256 ± 0.054 | 0.397 ± 0.068 |
| A-GEM | 0.344 ± 0.013 | 0.984 ± 0.000 | 0.861 ± 0.009 | 0.181 ± 0.003 | 0.100 ± 0.008 |
| LoRA-per-task | 0.184 ± 0.004 | 0.987 ± 0.001 | 0.625 ± 0.013 | 0.137 ± 0.006 | 0.024 ± 0.002 |
| Task Arithmetic | 0.525 ± 0.025 | 0.979 ± 0.004 | 0.565 ± 0.026 | 0.158 ± 0.027 | 0.034 ± 0.003 |
| TIES | 0.552 ± 0.027 | 0.983 ± 0.003 | 0.675 ± 0.012 | 0.166 ± 0.015 | 0.033 ± 0.004 |
| Multi-teacher distill. | 0.882 ± 0.004 | 0.986 ± 0.000 | 0.769 ± 0.007 | 0.262 ± 0.023 | 0.452 ± 0.013 |
| Continual distill. | 0.533 ± 0.038 | 0.980 ± 0.001 | 0.816 ± 0.004 | 0.210 ± 0.041 | 0.331 ± 0.045 |

![Critical-difference diagram](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/cd_diagram.png)

**Ranking across the 5 benchmarks (ACC):**

| Rank | Method |
|---|---|
| 1.0 | Joint |
| 3.0 | DER++ |
| 4.0 | Multi-teacher distillation |
| 4.2 | ER |
| 6.0 | Continual distillation |
| 6.4 | LwF |
| 6.6 | A-GEM |
| 8.6 | TIES |
| 8.8 | Naive |
| 9.6 | EWC |
| 9.8 | LoRA-per-task |
| 10.0 | Task Arithmetic |

The Friedman test gives p = 1.1·10⁻⁴, and the Nemenyi critical difference is 7.45 (α = 0.05). With only 5
benchmarks the CD is wide: only Joint is significantly better than TIES, Naive, EWC, LoRA-per-task and Task
Arithmetic. Before v0.2, multi-teacher distillation ranked last (9.8) with under-trained defaults.

## Findings

1. **Replay and multi-teacher distillation lead.** The best non-Joint method on each benchmark:
   - DER++ on Permuted MNIST (0.901), and tied with LwF on Split MNIST Task-IL (0.990)
   - ER on Split CIFAR-10 (0.302)
   - multi-teacher distillation on Split MNIST Class-IL (0.882) and text5 (0.452)

   Average ranks: DER++ 3.0, distillation 4.0, ER 4.2. Replay also has the smallest stability gap, for example
   0.151 vs 0.995 for Naive on Split MNIST Class-IL.
2. **Multi-teacher distillation buys accuracy with experts.**
   - It keeps every task expert: 11 MB of stored models on Split MNIST against 1.8 MB for a single model.
   - It trains every expert as well: 1,915 s per text5 run against 1,044 s for DER++.
   - In return it forgets little and is well calibrated across tasks, with BWT −0.066 and task-recency bias 0.04
     on Split MNIST Class-IL.
   - On Permuted MNIST it falls below Naive (0.769 vs 0.808): the 500-sample buffer leaves 50 samples for each
     of the 10 input distributions.
3. **Teacher → student continual distillation is the cheap variant.** It never stores an expert: the student is
   the only model kept (1.8 MB). It beats Naive on every Class- and Domain-IL benchmark:
   - Split MNIST Class-IL 0.533 vs 0.196
   - Permuted MNIST 0.816 vs 0.808, ahead of multi-teacher distillation
   - CIFAR-10 0.210 vs 0.181
   - text5 0.331 vs 0.097, on par with ER

   Old knowledge reaches the student only through self-distillation on the buffer, so errors compound task by
   task: BWT −0.566 and recency bias 0.53 on Split MNIST Class-IL.
4. **Regularisation alone does not solve Class-IL.** EWC (0.198) and LwF (0.225) stay at the Naive level (0.196)
   on Split MNIST Class-IL, with task-recency bias ≈ 1.0: almost every old-task sample is predicted as a
   newest-task class. The same holds on text5 (0.105 vs 0.097). Tuning does not change this (see
   [Tuned comparison](#tuned-comparison)).
5. **EWC is sensitive to λ.** With the literature default λ = 100 and an online (accumulated) Fisher, every seed
   diverged on Permuted MNIST and Split CIFAR-10. With λ = 30, chosen on validation accuracy, Permuted MNIST
   trains stably and reaches 0.821, above Naive (0.808).
6. **A-GEM helps in Domain-IL but not in Class-IL.** It reaches 0.861 on Permuted MNIST but only 0.344 on Split
   MNIST Class-IL, and stays at the Naive level on CIFAR-10 and text5. This matches the known weakness of
   gradient projection under class-incremental output heads.
7. **Merging trails replay.**
   - TIES is consistently above Task Arithmetic: 0.552 vs 0.525 on Split MNIST Class-IL, 0.675 vs 0.565 on
     Permuted MNIST.
   - Merged models show little recency bias (0.12–0.13), but learning accuracy drops to 0.77–0.80 because the
     experts interfere.
   - As a starting point for distillation, the merged model is what made distillation work (see the tuned
     comparison).
8. **Task-IL is easy at this scale.** Every method reaches ≥ 0.979 on Split MNIST Task-IL.
9. **Language models trade task learning against general ability.**
   - Full fine-tuning methods raise the WikiText perplexity of the original LM head by +4,500 to +14,000.
   - LoRA-per-task (+186) and TIES (+1,709) keep it low but barely learn the tasks (learning accuracy
     0.21–0.25).
   - Multi-teacher distillation has the best balance: the highest text5 accuracy (0.452) at +3,643.
10. **LoRA-per-task on vision uses a frozen randomly initialised backbone** (no pretraining), which limits it to
    0.184 on Split MNIST Class-IL and 0.625 on Permuted MNIST. It is a meaningful baseline only with pretrained
    backbones.

![Continual evaluation on Split MNIST Class-IL](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/split_mnist_class_stream_curves.png)

*Continual evaluation on Split MNIST Class-IL: accuracy of each task over training steps. Dashed lines are task
boundaries. Naive, EWC, LwF and LoRA collapse to zero on old tasks immediately, while ER and DER++ keep them.*

![Accuracy matrices on Split MNIST Class-IL](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/split_mnist_class_acc_heatmap.png)

## Tuned comparison

Defaults from the literature are not always good settings. `cpu_tuned` gives every method with hyperparameters a
small grid (`method_sweep`) on the two MNIST benchmarks:

- EWC: λ ∈ {1, 3, 10, 30, 100, 300}
- LwF: α ∈ {0.5, 1, 2, 4}
- DER++: α ∈ {0.1, 0.3, 1} × β ∈ {0.5, 1}
- TIES: k ∈ {0.1, 0.2, 0.5}
- Multi-teacher distillation: init {θ₀, TIES} × routing {entropy, routed} × epochs {1, 10, 30}

For each method, the variant with the best mean **validation** accuracy over seeds is selected (spec §5.3);
diverged seeds score 0. Optimiser, learning rate, epochs, batch size and memory budget stay fixed by the
fairness rules. "Default" is the literature default from the main study (v0.1 defaults for distillation).

| Method | Split MNIST Class-IL: selected | default | tuned | Permuted MNIST: selected | default | tuned |
|---|---|---|---|---|---|---|
| Naive | — | 0.196 | 0.196 | — | 0.808 | 0.808 |
| Joint | — | 0.951 | 0.951 | — | 0.911 | 0.911 |
| EWC | λ=300 | 0.198 | **0.203** | λ=30 | diverged 3/3 | **0.821** |
| LwF | α=1 | 0.225 | **0.225** | α=0.5 | 0.809 | **0.822** |
| ER | — | 0.856 | 0.856 | — | 0.866 | 0.866 |
| DER++ | α=1, β=1 | 0.861 | **0.891** | α=0.3, β=1 | 0.901 | **0.904** |
| A-GEM | — | 0.344 | 0.344 | — | 0.861 | 0.861 |
| LoRA-per-task | — | 0.184 | 0.184 | — | 0.625 | 0.625 |
| Task Arithmetic | — | 0.525 | 0.525 | — | 0.565 | 0.565 |
| TIES | k=0.1 | 0.552 | **0.582** | k=0.2 | 0.675 | **0.675** |
| Multi-teacher distill. | epochs 10, TIES init, routed | 0.204 | **0.882** | epochs 30, TIES init, routed | 0.091 | **0.781** |
| Continual distill. | — | 0.533 | 0.533 | — | 0.816 | 0.816 |

- **Tuning changes the verdict for distillation.** The v0.1 defaults trained the student for 16 SGD steps per task
  from θ₀ with entropy-mixed targets. A TIES-merged start with routed targets lifts Split MNIST Class-IL from
  0.204 to 0.882 and Permuted MNIST from 0.091 to 0.781. These settings are now the defaults.
- **Tuning does not rescue regularisation in Class-IL.** No λ makes EWC differ from Naive on Split MNIST
  Class-IL (0.197–0.203), and no α lifts LwF above 0.225. On Permuted MNIST, λ = 30 avoids the divergence of
  λ = 100 and gives 0.821 (Naive 0.808).
- DER++ gains 0.03 with stronger replay terms (α = β = 1). TIES gains 0.03 with sparser task vectors (k = 0.1) on
  Split MNIST Class-IL.
- Average ranks over the two benchmarks, using the selected variants:
  - Joint 1.0, DER++ 2.0, ER 3.5
  - A-GEM 6.0, multi-teacher distillation 6.0, continual distillation 6.5
  - LwF 7.0, TIES 7.5, EWC 8.0
  - Naive 9.5, TA 9.5, LoRA-per-task 11.5

  Two benchmarks are too few for significance (Friedman p = 0.13).

## Teacher → smaller student

`cpu_student` distils the 2×400 MLP experts (478k parameters) into a 2×100 student (89.6k parameters, 5.3×
fewer). A different architecture cannot start from merged expert weights, so both students start from their own
initialisation. The fair full-size reference for multi-teacher distillation is therefore the θ₀-initialised
variant of the tuned comparison.

| Method | Student | Split MNIST Class-IL | Permuted MNIST | Stored model bytes (Split / Permuted) |
|---|---|---|---|---|
| Multi-teacher distill. | full size (θ₀ init) | 0.856 | 0.534 | 10.9 / 20.1 MB (experts + student) |
| Multi-teacher distill. | 2×100 | 0.857 ± 0.010 | 0.464 ± 0.010 | 9.5 / 18.6 MB (experts + student) |
| Continual distill. | full size | 0.533 ± 0.038 | 0.816 ± 0.004 | 1.8 MB (student only) |
| Continual distill. | 2×100 | 0.490 ± 0.034 | 0.798 ± 0.001 | **0.34 MB** (student only) |

- On Split MNIST Class-IL the small student matches the full-size one (0.857 vs 0.856). Permuted MNIST (10 tasks,
  all 10 classes shared) costs it 0.07.
- Continual distillation never stores an expert. Its 2×100 student keeps 98 % of the full-size accuracy on
  Permuted MNIST, in 0.34 MB. That is the smallest deployable result in this study.

## Per-benchmark details

Means over 3 seeds.
- `wall_time` is training time per run. Joint and both distillation methods were re-run for v0.2 while other jobs
  shared the CPU, and wall times of runs from different sessions vary by up to 2× with load. It includes expert training for consolidation methods and Fisher
  computation for EWC.
- `buffer_bytes` counts the stored samples (or the distillation transfer set) and DER++ logits.
- `state_bytes` counts extra state kept between tasks: the EWC Fisher diagonal and anchor, and the LwF teacher.
- EWC has no row on Permuted MNIST and Split CIFAR-10 because every seed diverged at λ = 100.

### Split MNIST (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | recency_bias | buffer_bytes | state_bytes | wall_time |
|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.196 | -0.992 | 0.990 | 0.000 | 0.995 | 1.000 | 0 | 0 | 11 s |
| Joint | 0.951 | — | — | — | — | 0.013 | 0 | 0 | 10 s |
| EWC | 0.198 | -0.990 | 0.990 | 0.000 | 0.995 | 0.998 | 0 | 3.65 MB | 38 s |
| LwF | 0.225 | -0.957 | 0.991 | 0.021 | 0.973 | 0.960 | 0 | 1.82 MB | 18 s |
| ER | 0.856 | -0.151 | 0.977 | 0.762 | 0.151 | 0.130 | 1.50 MB | 0 | 14 s |
| DER++ | 0.861 | -0.156 | 0.986 | 0.753 | 0.151 | 0.130 | 1.52 MB | 0 | 17 s |
| A-GEM | 0.344 | -0.807 | 0.990 | 0.103 | 0.854 | 0.807 | 1.50 MB | 0 | 36 s |
| LoRA-per-task | 0.184 | -0.987 | 0.974 | 0.000 | 0.988 | 1.000 | 0 | 0 | 17 s |
| Task Arithmetic | 0.525 | -0.306 | 0.770 | — | — | 0.130 | 0 | 0 | 15 s |
| TIES | 0.552 | -0.305 | 0.796 | — | — | 0.124 | 0 | 0 | 16 s |
| Multi-teacher distill. | 0.882 | -0.066 | 0.934 | — | — | 0.041 | 1.50 MB | 0 | 21 s |
| Continual distill. | 0.533 | -0.566 | 0.986 | — | — | 0.529 | 1.50 MB | 0 | 23 s |

### Split MNIST (Task-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | buffer_bytes | state_bytes | wall_time |
|---|---|---|---|---|---|---|---|---|
| Naive | 0.980 | -0.013 | 0.990 | 0.969 | 0.019 | 0 | 0 | 10 s |
| Joint | 0.992 | — | — | — | — | 0 | 0 | 10 s |
| EWC | 0.983 | -0.008 | 0.990 | 0.973 | 0.015 | 0 | 3.65 MB | 39 s |
| LwF | 0.990 | 0.000 | 0.990 | 0.992 | 0.003 | 0 | 1.82 MB | 18 s |
| ER | 0.983 | -0.005 | 0.988 | 0.983 | 0.008 | 1.50 MB | 0 | 14 s |
| DER++ | 0.990 | -0.002 | 0.992 | 0.985 | 0.008 | 1.52 MB | 0 | 18 s |
| A-GEM | 0.984 | -0.007 | 0.990 | 0.980 | 0.010 | 1.50 MB | 0 | 35 s |
| LoRA-per-task | 0.987 | 0.000 | 0.987 | 0.993 | 0.000 | 0 | 0 | 10 s |
| Task Arithmetic | 0.979 | -0.005 | 0.983 | — | — | 0 | 0 | 15 s |
| TIES | 0.983 | -0.006 | 0.988 | — | — | 0 | 0 | 15 s |
| Multi-teacher distill. | 0.986 | -0.002 | 0.988 | — | — | 1.50 MB | 0 | 22 s |
| Continual distill. | 0.980 | -0.010 | 0.988 | — | — | 1.50 MB | 0 | 24 s |

### Permuted MNIST (Domain-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | buffer_bytes | state_bytes | wall_time |
|---|---|---|---|---|---|---|---|---|
| Naive | 0.808 | -0.093 | 0.892 | 0.772 | 0.024 | 0 | 0 | 20 s |
| Joint | 0.911 | — | — | — | — | 0 | 0 | 19 s |
| LwF | 0.809 | -0.088 | 0.888 | 0.770 | 0.022 | 0 | 1.82 MB | 33 s |
| ER | 0.866 | -0.047 | 0.909 | 0.841 | 0.019 | 1.50 MB | 0 | 27 s |
| DER++ | 0.901 | -0.002 | 0.902 | 0.882 | -0.002 | 1.52 MB | 0 | 34 s |
| A-GEM | 0.861 | -0.034 | 0.892 | 0.831 | 0.003 | 1.50 MB | 0 | 67 s |
| LoRA-per-task | 0.625 | -0.094 | 0.710 | 0.590 | 0.034 | 0 | 0 | 52 s |
| Task Arithmetic | 0.565 | -0.150 | 0.701 | — | — | 0 | 0 | 36 s |
| TIES | 0.675 | -0.107 | 0.771 | — | — | 0 | 0 | 39 s |
| Multi-teacher distill. | 0.769 | -0.055 | 0.819 | — | — | 1.50 MB | 0 | 50 s |
| Continual distill. | 0.816 | -0.049 | 0.861 | — | — | 1.48 MB | 0 | 47 s |

### Split CIFAR-10 (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | recency_bias | buffer_bytes | state_bytes | wall_time |
|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.181 | -0.799 | 0.820 | 0.000 | 0.795 | 1.000 | 0 | 0 | 238 s |
| Joint | 0.398 | — | — | — | — | 0.052 | 0 | 0 | 421 s |
| LwF | 0.173 | -0.793 | 0.807 | 0.000 | 0.787 | 1.000 | 0 | 4.18 MB | 439 s |
| ER | 0.302 | -0.670 | 0.838 | 0.063 | 0.686 | 0.647 | 5.86 MB | 0 | 415 s |
| DER++ | 0.256 | -0.665 | 0.788 | 0.066 | 0.659 | 0.763 | 5.88 MB | 0 | 930 s |
| A-GEM | 0.181 | -0.817 | 0.834 | 0.000 | 0.815 | 1.000 | 5.86 MB | 0 | 3465 s |
| LoRA-per-task | 0.137 | -0.722 | 0.714 | 0.000 | 0.732 | 1.000 | 0 | 0 | 1635 s |
| Task Arithmetic | 0.158 | -0.240 | 0.350 | — | — | 0.008 | 0 | 0 | 430 s |
| TIES | 0.166 | -0.238 | 0.357 | — | — | 0.017 | 0 | 0 | 431 s |
| Multi-teacher distill. | 0.262 | -0.234 | 0.449 | — | — | 0.202 | 5.86 MB | 0 | 490 s |
| Continual distill. | 0.210 | -0.709 | 0.777 | — | — | 0.723 | 5.86 MB | 0 | 565 s |

### text5 + Pythia-70M (Class-IL)

| Method | acc | bwt | learning_acc | min_acc | stability_gap | general_ppl_delta | buffer_bytes | state_bytes | wall_time |
|---|---|---|---|---|---|---|---|---|---|
| Naive | 0.097 | -0.589 | 0.568 | 0.002 | 0.498 | +7481 | 0 | 0 | 461 s |
| Joint | 0.529 | — | — | — | — | +4488 | 0 | 0 | 674 s |
| EWC | 0.105 | -0.588 | 0.575 | 0.003 | 0.494 | +7541 | 0 | 340.94 MB | 1199 s |
| LwF | 0.105 | -0.606 | 0.590 | 0.000 | 0.480 | +8488 | 0 | 268.72 MB | 816 s |
| ER | 0.329 | -0.294 | 0.564 | 0.292 | 0.068 | +14037 | 0.39 MB | 0 | 733 s |
| DER++ | 0.397 | -0.231 | 0.582 | 0.284 | 0.126 | +6036 | 0.42 MB | 0 | 1044 s |
| A-GEM | 0.100 | -0.602 | 0.582 | 0.001 | 0.505 | +7056 | 0.39 MB | 0 | 1563 s |
| LoRA-per-task | 0.024 | -0.233 | 0.210 | 0.000 | 0.187 | +186 | 0 | 0 | 755 s |
| Task Arithmetic | 0.034 | -0.268 | 0.249 | — | — | +5743 | 0 | 0 | 628 s |
| TIES | 0.033 | -0.277 | 0.254 | — | — | +1709 | 0 | 0 | 674 s |
| Multi-teacher distill. | 0.452 | -0.050 | 0.492 | — | — | +3643 | 0.39 MB | 0 | 1915 s |
| Continual distill. | 0.331 | -0.260 | 0.539 | — | — | +12459 | 0.39 MB | 0 | 916 s |

## Throughput

![Training throughput](https://raw.githubusercontent.com/JonPark0/contilab/main/docs/assets/results/bench_bars.png)

See [Performance](Performance.md). For the MLP, training 8 seeds in one `vmap` ensemble gives 3.8× the
throughput of one seed (40.7k vs 10.8k samples/s). For the reduced ResNet-18, bf16 + `torch.compile` gives 2.3×.

## Success criteria

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| 1 | Smoke profile: every method and consolidation path finishes within 60 s; all tests pass | Met | 66 smoke runs within the 60 s test on an idle 4-core CPU; 178 tests pass on Python 3.11 and 3.13 in CI (5 network tests skipped by default) |
| 2 | CPU profile matrix completes in the session and produces a report | Met, with the budget deviations listed above | 180/180 runs finished (174 done, 6 diverged, 0 failed); `report.md` / `report.html` generated |
| 3 | Split MNIST Class-IL ordering Naive < {EWC, LwF} < {ER, DER++} < Joint (mean ACC) | Met on means; EWC ≈ Naive | 0.196 < 0.198 / 0.225 < 0.856 / 0.861 < 0.951. The EWC–Naive gap (0.002) is within seed noise, and 3 seeds cannot give p < 0.25 in an exact paired test |
| 4 | `contilab bench` reports vmap / compile speed-ups | Met | MLP vmap E=8: 3.8×; ResNet bf16 + compile: 2.3×; ResNet vmap: no gain (reported) |
| 5 | Equivalence and consistency tests pass | Met | vmap members = single runs, compile = eager, EWC(λ=0) / ER(M=0) / DER++(0,0) = Naive, merge identities, resume = uninterrupted run |

## Limitations

- **Statistical power:** 3 seeds per configuration are enough for means and spreads but not for per-benchmark
  significance tests.
- **Scale:** this is a CPU-scale study. CIFAR-10 at 2 epochs and text5 at 1,000 texts per task are far below
  their converged regimes; read absolute numbers as relative comparisons.
- **Tuning scope:** hyperparameters were tuned only on the two MNIST benchmarks. CIFAR-10 and text5 use the
  defaults, including the distillation defaults chosen on MNIST.
- **Head assembly in merging:** Class-IL merging assembles classifier rows from different experts, leaving
  logit-scale mismatch between tasks.
- **Distillation cost:** both distillation methods first train one expert per task. Their wall time and FLOPs
  include that, but a deployment that already has the experts would only pay the distillation.

## Reproduce

```bash
contilab run configs/experiments/cpu_main.yaml      # ~17 h of training on 4 CPU cores (resumable)
contilab run configs/experiments/cpu_tuned.yaml     # ~25 min
contilab run configs/experiments/cpu_student.yaml   # ~10 min
contilab report runs/cpu_main
```
