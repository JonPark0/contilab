# Metrics

contilab computes metrics **after** training from raw data stored per run, so a metric can be added later with
`contilab metrics <exp> --add <name>`. A metric that is undefined for a run returns no value with a reason (for
example "needs at least 2 tasks") instead of a misleading number.

Notation: $T$ tasks; $R_{i,j}$ = test accuracy on task $j$ after training task $i$; $b_j$ = accuracy of the
initial model on task $j$.

## Raw data and tiers

| Tier | Raw records collected |
|---|---|
| `core` | accuracy matrix, initial accuracy, predictions/confidences per task end, cost |
| `extended` | + continual-evaluation stream (every ρ steps), learning curves (first β batches), probe features, neuron activations, LM perplexity |
| `full` | + parameter snapshots, loss-interpolation curves, gradient-cosine matrices, few-shot recovery |

## Accuracy-matrix metrics (`core`)

| Metric | Definition | Reference |
|---|---|---|
| `acc` | $\frac{1}{T}\sum_j R_{T-1,j}$ | Lopez-Paz & Ranzato 2017 |
| `aia` | $\frac{1}{T}\sum_i \frac{1}{i+1}\sum_{j\le i} R_{i,j}$ (average incremental accuracy) | Rebuffi et al. 2017 |
| `bwt` | $\frac{1}{T-1}\sum_{j<T-1}(R_{T-1,j}-R_{j,j})$ | Lopez-Paz & Ranzato 2017 |
| `fwt` | $\frac{1}{T-1}\sum_{j\ge1}(R_{j-1,j}-b_j)$ — undefined in Class-IL, where a task's classes are masked out before it is learned | Lopez-Paz & Ranzato 2017 |
| `forgetting` | mean of $f_j=\max_{l\in[j,T-2]}R_{l,j}-R_{T-1,j}$ | Chaudhry et al. 2018 |
| `relative_forgetting` | mean of $f_j / \max_l R_{l,j}$ | — |
| `learning_acc` | $\frac{1}{T}\sum_j R_{j,j}$ (plasticity) | Riemer et al. 2019 |
| `intransigence` | $\frac{1}{T}\sum_j(a^*_j-R_{j,j})$ with $a^*$ from the Joint run of the same preset and seed | Chaudhry et al. 2018 |

## Continual evaluation (`extended`)

$A(k,n)$ = accuracy on task $k$ at step $n$, sampled every ρ steps on a fixed subset.

| Metric | Definition | Reference |
|---|---|---|
| `min_acc` | mean over $k<T-1$ of the minimum $A(k,n)$ after task $k$ was learned | De Lange et al. 2023 |
| `wc_acc` | $\frac{1}{T}A(T-1,\text{end}) + (1-\frac{1}{T})\,\text{min-ACC}$ | De Lange et al. 2023 |
| `stability_gap` | accuracy of task $k$ at the end of task $k$ minus its minimum while task $k+1$ trains, averaged | De Lange et al. 2023 |
| `windowed_forgetting` | largest drop within a window Δ = 10 % of a task's length; undefined when Δ < ρ | De Lange et al. 2023 |
| `aaa` | average over evaluation points of the mean accuracy on seen tasks (anytime accuracy) | Caccia et al. 2022 |
| `lca` | $\frac{1}{\beta+1}\sum_{b=0}^{\beta}Z_b$, accuracy on the current task during its first β batches | Chaudhry et al. 2019 |

## Representation (`extended`)

| Metric | Definition | Reference |
|---|---|---|
| `probe_acc` | matrix $P_{i,j}$ of linear-probe accuracy on frozen features | Davari et al. 2022 |
| `repr_forgetting` / `head_forgetting` | probe drop $P_{j,j}-P_{T-1,j}$, and the rest of the accuracy drop attributed to the classifier head | Davari et al. 2022 |
| `cka_final`, `cka_matrix` | linear CKA between features of the same samples at different times | Kornblith et al. 2019 |
| `feature_drift` | mean normalised L2 movement of a task's features from when it was learned to the end | Ramasesh et al. 2021 |

## Plasticity loss (`extended`, `full`)

| Metric | Definition | Reference |
|---|---|---|
| `dormant_ratio` (+ curve) | share of neurons with normalised mean activation $\le\tau$ (τ = 0.025) | Sokar et al. 2023 |
| `effective_rank`, `srank` (+ curve) | exp-entropy of normalised singular values; smallest k covering 99 % of singular mass | Roy & Vetterli 2007; Kumar et al. 2021 |
| `weight_norms` | per-layer weight norm after each task | Dohare et al. 2024 |

Structural plasticity diagnostics are symptoms rather than guarantees; report them next to `learning_acc` and
`lca`.

## Bias and calibration (`core`)

| Metric | Definition | Reference |
|---|---|---|
| `recency_bias` | Class-IL: share of old-task samples predicted as the newest task's classes | Wu et al. 2019; Masana et al. 2022 |
| `task_confusion` | true-task × predicted-task matrix | Masana et al. 2022 |
| `ece`, `ece_per_task` | expected calibration error, 15 bins | Guo et al. 2017 |

## Recovery, landscape, LLMs (`full`, `extended`)

| Metric | Definition | Reference |
|---|---|---|
| `recovery` | $(acc_{after}-acc_{before})/(R_{j,j}-acc_{before})$ after 20 SGD steps on 50 samples of an old task: high recovery suggests *spurious* forgetting | Zheng et al., ICLR 2025 |
| `loss_barrier` | $\max_\alpha L((1-\alpha)\theta_k+\alpha\theta_{k+1}) - [(1-\alpha)L(\theta_k)+\alpha L(\theta_{k+1})]$ on task $k$ | Mirzadeh et al. 2020 |
| `grad_interference` | mean off-diagonal cosine between task gradients | — |
| `general_ppl_delta` (+ curve) | WikiText perplexity of the original LM head after training minus before (general-ability retention) | Wang et al. 2023 (TRACE) |

## Cost (`core`)

`wall_time`, `samples_per_s`, `peak_mem` (per run group), `trainable_params`, `buffer_bytes`, `model_bytes`,
`total_flops` (measured with `FlopCounterMode` per member, including replay, reference gradients, experts and
distillation).

## Statistics

- mean ± std over seeds, bootstrap 95 % confidence intervals (10,000 resamples)
- paired test against a baseline method: Wilcoxon signed-rank for n ≥ 6 seeds, otherwise an exact sign-flip
  permutation test (with 3 seeds the smallest possible p is 0.25, so the report warns about statistical power)
- across benchmarks: average ranks, Friedman test, Nemenyi critical difference (Demšar 2006)

## Figures

Accuracy-matrix heatmaps, continual-evaluation curves with task boundaries, task-confusion matrices, CKA
heatmaps, per-layer weight drift, gradient-interference heatmaps, PCA and t-SNE of features, loss-interpolation
curves, critical-difference diagrams and throughput bars.
