# contilab — Continual Learning Benchmark, Distillation & Model-Merging Framework (PyTorch)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Docs: wiki](https://img.shields.io/badge/docs-wiki-green.svg)](https://github.com/JonPark0/contilab/wiki)

**contilab** is a research framework for **continual learning** (also called *lifelong learning* or
*incremental learning*): training a model on a sequence of tasks without **catastrophic forgetting**.
It is built for **empirical studies** — fair, reproducible comparisons of methods on vision and
language (LLM) benchmarks — and is designed around three goals:

1. **Speed.** Several random seeds are trained *simultaneously* as one stacked-parameter ensemble
   (`torch.func.vmap` + `functional_call`), with optional `torch.compile` and bf16 autocast and a
   DataLoader-free, device-resident data pipeline.
2. **Evaluation depth.** 30+ metrics from the continual-learning literature — accuracy matrix,
   forgetting, backward/forward transfer, the **stability gap**, worst-case accuracy, linear probing
   and **CKA**, **plasticity loss** (dormant neurons, effective rank), task-recency bias, calibration,
   few-shot recovery, loss barriers, gradient interference and **LLM general-ability retention** —
   computed *post hoc* from stored raw data, with bootstrap confidence intervals, paired tests and
   Friedman/Nemenyi critical differences.
3. **Distillation and consolidation.** Besides sequential training, contilab compares
   *train-then-consolidate* pipelines that merge per-task experts into one model with
   **Task Arithmetic**, **TIES-Merging** or **multi-teacher knowledge distillation**.

## Features

| Area | What you get |
|---|---|
| Scenarios | Task-incremental (Task-IL), class-incremental (Class-IL), domain-incremental (Domain-IL) |
| Methods | Naive fine-tuning, Joint (upper bound), **EWC** (online), **LwF**, **Experience Replay (ER)**, **DER++**, **A-GEM**, **LoRA-per-task** |
| Consolidation | **Task Arithmetic**, **TIES-Merging** (full weights or LoRA ΔW), **multi-teacher distillation** with full / buffer / proxy transfer data |
| Benchmarks | Split MNIST, Permuted MNIST, Split CIFAR-10, Split CIFAR-100, **text5** (AG News → Yelp → Amazon → DBpedia → Yahoo Answers) with Hugging Face LMs (e.g. Pythia, Qwen2.5) |
| Metrics | 30+ quantitative metrics + figures (accuracy heatmaps, stability-gap curves, confusion, CKA, drift, t-SNE/PCA, loss barriers, critical-difference diagrams) |
| Statistics | Mean ± std over seeds, bootstrap 95 % CI, Wilcoxon / exact permutation tests, Friedman + Nemenyi CD |
| Engineering | Seed-parallel `vmap` ensembles, `torch.compile`, bf16, idempotent experiment matrix, per-task checkpoints and resume, failure isolation, divergence detection, Markdown/HTML reports |

## Quick start

```bash
git clone https://github.com/JonPark0/contilab && cd contilab
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# every method on tiny synthetic streams (~40 s on 4 CPU cores)
.venv/bin/contilab run configs/experiments/smoke_all.yaml
.venv/bin/contilab report runs/smoke_all          # runs/smoke_all/report.md + report.html

# the CPU study: Split/Permuted MNIST, Split CIFAR-10, text5 × 11 methods × 3 seeds
.venv/bin/contilab run configs/experiments/cpu_main.yaml
.venv/bin/contilab run configs/experiments/cpu_main.yaml "matrix.preset=[split_mnist_class]" train.epochs=2

# add a metric later without re-training (raw data are stored)
.venv/bin/contilab metrics runs/cpu_main --add cka_final recovery

# throughput: eager vs compile × ensemble width
.venv/bin/contilab bench --model mlp --E 1,2,4,8 --compile off,on
```

Datasets download from their original mirrors into `$CONTILAB_CACHE` (default `~/.cache/contilab`);
pretrained language models come from the Hugging Face Hub.

## Methods and references

| Method | Family | Reference |
|---|---|---|
| EWC (online) | regularisation | Kirkpatrick et al., *PNAS* 2017; Schwarz et al., *ICML* 2018 |
| LwF | distillation-based regularisation | Li & Hoiem, *TPAMI* 2017 |
| ER (reservoir) | replay | Chaudhry et al., 2019 (*On Tiny Episodic Memories*); Vitter 1985 |
| DER++ | replay + logit distillation | Buzzega et al., *NeurIPS* 2020 |
| A-GEM | gradient projection | Chaudhry et al., *ICLR* 2019 |
| LoRA-per-task | parameter isolation | Hu et al., *ICLR* 2022 |
| Task Arithmetic | model merging | Ilharco et al., *ICLR* 2023 |
| TIES-Merging | model merging | Yadav et al., *NeurIPS* 2023 |
| Multi-teacher distillation | consolidation | Hinton et al., 2015 |

## Metrics (excerpt)

| Axis | Metrics | References |
|---|---|---|
| Overall | ACC, Average Incremental Accuracy | Lopez-Paz & Ranzato 2017; Rebuffi et al. 2017 |
| Stability | BWT, forgetting, relative forgetting | Lopez-Paz & Ranzato 2017; Chaudhry et al. 2018 |
| Plasticity | FWT, intransigence, learning accuracy, LCA | Chaudhry et al. 2018/2019; Riemer et al. 2019 |
| Continual evaluation | min-ACC, WC-ACC, stability gap, windowed forgetting, anytime accuracy | De Lange et al. 2023; Caccia et al. 2022 |
| Representation | linear-probe forgetting (representation vs head), CKA, feature drift | Davari et al. 2022; Kornblith et al. 2019 |
| Plasticity loss | dormant-neuron ratio, effective rank, srank | Sokar et al. 2023; Kumar et al. 2021; Dohare et al. 2024 |
| Bias & calibration | task-recency bias, task confusion, ECE | Wu et al. 2019; Masana et al. 2022 |
| Recovery | few-shot relearning recovery (true vs spurious forgetting) | Zheng et al., *ICLR* 2025 |
| Landscape | linear-interpolation loss barrier, gradient interference | Mirzadeh et al. 2020 |
| LLMs | general-ability retention (perplexity drift) | Wang et al. 2023 (TRACE) |
| Cost | wall time, throughput, peak memory, FLOPs, trainable params, buffer/model bytes | Díaz-Rodríguez et al. 2018; Prabhu et al. 2023 |

Full definitions and formulas: [Metrics](docs/wiki/Metrics.md).

## Performance

Training throughput on 4 CPU cores (Intel Xeon @ 2.1 GHz with AMX), samples/s summed over ensemble members:

| Model | Baseline (fp32, eager, one seed) | Best configuration | Speed-up |
|---|---|---|---|
| MLP (2×400) | 10.8 k | fp32 eager, 8 seeds in one `vmap` ensemble: **40.7 k** | 3.8× |
| Reduced ResNet-18 | 318 | bf16 autocast + `torch.compile`, one seed: **730** | 2.3× |

Small models gain most from training seeds together; convolutional models gain from bf16 + compile.
Each benchmark preset selects its fastest configuration. Details: [Performance](docs/wiki/Performance.md).

## Documentation

- Wiki: <https://github.com/JonPark0/contilab/wiki> (mirrored in [`docs/wiki/`](docs/wiki))
- Design spec: [`docs/design/2026-10-22-contilab-design.md`](docs/design/2026-10-22-contilab-design.md)

## Citing

If contilab helps your research, please cite it (GitHub's *Cite this repository* uses [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{contilab2026,
  author  = {Park, Yeochoon},
  title   = {contilab: a performance-first framework for continual learning benchmarking, distillation and model merging},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/JonPark0/contilab}
}
```

## 한국어 요약

**contilab**은 지속 학습(Continual Learning, 연속 학습·평생 학습) 연구와 실증을 위한 PyTorch 프레임워크입니다.
여러 태스크를 순서대로 학습할 때 생기는 **파국적 망각(catastrophic forgetting)** 을 비전·언어(LLM) 모델에서 공정하게 비교합니다.

- **빠른 실험**: 여러 시드를 하나의 앙상블로 동시에 학습합니다(`vmap`). 여기에 `torch.compile`, bf16, DataLoader 없는 디바이스 상주 데이터를 씁니다.
- **풍부한 평가 지표 30여 종**: 다음 지표들을 저장된 원자료에서 사후 계산하므로, 재학습 없이 지표를 추가할 수 있습니다.
  - 정확도 행렬, BWT/FWT, 망각
  - 안정성 격차(stability gap), 최악 정확도
  - 선형 probe·CKA
  - 가소성 손실
  - 최근 태스크 편향, 보정(ECE)
  - 복구성, 손실 장벽
  - LLM 일반 능력 보존
- **통계**: 부트스트랩 신뢰구간, 대응 검정, Friedman/Nemenyi 임계 차이를 제공합니다.
- **증류·병합**: 태스크별 전문가 모델을 Task Arithmetic, TIES-Merging, 다중 교사 지식 증류로 하나로 통합해 순차 학습과 비교합니다.
- **지원 범위**
  - 메서드: EWC, LwF, ER, DER++, A-GEM, 태스크별 LoRA
  - 벤치마크: Split/Permuted MNIST, Split CIFAR, text5(AG News·Yelp·Amazon·DBpedia·Yahoo)

자세한 문서는 [위키](https://github.com/JonPark0/contilab/wiki)를 참고하세요.

## License

[MIT](LICENSE)
