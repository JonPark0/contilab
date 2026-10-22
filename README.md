# contilab

Performance-first continual learning framework for benchmarking, distillation and consolidation —
vision and language models behind one interface.

- **Methods:** Naive, Joint, EWC (online), LwF, ER, DER++, A-GEM, LoRA-per-task
- **Train-then-consolidate:** Task Arithmetic, TIES, multi-teacher distillation (full / buffer / proxy transfer data)
- **Scenarios:** Task-IL, Class-IL, Domain-IL
- **Benchmarks:** synthetic, Split/Permuted MNIST, Split CIFAR-10/100, text5 (AG News → Yelp → Amazon → DBpedia → Yahoo)
- **Speed:** seeds are trained together as one stacked ensemble (`vmap(functional_call)`), `torch.compile`, bf16 when the hardware has it, DataLoader-free device-resident data
- **Metrics:** 30+ quantitative metrics (ACC/AIA/BWT/FWT/forgetting, min-ACC/WC-ACC/stability gap, probes/CKA, dormant neurons/rank, recency bias/ECE, recovery, loss barrier, gradient interference, LLM perplexity drift, cost) computed post hoc from stored raw data, plus statistics (bootstrap CI, paired tests, Friedman/Nemenyi CD)

Design: `docs/design/2026-10-22-contilab-design.md`

## Install

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

## Use

```bash
contilab run configs/experiments/smoke_all.yaml                 # every method on synthetic data, ~40 s on 4 CPUs
contilab run configs/experiments/cpu_main.yaml                  # CPU empirical study (hours)
contilab run configs/experiments/cpu_main.yaml "matrix.preset=[split_mnist_class]" train.epochs=2
contilab report runs/cpu_main                                   # runs/cpu_main/report.md + report.html
contilab metrics runs/cpu_main --add cka_final recovery         # add metrics later, no re-training
contilab bench --model mlp --E 1,2,4,8 --compile off,on         # ensemble / compile throughput
contilab data prepare split_cifar10                             # pre-download and cache a benchmark
```

Profiles (`configs/profiles/{smoke,cpu,gpu}.yaml`) hold scale; experiments (`configs/experiments/*.yaml`) hold the
matrix. Any config key can be overridden as `key=value`. Runs that already finished are skipped, interrupted runs
resume from their last task checkpoint.

Datasets download from the original mirrors (GCS for MNIST, fast.ai S3 for CIFAR and the text corpora) into
`$CONTILAB_CACHE` (default `~/.cache/contilab`). Pretrained language models come from the Hugging Face Hub.

## Test

```bash
.venv/bin/pytest                 # offline suite
.venv/bin/pytest --run-network   # + dataset download tests
```
