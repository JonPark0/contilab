# Running Experiments

## Commands

| Command | Purpose |
|---|---|
| `contilab run <experiment.yaml> [key=value ...] [--root runs]` | Run (or resume) an experiment matrix and compute metrics |
| `contilab report <runs/exp>` | Write `report.md` and a single-file `report.html` |
| `contilab metrics <runs/exp> --add <name ...> [--tier T]` | Compute metrics later from stored raw data |
| `contilab bench [--model mlp\|resnet18r\|hf] [--E 1,2,4,8] [--compile off,on]` | Throughput benchmark |
| `contilab data prepare <benchmark>` | Download and cache a dataset |

## Run directory

```
runs/<experiment>/
  experiment.yaml          # expanded experiment definition
  <preset>-<method>-s<seed>-<hash>/
    config.yaml            # fully resolved configuration
    env.json               # versions, hardware, compile/precision decisions and reasons
    status.json            # pending | running | done | diverged | failed (+ traceback)
    stream.json            # scenario and class table (metrics never rebuild datasets)
    raw/*.jsonl, raw/*.safetensors
    metrics.json
  _ckpt/                   # per-group checkpoints of unfinished runs
  results.csv              # one row per run, one column per scalar metric
  report.md, report.html, figs/
```

## Resume and failure handling

- Re-running the same command skips `done` and `diverged` runs.
- An interrupted group resumes from its last finished task: parameters, strategy state (buffers, Fisher,
  snapshots) and every random-generator state are restored, giving the same result as an uninterrupted run.
- A run that restarts from scratch clears its old raw records first.
- An exception marks the run `failed` with its traceback; other runs continue. A group that fails before its first
  optimisation step (for example a vmap-incompatible operation) is retried member by member.
- A member whose loss becomes NaN or exceeds 1000 × its first loss is marked `diverged` and kept in the results.

## Reports

`report.md` / `report.html` contain, per benchmark: the main table (mean ± std, `*` = significant against the
baseline), per-axis tables (stability, plasticity, representation, plasticity loss, bias and calibration,
recovery, efficiency), figures of a representative seed, evaluation settings (ρ, β, Δ), cross-benchmark ranks with
a critical-difference diagram, the throughput chart and a limitations section.
