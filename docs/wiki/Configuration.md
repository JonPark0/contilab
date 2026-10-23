# Configuration

An experiment is a YAML file that names a **profile** and a **matrix**:

```yaml
# configs/experiments/cpu_main.yaml
name: cpu_main
profile: cpu
matrix:
  preset: [split_mnist_class, split_mnist_task, permuted_mnist, split_cifar10_class, text5_class]
  method.name: [naive, joint, ewc, lwf, er, derpp, agem, lora_per_task,
                'consolidate:ta', 'consolidate:ties', 'consolidate:distill']
  seed: [0, 1, 2]
workers: 2
threads_per_worker: 2
```

A **profile** (`configs/profiles/{smoke,cpu,gpu}.yaml`) holds `defaults` and named `presets` (benchmark + model +
training scale).

## Merge order

Later entries win:

1. dataclass defaults (`contilab/core/config.py`)
2. `profile.defaults`
3. the preset
4. `experiment.overrides`
5. matrix axes (dotted paths such as `method.name`, `consolidate.transfer`)
6. `sweep` values
7. command-line `key=value`

Unknown keys are errors.

## Command-line overrides

```bash
contilab run configs/experiments/cpu_main.yaml train.epochs=2 optim.lr=0.01
contilab run configs/experiments/cpu_main.yaml "matrix.preset=[split_mnist_class]" "matrix.seed=[0]"
```

`matrix.<axis>=[...]` replaces a matrix axis.

## Sweeps and variants

```yaml
sweep:
  method.params.lam: [1, 10, 100]
```

Any extra matrix axis or sweep key becomes a `variant` column in `results.csv` and its own row in report tables.

## Main keys

| Section | Keys (default) |
|---|---|
| `benchmark` | `name`, `scenario` (class), `n_train`, `n_test`, `n_tasks`, `max_len` (128), `tokenizer`, `task_order_seed` |
| `model` | `name` (mlp \| resnet18r \| hf), `hidden` (400), `depth` (2), `nf` (20), `hf_name`, `lora.{r: 8, alpha: 16, targets, lr: 1e-3}` |
| `optim` | `name` (sgd \| adamw), `lr` (0.03), `weight_decay`, `momentum`, `clip` |
| `train` | `epochs` (1), `batch_size` (32), `eval_batch_size` (256) |
| `engine` | `device` (auto), `compile` (auto), `precision` (auto), `max_E` (8) |
| `method` | `name`, `params` (strategy-specific, merged over per-modality defaults) |
| `consolidate` | `lam_grid` (0.1…1.0), `ties_k` (0.2), `init` (base), `routing` (entropy), `transfer` (buffer), `M` (500), `temperature` (2), `epochs` (1), `experts_lora` (false), `init_seed` (0) |
| `metrics` | `tier` (core \| extended \| full), `rho` (50), `stream_subset` (200), `beta` (10), `probe_size` (256), `recovery_n` (50), `recovery_steps` (20), `window_frac` (0.1), `ece_bins` (15), `dormant_tau` (0.025), `srank_delta` (0.01), `ppl_n` (200), `barrier_points` (11) |
