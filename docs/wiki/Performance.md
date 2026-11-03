---
title: "Performance: vmap seed ensembles, torch.compile and bf16 for continual learning"
description: "How contilab trains several seeds at once with torch.func.vmap, uses torch.compile and bf16, avoids a DataLoader, and the measured throughput on a 4-core CPU."
nav_order: 11
nav_title: "Performance"
---

# Performance

contilab is designed so that the expensive part of continual-learning studies — many methods × benchmarks ×
seeds — runs as fast as the hardware allows.

## What makes it fast

1. **Seed ensembles with `vmap`.** Runs that differ only by seed are trained as one ensemble of E members:
   parameters are stacked `[E, ...]` and the forward/backward pass runs once through
   `vmap(functional_call(...))`. Small models are dominated by Python and kernel-launch overhead, which this
   amortises.
2. **No DataLoader.** Datasets are preprocessed once, cached as safetensors and kept on the device; batches are
   gathered with index tensors. Replay buffers store indices, not copies.
3. **`torch.compile`** of the batched loss (`engine.compile: auto|on|off`). A compilation failure, on the
   first step or a later recompilation, falls back to eager and is recorded.
   - The batch axis is marked dynamic, so the last batch of a task, a filling replay buffer and A-GEM's reference
     batch (whose size changes at almost every step) reuse one graph instead of compiling one per size.
   - The graph limit per compiled function is the number of tasks + 8 (Dynamo's default is 8). LoRA-per-task
     needs one graph per task, because the active adapter is chosen in Python. Past the limit, Dynamo would
     silently run the function eagerly.
   - Under `vmap` (E > 1) some batching rules fix the batch size, so those graphs stay static.
4. **bf16 autocast** (`engine.precision: auto|bf16|fp32`). `auto` enables it on CUDA with bf16 support or CPUs
   with AMX / AVX512-BF16.
5. **Process-level parallelism.** Groups run in a process pool (`workers × threads_per_worker`); language-model
   groups get the whole machine.
6. **No repeated work on the hot path.**
   - The class table is built once per stream and device.
   - LwF runs one student forward per step.
   - Distillation samples from the run's data pool instead of rebuilding it.
   - The probe metrics share one fitted probe matrix.

A group that runs out of memory is re-run as two smaller groups, down to single members if needed. A strategy
that cannot be vmapped runs member by member. Both cases are recorded in each run's `env.json`.

## Measured throughput

`contilab bench`, 4 CPU cores (Intel Xeon @ 2.1 GHz, AMX), samples/s summed over members. The bench measures
`--precision fp32,auto` by default; `auto` is what the profiles train with (bf16 on this CPU).

| Model | Precision | Compile | E = 1 | E = 2 | E = 4 | E = 8 |
|---|---|---|---|---|---|---|
| MLP 2×400 | fp32 | off | 10,806 | 18,686 | 31,450 | **40,751** |
| MLP 2×400 | fp32 | on | 13,698 | 17,451 | 31,573 | 31,431 |
| MLP 2×400 | bf16 | off | 9,195 | — | — | 28,581 |
| MLP 2×400 | bf16 | on | 15,573 | — | — | 37,006 |
| Reduced ResNet-18 | fp32 | off | 318 | 266 | 314 | — |
| Reduced ResNet-18 | fp32 | on | 422 | 305 | 315 | — |
| Reduced ResNet-18 | bf16 | on | **730** | — | — | — |

A later run of `contilab bench --model resnet18r --E 1` on the same machine, two repetitions:

| Model | Precision | Compile | E = 1 |
|---|---|---|---|
| Reduced ResNet-18 | fp32 | off | 361–364 |
| Reduced ResNet-18 | fp32 | on | 381–455 |
| Reduced ResNet-18 | bf16 | off | 586–652 |
| Reduced ResNet-18 | bf16 | on | 595–613 |

Takeaways:

- For small models, training 8 seeds together is **3.8×** faster than one seed at a time; on this CPU fp32 eager
  beats bf16 and compile for the MLP.
- For convolutional models vmap brings nothing on CPU, but bf16 + compile gives **2.3×**. Most of it comes from
  bf16: in the later run, compile added nothing over bf16 eager at steady state, and compiling one ResNet graph
  took about 50 s.
- The `cpu` profile therefore uses `max_E: 8`, fp32, eager for MNIST presets and `max_E: 1`, bf16, compile for
  CIFAR. Run `contilab bench` on your own hardware and set `engine` per preset accordingly.

## Practical notes

- **Do not oversubscribe CPU threads.** PyTorch's OpenMP threads busy-wait; running a second CPU-heavy job next to
  a training process (even niced) slowed a language-model run by about 6×.
- **Memory.** Every ensemble member holds its own parameters, gradients and optimiser state; `engine.max_E` caps
  the group size (expert training for consolidation honours it too).
- Equivalence of the fast paths is tested: vmapped members equal individual runs, compiled equals eager.
