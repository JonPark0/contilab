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
3. **`torch.compile`** of the batched loss (`engine.compile: auto|on|off`). A failure on the first step falls
   back to eager and is recorded.
4. **bf16 autocast** (`engine.precision: auto|bf16|fp32`). `auto` enables it on CUDA with bf16 support or CPUs
   with AMX / AVX512-BF16.
5. **Process-level parallelism.** Groups run in a process pool (`workers × threads_per_worker`); language-model
   groups get the whole machine.

## Measured throughput

`contilab bench`, 4 CPU cores (Intel Xeon @ 2.1 GHz, AMX), samples/s summed over members:

| Model | Precision | Compile | E = 1 | E = 2 | E = 4 | E = 8 |
|---|---|---|---|---|---|---|
| MLP 2×400 | fp32 | off | 10,806 | 18,686 | 31,450 | **40,751** |
| MLP 2×400 | fp32 | on | 13,698 | 17,451 | 31,573 | 31,431 |
| MLP 2×400 | bf16 | off | 9,195 | — | — | 28,581 |
| MLP 2×400 | bf16 | on | 15,573 | — | — | 37,006 |
| Reduced ResNet-18 | fp32 | off | 318 | 266 | 314 | — |
| Reduced ResNet-18 | fp32 | on | 422 | 305 | 315 | — |
| Reduced ResNet-18 | bf16 | on | **730** | — | — | — |

Takeaways:

- For small models, training 8 seeds together is **3.8×** faster than one seed at a time; on this CPU fp32 eager
  beats bf16 and compile for the MLP.
- For convolutional models vmap brings nothing on CPU, but bf16 + compile gives **2.3×**.
- The `cpu` profile therefore uses `max_E: 8`, fp32, eager for MNIST presets and `max_E: 1`, bf16, compile for
  CIFAR. Run `contilab bench` on your own hardware and set `engine` per preset accordingly.

## Practical notes

- **Do not oversubscribe CPU threads.** PyTorch's OpenMP threads busy-wait; running a second CPU-heavy job next to
  a training process (even niced) slowed a language-model run by about 6×.
- **Memory.** Every ensemble member holds its own parameters, gradients and optimiser state; `engine.max_E` caps
  the group size (expert training for consolidation honours it too).
- Equivalence of the fast paths is tested: vmapped members equal individual runs, compiled equals eager.
