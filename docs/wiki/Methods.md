# Methods

## Scenarios

contilab uses **one shared classification head** for all scenarios and expresses the scenario as a per-sample
logit mask (disallowed logits are set to −1e9):

| Scenario | Training mask | Evaluation mask |
|---|---|---|
| Task-IL (task-incremental) | classes of the sample's own task | classes of the evaluated task |
| Class-IL (class-incremental) | all classes seen so far | all classes seen so far |
| Domain-IL (domain-incremental) | all (shared) classes | all classes |

Replayed samples always use **their own** task's mask, which matters in Task-IL.

## Strategies

All strategies share the same optimiser, epochs, batch size and memory budget. Defaults (vision / text):

| Strategy | Key | Idea | Defaults |
|---|---|---|---|
| Naive fine-tuning | `naive` | Train on each task in turn (lower bound) | — |
| Joint | `joint` | All tasks merged into one task (upper bound); evaluated per original task | — |
| EWC (online) | `ewc` | Quadratic penalty `λ/2 · Σ F (θ − θ*)²` with a diagonal empirical Fisher, accumulated `F ← γF + F_new` | λ = 100 / 10, γ = 1, 1024 Fisher samples |
| LwF | `lwf` | Distil the previous model's outputs on the current batch: `T² · KL(p_old ‖ p_new)` on old classes | T = 2, α = 1 |
| ER | `er` | Reservoir buffer of **dataset indices**; each step trains on current + replay batch | M = 500 / 200 |
| DER++ | `derpp` | ER + MSE to logits stored at insertion time + CE on a second replay batch | M = 500 / 200, α = 0.1, β = 0.5 |
| A-GEM | `agem` | Project the gradient when it conflicts with a reference gradient from **previous tasks'** memory | M = 500 / 200, reference batch 256 / 64 |
| LoRA-per-task | `lora_per_task` | Frozen backbone, one LoRA adapter per task, shared head; Task-IL selects the task's adapter, Class/Domain-IL merges seen adapters with a per-seed λ chosen on validation data | r = 8, α = 16, lr 1e-3 with AdamW |

Implementation notes:

- **EWC** computes per-sample gradients with nested `vmap(grad)`, chunked to a 1 GiB memory budget, so it also
  runs on language models.
- **Replay buffers** store global indices into the device-resident `TrainPool`, not copies of samples.
- **A-GEM** projects per member: `g ← g − (g·g_ref / ‖g_ref‖²) g_ref` only when `g·g_ref < 0`.
- **LoRA** is implemented in-house (no PEFT) for `nn.Linear` and `nn.Conv2d`; adapters merge as
  `ΔW = (α/r)·B·A`. A per-member scale buffer lets each seed use its own merge coefficient.
- Strategies draw randomness from generators derived from each member's seed, never from the data-order
  generator, so e.g. `EWC(λ=0)` reproduces `naive` exactly.

## Choosing hyperparameters

Defaults follow the literature. Sweep them with the experiment's `sweep:` block (see
[Configuration](Configuration.md)); each value appears as its own row (`ewc [method.params.lam=10]`) in tables.
