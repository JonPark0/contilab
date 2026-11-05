---
title: "Continual learning methods: EWC, LwF, experience replay, ER-ACE, DER++, A-GEM, GEM, LoRA-per-task and O-LoRA"
description: "Task-, class- and domain-incremental scenarios and the continual-learning strategies in contilab: Naive fine-tuning, Joint training, EWC, LwF, Experience Replay, ER-ACE, DER++, A-GEM, GEM, LoRA-per-task and O-LoRA, with their hyperparameters."
nav_order: 5
nav_title: "Methods"
---

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
| EWC (online) | `ewc` | Quadratic penalty `λ/2 · Σ F (θ − θ*)²` with a diagonal empirical Fisher, accumulated `F ← γF + F_new` | λ = 30 / 10, γ = 1, 1024 Fisher samples, stiffness cap 1 |
| LwF | `lwf` | Distil the previous model's outputs on the current batch: `T² · KL(p_old ‖ p_new)` on old classes | T = 2, α = 1 |
| ER | `er` | Reservoir buffer of **dataset indices**; each step trains on current + replay batch | M = 500 / 200 |
| ER-ACE | `er_ace` | ER with an asymmetric cross-entropy: the incoming batch's softmax covers only the classes present in it, the replay batch's all allowed classes (Caccia et al. 2022) | M = 500 / 200 |
| DER++ | `derpp` | ER + MSE to logits stored at insertion time + CE on a second replay batch | M = 500 / 200, α = 0.1, β = 0.5 |
| A-GEM | `agem` | Project the gradient when it conflicts with a reference gradient from **previous tasks'** memory | M = 500 / 200, reference batch 256 / 64 |
| GEM | `gem` | One reference gradient **per previous task**; when the gradient conflicts with any of them, project it by the dual QP (Lopez-Paz & Ranzato 2017) | M = 500 / 200, reference batch 256 / 64, margin 0.5 |
| LoRA-per-task | `lora_per_task` | Frozen backbone, one LoRA adapter per task, shared head; Task-IL selects the task's adapter, Class/Domain-IL merges seen adapters with a per-seed λ chosen on validation data | r = 8, α = 16, lr 1e-3 with AdamW |
| O-LoRA | `olora` | Frozen backbone, a new LoRA adapter per task trained while the earlier ones stay active and frozen, with a penalty that keeps its subspace orthogonal to theirs; inference sums all adapters, without task identity (Wang et al. 2023) | r = 8, α = 16, λ₁ = 0.5, λ₂ = 0, lr 1e-3 with AdamW |

Implementation notes:

- **EWC** computes per-sample gradients with nested `vmap(grad)`, chunked to a 1 GiB memory budget, so it also
  runs on language models. Under SGD a parameter whose stiffness `lr · λ · F` exceeds 2 oscillates with growing
  amplitude around its anchor, and the accumulated Fisher only grows, so long streams used to diverge. The stored
  Fisher is clamped at `max_stiffness / (lr · λ)` (`method.params.max_stiffness`, default 1; `null` turns it off),
  which leaves every parameter below the bound unchanged. Adam-type optimisers normalise the step, so nothing is
  clamped for them.
- **Replay buffers** store global indices into the device-resident `TrainPool`, not copies of samples.
- **A-GEM** projects per member: `g ← g − (g·g_ref / ‖g_ref‖²) g_ref` only when `g·g_ref < 0`.
- **GEM** uses the same memory and reference-batch size as A-GEM, so the two differ only in the projection:
  - The reference batch is split evenly over the previous tasks, and each part gives that task's gradient `G_k`.
  - The original keeps a fixed memory per task and uses all of it every step.
  - Each part is a separate forward and backward pass, so a step costs about as much as A-GEM's.
  - When `G_k · g < 0` for some k, the update is `g + Gᵀv`, with v the solution of the dual
    `min ½ vᵀ(GGᵀ + εI)v + (Gg)ᵀv` subject to `v ≥ margin`; the margin (0.5) and ε (10⁻³) are those of the
    reference code.
  - The QP is solved exactly, per member and in batch: every possible set of free variables is solved and the
    best feasible solution kept. This is 2^K small linear systems for K ≤ 10 previous tasks; above that it
    falls back to coordinate descent.
- **ER-ACE** runs two forward passes (incoming and replay batch) and adds their losses, like the reference
  implementation, instead of ER's single pass over the concatenated batch.
- **O-LoRA** penalises `λ₁ · Σᵢ |A_i A_tᵀ|` (L1 over the entries, as in the reference code) between the new adapter
  `A_t` and every earlier adapter `A_i` of the same module. The original is a generative T5/LLaMA setup; here the
  backbone feeds contilab's shared classification head, which keeps training across tasks as in LoRA-per-task.
- **LoRA** is implemented in-house (no PEFT) for `nn.Linear` and `nn.Conv2d`; adapters merge as
  `ΔW = (α/r)·B·A`. A per-member scale buffer lets each seed use its own merge coefficient.
- Strategies draw randomness from generators derived from each member's seed, never from the data-order
  generator, so e.g. `EWC(λ=0)` reproduces `naive` exactly.

## Choosing hyperparameters

Defaults follow the literature. Sweep them with the experiment's `sweep:` block (see
[Configuration](Configuration.md)); each value appears as its own row (`ewc [method.params.lam=10]`) in tables.
