# Consolidation: merging and distillation

Besides training one model sequentially, contilab can **train one expert per task and consolidate** the experts
into a single model after every task. Because consolidation happens after each task, the accuracy matrix and
all metrics are defined exactly as for sequential methods.

## Experts

- Every expert starts from the same θ₀ (`consolidate.init_seed`), which merging requires.
- Expert *t* trains only on task *t* (full fine-tuning, or a single LoRA adapter + head with
  `consolidate.experts_lora: true`).
- Experts of equal-sized tasks train together in one `vmap` ensemble, up to `engine.max_E` members.
- Experts are cached on disk by configuration hash, so TA, TIES and distillation reuse them.

## Merging

| Method | Key | Rule |
|---|---|---|
| Task Arithmetic | `consolidate:ta` | θ = θ₀ + λ · Σ τᵢ, with τᵢ = θᵢ − θ₀ (Ilharco et al. 2023) |
| TIES-Merging | `consolidate:ties` | trim each τᵢ to its top-k magnitudes (k = 20 %) → elect the sign of Σ τᵢ → average only the agreeing values → θ₀ + λ · τ (Yadav et al. 2023) |

- λ is chosen per seed from `{0.1, …, 1.0}` by validation accuracy on the tasks seen so far.
- With LoRA experts the task vector is the adapter's `ΔW = (α/r)·B·A`, never A and B separately.
- The classifier head is assembled row by row: each class takes the row of the expert of the task that owns it
  (Domain-IL averages heads). In Class-IL this leaves logit-scale mismatch between experts — a known limitation
  that distillation can correct.

## Multi-teacher distillation

`consolidate:distill` trains a student with `T² · KL(target ‖ softmax(z/T))`:

- **Student initialisation** `consolidate.init`: `base` (θ₀), `ta` or `ties`.
- **Targets**: teacher *k* only speaks for its own classes. With `routing: routed` each sample uses its own
  task's teacher (always the case in Task-IL); with `routing: entropy` teachers are mixed by `softmax(−H)` of
  their prediction entropy. Targets are renormalised onto the classes the student may predict.
- **Transfer data** (`consolidate.transfer`), three levels of data availability:
  - `full`: all training data of the tasks seen so far (reference upper bound)
  - `buffer`: M samples in total, split evenly over tasks (same budget as replay)
  - `proxy`: unlabelled data from another source (CIFAR-10 for MNIST, CIFAR-100 ↔ CIFAR-10, WikiText for text)
- Teacher logits are computed once per consolidation step and reused for every distillation epoch.

## Cost accounting

The cost of a consolidation method includes training all experts and the distillation steps; model bytes include
the stored experts.

## Extension point

`Consolidator(student_factory=...)` and `on_new_expert(...)` are reserved for teacher → student continual
distillation into a different architecture.
