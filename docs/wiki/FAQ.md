# FAQ

**How is contilab different from other continual-learning libraries?**
It focuses on empirical comparison: seed-parallel training for throughput, metrics computed post hoc from raw
data (including continual evaluation, representation, plasticity-loss, calibration and recovery metrics), proper
statistics, and train-then-consolidate baselines (model merging and distillation) evaluated with the same metrics
as sequential methods.

**Does it support large language models?**
Yes. Any Hugging Face causal LM can be used through `model.name: hf`; LoRA targets the attention projections. The
`gpu` profile uses Qwen2.5-1.5B on text5; the `cpu` profile uses Pythia-70M.

**Why is EWC marked `diverged` in some runs?**
With the literature default λ = 100 and an accumulated (online) Fisher, the penalty becomes too stiff for the
learning rate on long or hard sequences, and the loss explodes. Diverged runs are kept and flagged rather than
dropped; sweep `method.params.lam` to tune it.

**Why is FWT empty in Class-IL?**
Before task j is learned, its classes are masked out of the Class-IL prediction, so its accuracy is zero by
construction and forward transfer is undefined.

**Why is windowed forgetting empty?**
It is undefined when the window Δ (10 % of a task's length) is shorter than the evaluation spacing ρ; lower
`metrics.rho`.

**Can I run on GPU?**
Yes: use the `gpu` profile or set `engine.device: cuda`. Mixed precision and compile are chosen automatically.

**How do I cite it?**
See [Citing](Citing.md).
