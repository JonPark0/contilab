# Reproducibility

- **Seeds.** Each run's model initialisation and data order come from its seed; strategies and distillation draw
  from generators derived from the seed, so a run's result does not depend on which other seeds share its
  ensemble.
- **Equivalence tests.** A vmapped member equals the same seed trained alone, and compiled equals eager (within
  `atol=1e-5` in fp32). bf16 and multi-threaded CPU kernels are not bit-exact across machines.
- **Recorded environment.** `env.json` stores Python, PyTorch and platform versions, the CPU capability, the
  thread count, the git commit, and the compile and precision decisions with their reasons.
- **Resolved configuration.** `config.yaml` stores every value used; `run_id` is a hash of it.
- **Caches.** Preprocessed datasets and trained experts are cached by configuration hash under
  `$CONTILAB_CACHE`.
- **Resume.** Checkpoints include generator states, so a resumed run matches an uninterrupted one (tested).
- **Statistics.** Report at least 5 seeds when you need significance tests; with 3 seeds the exact permutation
  test cannot reach p < 0.25 and the report says so.
