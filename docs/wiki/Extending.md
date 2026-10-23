# Extending contilab

## Add a strategy

A strategy has a **per-member hot path** (vmapped and optionally compiled by the engine) and **host-side hooks**.

```python
import torch.nn.functional as F
from contilab.strategies.base import Strategy, register_strategy, cross_entropy


@register_strategy("my_method")
class MyMethod(Strategy):
    DEFAULTS = {"vision": {"strength": 1.0}, "text": {"strength": 0.1}}

    def __init__(self, strength: float = 1.0):
        super().__init__()
        self.strength = strength

    # hot path: ONE member; p = that member's parameters, s = self.state() sliced to the member
    def loss(self, p, s, batch, replay, f):
        z = f.logits(p, batch.x, batch.attn_mask, batch.mask)
        loss = cross_entropy(z, batch.y)
        if "anchor.head.weight" in s:
            loss = loss + self.strength * (p["head.weight"] - s["anchor.head.weight"]).pow(2).sum()
        return loss, z

    def state(self):              # stacked [E, ...] tensors passed into the hot path
        return self.tensors

    def after_task(self, ctx, task):  # host side, runs outside vmap/compile
        self.tensors["anchor.head.weight"] = ctx.ens.params["head.weight"].detach().clone()
```

Then use `method.name: my_method` in an experiment. Other hooks: `prepare_model`, `trainable_names`,
`before_task`, `replay_batch`, `after_step`, `eval_setup`, `ref_loss` + `transform_grad` (gradient surgery, see
A-GEM), `state_dict` / `load_state_dict` (checkpoint resume). Draw randomness from
`derived_gens(ctx, salt)`, never from `ctx.gens`, so data order stays identical across methods.

Checklist: a test that the method with its strength set to 0 equals `naive`, and a vmap-equivalence test
(E = k members equal k single runs).

## Add a metric

```python
from contilab.metrics.registry import metric, MetricUnavailable
from contilab.metrics.accuracy import acc_matrix_of


@metric("final_min_task_acc", requires=("acc_matrix",), tier="core")
def final_min_task_acc(log, mctx):
    R = acc_matrix_of(log)
    if R.shape[0] < 2:
        raise MetricUnavailable("needs at least 2 tasks")
    return float(R[-1].min())
```

Put it in a module listed in `contilab/metrics/registry.py` (`_MODULES`). Because raw data are stored, existing
experiments get the new metric with `contilab metrics runs/<exp> --add final_min_task_acc`.

## Add a benchmark

Write a builder that returns a `Stream` (tasks with `train`/`val`/`test` `TensorSplit`s and global class ids),
cache preprocessing with `cached_tensors`, and register its name in `contilab/data/registry.py`. Add a preset to a
profile to run it.

## Add a consolidation method

Subclass `Consolidator`, implement `consolidate(experts, upto, cctx) -> params`, and register it in
`build_consolidator` under `consolidate:<name>`.
