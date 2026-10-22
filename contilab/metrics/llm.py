"""General-ability retention for language models (TRACE's General Ability Delta, perplexity form)."""

from __future__ import annotations

from contilab.metrics.registry import MetricUnavailable, metric


def _ppl(log) -> dict[int, float]:
    out = {r["task"]: r["ppl"] for r in log.records("general_ppl")}
    if -1 not in out or len(out) < 2:
        raise MetricUnavailable("needs base and post-training perplexity")
    return out


@metric("general_ppl_delta", ("general_ppl",), "extended")
def general_ppl_delta(log, mctx):
    p = _ppl(log)
    return float(p[max(p)] - p[-1])


@metric("general_ppl_curve", ("general_ppl",), "extended")
def general_ppl_curve(log, mctx):
    p = _ppl(log)
    return {"per_task": [p[t] for t in sorted(p)]}
