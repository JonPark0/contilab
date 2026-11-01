"""`contilab` command line (§7.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(a) -> int:
    from contilab.runner.config import load_experiment
    from contilab.runner.scheduler import run_experiment

    try:
        exp = load_experiment(Path(a.config), a.overrides)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = run_experiment(exp, Path(a.root))
    print(f"results: {out / 'results.csv'}")
    return 0


def _cmd_report(a) -> int:
    from contilab.report.render import build_report

    md, html = build_report(Path(a.exp_dir))
    print(f"report: {md}\nhtml:   {html}")
    return 0


def _cmd_metrics(a) -> int:
    from contilab.metrics.registry import METRICS, load_all
    from contilab.runner.finalize import finalize_experiment

    load_all()
    unknown = [m for m in a.add if m not in METRICS]
    if unknown:
        print(f"error: unknown metric(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    print(f"results: {finalize_experiment(Path(a.exp_dir), add=a.add, tier=a.tier)}")
    return 0


def _cmd_bench(a) -> int:
    from contilab.runner.bench import run_bench

    rows = run_bench(a.model, [int(e) for e in a.E.split(",")], a.steps, a.compile.split(","), Path(a.out), a.device,
                     a.precision.split(","))
    print(f"{'model':10} {'E':>3} {'compile':8} {'precision':9} {'samples/s':>10} {'step ms':>8} {'vmap×':>6}")
    for r in rows:
        used = "on" if r["compile_used"] else "off"
        print(f"{r['model']:10} {r['E']:>3} {used:8} {r['precision_used']:9} {r['samples_per_s']:>10.0f} "
              f"{r['step_ms']:>8.2f} {r['vmap_speedup']:>6.2f}")
    return 0


def _cmd_data(a) -> int:
    from contilab.core.config import BenchmarkConfig
    from contilab.data.registry import build_stream

    s = build_stream(BenchmarkConfig(name=a.benchmark, tokenizer=a.tokenizer))
    print(f"{s.name}: {len(s.tasks)} tasks, {sum(len(t.train) for t in s.tasks)} train samples cached")
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="contilab", description="Performance-first continual learning framework")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment matrix")
    r.add_argument("config")
    r.add_argument("overrides", nargs="*", help="key=value (matrix.<axis>=[...] replaces a matrix axis)")
    r.add_argument("--root", default="runs")
    r.set_defaults(fn=_cmd_run)
    rp = sub.add_parser("report", help="build report.md / report.html")
    rp.add_argument("exp_dir")
    rp.set_defaults(fn=_cmd_report)
    m = sub.add_parser("metrics", help="compute metrics post hoc from stored raw data")
    m.add_argument("exp_dir")
    m.add_argument("--add", nargs="+", required=True)
    m.add_argument("--tier", default=None)
    m.set_defaults(fn=_cmd_metrics)
    b = sub.add_parser("bench", help="throughput: eager vs compile × ensemble width")
    b.add_argument("--model", default="mlp", choices=["mlp", "resnet18r", "hf"])
    b.add_argument("--E", default="1,2,4,8")
    b.add_argument("--steps", type=int, default=50)
    b.add_argument("--compile", default="off,on")
    b.add_argument("--precision", default="fp32,auto", help="fp32 | bf16 | fp16 | auto (what the profiles use)")
    b.add_argument("--device", default="auto")
    b.add_argument("--out", default="runs/bench")
    b.set_defaults(fn=_cmd_bench)
    d = sub.add_parser("data", help="dataset utilities")
    dsub = d.add_subparsers(dest="data_cmd", required=True)
    dp = dsub.add_parser("prepare", help="download and cache a benchmark")
    dp.add_argument("benchmark")
    dp.add_argument("--tokenizer", default="EleutherAI/pythia-70m")
    dp.set_defaults(fn=_cmd_data)
    a, extra = p.parse_known_args(argv)
    # before Python 3.12, argparse leaves positionals that follow an option (`run exp.yaml --root r k=v`) unparsed
    if extra and a.cmd == "run" and all("=" in e and not e.startswith("-") for e in extra):
        a.overrides = [*a.overrides, *extra]
    elif extra:
        p.error(f"unrecognized arguments: {' '.join(extra)}")
    return a


def main(argv: list[str] | None = None) -> int:
    a = _parse(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
