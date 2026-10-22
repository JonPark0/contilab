"""report.md + single-file report.html (figures embedded) for one experiment directory (§7.6)."""

from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import pandas as pd
import yaml

from contilab.metrics.stats import low_power_warning
from contilab.report import figures
from contilab.report.tables import axis_tables, main_table, methods_in, rank_table

AXIS_TITLES = {"stability": "안정성", "plasticity": "가소성", "representation": "표현", "plasticity_loss": "가소성 손실",
               "bias_calibration": "편향·신뢰도", "recovery": "복구성·지형", "efficiency": "효율"}
LIMITATIONS = [
    "Class-IL 병합(TA/TIES)은 헤드 행을 전문가별로 가져오므로 전문가 간 로짓 스케일 불일치가 남는다; 증류가 이를 보정하는지가 비교 축이다.",
    "CPU 프로파일은 데이터를 서브샘플하고 소형 모델을 쓰므로 절대 성능보다 메서드 간 상대 비교로 읽어야 한다.",
    "연속 평가 지표(min-ACC, WC-ACC, stability gap, WF)는 ρ 스텝 간격 표본이므로 ρ보다 짧은 낙폭은 놓칠 수 있다.",
]


def _representatives(exp_dir: Path, df: pd.DataFrame, preset: str) -> dict[str, Path]:
    out = {}
    d = df[(df.preset == preset) & df.status.isin(["done", "diverged"])]
    for m in methods_in(d):
        r = d[d.method == m].sort_values("seed").iloc[0]
        out[m] = exp_dir / r["run_id"]
    return out


def _env_summary(exp_dir: Path) -> list[str]:
    envs = sorted(exp_dir.glob("*/env.json"))
    if not envs:
        return []
    e = json.loads(envs[0].read_text())
    lines = [f"- torch {e.get('torch')} · python {e.get('python')} · {e.get('platform')}",
             f"- device {e.get('device')} · CPU capability {e.get('cpu_capability')} · threads {e.get('threads')}",
             f"- precision: {e.get('precision')}"]
    comp = {json.loads(p.read_text()).get("compile", {}).get("reason", "") for p in envs}
    lines.append(f"- compile: {'; '.join(sorted(c for c in comp if c))}")
    return lines


def build_report(exp_dir: Path) -> tuple[Path, Path]:
    exp_dir = Path(exp_dir)
    df = pd.read_csv(exp_dir / "results.csv")
    exp = yaml.safe_load((exp_dir / "experiment.yaml").read_text()) if (exp_dir / "experiment.yaml").exists() else {}
    baseline = exp.get("baseline", "naive")
    figs_root = exp_dir / "figs"
    md = [f"# contilab report — {exp.get('name', exp_dir.name)}", ""]
    st = df.status.value_counts().to_dict()
    md += [f"runs: {len(df)} ({', '.join(f'{k} {v}' for k, v in st.items())}) · baseline for significance (*, p<0.05): `{baseline}`", ""]
    n_seeds = int(df.groupby(["preset", "method"]).seed.nunique().min()) if len(df) else 0
    warn = low_power_warning(n_seeds)
    if warn:
        md += [f"> {warn}", ""]
    env = _env_summary(exp_dir)
    if env:
        md += ["## 실행 환경", ""] + env + [""]
    failed = df[df.status == "failed"]
    if len(failed):
        md += ["## 실패한 run", ""] + [f"- `{r}`" for r in failed.run_id] + [""]
    for preset in list(dict.fromkeys(df.preset)):
        md += [f"## {preset}", "", "### 주요 지표", "", main_table(df, preset, baseline)]
        for axis, tab in axis_tables(df, preset, baseline).items():
            md += [f"### {AXIS_TITLES.get(axis, axis)}", "", tab]
        runs = _representatives(exp_dir, df, preset)
        out = figs_root / preset
        made = [p for fn in figures.PRESET_FIGURES if (p := fn(runs, out)) is not None]
        if made:
            md += ["### 그림 (대표 seed)", ""] + [f"![{p.stem}]({p.relative_to(exp_dir)})" for p in made] + [""]
    if df.preset.nunique() >= 2:
        tab, stats = rank_table(df, "acc")
        md += ["## 벤치마크 간 순위 (ACC)", "", tab]
        cd = figures.cd_diagram(stats["avg_ranks"], stats["cd"], figs_root)
        if cd:
            md += [f"![cd_diagram]({cd.relative_to(exp_dir)})", ""]
    bench = figures.bench_bars(exp_dir.parent / "bench" / "bench.json", figs_root)
    if bench:
        md += ["## 처리량 (bench)", "", f"![bench_bars]({bench.relative_to(exp_dir)})", ""]
    md += ["## 한계", ""] + [f"- {l}" for l in LIMITATIONS] + ([f"- {warn}"] if warn else []) + [""]
    md_path = exp_dir / "report.md"
    md_path.write_text("\n".join(md))
    html_path = exp_dir / "report.html"
    html_path.write_text(_to_html("\n".join(md), exp_dir, exp.get("name", exp_dir.name)))
    return md_path, html_path


_IMG = re.compile(r"^!\[(.*?)\]\((.*?)\)$")


def _inline(s: str) -> str:
    s = html.escape(s)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def _to_html(md: str, base: Path, title: str) -> str:
    out, table = [], []

    def flush_table():
        if not table:
            return
        rows = [r.strip().strip("|").split("|") for r in table if not re.match(r"^\|[-| ]+\|$", r.strip())]
        h = "".join(f"<th>{_inline(c.strip())}</th>" for c in rows[0])
        b = "".join("<tr>" + "".join(f"<td>{_inline(c.strip())}</td>" for c in r) + "</tr>" for r in rows[1:])
        out.append(f"<div class='tw'><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>")
        table.clear()

    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if m := _IMG.match(line):
            data = base64.b64encode((base / m.group(2)).read_bytes()).decode()
            out.append(f"<figure><img alt='{html.escape(m.group(1))}' src='data:image/png;base64,{data}'>"
                       f"<figcaption>{html.escape(m.group(1))}</figcaption></figure>")
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{_inline(line[level:].strip())}</h{level}>")
        elif line.startswith("> "):
            out.append(f"<p class='warn'>{_inline(line[2:])}</p>")
        elif line.startswith("- "):
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    css = ("body{font-family:system-ui,sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1200px;margin:auto;padding:16px}"
           "table{border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #e4e3df;padding:4px 8px;text-align:right}"
           "th:first-child,td:first-child{text-align:left}.tw{overflow-x:auto}img{max-width:100%}"
           ".warn{background:#fff4e0;padding:8px}code{background:#f0efec;padding:0 3px}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(out)}</body></html>"
