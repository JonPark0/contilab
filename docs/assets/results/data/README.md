# Per-run results of the CPU study

One row per run (preset × method × variant × seed). These are the `results.csv` files that `contilab run`
writes, copied from the run directories. The raw logs (about 3 GB) are not kept in the repository.

| File | Experiment config | Runs |
|---|---|---|
| `cpu_main.csv` | `configs/experiments/cpu_main.yaml` + `cpu_main_seeds.yaml` (seeds 3–5 on MNIST) | 324 |
| `cpu_tuned.csv` | `configs/experiments/cpu_tuned.yaml` | 228 |
| `cpu_student.csv` | `configs/experiments/cpu_student.yaml` | 12 |

- **Identifier columns**: `run_id`, `preset`, `method`, `variant`, `seed`, `status`. `cpu_tuned` adds `selected`
  (chosen on validation accuracy) and `cell` (comparison-axis part of the variant).
- **Metric columns**: every other column is a scalar metric. Definitions are on the
  [Metrics](https://github.com/JonPark0/contilab/wiki/Metrics) page.
- **Empty cells**: the metric is undefined for that run. Examples are FWT in Class-IL, stream metrics for
  consolidation methods, and anything for a diverged run.

Rebuild the tables of the Results page with pandas:

```python
import pandas as pd
from contilab.report.tables import main_table, rank_table

df = pd.read_csv("docs/assets/results/data/cpu_main.csv")
print(main_table(df, "split_mnist_class"))
print(rank_table(df)[0])
```
