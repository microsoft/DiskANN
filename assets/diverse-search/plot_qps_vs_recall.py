#!/usr/bin/env python3
"""Generate the QPS-vs-recall scatter charts for the diverse-search benchmark doc.

Reads chart_data.json (recall/QPS at L=100 for each method and dataset) and
writes one PNG per dataset next to this script. Kept in the repo so the charts
can be regenerated whenever the benchmark numbers change.

Usage:
    python plot_qps_vs_recall.py

Requires matplotlib (pip install matplotlib).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "chart_data.json"


def main() -> None:
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    methods = data["methods"]
    colors = data["colors"]

    for chart in data["charts"]:
        fig, ax = plt.subplots(figsize=(9, 6))

        for method in methods:
            point = chart["points"][method]
            recall = point["recall"]
            qps = point["qps"]
            ax.scatter(
                recall,
                qps,
                s=160,
                color=colors[method],
                label=method,
                zorder=3,
            )
            ax.annotate(
                method,
                (recall, qps),
                textcoords="offset points",
                xytext=(8, 8),
                color=colors[method],
                fontsize=10,
            )

        ax.set_title(f"{chart['title']}\nQPS vs recall at L=100")
        ax.set_xlabel("Recall @10 (%)  -  higher is better")
        ax.set_ylabel("QPS  -  higher is better")
        ax.grid(True, color="0.9", zorder=0)
        ax.legend(title="Approach", loc="lower left")

        out_path = HERE / chart["output"]
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
