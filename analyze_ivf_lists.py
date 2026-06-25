"""Analyze graph-IVF posting-list size distributions from .graphivf_meta files.

The meta file is a 52-byte little-endian header followed by num_clusters u32
counts:
  u32 magic, u32 version, u32 metric, u32 element_size, u32 dim,
  u64 num_points, u64 num_clusters,
  u32 degree, u32 l_build, f32 slack, f32 alpha,
  then [u32; num_clusters] per-cluster point counts.
"""

import struct
import sys

import numpy as np

HEADER = struct.Struct("<5I2Q2I2f")  # 52 bytes
DATA_DIR = "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4"

INDEXES = [
    ("2048", "graphivf_index.graphivf_meta"),
    ("16384 (100K sample)", "graphivf_index_16384.graphivf_meta"),
    ("16384 (full corpus)", "graphivf_index_16384_full.graphivf_meta"),
    ("40960 (400K sample)", "graphivf_index_40960.graphivf_meta"),
    ("40960 (full refine)", "graphivf_index_40960_full.graphivf_meta"),
]


def load_counts(path):
    with open(path, "rb") as f:
        raw = f.read()
    (magic, version, metric, elem, dim, npts, nclu, deg, lb, slack, alpha) = (
        HEADER.unpack_from(raw, 0)
    )
    counts = np.frombuffer(raw, dtype="<u4", count=nclu, offset=HEADER.size)
    return counts.astype(np.int64), npts, dim


def stats(counts, npts):
    c = counts
    total = int(c.sum())
    mean = c.mean()
    # Size-weighted mean = expected list size for a uniformly random *point*
    # (= sum c_i^2 / sum c_i). Equals mean only for a perfectly balanced index;
    # grows with imbalance. This tracks bytes-read-per-probe because queries
    # land in dense regions, which are exactly the big lists.
    sw_mean = (c.astype(np.float64) ** 2).sum() / total
    # Gini coefficient of the list sizes.
    cs = np.sort(c)
    n = len(cs)
    gini = (2.0 * np.arange(1, n + 1) - n - 1).dot(cs) / (n * cs.sum())
    return {
        "clusters": n,
        "points": total,
        "mean": mean,
        "std": c.std(),
        "cv": c.std() / mean,
        "min": int(c.min()),
        "p50": float(np.percentile(c, 50)),
        "p90": float(np.percentile(c, 90)),
        "p99": float(np.percentile(c, 99)),
        "p999": float(np.percentile(c, 99.9)),
        "max": int(c.max()),
        "empty": int((c == 0).sum()),
        "sw_mean": sw_mean,
        "gini": gini,
    }


def histogram(counts, bins):
    hist, edges = np.histogram(counts, bins=bins)
    maxh = hist.max()
    lines = []
    for i, h in enumerate(hist):
        lo, hi = int(edges[i]), int(edges[i + 1])
        bar = "#" * int(round(40 * h / maxh)) if maxh else ""
        pct = 100.0 * h / len(counts)
        lines.append(f"  [{lo:>5},{hi:>5}) {h:>6} ({pct:5.1f}%) {bar}")
    return "\n".join(lines)


def main():
    loaded = []
    for name, fname in INDEXES:
        path = f"{DATA_DIR}/{fname}"
        try:
            counts, npts, dim = load_counts(path)
        except FileNotFoundError:
            print(f"SKIP {name}: {path} not found")
            continue
        loaded.append((name, counts, npts, dim))

    # Summary table.
    cols = [
        ("clusters", "{:>8d}"),
        ("points", "{:>9d}"),
        ("mean", "{:>7.1f}"),
        ("std", "{:>7.1f}"),
        ("cv", "{:>6.2f}"),
        ("min", "{:>5d}"),
        ("p50", "{:>6.0f}"),
        ("p90", "{:>6.0f}"),
        ("p99", "{:>6.0f}"),
        ("p999", "{:>7.0f}"),
        ("max", "{:>6d}"),
        ("empty", "{:>6d}"),
        ("sw_mean", "{:>8.1f}"),
        ("gini", "{:>5.3f}"),
    ]
    header = "index".ljust(22) + "".join(
        k.rjust(int(fmt.split(">")[1].split("d")[0].split(".")[0]) + 1)
        for k, fmt in cols
    )
    print(header)
    print("-" * len(header))
    per_index = []
    for name, counts, npts, dim in loaded:
        s = stats(counts, npts)
        per_index.append((name, counts, s))
        row = name.ljust(22) + "".join(fmt.format(s[k]) for k, fmt in cols)
        print(row)

    # Per-index histograms over a shared bin scheme.
    for name, counts, s in per_index:
        print()
        print(f"=== {name}: list-size histogram ===")
        hi = int(np.percentile(counts, 99.5))
        bins = np.linspace(0, max(hi, 1), 21)
        print(histogram(counts, bins))
        print(
            f"  (mean {s['mean']:.1f}, size-weighted mean {s['sw_mean']:.1f}, "
            f"max {s['max']}, empty {s['empty']})"
        )


if __name__ == "__main__":
    main()
