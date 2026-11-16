# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Validate complete measurements and plot paired speed/quality statistics."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def metadata(output):
    path = output / 'environment.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def summarize(config, rows):
    expected = {(c['name'], seed, repeat, r['algorithm'], r['epochs']) for c in config['cases']
                for seed in config['seeds'] for repeat in range(config['repeats']) for r in config['runs']}
    keys = [(r['case'], r['seed'], r['repeat'], r['algorithm'], r['epochs']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('missing or duplicate measurements; refusing partial results')
    groups, per_seed = defaultdict(list), {}
    for r in rows:
        groups[(r['case'], r['algorithm'], r['epochs'], r['seed'])].append(r)
    for key, samples in groups.items():
        for field in ['train_mse', 'test_mse', 'recall_at10']:
            if len({r[field] for r in samples}) != 1:
                raise ValueError('quality changed between deterministic timing repeats')
        times = [r['seconds'] for r in sorted(samples, key=lambda r: r['repeat'])]
        if not all(np.isfinite(t) and t > 0 for t in times):
            raise ValueError('invalid timing')
        value = {k: samples[0][k] for k in ['seed', 'train_mse', 'test_mse', 'recall_at10']}
        per_seed[key] = dict(value, seconds=median(times), timing_samples=times)
    summary = []
    for case in config['cases']:
        for run in config['runs']:
            samples = [per_seed[(case['name'], run['algorithm'], run['epochs'], seed)] for seed in config['seeds']]
            times = [r['seconds'] for r in samples]
            result = dict(case=case['name'], mode=case['mode'], algorithm=run['algorithm'], epochs=run['epochs'],
                          seconds=median(times), time_q1=float(np.percentile(times, 25)), time_q3=float(np.percentile(times, 75)),
                          train_mse=median(r['train_mse'] for r in samples), test_mse=median(r['test_mse'] for r in samples),
                          recall_at10=median(r['recall_at10'] for r in samples) if case['mode'] == 'pq' else None,
                          per_seed=samples, comparisons={})
            for budget in [5, 12, 25]:
                baseline = [per_seed[(case['name'], 'lloyd', budget, seed)] for seed in config['seeds']]
                deltas = [100 * (r['test_mse'] / b['test_mse'] - 1) for r, b in zip(samples, baseline)]
                result['comparisons'][str(budget)] = dict(
                    speedup=median(b['seconds'] for b in baseline) / median(times),
                    paired_speedups=[b['seconds'] / r['seconds'] for r, b in zip(samples, baseline)],
                    paired_test_mse_change_pct=median(deltas), test_change_q1=float(np.percentile(deltas, 25)), test_change_q3=float(np.percentile(deltas, 75)),
                    ratio_of_median_test_mse_change_pct=100 * (result['test_mse'] / median(b['test_mse'] for b in baseline) - 1),
                    paired_train_mse_change_pct=median(100 * (r['train_mse'] / b['train_mse'] - 1) for r, b in zip(samples, baseline)),
                    test_quality_matches=sum(r['test_mse'] <= b['test_mse'] for r, b in zip(samples, baseline)),
                    train_quality_matches=sum(r['train_mse'] <= b['train_mse'] for r, b in zip(samples, baseline)),
                    paired_recall_change_pp=median(100 * (r['recall_at10'] - b['recall_at10']) for r, b in zip(samples, baseline)) if case['mode'] == 'pq' else None,
                )
            summary.append(result)
    return summary


def plot_quality(config, summary, output):
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})
    fig = plt.figure(figsize=(11.6, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for case, ax in zip(config['cases'], axes):
        for algorithm, color, marker in [('lloyd', '#2869b2', 's'), ('emeans', '#ce5b28', 'o')]:
            points = sorted([r for r in summary if r['case'] == case['name'] and r['algorithm'] == algorithm], key=lambda r: r['epochs'])
            x = np.array([r['seconds'] for r in points])
            y = np.array([r['comparisons']['25']['paired_test_mse_change_pct'] for r in points])
            xerr = [x - [r['time_q1'] for r in points], [r['time_q3'] for r in points] - x]
            yerr = [y - [r['comparisons']['25']['test_change_q1'] for r in points], [r['comparisons']['25']['test_change_q3'] for r in points] - y]
            label = 'e-means (random init)' if algorithm == 'emeans' else ('Lloyd (same random init)' if case['mode'] == 'clustering' else 'LightPQ (k-means++ init)')
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, color=color, label=label, capsize=3, linewidth=1.5)
            for px, py, point in zip(x, y, points):
                ax.annotate(str(point['epochs']), (px, py), xytext=(6, 7), textcoords='offset points', color=color)
                if algorithm == 'emeans' and point['epochs'] == 5:
                    ax.scatter([px], [py], s=150, facecolors='none', edgecolors=color, linewidths=1.5)
        ax.axhline(0, color='#666666', linestyle='--', linewidth=0.8)
        ax.set_xscale('log'); ax.margins(x=0.12, y=0.15); ax.grid(alpha=0.2)
        details = f"{case['train_rows']:,} training rows; k={case['clusters']}"
        details += f"; {case['partitions']} x 8D" if case['mode'] == 'pq' else '; 128D'
        ax.set_title(case['name'] + '\n' + details)
        ax.set_xlabel('Training seconds, including initialization (log scale)')
        ax.set_ylabel('Held-out MSE vs Lloyd@25 (%)')
        ax.legend(frameon=False, fontsize=9)
    revision = metadata(output).get('implementation_commit', 'unrecorded')[:8]
    fig.suptitle(f"Rust e-means {revision}: speed versus quality\n{len(config['seeds'])} seeds x {config['repeats']} timing repeats; error bars = IQR; labels = epoch budgets", fontsize=15)
    fig.savefig(output / 'speed-quality.png', dpi=160, facecolor='white'); plt.close(fig)


def plot_speedups(config, summary, output):
    items = []
    for case in config['cases']:
        candidate = next(r for r in summary if r['case'] == case['name'] and r['algorithm'] == 'emeans' and r['epochs'] == 5)
        for budget in [5, 12, 25]:
            items.append((case['name'], budget, candidate['comparisons'][str(budget)]))
    fig, (ax, text_ax) = plt.subplots(1, 2, figsize=(11.6, 6.8), gridspec_kw={'width_ratios': [1.5, 1]}, constrained_layout=True)
    y = np.arange(len(items))
    ax.barh(y, [v['speedup'] for _, _, v in items], color='#ce5b28', alpha=0.8, height=0.65)
    for i, (_, _, values) in enumerate(items):
        ax.scatter(values['paired_speedups'], np.full(len(values['paired_speedups']), i), color='#333333', s=12, zorder=3)
        label_x = max(values['paired_speedups'] + [values['speedup']]) + 0.08
        ax.text(label_x, i, f"{values['speedup']:.2f}x", va='center', fontsize=11)
    ax.axvline(1, linestyle='--', color='#666666', linewidth=1)
    ax.set_yticks(y, [f'{name} / baseline @{budget}' for name, budget, _ in items])
    ax.invert_yaxis(); ax.set_xlabel('Speedup of default e-means (5 epochs)')
    ax.set_xlim(0, max(max(v['paired_speedups'] + [v['speedup']]) for _, _, v in items) * 1.2)
    ax.grid(axis='x', alpha=0.2)
    text_ax.set_ylim(ax.get_ylim()); text_ax.set_xlim(0, 1); text_ax.axis('off')
    text_ax.text(0, -0.75, 'MSE change       Recall change      MSE wins', fontsize=10, weight='bold')
    for i, (_, _, values) in enumerate(items):
        recall = 'n/a' if values['paired_recall_change_pp'] is None else f"{values['paired_recall_change_pp']:+.2f} pp"
        text_ax.text(0, i, f"{values['paired_test_mse_change_pct']:+.2f}%", va='center')
        text_ax.text(0.37, i, recall, va='center')
        text_ax.text(0.81, i, f"{values['test_quality_matches']}/{len(config['seeds'])}", va='center')
    revision = metadata(output).get('implementation_commit', 'unrecorded')[:8]
    fig.suptitle(f'Rust e-means {revision}: measured speedup and quality\nBars = ratios of median times; dots = paired seed ratios; both include initialization', fontsize=14)
    fig.savefig(output / 'speedup-quality.png', dpi=160, facecolor='white'); plt.close(fig)


def main():
    config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in Path(sys.argv[2]).read_text(encoding='utf-8').splitlines()]
    output = Path(sys.argv[3]); output.mkdir(parents=True, exist_ok=True)
    summary = summarize(config, rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_quality(config, summary, output)
    plot_speedups(config, summary, output)
    write_markdown(config, summary, len(rows), output)
    print(f'Validated {len(rows)} measurements and wrote {len(summary)} aggregate rows')


def write_markdown(config, summary, count, output):
    environment = metadata(output)
    revision = environment.get('implementation_commit', 'unrecorded')
    lines = [f"# e-means performance: {environment.get('date', 'date not recorded')}", '',
             f'{count} complete timed fits: {len(config["seeds"])} seeds x {config["repeats"]} repeats, three cases and eight budgets.', '',
             f"Measured on {environment.get('cpu', 'unrecorded CPU')}, {environment.get('os', 'unrecorded OS')}. Build and thread configuration are recorded in environment.json. "
             'Initialization and in-RAM training preparation are included; disk I/O, scoring and logging are excluded.', '',
             f'The measured algorithm is commit {revision}, including its composable-backend/Strided refactor. '
             'The benchmark uses the cached default SIMD backend. This replaces the earlier prototype/hardware measurements.', '',
             '## Speed versus quality', '', '![Speed versus quality](speed-quality.png)', '',
             'Left is faster; below zero means lower held-out MSE than the corresponding Lloyd@25 baseline. '
             'Labels are independent epoch budgets, not checkpoints. The outlined point is the default e-means budget of 5 epochs. '
             'Error bars are IQRs across five seeds, not confidence intervals.', '',
             '## Default e-means: five epochs', '', '![Speedup and quality](speedup-quality.png)', '',
             'Clustering uses matched random initial centers. PQ compares actual LightPQ (k-means++ + Lloyd) with '
             'the supplied random-initialization e-means adapter. PQ therefore changes initialization and optimizer together.', '',
             '| Case | Baseline epochs | Baseline seconds | e-means seconds | Speedup | Paired test MSE change | MSE wins | Paired Recall@10 change |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in config['cases']:
        candidate = next(r for r in summary if r['case'] == case['name'] and r['algorithm'] == 'emeans' and r['epochs'] == 5)
        for budget in [5, 12, 25]:
            baseline = next(r for r in summary if r['case'] == case['name'] and r['algorithm'] == 'lloyd' and r['epochs'] == budget)
            values = candidate['comparisons'][str(budget)]
            recall = 'n/a' if values['paired_recall_change_pp'] is None else f"{values['paired_recall_change_pp']:+.2f} pp"
            lines.append(f"| {case['name']} | {budget} | {baseline['seconds']:.4f} | {candidate['seconds']:.4f} | {values['speedup']:.2f}x | {values['paired_test_mse_change_pct']:+.3f}% | {values['test_quality_matches']}/{len(config['seeds'])} | {recall} |")
    lines += ['', '## All measured budgets', '',
              'MSE and recall changes below are paired per-seed medians relative to Lloyd@25. Negative MSE change and positive recall change are better. '
              'Speedup is the ratio of aggregate median times, after taking the median of three repeats within each seed. '
              'All three repeats must have bit-identical centers; their quality values are not independent samples.', '',
              '| Case | Method | Epochs | Seconds | Test MSE | Paired MSE change | Train/test MSE wins | Recall@10 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for point in summary:
        values = point['comparisons']['25']
        recall = 'n/a' if point['recall_at10'] is None else f"{point['recall_at10'] * 100:.2f}%"
        lines.append(f"| {point['case']} | {point['algorithm']} | {point['epochs']} | {point['seconds']:.4f} | {point['test_mse']:.3f} | {values['paired_test_mse_change_pct']:+.3f}% | {values['train_quality_matches']}/5, {values['test_quality_matches']}/5 | {recall} |")
    lines += ['', '## Limits and reproducibility', '',
              '- This is CPU training performance, not GPU performance or end-to-end DiskANN index construction.',
              '- Recall is exhaustive f64 PQ ADC against f64 exact squared-L2 ground truth, using the production byte encoder. It is not graph-search recall/QPS.',
              '- SIFT retrieval uses 100,000 database vectors and the first 1,000 supplied queries; YFCC uses 10,000 vectors and 100 queries. The small YFCC query set limits generalization.',
              '- Lower reconstruction MSE does not imply better recall. Both metrics and per-seed results are retained.',
              '- Fixed budgets do not establish that Lloyd cannot improve after 25 iterations. No universal 10x claim follows from these cases.',
              '- [Method and reproduction commands](../../README.md)',
              '- [Exact run configuration](config.json)',
              '- [Raw measurements](measurements.jsonl)',
              '- [Full statistics, including individual seeds and repeat timings](summary.json)',
              '- [Dataset hashes and split provenance](datasets.json)',
              '- [Measured source, binary and hardware metadata](environment.json)',
              '- [Dependency lock snapshot](dependency-lock.toml)', '',
              'For exact dependency versions, copy dependency-lock.toml to Cargo.lock in a fresh checkout of the recorded benchmark commit before building with --locked.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
