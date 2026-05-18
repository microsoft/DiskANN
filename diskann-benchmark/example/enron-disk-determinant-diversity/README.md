# Enron disk determinant-diversity benchmark results

This folder contains local benchmark configs and raw outputs for DiskANN determinant-diversity search on the Mimir Enron disk index.

## Dataset

Azure Artifacts Universal Package:

- Organization: `https://dev.azure.com/msdata/`
- Project: `CosmosDB`
- Feed: `DiskANN_Test_Data`
- Package: `mimir-t1-email-test-enron-index`
- Version: `0.0.1`

Download from the DiskANN repo root:

```powershell
az artifacts universal download `
  --organization https://dev.azure.com/msdata/ `
  --project CosmosDB `
  --scope project `
  --feed DiskANN_Test_Data `
  --name mimir-t1-email-test-enron-index `
  --version 0.0.1 `
  --path .\tmp\mimir-t1-email-test-enron-index
```

The package includes a prebuilt FP16 cosine-normalized disk index:

`tmp\mimir-t1-email-test-enron-index\normalized_dim_384_vector_fp16_1087932_vectors_r_59_l_100*`

## Commands

Run from the DiskANN repo root.

PR858-style parameters matching `openai-disk-determinant-diversity-compare.json` (`eta=0.01`, `power=2.0`) with `search_list=[100,200,400]`:

```powershell
cargo run --release --package diskann-benchmark --features disk-index -- run `
  --input-file diskann-benchmark\example\enron-disk-determinant-diversity\enron-pr858-params-sl100-200-400-k10.json `
  --output-file .\tmp\enron-pr858-params-sl100-200-400-k10-output.json
```

Eta/power sweep (`eta in {0.1,0.01,0.001}`, `power in {1,2,3}`) with `search_list=[1000,400]`:

```powershell
cargo run --release --package diskann-benchmark --features disk-index -- run `
  --input-file diskann-benchmark\example\enron-disk-determinant-diversity\enron-detdiv-eta-power-sweep-sl1000-400-k10.json `
  --output-file .\tmp\enron-detdiv-eta-power-sweep-sl1000-400-k10-output.json
```

## PR858-style parameter results

| Mode | L | Recall@10 | QPS | Mean latency |
|---|---:|---:|---:|---:|
| Baseline | 100 | 95.31 | 596.34 | 13.37 ms |
| Baseline | 200 | 97.58 | 336.56 | 23.67 ms |
| Baseline | 400 | 98.93 | 168.04 | 47.35 ms |
| Det-div eta=0.01 power=2 | 100 | 0.09 | 349.51 | 22.66 ms |
| Det-div eta=0.01 power=2 | 200 | 0.09 | 183.51 | 43.48 ms |
| Det-div eta=0.01 power=2 | 400 | 0.05 | 92.17 | 86.59 ms |

## Eta/power sweep summary

All eta/power combinations in the sweep produced near-zero vector Recall@10 for determinant-diversity while the baseline remained high.

| Mode | eta | power | L | Recall@10 | QPS | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | - | - | 1000 | 99.58 | 70.88 | 112.53 ms |
| Baseline | - | - | 400 | 98.93 | 173.54 | 45.91 ms |
| Det-div | 0.1 | 1 | 1000 | 0.00 | 37.33 | 213.70 ms |
| Det-div | 0.1 | 1 | 400 | 0.06 | 92.38 | 86.30 ms |
| Det-div | 0.1 | 2 | 1000 | 0.00 | 37.54 | 212.54 ms |
| Det-div | 0.1 | 2 | 400 | 0.05 | 93.72 | 85.14 ms |
| Det-div | 0.1 | 3 | 1000 | 0.00 | 37.46 | 212.96 ms |
| Det-div | 0.1 | 3 | 400 | 0.04 | 93.45 | 85.43 ms |
| Det-div | 0.01 | 1 | 1000 | 0.00 | 37.47 | 212.94 ms |
| Det-div | 0.01 | 1 | 400 | 0.06 | 93.84 | 84.96 ms |
| Det-div | 0.01 | 2 | 1000 | 0.00 | 37.78 | 211.19 ms |
| Det-div | 0.01 | 2 | 400 | 0.05 | 92.60 | 86.24 ms |
| Det-div | 0.01 | 3 | 1000 | 0.00 | 37.78 | 211.21 ms |
| Det-div | 0.01 | 3 | 400 | 0.04 | 93.92 | 84.92 ms |
| Det-div | 0.001 | 1 | 1000 | 0.00 | 36.49 | 218.69 ms |
| Det-div | 0.001 | 1 | 400 | 0.06 | 90.73 | 87.84 ms |
| Det-div | 0.001 | 2 | 1000 | 0.00 | 37.75 | 211.50 ms |
| Det-div | 0.001 | 2 | 400 | 0.05 | 93.40 | 85.42 ms |
| Det-div | 0.001 | 3 | 1000 | 0.00 | 37.84 | 211.07 ms |
| Det-div | 0.001 | 3 | 400 | 0.04 | 93.69 | 84.98 ms |

Recall is vector-level Recall@10 from `diskann-benchmark` against vector-ID ground truth.
