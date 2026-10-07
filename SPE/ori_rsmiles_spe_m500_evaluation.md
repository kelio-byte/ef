# ori_rsmiles_spe_m500_evaluation

Evaluation date: 2026-08-20

## Evaluation settings

- Dataset: `datasets/USPTO_50K_PtoR_aug20_SPE_m500`
- Evaluation split: `evaluation_v2/dev_unique1000_aug20`
- Sampler: `euler`
- Samples per product: `9`
- Sampling steps: `100`
- Scheduler: `cubic`
- Batch size: `32`
- Seed: `42`
- Augmentation: `20`
- `n_best`: `10`
- Scoring processes: `12`
- Device: `cuda`

## Checkpoint source

`checkpoints/retro_spe_m500_rsmiles_600k/USPTO_50K_PtoR_aug20_SPE_m500/2026-08-20_02-04-36`

## Summary

| Step | Top-1 | Top-3 | Top-5 | Top-10 | Oracle-any |
|---|---:|---:|---:|---:|---:|
| 300K | 42.4% | 60.5% | 64.0% | 65.9% | 81.7% |
| 450K | 43.5% | 62.0% | 65.1% | 66.6% | 82.9% |
| 490K | 41.4% | 59.8% | 63.4% | 64.7% | 81.5% |
| 500K | 42.6% | 60.7% | 64.2% | 65.5% | 81.7% |
| 550K | 45.5% | 61.5% | 64.9% | 66.7% | 83.4% |
| 600K | 43.8% | 61.9% | 64.9% | 66.0% | 82.8% |

## Detailed Top-N accuracy

| Step | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | Top-6 | Top-7 | Top-8 | Top-9 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300K | 42.4% | 55.4% | 60.5% | 63.3% | 64.0% | 64.7% | 65.3% | 65.5% | 65.7% | 65.9% |
| 450K | 43.5% | 56.9% | 62.0% | 63.6% | 65.1% | 65.5% | 65.9% | 66.1% | 66.4% | 66.6% |
| 490K | 41.4% | 54.5% | 59.8% | 62.3% | 63.4% | 64.0% | 64.3% | 64.3% | 64.3% | 64.7% |
| 500K | 42.6% | 55.3% | 60.7% | 63.0% | 64.2% | 64.6% | 64.9% | 65.1% | 65.2% | 65.5% |
| 550K | 45.5% | 57.9% | 61.5% | 63.3% | 64.9% | 65.7% | 66.1% | 66.3% | 66.5% | 66.7% |
| 600K | 43.8% | 57.4% | 61.9% | 63.8% | 64.9% | 65.4% | 65.8% | 65.8% | 65.9% | 66.0% |

## Coverage diagnostics

| Step | Oracle-any | Covered but outside Top-3 | Mean target final rank when covered |
|---|---:|---:|---:|
| 300K | 81.7% (817/1000) | 21.2% (212/1000) | 8.033 |
| 450K | 82.9% (829/1000) | 20.9% (209/1000) | 8.528 |
| 490K | 81.5% (815/1000) | 21.7% (217/1000) | 8.097 |
| 500K | 81.7% (817/1000) | 21.0% (210/1000) | 8.050 |
| 550K | 83.4% (834/1000) | 21.9% (219/1000) | 7.995 |
| 600K | 82.8% (828/1000) | 20.9% (209/1000) | 7.797 |

## Output directories

- `results/orig_rsmiles_spe_m500_step300k_dev_unique1000_euler_n9_seed42`
- `results/orig_rsmiles_spe_m500_step450k_dev_unique1000_euler_n9_seed42`
- `results/orig_rsmiles_spe_m500_step490k_dev_unique1000_euler_n9_seed42`
- `results/orig_rsmiles_spe_m500_step500k_dev_unique1000_euler_n9_seed42`
- `results/orig_rsmiles_spe_m500_step550k_dev_unique1000_euler_n9_seed42`
- `results/orig_rsmiles_spe_m500_step600k_dev_unique1000_euler_n9_seed42`

## Log files

- `logs/orig_rsmiles_spe_m500_step300k_dev1000_euler_n9.log`
- `logs/orig_rsmiles_spe_m500_step450k_dev1000_euler_n9.log`
- `logs/orig_rsmiles_spe_m500_step490k_dev1000_euler_n9.log`
- `logs/orig_rsmiles_spe_m500_step500k_dev1000_euler_n9.log`
- `logs/orig_rsmiles_spe_m500_step550k_dev1000_euler_n9.log`
- `logs/orig_rsmiles_spe_m500_step600k_dev1000_euler_n9.log`
