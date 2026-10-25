# Multivariate Weather Forecasting with a Quantile CNN-LSTM architecture, Adapted with LoRA / DoRA

IE-643 course project, team **The Epoch-alypse Survivors**
Jiya Gada (24B1013) and Dwija Patel (24B0995)

A Quantile CNN-LSTM forecasts the next 12 six-hourly steps of temperature anomaly from the past 72 steps of five ERA5 variables, and outputs a 90% prediction interval directly (quantiles 0.05 / 0.5 / 0.95). The model is pretrained on 72 grid points, then adapted to 9 held-out grid points with full fine-tuning, LoRA or DoRA. We compare these against zero-shot transfer and training from scratch, separately on normal and extreme events, with and without split-conformal calibration.

## Method in brief

| Step | What happens |
|---|---|
| Data | ERA5 single levels (CDS API), 6-hourly, 2015-2023, area 18-20N / 72-74E (81 grid points) |
| Features | Temperature, Dewpoint, Wind_Speed (from u/v), Cloud_Cover, Precipitation. The target is the temperature anomaly |
| Preprocessing (leak-free) | Per-location (day-of-year, hour) climatology from training years only, smoothed over +-7 days; anomaly; min-max scaling; 95th-percentile peaks-over-threshold extreme flag (training years only) |
| Windows | 72 input steps, 12 target steps |
| Splits | Train 2015-2020, validation 2021-2022, test 2023. 72 source and 9 target locations (fixed with `RandomState(0)`) |
| Model | Conv1d(5->64, k=3) + ReLU + MaxPool(2), 2-layer LSTM (64->50), Linear(50 -> 12x3). 46,460 parameters. Outputs are sorted to avoid quantile crossing |
| Loss | Pinball (quantile) loss; extreme targets get weight 3 during adaptation, weight 1 in pretraining |
| Optimiser | Adam, batch size 256, constant learning rate (no schedule), early stopping on a validation subset of at most 20,000 windows |
| Learning rates | Pretrain 5e-4 (up to 15 epochs, patience 6); scratch 2e-3; full fine-tune 5e-4; LoRA / DoRA 2e-3 (up to 20 epochs, patience 4) |
| LoRA / DoRA | Rank 4, alpha 8, on the conv weight, both LSTM layers (input and hidden weights) and the output layer; output bias also trained. 4,752 (10.2%) and 5,652 (12.2%) trainable parameters. Applied through `torch.func.functional_call` |
| Calibration | Split-conformal (CQR-style): margin from validation scores `max(lo - y, y - hi)`, applied to both interval bounds on test |
| Metrics | RMSE and MAE (median forecast), PICP (nominal 0.90), MPIW, reported for normal and extreme steps, mean +- std over seeds 0, 1, 2 |

## Repository contents

| File | Purpose |
|---|---|
| `model.py` | Full pipeline: download, preprocessing, pretraining, adaptation, evaluation, conformal calibration, Gradio demo (corresponding to E2) |
| `show_model.py` | Reopens the Gradio demo from saved checkpoints, with no training |
| `IE_643_Project_Report.pdf` | Report for the Course Project conducted and the results and conclusions |
| `era5_multivariate_data_{year}.nc` | ERA5 Reanalysis dataset for the corresponding years downloaded from cdsapi |
| `results_all_seeds.csv`, `results_summary.csv` | Written by the pipeline after a run |
| `base_seed{S}.pt`, `lora_r4_seed{S}.pt`, `dora_r4_seed{S}.pt` | Saved checkpoints (pretrained base, adapters) |
| `E0` | Contains the results from the experiment E0 |
| `E3` | Contains the results, checkpoints and the model used for experiment E3 |
| `E2 (main experiment)` | Contains the results, checkpoints and the model used for experiment E2 |

## Setup

Python 3.9+ is recommended.

```bash
pip install numpy pandas scipy torch xarray netcdf4 cdsapi gradio matplotlib
```

ERA5 download needs a free Copernicus CDS account:

1. Create an account at https://cds.climate.copernicus.eu and accept the ERA5 licence.
2. Put your API key in `~/.cdsapirc`.

Downloads are queued by CDS and can take hours. Existing `era5_multivariate_data_<year>.nc` files are skipped, so a stopped download resumes where it left off.

## Running

```bash
# full experiment: three seeds, then the web demo
python model.py

# one quick seed, without the demo
python model.py --seeds 0 --no-gradio

# reuse a pretrained base model
python model.py --base-ckpt base_seed0.pt
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--epochs-pre` | 15 | Max pretraining epochs |
| `--epochs-ad` | 20 | Max adaptation epochs |
| `--rank` | 4 | LoRA / DoRA rank |
| `--alpha` | 8 | LoRA / DoRA scaling |
| `--tail-weight` | 3.0 | Pinball weight on extreme targets during adaptation |
| `--n-target` | 9 | Number of held-out target grid points |
| `--seeds` | 0,1,2 | Comma-separated seeds |
| `--base-ckpt` | none | Load a pretrained state_dict and skip pretraining |
| `--no-gradio` | off | Do not launch the web interface |

A full three-seed run on CPU takes roughly 3-4 hours. A pretrained base saved as `base_seed{S}.pt` is reused automatically on the next run.

## Output

- Console table and `results_summary.csv`: mean +- std over seeds, for zero-shot, scratch, full fine-tune, LoRA and DoRA, on normal and extreme events, raw and with conformal calibration.
- `results_all_seeds.csv`: every seed and every row.
- Gradio demo: pick a 2023 test window of the target locations. It plots the history, the true future, the median forecast, the conformally widened 90% interval and the extreme steps (orange markers). It uses the final DoRA model.

## Reopening the demo without retraining

Training saves the adapted models. To reopen the interface from them:

```bash
python show_model.py
```

## Main results (three seeds, 2023 test year, target locations)

On hot extremes, LoRA reaches the lowest RMSE (0.127) against 0.131-0.135 for the other models, while training about 10% of the parameters. Normal-event accuracy is similar for all models. Raw interval coverage on extremes is only about 0.57-0.72 against the nominal 0.90, and marginal split-conformal calibration does not fix it. When cold extremes are included, the advantage of LoRA disappears.

## Notes and limitations

- Conformal calibration gives marginal coverage over all targets, not coverage conditional on extreme events.
- Planned extensions: extreme-aware (Mondrian and adaptive) conformal calibration, a low-data adaptation study and a forecast-horizon analysis.

## References

- Cheung et al., Quantile deep learning models for multi-step ahead time series prediction, Applied Soft Computing, 2025.
- Shen et al., Multi-Scale CNN-LSTM-Attention for temperature prediction in Eastern China, arXiv:2412.07997.
- Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, ICLR 2022.
- Liu et al., DoRA: Weight-Decomposed Low-Rank Adaptation, ICML 2024.
- Romano, Patterson and Candes, Conformalized Quantile Regression, NeurIPS 2019.
- Gibbs and Candes, Adaptive Conformal Inference Under Distribution Shift, NeurIPS 2021.
