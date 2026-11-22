"""Launch a Gradio demo for a saved matched E4 experiment checkpoint.

Edit the settings below to select a head, weighting, seed, adapter, and
calibration variant. The corresponding experiment checkpoint must exist.
"""
import os
import sys

import numpy as np
import torch


# Select an experiment checkpoint and interval calibration.
SEED = 2
HEAD = "gauss_nll"              # pinball, expectile, gauss_nll, t_nll, crps
WEIGHTING = "sym"          # none, sym, asym, magnitude, focal, covreg
ADAPTER = "dora"              # dora, lora, full
CALIBRATION = "raw"      # raw, cqr_global, cqr_asym, cqr_mondrian
RANK = 4
ALPHA = 8
N_TARGET = 9
REGIME_PCT = 10
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

E4_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(E4_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "E3"))
sys.path.insert(0, E4_DIR)

import modelE3
import tail_loss_experiment as experiment


def load_model():
    if HEAD not in experiment.HEADS:
        raise ValueError(f"Unknown HEAD {HEAD!r}; choose from {experiment.HEADS}.")
    if WEIGHTING not in experiment.WEIGHTINGS:
        raise ValueError(f"Unknown WEIGHTING {WEIGHTING!r}; choose from {experiment.WEIGHTINGS}.")
    if ADAPTER not in ("dora", "lora", "full"):
        raise ValueError("ADAPTER must be 'dora', 'lora', or 'full'.")
    if CALIBRATION not in ("raw", "cqr_global", "cqr_asym", "cqr_mondrian"):
        raise ValueError("Choose a deployable calibration: raw, cqr_global, cqr_asym, or cqr_mondrian.")

    base = experiment.new_base(HEAD, len(modelE3.FEATURES))
    if ADAPTER == "full":
        model = base
    else:
        model = experiment.AdaptedModel(
            base, r=RANK, alpha=ALPHA, dora=(ADAPTER == "dora")
        )

    tag = f"{ADAPTER}_{HEAD}_{WEIGHTING}_seed{SEED}"
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{tag}.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model, tag


def load_data():
    os.chdir(REPO_ROOT)
    modelE3.download_era5_data()
    df, features = modelE3.load_multi_location_era5()
    locations = sorted(df["location_id"].unique())
    target_locations = set(np.random.RandomState(0).choice(locations, N_TARGET, replace=False))
    splits = experiment.build_splits_tails(df, features, target_locations)
    return splits[("target", "val")], splits[("target", "test")]


def get_test_intervals(model, tag, val, test):
    val_pred = experiment.predict_head(model, val[0], HEAD)
    test_pred = experiment.predict_head(model, test[0], HEAD)
    calibrated = experiment.calibrate_variants(
        val_pred, val[1], val[2], test_pred, test[2], REGIME_PCT
    )

    if CALIBRATION == "cqr_asym":
        calibration_path = os.path.join(CHECKPOINT_DIR, f"{tag}_cqr_asym.npz")
        if not os.path.isfile(calibration_path):
            raise FileNotFoundError(f"Asymmetric calibration file not found: {calibration_path}")
        with np.load(calibration_path) as saved:
            lower_margin = float(saved["lower"])
            upper_margin = float(saved["upper"])
        return (
            test_pred[..., 0] - lower_margin,
            test_pred[..., 1],
            test_pred[..., 2] + upper_margin,
        )

    lower, upper = calibrated[CALIBRATION]
    return lower, test_pred[..., 1], upper


def make_predict_interface(model, test, intervals):
    import matplotlib.pyplot as plt

    X, Y, C = test
    low, med, high = intervals

    def predict_interface(window_idx, history_shown):
        i = int(window_idx)
        history_steps = int(history_shown)
        steps = np.arange(1, modelE3.PRED_LEN + 1)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            np.arange(-history_steps + 1, 1),
            X[i, -history_steps:, 0],
            color="gray",
            label="History",
        )
        ax.plot(steps, Y[i], "k--", label="Actual")
        ax.plot(steps, med[i], color="blue", marker="o", label="Median forecast")
        ax.fill_between(steps, low[i], high[i], color="red", alpha=0.2, label="Prediction interval")

        upper_extremes = np.where(C[i] == 1)[0]
        lower_extremes = np.where(C[i] == 2)[0]
        if len(upper_extremes):
            ax.scatter(upper_extremes + 1, Y[i][upper_extremes], color="darkorange",
                       zorder=5, s=55, label="Upper-tail target")
        if len(lower_extremes):
            ax.scatter(lower_extremes + 1, Y[i][lower_extremes], color="purple",
                       zorder=5, s=55, label="Lower-tail target")

        ax.axvline(0.5, color="lightgray", ls=":")
        ax.set_xlabel("Forecast step (6-hour increments)")
        ax.set_ylabel("Scaled temperature anomaly")
        ax.set_title(f"E4 forecast: {HEAD} / {WEIGHTING} / {CALIBRATION}")
        ax.legend(loc="best", fontsize=8)
        return fig

    return predict_interface


def main():
    model, tag = load_model()
    val, test = load_data()
    intervals = get_test_intervals(model, tag, val, test)

    print(f"Loaded {tag} with {CALIBRATION} calibration.")
    for subset, mask in (
        ("upper", test[2] == 1),
        ("lower", test[2] == 2),
        ("extreme", test[2] > 0),
        ("all", np.ones_like(test[2], dtype=bool)),
    ):
        print(f"{subset}: {experiment.interval_metrics(test[1][mask], intervals[0][mask], intervals[1][mask], intervals[2][mask])}")

    import gradio as gr

    interface = gr.Interface(
        fn=make_predict_interface(model, test, intervals),
        inputs=[
            gr.Slider(
                minimum=0,
                maximum=len(test[0]) - 1,
                step=1,
                value=0,
                label="2023 test-window index (target locations)",
            ),
            gr.Slider(
                minimum=24,
                maximum=72,
                step=12,
                value=48,
                label="History steps shown (model input remains 72)",
            ),
        ],
        outputs="plot",
        title="E4 Tail-Aware Weather Forecast Demo",
    )
    interface.launch(share=True)


if __name__ == "__main__":
    main()
