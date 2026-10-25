"""
Multivariate weather forecasting with a Quantile CNN-LSTM, adapted with LoRA / DoRA.

Pipeline
  1. Load ERA5 (same download/loader as before, now with Dewpoint as a 5th feature)
  2. Leak-free preprocessing: climatology, scaling and the POT threshold are all
     computed on the TRAIN period only (train 2015-2020, val 2021-2022, test 2023)
  3. Pretrain the base Quantile CNN-LSTM on SOURCE grid points
  4. Adapt to held-out TARGET grid points with: full fine-tune, LoRA, DoRA
  5. Compare zero-shot / scratch / full-FT / LoRA / DoRA on normal vs extreme events,
     over several seeds (mean +- std), before and after split-conformal calibration
  6. (optional) Gradio demo that uses the real adapted model on real test windows
"""
import os, glob, zipfile, copy, math, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import genpareto

warnings.filterwarnings("ignore", message=".*RNN module weights.*")

SEQ_LEN, PRED_LEN = 72, 12
QUANTILES = [0.05, 0.5, 0.95]
FEATURES = ['Temperature', 'Dewpoint', 'Wind_Speed', 'Cloud_Cover', 'Precipitation']
TRAIN_END = pd.Timestamp('2021-01-01')   # train: 2015-2020
VAL_END = pd.Timestamp('2023-01-01')     # val: 2021-2022, test: 2023
EXTREME_PCT = 95                         # POT threshold


# ==========================================
# 1. Data retrieval + loading
# ==========================================
def download_era5_data():
    import cdsapi
    years = [str(y) for y in range(2015, 2024)]   # 2015-2023
    c = cdsapi.Client()
    for year in years:
        filename = f'era5_multivariate_data_{year}.nc'
        if os.path.exists(filename):
            print(f"File {filename} already exists. Skipping.")
            continue
        print(f"Downloading ERA5 data for {year}...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                             'total_cloud_cover', 'total_precipitation', '2m_dewpoint_temperature'],
                'year': [year],
                'month': [str(i).zfill(2) for i in range(1, 13)],
                'day': [str(i).zfill(2) for i in range(1, 32)],
                'time': [f"{str(i).zfill(2)}:00" for i in range(0, 24, 6)],
                'area': [20.0, 72.0, 18.0, 74.0],
            },
            filename)


def load_multi_location_era5(file_pattern='era5_multivariate_data_*.nc'):
    import xarray as xr
    extracted_files = []
    for path in sorted(glob.glob(file_pattern)):
        if zipfile.is_zipfile(path):
            extract_dir = f"extracted_{os.path.basename(path).split('.')[0]}"
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(extract_dir)
            extracted_files.extend(glob.glob(f"{extract_dir}/*.nc"))
        else:
            extracted_files.append(path)
    if not extracted_files:
        raise ValueError("No NetCDF files found.")

    dfs = []
    for f in extracted_files:
        with xr.open_dataset(f, engine='netcdf4') as ds:
            if 'valid_time' in ds.coords or 'valid_time' in ds.dims:
                ds = ds.rename({'valid_time': 'time'})
            dfs.append(ds.to_dataframe().reset_index())
    df_raw = pd.concat(dfs, ignore_index=True)

    col_mapping = {'t2m': 'Temperature', '2t': 'Temperature', 'd2m': 'Dewpoint', '2d': 'Dewpoint',
                   'tcc': 'Cloud_Cover', 'tp': 'Precipitation', '10u': 'u10', '10v': 'v10'}
    df_raw.rename(columns={k: v for k, v in col_mapping.items() if k in df_raw.columns}, inplace=True)
    df_raw['Wind_Speed'] = np.sqrt(df_raw['u10'] ** 2 + df_raw['v10'] ** 2)

    lat_col = 'latitude' if 'latitude' in df_raw.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df_raw.columns else 'lon'
    df_raw['location_id'] = df_raw[lat_col].round(2).astype(str) + '_' + df_raw[lon_col].round(2).astype(str)

    feats = [f for f in FEATURES if f in df_raw.columns]
    assert feats[0] == 'Temperature', "Temperature must be the first feature (it is the target)."
    df = df_raw.groupby(['location_id', 'time'])[feats].first().reset_index()
    df[feats] = df.groupby('location_id')[feats].transform(lambda g: g.ffill().bfill())
    df = df.dropna().sort_values(['location_id', 'time'])
    print(f"Loaded {len(df)} rows across {df['location_id'].nunique()} locations.")
    return df, feats


# ==========================================
# 2. Leak-free preprocessing (train-period statistics only)
# ==========================================
def preprocess_location(group, feats):
    """Climatological anomaly -> min-max scale -> POT extremes. Everything is fit on train period only."""
    group = group.sort_values('time')
    times = pd.DatetimeIndex(group['time'])
    vals = group[feats].to_numpy(dtype=np.float64)
    train_mask = np.asarray(times < TRAIN_END)
    doy, hr = times.dayofyear.values, times.hour.values

    anom = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        d = pd.DataFrame({'d': doy[train_mask], 'h': hr[train_mask], 'v': vals[train_mask, j]})
        m = d.groupby(['d', 'h'])['v'].mean().unstack().reindex(index=range(1, 367))
        ext = pd.concat([m.iloc[-7:], m, m.iloc[:7]])                       # circular smoothing over day-of-year
        sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
        sm = sm.interpolate(limit_direction='both').to_numpy()
        hi_idx = m.columns.get_indexer(hr)
        anom[:, j] = vals[:, j] - sm[doy - 1, hi_idx]

    lo, hi = anom[train_mask].min(0), anom[train_mask].max(0)
    scaled = (anom - lo) / (hi - lo + 1e-8)

    thr = np.percentile(anom[train_mask, 0], EXTREME_PCT)                   # POT threshold u (train only)
    is_ext = anom[:, 0] > thr
    exc = anom[train_mask, 0][anom[train_mask, 0] > thr] - thr
    gpd = genpareto.fit(exc, floc=0) if len(exc) > 20 else (np.nan, 0, np.nan)  # diagnostic: tail shape/scale
    return dict(scaled=scaled.astype(np.float32), ext=is_ext, times=times, gpd=gpd)


def make_windows(scaled, ext, times):
    T = len(scaled)
    N = T - SEQ_LEN - PRED_LEN + 1
    if N <= 0:
        return None
    X = sliding_window_view(scaled, SEQ_LEN, axis=0)[:N].transpose(0, 2, 1)
    Y = sliding_window_view(scaled[SEQ_LEN:, 0], PRED_LEN)[:N]
    E = sliding_window_view(ext[SEQ_LEN:], PRED_LEN)[:N]
    t_last = times[SEQ_LEN + PRED_LEN - 1: SEQ_LEN + PRED_LEN - 1 + N]      # time of last target step
    return (np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(Y, dtype=np.float32),
            np.ascontiguousarray(E), np.asarray(t_last))


def build_splits(df, feats, target_locs):
    """Returns dict[(domain, split)] -> (X, Y, E) with domain in {source, target}."""
    bucket = {}
    gpd_report = []
    for loc, g in df.groupby('location_id'):
        pre = preprocess_location(g, feats)
        w = make_windows(pre['scaled'], pre['ext'], pre['times'])
        if w is None:
            continue
        X, Y, E, tl = w
        dom = 'target' if loc in target_locs else 'source'
        gpd_report.append(pre['gpd'])
        for name, mask in [('train', tl < np.datetime64(TRAIN_END)),
                           ('val', (tl >= np.datetime64(TRAIN_END)) & (tl < np.datetime64(VAL_END))),
                           ('test', tl >= np.datetime64(VAL_END))]:
            bucket.setdefault((dom, name), []).append((X[mask], Y[mask], E[mask]))
    out = {k: tuple(np.concatenate(z, 0) for z in zip(*v)) for k, v in bucket.items()}
    # g = np.array([x[0] for x in gpd_report if not np.isnan(x[0])])
    # if len(g):
    #     print(f"GPD tail shape xi over locations: mean={g.mean():.3f}, min={g.min():.3f}, max={g.max():.3f}")
    return out


# ==========================================
# 3. Model, adapters, loss
# ==========================================
class QuantileCNNLSTM(nn.Module):
    def __init__(self, input_size, pred_len, quantiles):
        super().__init__()
        self.quantiles, self.pred_len = quantiles, pred_len
        self.cnn = nn.Sequential(nn.Conv1d(input_size, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(64, 50, batch_first=True, num_layers=2)
        self.fc = nn.Linear(50, pred_len * len(quantiles))

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(-1, self.pred_len, len(self.quantiles))


class LowRankAdapter(nn.Module):
    """W' = W + (alpha/r) B A   (LoRA).   With dora=True: W' = m * (W + BA) / ||W + BA||_row  (DoRA)."""
    def __init__(self, weight, r, alpha, dora):
        super().__init__()
        out = weight.shape[0]
        in_flat = weight[0].numel()
        self.shape, self.scale, self.dora = tuple(weight.shape), alpha / r, dora
        self.A = nn.Parameter(torch.empty(r, in_flat))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(out, r))                      # zero init -> starts identical to base
        if dora:
            self.m = nn.Parameter(weight.detach().reshape(out, -1).norm(dim=1, keepdim=True).clone())

    def forward(self, W):
        W2 = W.reshape(W.shape[0], -1) + self.scale * (self.B @ self.A)
        if self.dora:
            W2 = self.m * W2 / (W2.norm(dim=1, keepdim=True) + 1e-8)
        return W2.reshape(self.shape)


class AdaptedModel(nn.Module):
    """Frozen base + LoRA/DoRA on Conv1d, both LSTM layers (ih & hh) and the output Linear.
    Uses functional_call so adapters also work on nn.LSTM's raw weight tensors."""
    def __init__(self, base, r=4, alpha=8, dora=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.fc.bias.requires_grad = True                          # tiny, useful for recalibrating the head
        names = ['cnn.0.weight', 'fc.weight']
        for l in range(base.lstm.num_layers):
            names += [f'lstm.weight_ih_l{l}', f'lstm.weight_hh_l{l}']
        self.names = names
        pd_ = dict(base.named_parameters())
        self.adapters = nn.ModuleDict({n.replace('.', '_'): LowRankAdapter(pd_[n], r, alpha, dora) for n in names})

    def forward(self, x):
        P = dict(self.base.named_parameters())
        new = {n: self.adapters[n.replace('.', '_')](P[n]) for n in self.names}
        return functional_call(self.base, new, (x,))


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def pinball_loss(y, pred, quantiles, w=None):
    loss = 0
    for i, q in enumerate(quantiles):
        e = y - pred[:, :, i]
        l = torch.maximum((q - 1) * e, q * e)
        loss = loss + (l * w).mean() if w is not None else loss + l.mean()
    return loss / len(quantiles)


# ==========================================
# 4. Training / inference / evaluation
# ==========================================
def predict(model, X, bs=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(torch.as_tensor(X[i:i + bs])).numpy())
    return np.sort(np.concatenate(outs, 0), axis=-1)                   # sort => no quantile crossing


def fit(model, train, val, epochs, lr, tail_w=1.0, bs=256, patience=4, tag=""):
    X, Y, E = (torch.as_tensor(a) for a in train)
    loader = DataLoader(TensorDataset(X, Y, E.float()), batch_size=bs, shuffle=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr)
    sel = np.linspace(0, len(val[0]) - 1, min(len(val[0]), 20000)).astype(int)   # evenly spaced val subset
    Xv, Yv = torch.as_tensor(val[0][sel]), torch.as_tensor(val[1][sel])
    best, best_state, bad = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        tot = 0
        for bx, by, be in loader:
            opt.zero_grad()
            loss = pinball_loss(by, model(bx), QUANTILES, 1 + (tail_w - 1) * be)
            loss.backward()
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = pinball_loss(Yv, model(Xv), QUANTILES).item()
        print(f"   [{tag}] epoch {ep + 1}/{epochs}  train {tot / len(loader):.4f}  val {vl:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def metrics(y, lo, med, hi):
    if y.size == 0:
        return dict(n=0, RMSE=np.nan, MAE=np.nan, PICP=np.nan, MPIW=np.nan)
    return dict(n=int(y.size),
                RMSE=float(np.sqrt(np.mean((y - med) ** 2))), MAE=float(np.mean(np.abs(y - med))),
                PICP=float(np.mean((y >= lo) & (y <= hi))), MPIW=float(np.mean(hi - lo)))


def evaluate(model, test):
    X, Y, E = test
    p = predict(model, X)
    lo, med, hi = p[..., 0], p[..., 1], p[..., 2]
    return {name: metrics(Y[m], lo[m], med[m], hi[m])
            for name, m in [('normal', ~E), ('extreme', E), ('all', np.ones_like(E, bool))]}


# ==========================================
# 5. Gradio demo (real model, real test windows)
# ==========================================
def make_predict_interface(model, test, margin=0.0):
    """Returns predict_interface(window_idx, history_shown) -> matplotlib figure.
    margin > 0 applies the conformal widening computed on the validation set."""
    import matplotlib
    import matplotlib.pyplot as plt
    X, Y, E = test
 
    def predict_interface(window_idx, history_shown):
        i = int(window_idx)
        h_n = int(history_shown)
        p = predict(model, X[i:i + 1])[0]                       # model always reads the full 72-step window
        lo, med, hi = p[:, 0] - margin, p[:, 1], p[:, 2] + margin
        hist = X[i, -h_n:, 0]                                    # scaled temperature anomaly (feature 0)
        steps = np.arange(1, PRED_LEN + 1)
 
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(-h_n + 1, 1), hist, color='gray', label='History (input)')
        ax.plot(steps, Y[i], 'k--', label='Actual')
        ax.plot(steps, med, color='blue', marker='o', label='Median forecast')
        ax.fill_between(steps, lo, hi, color='red', alpha=0.2, label='90% interval')
        ext = np.where(E[i])[0]
        if len(ext):
            ax.scatter(ext + 1, Y[i][ext], color='orange', zorder=5, s=60, label='Extreme event')
        ax.axvline(0.5, color='lightgray', ls=':')
        ax.set_xlabel("Time step relative to forecast start (6 h each)")
        ax.set_ylabel("Scaled temperature anomaly")
        ax.set_title("Forecast with Calibrated Uncertainty Estimates")
        ax.legend(loc='best', fontsize=8)
        return fig
 
    return predict_interface

def launch_demo(model, test, margin=0.0):
    import gradio as gr
    predict_interface = make_predict_interface(model, test, margin)
    iface = gr.Interface(
        fn=predict_interface,
        inputs=[
            gr.Slider(minimum=0, maximum=len(test[0]) - 1, step=1, value=0, label="Test window index (2023, target locations)"),
            gr.Slider(minimum=24, maximum=72, step=12, value=48, label="History steps shown on plot (model always uses 72)"),
        ],
        outputs="plot",
        title="Multivariate Extreme Weather Forecaster",
    )
    iface.launch(share=False)


# ==========================================
# 6. Split-conformal calibration (fit on validation, applied to test)
# ==========================================
def conformal_margin(model, val, target=0.90):
    """CQR-style: s = max(lo - y, y - hi); margin = finite-sample (target) quantile of s on validation."""
    p = predict(model, val[0])
    y = val[1]
    s = np.maximum(p[..., 0] - y, y - p[..., 2]).ravel()
    n = len(s)
    k = min(n - 1, int(math.ceil((n + 1) * target)) - 1)
    return float(np.partition(s, k)[k])


def evaluate_conformal(model, val, test, target=0.90):
    q = conformal_margin(model, val, target)
    X, Y, E = test
    p = predict(model, X)
    lo, med, hi = p[..., 0] - q, p[..., 1], p[..., 2] + q
    return {name: metrics(Y[m], lo[m], med[m], hi[m]) for name, m in [('normal', ~E), ('extreme', E)]}, q


# ==========================================
# 7. One full experiment for one seed
# ==========================================
def run_seed(D, n_feats, seed, args):
    torch.manual_seed(seed); np.random.seed(seed)
    tgt_train, tgt_val, tgt_test = D[('target', 'train')], D[('target', 'val')], D[('target', 'test')]

    base = QuantileCNNLSTM(n_feats, PRED_LEN, QUANTILES)
    ckpt = f'base_seed{seed}.pt'
    if args.base_ckpt:
        base.load_state_dict(torch.load(args.base_ckpt, map_location='cpu'))
        print(f"Loaded checkpoint {args.base_ckpt}")
    elif os.path.exists(ckpt):
        base.load_state_dict(torch.load(ckpt)); print(f"Loaded {ckpt}")
    else:
        print(f"[seed {seed}] Pretraining base model on SOURCE locations...")
        fit(base, D[('source', 'train')], D[('source', 'val')], args.epochs_pre, lr=5e-4, patience=6, tag="pretrain")
        torch.save(base.state_dict(), ckpt)

    models = {'Zero-shot base': base}
    counts = {'Zero-shot base': 0}

    print(f"[seed {seed}] Scratch on target only...")
    m = QuantileCNNLSTM(n_feats, PRED_LEN, QUANTILES)
    fit(m, tgt_train, tgt_val, args.epochs_ad, 2e-3, args.tail_weight, tag="scratch")
    models['Scratch (target only)'], counts['Scratch (target only)'] = m, count_trainable(m)

    print(f"[seed {seed}] Full fine-tune...")
    m = copy.deepcopy(base)
    fit(m, tgt_train, tgt_val, args.epochs_ad, 5e-4, args.tail_weight, tag="full-FT")
    models['Full fine-tune'], counts['Full fine-tune'] = m, count_trainable(m)

    for name, dora in [('LoRA', False), ('DoRA', True)]:
        key = f'{name} r={args.rank}'
        print(f"[seed {seed}] {key}...")
        m = AdaptedModel(copy.deepcopy(base), r=args.rank, alpha=args.alpha, dora=dora)
        fit(m, tgt_train, tgt_val, args.epochs_ad, 2e-3, args.tail_weight, tag=name)
        models[key], counts[key] = m, count_trainable(m)
        torch.save(m.state_dict(), f'{name.lower()}_r{args.rank}_seed{seed}.pt')

    rows = []
    for name, m in models.items():
        res = evaluate(m, tgt_test)
        cres, q = evaluate_conformal(m, tgt_val, tgt_test)
        for variant, r_all in [('raw', res), ('+conformal', cres)]:
            for subset in ('normal', 'extreme'):
                r = r_all[subset]
                rows.append(dict(seed=seed, Model=name, Calibration=variant, Subset=subset, Trainable=counts[name],
                                 n=r['n'], RMSE=r['RMSE'], MAE=r['MAE'], PICP=r['PICP'], MPIW=r['MPIW'],
                                 margin=q if variant == '+conformal' else 0.0))
    return pd.DataFrame(rows), models


def summarize(results, total_params):
    g = results.groupby(['Model', 'Calibration', 'Subset'], sort=False)
    mean, std = g[['RMSE', 'MAE', 'PICP', 'MPIW']].mean(), g[['RMSE', 'MAE', 'PICP', 'MPIW']].std().fillna(0)
    out = mean.copy()
    for c in mean.columns:
        out[c] = [f"{a:.3f}+-{b:.3f}" for a, b in zip(mean[c], std[c])]
    tr = g['Trainable'].first()
    out.insert(0, 'Trainable', [f"{t} ({100 * t / total_params:.1f}%)" for t in tr])
    return out.reset_index(), mean.reset_index()


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs-pre', type=int, default=15)
    ap.add_argument('--epochs-ad', type=int, default=20)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=8)
    ap.add_argument('--tail-weight', type=float, default=3.0, help='pinball weight on extreme targets during adaptation')
    ap.add_argument('--n-target', type=int, default=9, help='held-out target grid points')
    ap.add_argument('--seeds', default='0,1,2', help='comma-separated seeds, e.g. 0 for a quick run')
    ap.add_argument('--base-ckpt', default=None, help="Pretrained state_dict (skips pretraining)")
    ap.add_argument('--no-gradio', action='store_true')
    args = ap.parse_args()

    download_era5_data()
    df, feats = load_multi_location_era5()
    locs = sorted(df['location_id'].unique())
    target_locs = set(np.random.RandomState(0).choice(locs, args.n_target, replace=False))  # fixed across seeds
    print(f"{len(locs) - len(target_locs)} source locations, {len(target_locs)} target locations")

    D = build_splits(df, feats, target_locs)
    for k, v in D.items():
        print(f"  {k}: X{v[0].shape}  extreme targets: {v[2].mean():.3%}")

    all_res, last_models = [], None
    for seed in [int(s) for s in args.seeds.split(',')]:
        res, last_models = run_seed(D, len(feats), seed, args)
        all_res.append(res)
    results = pd.concat(all_res, ignore_index=True)
    results.to_csv('results_all_seeds.csv', index=False)

    total = sum(p.numel() for p in QuantileCNNLSTM(len(feats), PRED_LEN, QUANTILES).parameters())
    pretty, _ = summarize(results, total)
    pretty.to_csv('results_summary.csv', index=False)
    n_seeds = results['seed'].nunique()
    print(f"\n--- RESULTS on target locations, 2023 test year, mean+-std over {n_seeds} seed(s) (nominal PICP = 0.90) ---")
    print(pretty.to_string(index=False))

    if not args.no_gradio:
        print("\n6. Launching Web Interface...")
        dora = last_models[f'DoRA r={args.rank}']
        q = conformal_margin(dora, D[('target', 'val')])        # same calibration as the +conformal table rows
        launch_demo(dora, D[('target', 'test')], margin=q)