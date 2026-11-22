"""
Tail-loss experiment with matched controls across output heads.

What was wrong before
  * Gaussian / Student-t / CRPS heads were pretrained with the tail-weighted loss, the pinball head with the plain one.
  * Expectile reused the pinball-pretrained weights instead of being pretrained itself.
  * The alternative heads were only run with the Tail-sym weight, so they had no like-for-like pinball counterpart
    besides one row, and the 'pinball' baseline was unweighted.

What this version does
  * Every head (pinball, expectile, gauss_nll, t_nll, crps) is pretrained on the source points with ITS OWN
    UNWEIGHTED loss, through the same fit_loss, epochs, learning rate and patience.
  * Every head is adapted under the same weighting schemes (--runs head:weighting), so
    pinball:sym vs gauss_nll:sym vs crps:sym differ only in the head / loss.
  * Delta columns: vsNone = against the same head without weighting; vsPinball = against pinball with the same weighting.
  * A paired per-seed table (same seed, same weighting, head vs pinball) shows whether a difference is larger
    than the seed-to-seed spread.

Weightings: none | sym | asym | magnitude | focal | covreg
Example:
    python tail_loss_experiment_matched.py --seeds 0,1,2,3,4
    python tail_loss_experiment_matched.py --runs pinball:none,pinball:sym,crps:none,crps:sym,t_nll:none,t_nll:sym
    python tail_loss_experiment_matched.py --smoke
"""
import os, math, copy, argparse, warnings
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm, t as student_t

E4_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(E4_DIR, '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'E3'))

from modelE3 import (QuantileCNNLSTM, AdaptedModel, make_windows,
                     SEQ_LEN, PRED_LEN, QUANTILES, FEATURES, TRAIN_END, VAL_END, EXTREME_PCT)

warnings.filterwarnings("ignore", message=".*RNN module weights.*")

TARGET = round(QUANTILES[-1] - QUANTILES[0], 10)
METRICS = ['PICP', 'MPIW', 'MissAbove', 'MissBelow', 'WinklerIS', 'RMSE', 'MAE']
HEADS = ['pinball', 'expectile', 'gauss_nll', 't_nll', 'crps']
WEIGHTINGS = ['none', 'sym', 'asym', 'magnitude', 'focal', 'covreg']
CAL_ORDER = ['raw', 'cqr_global', 'cqr_asym', 'cqr_mondrian', 'oracle_tail']
SUBSETS = ['normal', 'upper', 'lower', 'extreme', 'all']
DEFAULT_RUNS = ','.join(
    [f'pinball:{w}' for w in WEIGHTINGS] + [f'{h}:{w}' for h in HEADS[1:] for w in ('none', 'sym')])


# ==========================================================
# 1. Data: tail-coded extremes (0 normal, 1 upper, 2 lower)
# ==========================================================
def preprocess_location_tails(group, feats):
    group = group.sort_values('time')
    times = pd.DatetimeIndex(group['time'])
    vals = group[feats].to_numpy(dtype=np.float64)
    train_mask = np.asarray(times < TRAIN_END)
    doy, hr = times.dayofyear.values, times.hour.values

    anom = np.zeros_like(vals)
    for j in range(vals.shape[1]):
        d = pd.DataFrame({'d': doy[train_mask], 'h': hr[train_mask], 'v': vals[train_mask, j]})
        m = d.groupby(['d', 'h'])['v'].mean().unstack().reindex(index=range(1, 367))
        ext = pd.concat([m.iloc[-7:], m, m.iloc[:7]])
        sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
        sm = sm.interpolate(limit_direction='both').to_numpy()
        hi_idx = m.columns.get_indexer(hr)
        anom[:, j] = vals[:, j] - sm[doy - 1, hi_idx]

    lo, hi = anom[train_mask].min(0), anom[train_mask].max(0)
    scaled = (anom - lo) / (hi - lo + 1e-8)
    up_thr = np.percentile(anom[train_mask, 0], EXTREME_PCT)
    lo_thr = np.percentile(anom[train_mask, 0], 100 - EXTREME_PCT)
    code = np.zeros(len(anom), dtype=np.int8)
    code[anom[:, 0] > up_thr] = 1
    code[anom[:, 0] < lo_thr] = 2
    return scaled.astype(np.float32), code, times


def build_splits_tails(df, feats, target_locs):
    bucket = {}
    for loc, g in df.groupby('location_id'):
        scaled, code, times = preprocess_location_tails(g, feats)
        w = make_windows(scaled, code, times)
        if w is None:
            continue
        X, Y, C, tl = w
        dom = 'target' if loc in target_locs else 'source'
        for name, mask in [('train', tl < np.datetime64(TRAIN_END)),
                           ('val', (tl >= np.datetime64(TRAIN_END)) & (tl < np.datetime64(VAL_END))),
                           ('test', tl >= np.datetime64(VAL_END))]:
            bucket.setdefault((dom, name), []).append((X[mask], Y[mask], C[mask]))
    return {k: tuple(np.concatenate(z, 0) for z in zip(*v)) for k, v in bucket.items()}


def make_synthetic_df(n_loc=6, seed=0):
    rng = np.random.RandomState(seed)
    times = pd.date_range('2015-01-01', '2023-12-31 18:00', freq='6h')
    t = np.arange(len(times))
    doy, hr = times.dayofyear.values, times.hour.values
    rows = []
    for l in range(n_loc):
        eps = rng.standard_t(4, size=(len(t), 5)) * 0.6
        z = np.zeros_like(eps)
        for i in range(1, len(t)):
            z[i] = 0.9 * z[i - 1] + eps[i]
        temp = 27 + 7 * np.sin(2 * np.pi * (doy - 100) / 365.25) + 2 * np.sin(2 * np.pi * hr / 24) + z[:, 0] + 0.4 * t / len(t)
        dew = 18 + 5 * np.sin(2 * np.pi * (doy - 120) / 365.25) + 0.6 * z[:, 0] + z[:, 1]
        wind = 3 + np.abs(z[:, 2]) + 0.5 * np.sin(2 * np.pi * doy / 365.25)
        cloud = np.clip(0.5 + 0.2 * np.sin(2 * np.pi * (doy - 180) / 365.25) + 0.1 * z[:, 3], 0, 1)
        prec = np.abs(z[:, 4]) * 1e-3
        rows.append(pd.DataFrame(dict(location_id=f"loc{l}", time=times, Temperature=temp, Dewpoint=dew,
                                      Wind_Speed=wind, Cloud_Cover=cloud, Precipitation=prec)))
    return pd.concat(rows, ignore_index=True), FEATURES


# ==========================================================
# 2. Heads, losses, weightings
# ==========================================================
PROB_HEADS = {'gauss_nll': 2, 't_nll': 3, 'crps': 2}
_QS = torch.tensor(QUANTILES, dtype=torch.float32)


class ProbabilisticCNNLSTM(nn.Module):
    """Same backbone as QuantileCNNLSTM; output is (mu, raw_sigma[, raw_nu]) per forecast step."""
    def __init__(self, input_size, pred_len, loss_name):
        super().__init__()
        self.pred_len = pred_len
        self.loss_name = loss_name
        self.n_out = PROB_HEADS[loss_name]
        self.cnn = nn.Sequential(nn.Conv1d(input_size, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(64, 50, batch_first=True, num_layers=2)
        self.fc = nn.Linear(50, pred_len * self.n_out)
        with torch.no_grad():
            b = self.fc.bias.view(pred_len, self.n_out)
            b[:, 1] = -2.25                                               # sigma starts near 0.1
            if loss_name == 't_nll':
                b[:, 2] = 2.0                                             # nu starts near 4

    def forward(self, x):
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).view(-1, self.pred_len, self.n_out)


def new_base(head, n_feats):
    if head in ('pinball', 'expectile'):
        return QuantileCNNLSTM(n_feats, PRED_LEN, QUANTILES)
    return ProbabilisticCNNLSTM(n_feats, PRED_LEN, head)


def _wmean(l, w):
    return (l * w).mean() if w is not None else l.mean()


def dist_params(out, head):
    mu = out[..., 0]
    sigma = F.softplus(out[..., 1]) + 1e-3
    nu = 2.1 + F.softplus(out[..., 2]) if head == 't_nll' else None
    return mu, sigma, nu


def head_loss(head, y, out, w=None):
    """Base loss of each head. w: optional (B,T) elementwise weights."""
    if head in ('pinball', 'expectile'):
        e = y.unsqueeze(-1) - out
        if head == 'pinball':
            l = torch.maximum((_QS - 1) * e, _QS * e)
        else:
            l = torch.abs(_QS - (e < 0).float()) * e ** 2
        return _wmean(l, None if w is None else w.unsqueeze(-1))
    mu, s, nu = dist_params(out, head)
    if head == 'gauss_nll':
        l = torch.log(s) + 0.5 * ((y - mu) / s) ** 2 + 0.5 * math.log(2 * math.pi)
    elif head == 't_nll':
        z2 = ((y - mu) / s) ** 2
        l = (torch.log(s) + 0.5 * torch.log(nu * math.pi) + torch.lgamma(nu / 2) - torch.lgamma((nu + 1) / 2)
             + (nu + 1) / 2 * torch.log1p(z2 / nu))
    else:                                                                 # crps, closed form for N(mu, s^2)
        z = (y - mu) / s
        cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        l = s * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))
    return _wmean(l, w)


def interval_torch(out, head):
    """lo, hi of the nominal 5-95% interval as tensors (for focal weights / coverage penalty)."""
    if head in ('pinball', 'expectile'):
        s = torch.sort(out[..., :3], dim=-1).values
        return s[..., 0], s[..., 2]
    mu, sigma, nu = dist_params(out, head)
    if head == 't_nll':
        z = torch.as_tensor(student_t.ppf(QUANTILES[-1], nu.detach().cpu().numpy()), dtype=mu.dtype)
    else:
        z = norm.ppf(QUANTILES[-1])
    return mu - sigma * z, mu + sigma * z


class Weighting:
    def __init__(self, kind, args, y_med, y_iqr):
        if kind not in WEIGHTINGS:
            raise ValueError(f"unknown weighting '{kind}'")
        self.kind, self.a, self.y_med, self.y_iqr = kind, args, y_med, y_iqr

    def weights(self, y, c, out, head):
        k, a = self.kind, self.a
        if k == 'none':
            return None
        if k in ('sym', 'covreg'):
            return 1 + (a.tail_weight - 1) * (c > 0).float()
        if k == 'asym':
            return 1 + (a.w_upper - 1) * (c == 1).float() + (a.w_lower - 1) * (c == 2).float()
        if k == 'magnitude':
            return 1 + a.gamma * ((y - self.y_med).abs() / self.y_iqr).clamp(max=a.zmax)
        lo, hi = interval_torch(out.detach(), head)                       # focal
        miss = F.relu(lo - y) + F.relu(y - hi)
        return 1 + a.gamma * (miss / self.y_iqr).clamp(max=a.zmax)

    def penalty(self, y, c, out, head):
        if self.kind != 'covreg':
            return 0.0
        lo, hi = interval_torch(out, head)
        inside = torch.sigmoid((hi - y) / self.a.soft_tau) * torch.sigmoid((y - lo) / self.a.soft_tau)
        pen = 0.0
        for m in ((c == 1).float(), (c == 2).float()):
            if m.sum() > 0:
                pen = pen + F.relu(TARGET - (inside * m).sum() / m.sum()) ** 2
        return self.a.lam_cov * pen


def make_loss(head, weighting):
    def fn(y, out, c):
        return head_loss(head, y, out, weighting.weights(y, c, out, head)) + weighting.penalty(y, c, out, head)
    return fn


# ==========================================================
# 3. Training, prediction
# ==========================================================
def fit_loss(model, train, val, loss_fn, plain_fn, epochs, lr, select='own', bs=256, patience=4, tag=""):
    X, Y, C = train
    loader = DataLoader(TensorDataset(torch.as_tensor(X), torch.as_tensor(Y), torch.as_tensor(C).long()),
                        batch_size=bs, shuffle=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sel = np.linspace(0, len(val[0]) - 1, min(len(val[0]), 20000)).astype(int)
    Xv, Yv, Cv = (torch.as_tensor(val[0][sel]), torch.as_tensor(val[1][sel]), torch.as_tensor(val[2][sel]).long())
    val_fn = loss_fn if select == 'own' else plain_fn
    best, best_state, bad = float('inf'), None, 0
    for ep in range(epochs):
        model.train(); tot = 0.0
        for bx, by, bc in loader:
            opt.zero_grad()
            loss = loss_fn(by, model(bx), bc)
            loss.backward(); opt.step(); tot += loss.item()
        model.eval()
        with torch.no_grad():
            vl = val_fn(Yv, model(Xv), Cv).item()
        print(f"   [{tag}] epoch {ep + 1}/{epochs}  train {tot / len(loader):.4f}  val({select}) {vl:.4f}")
        if vl < best - 1e-6:
            best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def to_interval(raw, head):
    """raw network output (numpy) -> (N, T, 3) = [lo, centre, hi]"""
    if head in ('pinball', 'expectile'):
        return np.sort(raw, axis=-1)
    mu = raw[..., 0]
    sigma = np.logaddexp(0, raw[..., 1]) + 1e-3
    if head == 't_nll':
        nu = 2.1 + np.logaddexp(0, raw[..., 2])
        zl, zh = student_t.ppf(QUANTILES[0], nu), student_t.ppf(QUANTILES[-1], nu)
    else:
        zl, zh = norm.ppf(QUANTILES[0]), norm.ppf(QUANTILES[-1])
    return np.stack([mu + sigma * zl, mu, mu + sigma * zh], axis=-1)


def predict_head(model, X, head, bs=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            outs.append(model(torch.as_tensor(X[i:i + bs])).numpy())
    return to_interval(np.concatenate(outs, 0), head)


def get_base(head, seed, D, n_feats, args, plain_fn):
    """Pretrained base for a head: unweighted own loss, same procedure for every head."""
    base = new_base(head, n_feats)
    legacy = os.path.join(E4_DIR, f"{args.ckpt_prefix}base_seed{seed}.pt")
    ckpt = os.path.join(E4_DIR, f"{args.ckpt_prefix}{head}_plainbase_seed{seed}.pt")
    if head == 'pinball' and args.reuse_pinball_base and os.path.exists(legacy):
        base.load_state_dict(torch.load(legacy, map_location='cpu'))
        print(f"Loaded legacy pinball base {legacy}")
    elif os.path.exists(ckpt):
        base.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f"Loaded {ckpt}")
    else:
        print(f"[seed {seed}] pretraining {head} base on SOURCE locations (unweighted {head} loss)...")
        torch.manual_seed(seed); np.random.seed(seed)
        fit_loss(base, D[('source', 'train')], D[('source', 'val')], plain_fn, plain_fn, args.epochs_pre, 5e-4,
                 select='plain', patience=6, tag=f'pretrain-{head}')
        torch.save(base.state_dict(), ckpt)
    return base


def adapt_model(base, args):
    m = copy.deepcopy(base)
    if args.adapter == 'full':
        return m, 5e-4
    return AdaptedModel(m, r=args.rank, alpha=args.alpha, dora=(args.adapter == 'dora')), 2e-3


# ==========================================================
# 4. Calibration + metrics (numpy)
# ==========================================================
def conf_q(scores, level):
    s = np.asarray(scores).ravel()
    n = len(s)
    if n == 0:
        return 0.0
    k = min(n - 1, int(math.ceil((n + 1) * level)) - 1)
    return float(np.partition(s, k)[k])


def calibrate_variants(pv, Yv, Cv, pt, Ct, regime_pct=10):
    a = 1 - TARGET
    lo_v, med_v, hi_v = pv[..., 0], pv[..., 1], pv[..., 2]
    lo_t, med_t, hi_t = pt[..., 0], pt[..., 1], pt[..., 2]
    out = {'raw': (lo_t, hi_t)}

    s_sym = np.maximum(lo_v - Yv, Yv - hi_v)
    q = conf_q(s_sym, TARGET)
    out['cqr_global'] = (lo_t - q, hi_t + q)

    s_lo, s_hi = lo_v - Yv, Yv - hi_v
    out['cqr_asym'] = (lo_t - conf_q(s_lo, 1 - a / 2), hi_t + conf_q(s_hi, 1 - a / 2))

    t_lo, t_hi = np.percentile(med_v, [regime_pct, 100 - regime_pct])
    regime = lambda m: np.where(m >= t_hi, 2, np.where(m <= t_lo, 0, 1))
    rv, rt = regime(med_v), regime(med_t)
    lo_m, hi_m = lo_t.copy(), hi_t.copy()
    for r in (0, 1, 2):
        mv, mt = rv == r, rt == r
        if mv.sum() == 0:
            continue
        lo_m[mt] = lo_t[mt] - conf_q(s_lo[mv], 1 - a / 2)
        hi_m[mt] = hi_t[mt] + conf_q(s_hi[mv], 1 - a / 2)
    out['cqr_mondrian'] = (lo_m, hi_m)

    lo_o, hi_o = lo_t.copy(), hi_t.copy()
    for c in (0, 1, 2):
        mv, mt = Cv == c, Ct == c
        if mv.sum() == 0:
            continue
        qo = conf_q(s_sym[mv], TARGET)
        lo_o[mt], hi_o[mt] = lo_t[mt] - qo, hi_t[mt] + qo
    out['oracle_tail'] = (lo_o, hi_o)
    return out


def interval_metrics(y, lo, med, hi):
    if y.size == 0:
        return dict(n=0, **{k: np.nan for k in METRICS})
    a = 1 - TARGET
    inside = (y >= lo) & (y <= hi)
    wink = (hi - lo) + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)
    return dict(n=int(y.size), PICP=float(inside.mean()), MPIW=float((hi - lo).mean()),
                MissAbove=float((y > hi).mean()), MissBelow=float((y < lo).mean()), WinklerIS=float(wink.mean()),
                RMSE=float(np.sqrt(np.mean((y - med) ** 2))), MAE=float(np.mean(np.abs(y - med))))


def subset_masks(C):
    return {'normal': C == 0, 'upper': C == 1, 'lower': C == 2, 'extreme': C > 0, 'all': np.ones_like(C, bool)}


def evaluate_model(model, head, weighting, seed, val, test, regime_pct):
    pv, pt = predict_head(model, val[0], head), predict_head(model, test[0], head)
    Yt, Ct = test[1], test[2]
    rows = []
    for cal, (lo, hi) in calibrate_variants(pv, val[1], val[2], pt, Ct, regime_pct).items():
        for sub, m in subset_masks(Ct).items():
            r = interval_metrics(Yt[m], lo[m], pt[..., 1][m], hi[m])
            rows.append(dict(seed=seed, Loss=f"{head}:{weighting}", Head=head, Weighting=weighting,
                             Calibration=cal, Subset=sub, Target=TARGET, **r))
    return rows


# ==========================================================
# 5. Summaries
# ==========================================================
def _with_deltas(mean, ref, on, tag):
    r = ref[on + ['PICP', 'MPIW', 'WinklerIS']].rename(columns={'PICP': 'rP', 'MPIW': 'rW', 'WinklerIS': 'rI'})
    out = mean.merge(r, on=on, how='left')
    out[f'dPICP_{tag}'] = 100 * (out['PICP'] - out['rP'])
    out[f'dMPIW%_{tag}'] = 100 * (out['MPIW'] / out['rW'] - 1)
    out[f'dIS%_{tag}'] = 100 * (out['WinklerIS'] / out['rI'] - 1)
    return out.drop(columns=['rP', 'rW', 'rI'])


def summarize(results, run_order):
    keys = ['Loss', 'Head', 'Weighting', 'Calibration', 'Subset']
    g = results.groupby(keys, sort=False)
    mean = g[METRICS + ['n', 'Target']].mean().reset_index()
    std = g[METRICS].std().fillna(0).reset_index()
    mean['Gap'] = mean['PICP'] - mean['Target']
    mean = _with_deltas(mean, mean[mean['Weighting'] == 'none'], ['Head', 'Calibration', 'Subset'], 'vsNone')
    mean = _with_deltas(mean, mean[mean['Head'] == 'pinball'], ['Weighting', 'Calibration', 'Subset'], 'vsPinball')

    pretty = mean[keys + ['n', 'Target']].copy()
    for c in ['PICP', 'MPIW', 'MissAbove', 'MissBelow', 'WinklerIS', 'RMSE']:
        pretty[c] = [f"{a:.3f}+-{b:.3f}" for a, b in zip(mean[c], std[c])]
    for c in [x for x in mean.columns if x.startswith('d')] + ['Gap']:
        pretty[c] = mean[c].round(3)
    for df_ in (mean, pretty):
        df_['Loss'] = pd.Categorical(df_['Loss'], run_order, ordered=True)
        df_['Calibration'] = pd.Categorical(df_['Calibration'], CAL_ORDER, ordered=True)
        df_['Subset'] = pd.Categorical(df_['Subset'], SUBSETS, ordered=True)
    mean = mean.sort_values(['Subset', 'Calibration', 'Loss']).reset_index(drop=True)
    pretty = pretty.sort_values(['Subset', 'Calibration', 'Loss']).reset_index(drop=True)
    return pretty, mean


def paired_vs_pinball(results, cal='raw'):
    """Per-seed differences head - pinball at the same weighting (same seed). Mean and std over seeds."""
    r = results[results['Calibration'] == cal]
    pb = r[r['Head'] == 'pinball'][['seed', 'Weighting', 'Subset', 'PICP', 'MPIW', 'WinklerIS']]
    pb = pb.rename(columns={'PICP': 'pP', 'MPIW': 'pW', 'WinklerIS': 'pI'})
    x = r[r['Head'] != 'pinball'].merge(pb, on=['seed', 'Weighting', 'Subset'])
    x['dPICP_pts'] = 100 * (x['PICP'] - x['pP'])
    x['dMPIW_pct'] = 100 * (x['MPIW'] / x['pW'] - 1)
    x['dIS_pct'] = 100 * (x['WinklerIS'] / x['pI'] - 1)
    x['IS_better'] = (x['WinklerIS'] < x['pI']).astype(int)
    g = x.groupby(['Subset', 'Weighting', 'Head'], sort=False)
    out = g[['dPICP_pts', 'dMPIW_pct', 'dIS_pct']].agg(['mean', 'std']).round(2)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out['seeds_IS_better'] = g['IS_better'].sum().astype(str) + '/' + g['IS_better'].count().astype(str)
    return out.reset_index()


def plot_tradeoff(mean, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    losses = list(dict.fromkeys(mean['Loss'].astype(str)))
    colors = {l: plt.cm.tab20(i % 20) for i, l in enumerate(losses)}
    marks = dict(zip(CAL_ORDER, ['o', 's', '^', 'D', 'x']))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, sub in zip(axes, ['upper', 'lower', 'normal']):
        for _, r in mean[mean['Subset'] == sub].iterrows():
            ax.scatter(r['MPIW'], r['PICP'], color=colors[str(r['Loss'])], marker=marks[str(r['Calibration'])], s=50)
        ax.axhline(TARGET, color='k', ls='--', lw=1)
        ax.set_title(f"{sub} targets"); ax.set_xlabel("MPIW (scaled units)"); ax.set_ylabel("observed PICP")
    h1 = [plt.Line2D([], [], color=c, marker='o', ls='', label=l) for l, c in colors.items()]
    h2 = [plt.Line2D([], [], color='gray', marker=m, ls='', label=k) for k, m in marks.items()]
    axes[0].legend(handles=h1 + h2, fontsize=6, loc='lower right', ncol=2)
    fig.suptitle(f"Coverage vs width (dashed = target {TARGET:.2f}); up and to the left is better")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ==========================================================
# 6. Driver
# ==========================================================
def parse_runs(text):
    runs = []
    for tok in [t.strip() for t in text.split(',') if t.strip()]:
        head, _, w = tok.partition(':')
        w = w or 'none'
        assert head in HEADS, f"unknown head '{head}' (choose from {HEADS})"
        assert w in WEIGHTINGS, f"unknown weighting '{w}' (choose from {WEIGHTINGS})"
        runs.append((head, w))
    for head in dict.fromkeys(h for h, _ in runs):                        # unweighted run of every head and pinball
        if (head, 'none') not in runs:
            runs.insert(0, (head, 'none'))
    if ('pinball', 'none') not in runs:
        runs.insert(0, ('pinball', 'none'))
    return list(dict.fromkeys(runs))


def run_seed(D, n_feats, seed, runs, args, y_med, y_iqr):
    tr, va, te = D[('target', 'train')], D[('target', 'val')], D[('target', 'test')]
    plain = {h: make_loss(h, Weighting('none', args, y_med, y_iqr)) for h in dict.fromkeys(h for h, _ in runs)}
    bases = {h: get_base(h, seed, D, n_feats, args, plain[h]) for h in plain}
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    rows = []
    for head, w in runs:
        name = f"{head}:{w}"
        print(f"[seed {seed}] adapting ({args.adapter}) {name}")
        torch.manual_seed(seed); np.random.seed(seed)
        model, lr = adapt_model(bases[head], args)
        torch.manual_seed(seed + 12345)                                   # same batch order for every run of a seed
        loss_fn = make_loss(head, Weighting(w, args, y_med, y_iqr))
        fit_loss(model, tr, va, loss_fn, plain[head], args.epochs_ad, lr, select=args.select, tag=name)
        tag = f"{args.adapter}_{head}_{w}_seed{seed}"
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{tag}.pt"))
        pv = predict_head(model, va[0], head)
        a = 1 - TARGET
        np.savez(os.path.join(args.checkpoint_dir, f"{tag}_cqr_asym.npz"),
                 lower=conf_q(pv[..., 0] - va[1], 1 - a / 2), upper=conf_q(va[1] - pv[..., 2], 1 - a / 2))
        rows += evaluate_model(model, head, w, seed, va, te, args.regime_pct)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=DEFAULT_RUNS, help="comma separated head:weighting, e.g. crps:sym,t_nll:none")
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--adapter', default='dora', choices=['dora', 'lora', 'full'])
    ap.add_argument('--epochs-pre', type=int, default=15)
    ap.add_argument('--epochs-ad', type=int, default=20)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=8)
    ap.add_argument('--n-target', type=int, default=9)
    ap.add_argument('--tail-weight', type=float, default=3.0)
    ap.add_argument('--w-upper', type=float, default=5.0)
    ap.add_argument('--w-lower', type=float, default=2.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--zmax', type=float, default=3.0)
    ap.add_argument('--lam-cov', type=float, default=0.2)
    ap.add_argument('--soft-tau', type=float, default=0.02)
    ap.add_argument('--select', default='own', choices=['own', 'plain'],
                    help="early stopping on the run's own (weighted) objective or on the head's unweighted loss")
    ap.add_argument('--regime-pct', type=float, default=10)
    ap.add_argument('--reuse-pinball-base', action='store_true',
                    help='reuse the old base_seed{S}.pt for pinball (faster, but then pretraining is not identical across heads)')
    ap.add_argument('--ckpt-prefix', default='')
    ap.add_argument('--checkpoint-dir', default=os.path.join(E4_DIR, 'checkpoints'))
    ap.add_argument('--out-prefix', default='matched_tail')
    ap.add_argument('--smoke', action='store_true', help='tiny synthetic run (no ERA5 needed)')
    args = ap.parse_args()
    if args.smoke:
        args.ckpt_prefix = 'smoke_'
        args.n_target, args.epochs_pre, args.epochs_ad, args.seeds = 2, 2, 2, '0'
    runs = parse_runs(args.runs)
    print("runs:", [f"{h}:{w}" for h, w in runs])

    if args.smoke:
        df, feats = make_synthetic_df()
    else:
        from modelE3 import download_era5_data, load_multi_location_era5
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        download_era5_data()
        df, feats = load_multi_location_era5()
    locs = sorted(df['location_id'].unique())
    target_locs = set(np.random.RandomState(0).choice(locs, args.n_target, replace=False))
    print(f"{len(locs) - len(target_locs)} source locations, {len(target_locs)} target locations")

    D = build_splits_tails(df, feats, target_locs)
    for k, v in D.items():
        print(f"  {k}: X{v[0].shape}  upper-tail targets {np.mean(v[2] == 1):.2%}  lower-tail targets {np.mean(v[2] == 2):.2%}")

    Ytr = D[('target', 'train')][1]
    y_med = float(np.median(Ytr))
    y_iqr = float(np.subtract(*np.percentile(Ytr, [75, 25]))) + 1e-8

    rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        rows += run_seed(D, len(feats), seed, runs, args, y_med, y_iqr)
    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(E4_DIR, f'{args.out_prefix}_results_all_seeds.csv'), index=False)

    pretty, mean = summarize(results, [f"{h}:{w}" for h, w in runs])
    pretty.to_csv(os.path.join(E4_DIR, f'{args.out_prefix}_results_pretty.csv'), index=False)
    mean.to_csv(os.path.join(E4_DIR, f'{args.out_prefix}_results_summary.csv'), index=False)
    paired = paired_vs_pinball(results, 'raw')
    paired.to_csv(os.path.join(E4_DIR, f'{args.out_prefix}_paired_vs_pinball_raw.csv'), index=False)
    plot_tradeoff(mean, os.path.join(E4_DIR, f'{args.out_prefix}_picp_vs_width.png'))

    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 500)
    n_seeds = results['seed'].nunique()
    cols = ['Loss', 'Calibration', 'n', 'Target', 'PICP', 'Gap', 'MPIW', 'MissAbove', 'MissBelow', 'WinklerIS',
            'dPICP_vsNone', 'dMPIW%_vsNone', 'dIS%_vsNone', 'dPICP_vsPinball', 'dMPIW%_vsPinball', 'dIS%_vsPinball']
    for sub in ['upper', 'lower', 'extreme', 'normal', 'all']:
        t = pretty[(pretty['Subset'] == sub) & pretty['Calibration'].isin(['raw', 'cqr_global'])]
        print(f"\n=== {sub.upper()} | test 2023 | mean+-std over {n_seeds} seed(s) | adapter={args.adapter} | target coverage {TARGET:.2f} ===")
        print("(vsNone: same head without weighting.  vsPinball: pinball with the same weighting.)")
        print(t[cols].to_string(index=False))
    print("\n=== PAIRED per-seed differences vs pinball at the same weighting (raw intervals) ===")
    print(paired[paired['Subset'].isin(['upper', 'lower', 'all'])].to_string(index=False))
    print(f"\nSaved {args.out_prefix}_results_all_seeds.csv, _results_pretty.csv, _results_summary.csv, "
          f"_paired_vs_pinball_raw.csv, _picp_vs_width.png")
