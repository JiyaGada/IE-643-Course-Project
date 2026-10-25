# reopen the Gradio demo from saved checkpoints (no training)
import numpy as np, torch
import model as w 

SEED = 2                              # last seed in the run
df, feats = w.load_multi_location_era5()
locs = sorted(df['location_id'].unique())
target_locs = set(np.random.RandomState(0).choice(locs, 9, replace=False))
D = w.build_splits(df, feats, target_locs)

base = w.QuantileCNNLSTM(len(feats), w.PRED_LEN, w.QUANTILES)
model = w.AdaptedModel(base, r=4, alpha=8, dora=True)
model.load_state_dict(torch.load(f'dora_r4_seed{SEED}.pt', map_location='cpu'))
model.eval()

q = w.conformal_margin(model, D[('target', 'val')])
w.launch_demo(model, D[('target', 'test')], margin=q)