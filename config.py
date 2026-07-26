"""Per-dataset hyperparameters.

These are the configurations used for every number reported in the paper. The
loss weights map to the joint objective as follows:

    L_total = w_recon * L_recon + L_sup + w_env * L_env + w_split * L_split

with L_split = -Var(m), the variance regulariser that pushes the attribution
towards 0 and 1. w_recon, w_env and w_split are w1, w2 and w3 in the paper.
"""

BASE = {
    'z_dim': 64,
    'hidden_dim': 128,
    'dropout': 0.3,
    'lr': 2e-3,
    'weight_decay': 1e-5,
    'class_weight_cap': 1,
    'grad_alpha': 1.0,
    'focal_gamma': 1.3,
    'num_neighbors': [-1, 50],
    'batch_size': 5120,
    'num_workers': 0,
    'total_epochs': 250,
    'warmup_epochs': 15,
    'anneal_epochs': 50,
    'min_temp': 0.15,
    'patience': 150,
    'val_interval': 1,
    'w_recon': 0.01,
    'w_env': 0.30,
    'w_split': 0.90,
    'has_edge_attr': False,
    'num_relations': 3,
}


DATASETS = {
    'yelpchi': dict(
        z_dim=64, hidden_dim=64, dropout=0.4, lr=2e-3,
        total_epochs=500, warmup_epochs=10, anneal_epochs=50, min_temp=0.3,
        w_recon=0.30, w_env=0.30, w_split=0.30,
        num_relations=3, has_edge_attr=False,
    ),
    'amazon': dict(
        z_dim=64, hidden_dim=128, dropout=0.3, lr=2e-3,
        total_epochs=250, warmup_epochs=15, anneal_epochs=50, min_temp=0.15,
        w_recon=0.01, w_env=0.30, w_split=0.90,
        focal_gamma=1.3, num_relations=3, has_edge_attr=False,
    ),
    't-finance': dict(
        z_dim=64, hidden_dim=128, dropout=0.4, lr=4.99e-3,
        total_epochs=500, warmup_epochs=22, anneal_epochs=57, min_temp=0.197,
        w_recon=0.294, w_env=0.210, w_split=0.472,
        focal_gamma=0.269, grad_alpha=0.069,
        num_neighbors=[50, 25], batch_size=10240,
        patience=200, num_relations=1, has_edge_attr=False,
    ),
    't-social': dict(
        z_dim=32, hidden_dim=128, dropout=0.035, lr=2.32e-3,
        total_epochs=150, warmup_epochs=15, anneal_epochs=50, min_temp=0.15,
        w_recon=0.0065, w_env=0.277, w_split=0.727,
        num_neighbors=[30, 15], batch_size=8192,
        num_relations=1, has_edge_attr=False,
    ),
    'healthcare': dict(
        z_dim=64, hidden_dim=32, dropout=0.3, lr=4.93e-3,
        total_epochs=200, warmup_epochs=20, anneal_epochs=77, min_temp=0.212,
        w_recon=0.0152, w_env=0.440, w_split=0.123,
        focal_gamma=1.679, grad_alpha=1.320,
        num_neighbors=[45, 20], batch_size=3000,
        patience=100, val_interval=5,
        num_relations=5, has_edge_attr=True, edge_feat_dim=32,
    ),
}


# The five initialisation seeds behind every mean and standard deviation in the
# paper. With --runs 5 and no explicit --seed these are the runs that are used.
SEEDS = {
    'yelpchi': [112, 114, 106, 108, 110],
    'amazon': [100, 102, 104, 106, 108],
    't-finance': [1, 2, 3, 4, 5],
    't-social': [10, 20, 30, 40, 50],
    'healthcare': [102, 104, 6, 8, 88],
}


# Grids for the sensitivity study of Figure 4. They are shared across datasets
# so that the x axes of the panels line up. w_split = 0 doubles as the
# "w/o L_split" row of the ablation table.
PARAM_GRIDS = {
    'w_recon': [0.0, 0.05, 0.1, 0.3, 0.5, 1.0],
    'w_env': [0.0, 0.05, 0.1, 0.3, 0.5, 1.0],
    'w_split': [0.0, 0.05, 0.1, 0.3, 0.5, 1.0],
    'z_dim': [8, 16, 32, 64, 128],
}


def get_config(dataset, overrides=None):
    """Return the configuration for a dataset, with optional overrides."""
    key = dataset.lower()
    if key not in DATASETS:
        raise ValueError(f'unknown dataset {dataset!r}; '
                         f'choose from {sorted(DATASETS)}')
    cfg = dict(BASE)
    cfg.update(DATASETS[key])
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
