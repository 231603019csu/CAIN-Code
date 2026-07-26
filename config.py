"""Per-dataset hyperparameters.

These are the configurations used for every number reported in the paper. The
loss weights map to the joint objective as follows:

    L_total = w_recon * L_recon + L_sup + w_env * L_env + w_split * L_split

"""

BASE = {
    'z_dim': 64,
    'hidden_dim': 128,
    'dropout': 0.3,
    'lr': 2e-3,
    'class_weight_cap': 1,
    'num_neighbors': [-1, 50],
    'batch_size': 5120,
    'num_workers': 0,
    'total_epochs': 250,
    'w_recon': 0.01,
    'w_env': 0.30,
    'w_split': 0.90,
    'has_edge_attr': False,
    'num_relations': 3,
}


DATASETS = {
    'yelpchi': dict(
        z_dim=64, hidden_dim=64, dropout=0.4, lr=2e-3,
        total_epochs=500, w_recon=0.30, w_env=0.30, w_split=0.30,
        num_relations=3, has_edge_attr=False,
    ),
    'amazon': dict(
        z_dim=64, hidden_dim=128, dropout=0.3, lr=2e-3,
        total_epochs=250, w_recon=0.01, w_env=0.30, w_split=0.90,
        num_relations=3, has_edge_attr=False,
    ),
    't-finance': dict(
        z_dim=64, hidden_dim=128, dropout=0.4, lr=4.99e-3,
        total_epochs=500, w_recon=0.294, w_env=0.210, w_split=0.472,
        num_neighbors=[50, 25], batch_size=10240,
        num_relations=1, has_edge_attr=False,
    ),
    't-social': dict(
        z_dim=32, hidden_dim=128, dropout=0.035, lr=2.32e-3,
        total_epochs=150, w_recon=0.0065, w_env=0.277, w_split=0.727,
        num_neighbors=[30, 15], batch_size=8192,
        num_relations=1, has_edge_attr=False,
    ),
    'healthcare': dict(
        z_dim=64, hidden_dim=32, dropout=0.3, lr=4.93e-3,
        total_epochs=200, w_recon=0.0152, w_env=0.440, w_split=0.123,
        num_neighbors=[45, 20], batch_size=3000,
        num_relations=5, has_edge_attr=True, edge_feat_dim=32,
    ),
}


# The five initialisation seeds behind every mean and standard deviation in the
# paper. With --runs 5 and no explicit --seed these are the runs that are used.
SEEDS = {
    'yelpchi': [106, 108, 110, 112, 114],
    'amazon': [100, 102, 104, 106, 108],
    't-finance': [1, 2, 3, 4, 5],
    't-social': [10, 20, 30, 40, 50],
    'healthcare': [6, 8, 88, 102, 104],
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
