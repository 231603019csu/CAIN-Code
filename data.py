"""Dataset loading and fraud-ratio construction.

Two input formats are supported:

  *.mat   the standard multi-relation review benchmarks (YelpChi, Amazon), with
          keys 'features', 'label' and one sparse adjacency per relation
  *.pt    a saved PyG Data object with x, edge_index, edge_type, y and, where
          available, edge_attr (used for T-Finance, T-Social and Healthcare)

Splits follow the protocol of the paper: a stratified 40/20/40 split over the
labelled nodes, computed once and then held fixed. Nodes with label -1 are
unlabelled; they stay in the graph and take part in message passing but never
contribute to a loss.
"""

import numpy as np
import scipy.io as sio
import torch
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

MAT_RELATIONS = {
    'yelpchi': ['net_rur', 'net_rtr', 'net_rsr'],
    'amazon': ['net_upu', 'net_usu', 'net_uvu'],
}


def _zscore(x):
    mu = x.mean(0)
    sigma = x.std(0).clamp(min=1e-8)
    return (x - mu) / sigma


def load_mat_dataset(mat_path, relations, split_seed=717):
    """Load a multi-relation .mat benchmark into a PyG Data object."""
    raw = sio.loadmat(mat_path)
    x = _zscore(torch.tensor(raw['features'].todense(), dtype=torch.float))
    y = torch.tensor(raw['label'].flatten(), dtype=torch.long)
    n = y.size(0)

    edge_index_parts, edge_type_parts = [], []
    for i, rel in enumerate(relations):
        coo = coo_matrix(raw[rel])
        src = torch.tensor(coo.row, dtype=torch.long)
        dst = torch.tensor(coo.col, dtype=torch.long)
        edge_index_parts.append(torch.stack([src, dst], dim=0))
        edge_type_parts.append(torch.full((src.size(0),), i, dtype=torch.long))

    data = Data(x=x,
                edge_index=torch.cat(edge_index_parts, dim=1),
                edge_type=torch.cat(edge_type_parts),
                y=y)
    return add_splits(data, split_seed)


def load_pyg_dataset(pt_path, split_seed=717):
    """Load a saved PyG Data object and attach the stratified split."""
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    if not hasattr(data, 'edge_type'):
        data.edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    if getattr(data, 'train_mask', None) is None:
        data = add_splits(data, split_seed)
    return data


def add_splits(data, split_seed=717):
    """Stratified 40/20/40 split over the labelled nodes."""
    n = data.y.size(0)
    labelled = (data.y != -1).nonzero(as_tuple=True)[0].numpy()

    idx_train, idx_rest = train_test_split(
        labelled, train_size=0.4, stratify=data.y[labelled].numpy(),
        random_state=split_seed)
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.67, stratify=data.y[idx_rest].numpy(),
        random_state=split_seed)

    for name, idx in [('train_mask', idx_train), ('val_mask', idx_val),
                      ('test_mask', idx_test)]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        setattr(data, name, mask)
    return data


def downsample_fraud(data, keep_fraction, seed=0):
    """Keep the given fraction of the fraud nodes in the training split.

    Validation and test are never touched, so evaluation is always against the
    true label distribution. Fraud nodes dropped from training remain in the
    graph for message passing; they simply stop contributing to the loss.

    For Amazon the five levels are not produced this way: they come from
    tightening the helpful-vote threshold that defines the labels, so each level
    is a separate .mat file (see the appendix of the paper).
    """
    if keep_fraction >= 1.0:
        return data

    rng = np.random.RandomState(seed)
    train_fraud = (data.train_mask & (data.y == 1)).nonzero(as_tuple=True)[0].numpy()
    n_keep = max(1, int(round(len(train_fraud) * keep_fraction)))
    keep = set(rng.choice(train_fraud, size=n_keep, replace=False).tolist())

    mask = data.train_mask.clone()
    for idx in train_fraud:
        if idx not in keep:
            mask[idx] = False
    data.train_mask = mask
    return data


def load_dataset(name, path, ratio=1.0, seed=0, split_seed=717):
    """Entry point used by train.py."""
    key = name.lower()
    if key in MAT_RELATIONS:
        data = load_mat_dataset(path, MAT_RELATIONS[key], split_seed)
    else:
        data = load_pyg_dataset(path, split_seed)
    return downsample_fraud(data, ratio, seed)


def describe(data, name=''):
    n_train = int(data.train_mask.sum())
    n_fraud = int((data.train_mask & (data.y == 1)).sum())
    print(f'[{name}] nodes={data.y.size(0)} edges={data.edge_index.size(1)} '
          f'relations={int(data.edge_type.max()) + 1} '
          f'train={n_train} train_fraud={n_fraud} '
          f'({100.0 * n_fraud / max(n_train, 1):.2f}%)')
