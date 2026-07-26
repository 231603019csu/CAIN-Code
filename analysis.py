"""Post-hoc analyses of a trained CAIN model.

Both analyses freeze the model and perturb or inspect its inputs; nothing is
retrained, which is what separates them from the ablation study.

Attribution-ranked edge removal (Figure 3). For every centre node the incoming
edges are ranked by the attribution m and the core weight of a fixed fraction is
set to zero, comparing two orderings: ascending m removes the edges the model
scores as most causal first, descending m removes those it scores as most
confounding first.

    python analysis.py edge-removal --dataset yelpchi \
        --data-path data/yelpchi.mat --runs 5 --out edge_removal.json

Attribution semantics (Table 4). The score of every one-hop edge whose centre
node comes from the test split is grouped by the labels of the two endpoints.

    python analysis.py semantics --dataset amazon \
        --data-path data/amazon.mat --runs 5 --out semantics.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from config import SEEDS, get_config
from data import load_dataset
from model import build_core_env_graph, get_temperature
from train import compute_metrics, run_once, set_seed

DROP_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def select_edges_per_node(m, dst, num_nodes, p, order):
    """Per centre node, mark the first floor(p * degree) incoming edges.

    order='ascending'  rank by increasing m, so the most causal edges go first
    order='descending' rank by decreasing m, so the most confounding go first
    """
    n_edges = m.numel()
    device = m.device
    if p <= 0:
        return torch.zeros(n_edges, dtype=torch.bool, device=device)

    key = m if order == 'ascending' else -m
    idx = torch.argsort(key)
    idx = idx[torch.argsort(dst[idx], stable=True)]  # group by centre, keep order
    sorted_dst = dst[idx]

    deg = torch.bincount(dst, minlength=num_nodes)
    offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                        torch.cumsum(deg, 0)[:-1]])
    rank_in_group = torch.arange(n_edges, device=device) - offset[sorted_dst]
    n_drop = (deg.float() * p).floor().long()

    remove = torch.zeros(n_edges, dtype=torch.bool, device=device)
    remove[idx[rank_in_group < n_drop[sorted_dst]]] = True
    return remove


@torch.no_grad()
def edge_removal(models, loader, cfg, device, threshold, epoch=None):
    """Sweep the removed fraction for both orderings; core branch only."""
    for mod in models.values():
        mod.eval()
    temp = 1.0 if epoch is None else get_temperature(
        epoch, cfg['warmup_epochs'], cfg['anneal_epochs'], cfg['min_temp'])
    causal = models['causal']
    results = {}

    for order in ('ascending', 'descending'):
        for p in DROP_RATIOS:
            probs, labels = [], []
            for batch in loader:
                batch = batch.to(device)
                y = batch.y[:batch.batch_size]
                valid = y != -1
                if not valid.any():
                    continue

                out = models['recon'](batch.x, batch.edge_index, batch.edge_type,
                                      batch.batch_size,
                                      getattr(batch, 'edge_attr', None))
                m = models['dcd'](batch.x, getattr(batch, 'edge_attr', None),
                                  out['error_node'], out['error_edge'],
                                  batch.edge_index, temperature=temp)

                core_g, _ = build_core_env_graph(batch, m)
                if p > 0:
                    remove = select_edges_per_node(
                        m, batch.edge_index[1], batch.x.size(0), p, order)
                    weight = core_g.edge_weight.clone()
                    weight[remove] = 0.0
                    core_g.edge_weight = weight

                # Only the core branch is evaluated: no env graph, no loss.
                logits = causal.clf_core(causal.encoder(core_g))
                probs.extend(torch.softmax(logits[valid], dim=1)[:, 1].cpu().numpy())
                labels.extend(y[valid].cpu().numpy())

            probs = np.array(probs)
            labels = np.array(labels)
            results[f'{order}@{p:.1f}'] = compute_metrics(
                labels, (probs >= threshold).astype(int), probs)
            print(f'    {order:>10s}  p={p:.0%}  '
                  f'PR-AUC={results[f"{order}@{p:.1f}"]["pr_auc"]:.4f}')
    return results


@torch.no_grad()
def semantics(models, loader, cfg, device, epoch=None):
    """Group the attribution by the labels of the two endpoints (Table 4)."""
    for mod in models.values():
        mod.eval()
    temp = 1.0 if epoch is None else get_temperature(
        epoch, cfg['warmup_epochs'], cfg['anneal_epochs'], cfg['min_temp'])

    stats = defaultdict(lambda: {'n': 0, 'sum_m': 0.0, 'n_core': 0})
    name = {0: 'benign', 1: 'fraud', -1: 'unlabelled'}

    for batch in loader:
        batch = batch.to(device)
        out = models['recon'](batch.x, batch.edge_index, batch.edge_type,
                              batch.batch_size, getattr(batch, 'edge_attr', None))
        m = models['dcd'](batch.x, getattr(batch, 'edge_attr', None),
                          out['error_node'], out['error_edge'],
                          batch.edge_index, temperature=temp)

        src, dst = batch.edge_index
        # Keep only edges whose centre node is one of the seed (test) nodes.
        keep = dst < batch.batch_size
        y = batch.y
        for centre, neighbour, score in zip(y[dst[keep]].tolist(),
                                            y[src[keep]].tolist(),
                                            m[keep].tolist()):
            cell = stats[(name[centre], name[neighbour])]
            cell['n'] += 1
            cell['sum_m'] += score
            cell['n_core'] += int(score < 0.5)

    table = {}
    for (centre, neighbour), cell in sorted(stats.items()):
        if centre == 'unlabelled':
            continue
        table[f'{centre}->{neighbour}'] = {
            'edges': cell['n'],
            'mean_m': cell['sum_m'] / max(cell['n'], 1),
            'core_pct': 100.0 * cell['n_core'] / max(cell['n'], 1),
        }
        row = table[f'{centre}->{neighbour}']
        print(f'    centre={centre:<7s} neighbour={neighbour:<10s} '
              f'edges={row["edges"]:>9d}  mean m={row["mean_m"]:.3f}  '
              f'core={row["core_pct"]:.1f}%')
    return table


def main():
    ap = argparse.ArgumentParser(description='Post-hoc analyses of CAIN.')
    ap.add_argument('task', choices=['edge-removal', 'semantics'])
    ap.add_argument('--dataset', required=True,
                    choices=['yelpchi', 'amazon', 't-finance', 't-social',
                             'healthcare'])
    ap.add_argument('--data-path', required=True)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=None,
                    help='override the seeds recorded in config.SEEDS')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config(args.dataset)
    seeds = ([args.seed + i for i in range(args.runs)] if args.seed is not None
             else SEEDS[args.dataset][:args.runs])
    all_runs = []

    for i, seed in enumerate(seeds):
        set_seed(seed)
        data = load_dataset(args.dataset, args.data_path, ratio=1.0, seed=seed)
        print(f'\n[run {i}] training')
        _, models, test_loader, threshold = run_once(
            data, cfg, device, run_idx=i, return_models=True)

        print(f'[run {i}] {args.task}')
        if args.task == 'edge-removal':
            all_runs.append(edge_removal(models, test_loader, cfg, device,
                                         threshold))
        else:
            all_runs.append(semantics(models, test_loader, cfg, device))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'dataset': args.dataset, 'task': args.task,
                       'runs': all_runs}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
