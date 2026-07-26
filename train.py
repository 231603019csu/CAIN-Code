"""Train and evaluate CAIN.

Main comparison (five runs at the original fraud ratio):
    python train.py --dataset amazon --data-path data/amazon.mat --runs 5

Robustness under a reduced fraud ratio:
    python train.py --dataset yelpchi --data-path data/yelpchi.mat --ratio 0.05

Model selection uses validation PR-AUC; the decision threshold is then tuned on
validation and applied unchanged to the test split.
"""

import argparse
import json
import random

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_fscore_support, recall_score)
from torch_geometric.loader import NeighborLoader

from config import PARAM_GRIDS, SEEDS, get_config
from data import describe, load_dataset
from model import (VARIANTS, build_model, combine_losses, forward_variant,
                   get_temperature)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def gmean_score(labels, preds):
    _, rec, _, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1],
                                                   zero_division=0)
    return float(np.sqrt(rec[0] * rec[1]))


def compute_metrics(labels, preds, probs):
    return {
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'pr_auc': float(average_precision_score(labels, probs)),
        'gmean': gmean_score(labels, preds),
        'recall': float(recall_score(labels, preds, zero_division=0)),
    }


def find_best_threshold(probs, labels):
    """Threshold maximising F1-macro on the validation split."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(labels, (probs >= t).astype(int), average='macro',
                      zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


@torch.no_grad()
def collect_probs(models, loader, cfg, variant, device):
    for m in models.values():
        m.eval()
    probs, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, _, _ = forward_variant(models, batch, cfg, variant,
                                       temperature=1.0)
        y = batch.y[:batch.batch_size]
        valid = y != -1
        probs.append(torch.softmax(logits[valid], dim=1)[:, 1].cpu().numpy())
        labels.append(y[valid].cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


# --------------------------------------------------------------------------
# One training run
# --------------------------------------------------------------------------
def run_once(data, cfg, device, variant='full', w_split=None, run_idx=0,
             verbose=True, return_models=False):
    in_dim = data.x.size(1)
    edge_feat_dim = cfg.get('edge_feat_dim') if cfg['has_edge_attr'] else None

    n_pos = int(data.y[data.train_mask].sum())
    n_neg = int(data.train_mask.sum()) - n_pos
    class_weight = torch.tensor(
        [1.0, min(n_neg / max(n_pos, 1), cfg['class_weight_cap'])]).to(device)

    loader_kw = dict(num_neighbors=cfg['num_neighbors'],
                     batch_size=cfg['batch_size'],
                     num_workers=cfg['num_workers'])
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                  shuffle=True, **loader_kw)
    val_loader = NeighborLoader(data, input_nodes=data.val_mask,
                                shuffle=False, **loader_kw)
    test_loader = NeighborLoader(data, input_nodes=data.test_mask,
                                 shuffle=False, **loader_kw)

    models = build_model(in_dim, cfg, cfg['num_relations'], edge_feat_dim,
                         class_weight, variant, device)
    params = [p for m in models.values() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg['lr'],
                           weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=10, min_lr=cfg['lr'] * 0.05)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_prauc, best_state, patience_cnt = -1.0, None, 0

    for epoch in range(1, cfg['total_epochs'] + 1):
        for m in models.values():
            m.train()
        warming_up = epoch <= cfg['warmup_epochs']
        temp = get_temperature(epoch, cfg['warmup_epochs'],
                               cfg['anneal_epochs'], cfg['min_temp'])

        for batch in train_loader:
            batch = batch.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, parts, m_scores = forward_variant(models, batch, cfg,
                                                     variant, temp)
                loss = combine_losses(parts, m_scores, cfg, warming_up, w_split)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            for mod in models.values():
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        if not warming_up and epoch % cfg['val_interval'] == 0:
            probs, labels = collect_probs(models, val_loader, cfg, variant, device)
            prauc = (float(average_precision_score(labels, probs))
                     if len(np.unique(labels)) > 1 else 0.0)
            sched.step(prauc)

            if prauc > best_prauc:
                best_prauc = prauc
                best_state = {k: {kk: vv.detach().cpu().clone()
                                  for kk, vv in mod.state_dict().items()}
                              for k, mod in models.items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= cfg['patience'] and epoch > 30:
                    break

    if best_state is not None:
        for k, mod in models.items():
            mod.load_state_dict({kk: vv.to(device)
                                 for kk, vv in best_state[k].items()})

    val_probs, val_labels = collect_probs(models, val_loader, cfg, variant, device)
    threshold, _ = find_best_threshold(val_probs, val_labels)
    test_probs, test_labels = collect_probs(models, test_loader, cfg, variant,
                                            device)
    metrics = compute_metrics(test_labels, (test_probs >= threshold).astype(int),
                              test_probs)
    if verbose:
        print('  run %d: ' % run_idx
              + '  '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
    if return_models:
        return metrics, models, test_loader, threshold
    return metrics


def aggregate(runs):
    return {k: (float(np.mean([r[k] for r in runs])),
                float(np.std([r[k] for r in runs]))) for k in runs[0]}


def resolve_seeds(args):
    """The five seeds behind the reported numbers, unless --seed overrides them."""
    if args.seed is not None:
        return [args.seed + i for i in range(args.runs)]
    seeds = SEEDS[args.dataset]
    if args.runs <= len(seeds):
        return seeds[:args.runs]
    return seeds + [seeds[-1] + i for i in range(1, args.runs - len(seeds) + 1)]


def evaluate_setting(args, cfg, device, variant='full', w_split=None):
    runs = []
    for i, seed in enumerate(resolve_seeds(args)):
        set_seed(seed)
        data = load_dataset(args.dataset, args.data_path, ratio=args.ratio,
                            seed=seed)
        if i == 0:
            describe(data, args.dataset)
        runs.append(run_once(data, cfg, device, variant, w_split, i))
    summary = aggregate(runs)
    print('  mean +- std: ' + '  '.join(f'{k}={mu:.4f}+-{sd:.4f}'
                                        for k, (mu, sd) in summary.items()))
    return {'runs': runs, 'summary': summary}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Train and evaluate CAIN.')
    ap.add_argument('--dataset', required=True,
                    choices=['yelpchi', 'amazon', 't-finance', 't-social',
                             'healthcare'])
    ap.add_argument('--data-path', required=True,
                    help='.mat file for YelpChi/Amazon, .pt file otherwise')
    ap.add_argument('--ratio', type=float, default=1.0,
                    help='fraction of the training fraud nodes to keep')
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=None,
                    help='override the seeds recorded in config.SEEDS')
    ap.add_argument('--variant', default='full', choices=list(VARIANTS),
                    help='ablation variant (Table 3)')
    ap.add_argument('--sweep', default=None, choices=sorted(PARAM_GRIDS),
                    help='hyperparameter to sweep (Figure 4)')
    ap.add_argument('--out', default=None, help='optional JSON output path')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_cfg = get_config(args.dataset)
    results = {}

    if args.sweep is None:
        results['default'] = evaluate_setting(args, base_cfg, device, args.variant)
    else:
        for value in PARAM_GRIDS[args.sweep]:
            cfg = dict(base_cfg)
            cfg[args.sweep] = int(value) if args.sweep == 'z_dim' else float(value)
            print(f'\n=== {args.sweep} = {cfg[args.sweep]} ===')
            # w_split is passed explicitly so that the sweep also covers the
            # w/o L_split ablation at w_split = 0.
            results[str(cfg[args.sweep])] = evaluate_setting(
                args, cfg, device, args.variant,
                w_split=cfg['w_split'] if args.sweep == 'w_split' else None)

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'dataset': args.dataset, 'ratio': args.ratio,
                       'variant': args.variant, 'sweep': args.sweep,
                       'results': results}, f, indent=2)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
