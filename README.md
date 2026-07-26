# CAIN Causal Attribution and Intervention for Fraud Detection on Graphs

Reference implementation for the KDD submission. CAIN learns a causal role for
every edge of a neighbourhood: a graph reconstruction module supplies a
label-free structural prior, a learnable decomposition turns that prior into a
per-edge attribution score `m` that splits the neighbourhood into a core
(`m -> 0`) and an env (`m -> 1`) subgraph, and an adversarial intervention keeps
label information out of the env branch so that the prediction rests on the core
alone.

## Files

| File | Contents |
| --- | --- |
| `model.py` | the three modules, the gradient reversal layer, and the ablation variants |
| `data.py` | dataset loading, the 40/20/40 stratified split, fraud-ratio construction |
| `config.py` | per-dataset hyperparameters and the sensitivity grids |
| `train.py` | training loop, evaluation, main comparison, ablation, sensitivity |
| `analysis.py` | attribution-ranked edge removal and attribution semantics |

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, PyTorch Geometric 2.x and one NVIDIA GPU.
CPU works but is slow on the larger graphs.

## Data

Place the datasets under `data/`. YelpChi and Amazon are the standard `.mat`
benchmarks with `features`, `label` and one sparse adjacency per relation.
T-Finance, T-Social and Healthcare are loaded from a saved PyTorch Geometric
`Data` object (`.pt`) holding `x`, `edge_index`, `edge_type`, `y` and, where
available, `edge_attr`.

Nodes labelled `-1` are unlabelled: they stay in the graph and take part in
message passing, but never contribute to a loss.

## Reproducing the paper

Main comparison, five runs at the original fraud ratio:

```bash
python train.py --dataset yelpchi --data-path data/yelpchi.mat --runs 5
```

Robustness under a reduced fraud ratio. `--ratio` keeps that fraction of the
training fraud nodes; validation and test are never modified:

```bash
for r in 0.05 0.1 0.3 0.6 1.0; do
  python train.py --dataset yelpchi --data-path data/yelpchi.mat --ratio $r
done
```

Analyses. Both freeze the trained model and only perturb or inspect its inputs:

```bash
python analysis.py edge-removal --dataset yelpchi --data-path data/yelpchi.mat
python analysis.py semantics    --dataset amazon  --data-path data/amazon.mat
```


## Notes on the objective

The joint loss is

```
L_total = w_recon * L_recon + L_sup + w_env * L_env + w_split * L_split
```

with `L_split = -Var(m)`, and `w_recon`, `w_env`, `w_split` written `w1`, `w2`,
`w3` in the paper. During the warm-up epochs only `L_sup` and `L_recon` are
active, so the attribution starts from a sensible prior. The temperature of the
attribution sigmoid is annealed from 1 down to `min_temp` after warm-up and
fixed to 1 at evaluation time. Model selection uses validation PR-AUC; the
decision threshold is tuned on validation and applied unchanged to test.
