"""CAIN: Causal Attribution and INtervention for node-level fraud detection.

Three modules, trained jointly:
  1. GraphReconstruction   node/edge reconstruction errors as a label-free prior
  2. CausalDecomposition   per-edge attribution score m, splitting the neighbourhood
                           into a core (m -> 0) and an env (m -> 1) subgraph
  3. CausalIntervention    classify from the core branch, strip label information
                           from the env branch with a gradient reversal layer

The ablation variants of the paper are selected with the `variant` argument of
`build_model` and `forward_variant`:

  full             the complete model
  no_dcd           w/o Decomposition: no core/env split. The reconstruction
                   signals are injected directly instead, by concatenating
                   error_node to the node features and using 1 - error_edge as
                   the edge weight of a single branch
  no_recon_shape   w/o Reconstruction: the mask network sees only the endpoint
                   features, so its input drops to 2 * node_feat_dim
  no_recon_random  a control for no_recon_shape in which the error channels are
                   resampled from U(0, 1) at every forward pass
  no_grl           w/o GRL: the env classifier is still trained, but the
                   gradient is no longer reversed, so the branch cooperates
                   instead of competing

The w/o L_split variant is obtained by setting w_split to 0 rather than by
changing the architecture, so it is handled in train.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv

VARIANTS = ('full', 'no_dcd', 'no_recon_shape', 'no_recon_random', 'no_grl')


# --------------------------------------------------------------------------
# Gradient reversal
# --------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    """Identity forwards, negated gradient scaled by alpha backwards."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# --------------------------------------------------------------------------
# Module 1: Graph Reconstruction
# --------------------------------------------------------------------------
class GraphReconstruction(nn.Module):
    """Two-layer relational GCN with a node decoder and an optional edge decoder.

    Returns min-max normalised node- and edge-level reconstruction errors. Without
    edge attributes the edge error falls back to 1 - sigmoid(z_src . z_dst).
    """

    def __init__(self, in_channels, hidden_channels, z_dim, num_relations,
                 dropout=0.3, edge_feat_dim=None):
        super().__init__()
        self.dropout = dropout
        self.has_edge_dec = edge_feat_dim is not None

        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.conv2 = RGCNConv(hidden_channels, z_dim, num_relations)
        self.norm2 = nn.LayerNorm(z_dim)

        self.node_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, in_channels))
        if self.has_edge_dec:
            self.edge_decoder = nn.Sequential(
                nn.Linear(2 * z_dim, hidden_channels), nn.ReLU(),
                nn.Linear(hidden_channels, edge_feat_dim))

    def forward(self, x, edge_index, edge_type, batch_size, edge_attr=None):
        src, dst = edge_index
        h = F.relu(self.norm1(self.conv1(x, edge_index, edge_type)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = F.relu(self.norm2(self.conv2(h, edge_index, edge_type)))

        x_hat = self.node_decoder(z)
        error_node = ((x - x_hat) ** 2).mean(dim=1).detach()
        error_node = (error_node - error_node.min()) / (
            error_node.max() - error_node.min() + 1e-8)
        recon_loss = F.mse_loss(x_hat, x.float())

        sim = torch.sigmoid((z[src] * z[dst]).sum(dim=1))
        if self.has_edge_dec and edge_attr is not None:
            edge_repr = torch.cat([z[src], z[dst]], dim=1)
            edge_attr_hat = self.edge_decoder(edge_repr)
            error_edge = ((edge_attr.float() - edge_attr_hat) ** 2).mean(dim=1).detach()
            error_edge = (error_edge - error_edge.min()) / (
                error_edge.max() - error_edge.min() + 1e-8)
            recon_loss = recon_loss + F.mse_loss(edge_attr_hat, edge_attr.float())
        else:
            error_edge = (1.0 - sim).detach()

        return {'z': z[:batch_size], 'error_node': error_node,
                'error_edge': error_edge, 'recon_loss': recon_loss}


# --------------------------------------------------------------------------
# Module 2: Learnable Causal Decomposition (Mask Network)
# --------------------------------------------------------------------------
class CausalDecomposition(nn.Module):
    """Per-edge attribution score m in (0, 1).

    The input concatenates the raw features of both endpoints, the three
    reconstruction errors and the edge attribute where available. A learnable
    global bias shifts the overall attribution level, and the temperature is
    annealed during training so the scores sharpen towards 0 and 1.

    With use_error=False the three error channels are dropped and the input is
    just the two endpoint feature vectors (the no_recon_shape variant).
    """

    def __init__(self, node_feat_dim, edge_feat_dim=None, use_error=True):
        super().__init__()
        self.has_edge = edge_feat_dim is not None
        self.use_error = use_error

        in_dim = node_feat_dim * 2 + (3 if use_error else 0)
        if self.has_edge:
            in_dim += edge_feat_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_attr, error_node, error_edge, edge_index,
                temperature=1.0):
        src, dst = edge_index
        parts = [x[src].float(), x[dst].float()]
        if self.use_error:
            parts += [error_node[src].unsqueeze(1),
                      error_node[dst].unsqueeze(1),
                      error_edge.unsqueeze(1)]
        if self.has_edge and edge_attr is not None:
            parts.append(edge_attr.float())

        inp = torch.cat(parts, dim=1)
        inp = F.layer_norm(inp, [inp.size(-1)])
        return torch.sigmoid((self.net(inp).squeeze(1) + self.bias) / temperature)


def build_core_env_graph(batch, m):
    """Split a mini-batch into the core (weight 1 - m) and env (weight m) views."""
    kw = dict(x=batch.x.float(), edge_index=batch.edge_index,
              batch_size=getattr(batch, 'batch_size', batch.num_nodes))
    if hasattr(batch, 'edge_type'):
        kw['edge_type'] = batch.edge_type
    if getattr(batch, 'edge_attr', None) is not None:
        kw['edge_attr'] = batch.edge_attr
    return Data(**kw, edge_weight=1.0 - m), Data(**kw, edge_weight=m)


# --------------------------------------------------------------------------
# Subgraph encoder
# --------------------------------------------------------------------------
class WeightedRGCNLayer(nn.Module):
    """Relational GCN layer whose messages are weighted by the attribution."""

    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.out_dim = out_dim
        self.W_rel = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.W_root = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.W_rel)
        nn.init.xavier_uniform_(self.W_root.weight)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        n = x.size(0)
        if edge_index.size(1) == 0:
            return self.W_root(x)

        src, dst = edge_index
        agg = torch.zeros(n, self.out_dim, device=x.device, dtype=x.dtype)
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            msg = x[src[mask]] @ self.W_rel[r]
            if edge_weight is not None:
                msg = msg * edge_weight[mask].unsqueeze(1)
            agg.scatter_add_(0, dst[mask].unsqueeze(1).expand_as(msg), msg)

        if edge_weight is not None:
            ws = torch.zeros(n, device=x.device, dtype=x.dtype)
            ws.scatter_add_(0, dst, edge_weight)
            agg = agg / (ws.unsqueeze(1) + 1e-8)
        else:
            deg = torch.zeros(n, device=x.device, dtype=x.dtype)
            deg.scatter_add_(0, dst, torch.ones(dst.size(0), device=x.device,
                                                dtype=x.dtype))
            agg = agg / deg.unsqueeze(1).clamp(min=1.0)
        return self.W_root(x) + agg


class SharedEncoder(nn.Module):
    """Two weighted layers, shared by the core and env branches so that the two
    representations live in the same embedding space."""

    def __init__(self, in_dim, hidden_dim, z_dim, num_relations):
        super().__init__()
        self.conv1 = WeightedRGCNLayer(in_dim, hidden_dim, num_relations)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = WeightedRGCNLayer(hidden_dim, z_dim, num_relations)

    def forward(self, data):
        x = data.x.float()
        ei = data.edge_index
        et = (data.edge_type if hasattr(data, 'edge_type')
              else torch.zeros(ei.size(1), dtype=torch.long, device=x.device))
        ew = getattr(data, 'edge_weight', None)
        h = F.relu(self.norm1(self.conv1(x, ei, et, ew)))
        z = self.conv2(h, ei, et, ew)
        return z[:data.batch_size]


# --------------------------------------------------------------------------
# Module 3: Causal Intervention
# --------------------------------------------------------------------------
class CausalIntervention(nn.Module):
    """Core classifier plus an adversarial env classifier behind the GRL.

    With use_grl=False the env classifier is still trained on the same loss, but
    the gradient is not reversed, so the branch cooperates rather than competes.
    This is the no_grl ablation, and it isolates the effect of the intervention
    from the effect of merely having a second branch.
    """

    def __init__(self, encoder, z_dim=64, class_weight=None, grad_alpha=1.0,
                 focal_gamma=2.0, use_grl=True):
        super().__init__()
        self.encoder = encoder
        self.grad_alpha = grad_alpha
        self.focal_gamma = focal_gamma
        self.use_grl = use_grl

        self.clf_core = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(z_dim // 2, 2))
        self.clf_env = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(),
            nn.Linear(z_dim // 2, 2))

        self.criterion = nn.CrossEntropyLoss(weight=class_weight)
        self.class_weight = class_weight

    def forward(self, core_g, env_g, labels):
        z_core = self.encoder(core_g)
        z_env = self.encoder(env_g)
        valid = labels != -1

        logits = self.clf_core(z_core)
        loss_sup = self.criterion(logits[valid], labels[valid])

        if valid.sum() > 1:
            z_rev = (GradReverse.apply(z_env, self.grad_alpha) if self.use_grl
                     else z_env)
            ce = F.cross_entropy(self.clf_env(z_rev)[valid], labels[valid],
                                 weight=self.class_weight, reduction='none')
            pt = torch.exp(-ce)
            loss_env = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            loss_env = torch.zeros((), device=z_env.device)

        return logits, {'loss_sup': loss_sup, 'loss_env': loss_env}


class SingleBranchIntervention(nn.Module):
    """Classifier for the no_dcd variant: one graph, no env branch, no adversary."""

    def __init__(self, encoder, z_dim=64, class_weight=None):
        super().__init__()
        self.encoder = encoder
        self.clf_core = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(z_dim // 2, 2))
        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, graph, labels):
        logits = self.clf_core(self.encoder(graph))
        valid = labels != -1
        return logits, {'loss_sup': self.criterion(logits[valid], labels[valid])}


def get_temperature(epoch, warmup_epochs=10, anneal_epochs=100, min_temp=0.3):
    """Anneal the attribution temperature from 1 down to min_temp after warm-up."""
    if epoch <= warmup_epochs:
        return 1.0
    progress = min(1.0, (epoch - warmup_epochs) / anneal_epochs)
    return float(1.0 - progress * (1.0 - min_temp))


# --------------------------------------------------------------------------
# Assembly and the shared forward pass
# --------------------------------------------------------------------------
def build_model(in_dim, cfg, num_relations, edge_feat_dim=None,
                class_weight=None, variant='full', device='cpu'):
    """Instantiate the modules a variant needs and return them in a dict."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}; choose from {VARIANTS}')

    models = {}
    # no_recon_* keep the reconstruction module out of the loss, but no_dcd and
    # the full model still need it.
    if variant in ('full', 'no_grl', 'no_dcd'):
        models['recon'] = GraphReconstruction(
            in_dim, cfg['hidden_dim'], cfg['z_dim'], num_relations,
            dropout=cfg.get('dropout', 0.3), edge_feat_dim=edge_feat_dim).to(device)

    if variant == 'no_dcd':
        # The prior is injected directly: error_node is concatenated to the node
        # features, so the encoder takes one extra input dimension.
        encoder = SharedEncoder(in_dim + 1, cfg['hidden_dim'], cfg['z_dim'],
                                num_relations).to(device)
        models['causal'] = SingleBranchIntervention(
            encoder, cfg['z_dim'], class_weight).to(device)
    else:
        models['dcd'] = CausalDecomposition(
            in_dim, edge_feat_dim=edge_feat_dim,
            use_error=(variant != 'no_recon_shape')).to(device)
        encoder = SharedEncoder(in_dim, cfg['hidden_dim'], cfg['z_dim'],
                                num_relations).to(device)
        models['causal'] = CausalIntervention(
            encoder, cfg['z_dim'], class_weight, cfg['grad_alpha'],
            cfg['focal_gamma'], use_grl=(variant != 'no_grl')).to(device)
    return models


def forward_variant(models, batch, cfg, variant, temperature=1.0):
    """One forward pass. Returns (logits, loss parts, attribution or None)."""
    labels = batch.y[:batch.batch_size]
    edge_attr = getattr(batch, 'edge_attr', None)
    parts = {}

    if variant == 'no_dcd':
        out = models['recon'](batch.x, batch.edge_index, batch.edge_type,
                              batch.batch_size, edge_attr)
        x_aug = torch.cat([batch.x.float(), out['error_node'].unsqueeze(1)], dim=1)
        graph = Data(x=x_aug, edge_index=batch.edge_index,
                     edge_type=batch.edge_type,
                     edge_weight=(1.0 - out['error_edge']),
                     batch_size=batch.batch_size)
        logits, loss_parts = models['causal'](graph, labels)
        parts.update(loss_parts)
        parts['recon_loss'] = out['recon_loss']
        return logits, parts, None

    if variant in ('full', 'no_grl'):
        out = models['recon'](batch.x, batch.edge_index, batch.edge_type,
                              batch.batch_size, edge_attr)
        error_node, error_edge = out['error_node'], out['error_edge']
        parts['recon_loss'] = out['recon_loss']
    elif variant == 'no_recon_random':
        error_node = torch.rand(batch.x.size(0), device=batch.x.device)
        error_edge = torch.rand(batch.edge_index.size(1), device=batch.x.device)
    else:  # no_recon_shape
        error_node, error_edge = None, None

    m = models['dcd'](batch.x, edge_attr, error_node, error_edge,
                      batch.edge_index, temperature=temperature)
    core_g, env_g = build_core_env_graph(batch, m)
    logits, loss_parts = models['causal'](core_g, env_g, labels)
    parts.update(loss_parts)
    return logits, parts, m


def combine_losses(parts, m, cfg, warming_up, w_split=None):
    """The joint objective of the paper, with the warm-up phase handled."""
    w_split = cfg['w_split'] if w_split is None else w_split
    loss = parts['loss_sup']
    if 'recon_loss' in parts:
        loss = loss + cfg['w_recon'] * parts['recon_loss']
    if warming_up or m is None:
        return loss
    return (loss
            + cfg['w_env'] * parts['loss_env']
            + w_split * (-m.var()))
