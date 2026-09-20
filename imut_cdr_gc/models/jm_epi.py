"""Antigen-conditioned model-only definitions from the executed JM-Epi source.

Model math and state key names are retained. Constructors accept explicit
components/config instead of importing a trainer or loading data/checkpoints.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import EpiConfig
from .fr_cdr import ProteinMLMContrastModel

class RBF(nn.Module):

    def __init__(self, num_k: int=16, dmin: float=0.0, dmax: float=20.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(dmin, dmax, num_k))
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        diff = d.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff * diff)

def make_sparse_attn_mask_with_bias(g: dict[str, torch.Tensor], rbf: RBF, rbf_mlp: nn.Module) -> torch.Tensor:
    edge_index = g['edge_index']
    edge_attr = g['edge_attr']
    n_nodes = g['x'].size(0)
    dev = edge_index.device
    length = n_nodes + 1
    mask = torch.full((length, length), -1000000000.0, dtype=torch.float32, device=dev)
    if n_nodes == 0:
        mask[0, 0] = 0.0
        return mask.contiguous()
    mask[0, :] = 0.0
    mask[:, 0] = 0.0
    idx = torch.arange(n_nodes, device=dev) + 1
    mask[idx, idx] = 0.0
    if edge_index.numel() > 0:
        src, dst = (edge_index[0], edge_index[1])
        phi = rbf(edge_attr[:, 0].to(dev))
        bias = rbf_mlp(phi).squeeze(-1)
        bias = bias / math.sqrt(float(phi.size(-1)) + 1e-08)
        bias = torch.clamp(bias, -2.0, 2.0).to(mask.dtype)
        mask[src + 1, dst + 1] = bias
    return mask.contiguous()

class GraphTransformerBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int=8, mlp_ratio: float=4.0, dropout: float=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.SiLU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None=None) -> torch.Tensor:
        q = k = v = self.ln1(x)
        additive_mask = None
        if attn_mask is not None:
            additive_mask = attn_mask.to(dtype=q.dtype, device=q.device).contiguous()
        out, _ = self.attn(q, k, v, attn_mask=additive_mask, need_weights=False)
        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x

class PocketGraphTransformer(nn.Module):

    def __init__(self, node_dim: int=30, hidden: int=256, layers: int=4, num_heads: int=8, rbf_num_k: int=16):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.blocks = nn.ModuleList([GraphTransformerBlock(hidden, num_heads=num_heads) for _ in range(layers)])
        self.rbf = RBF(num_k=rbf_num_k)
        self.rbf_mlp = nn.Sequential(nn.Linear(rbf_num_k, rbf_num_k), nn.SiLU(), nn.Linear(rbf_num_k, 1))
        self.readout = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        nn.init.trunc_normal_(self.cls, std=0.02)
        with torch.no_grad():
            nn.init.zeros_(self.rbf_mlp[-1].weight)
            nn.init.zeros_(self.rbf_mlp[-1].bias)

    def forward(self, graph: dict[str, torch.Tensor], return_nodes: bool=False):
        dev = self.node_proj.weight.device
        x = graph['x'].to(device=dev, dtype=torch.float32)
        if x.size(0) == 0:
            cond = torch.zeros(self.readout[-1].out_features, device=dev, dtype=torch.float32)
            nodes = torch.zeros(1, 0, self.node_proj.out_features, device=dev, dtype=torch.float32)
            return (cond, nodes.squeeze(0)) if return_nodes else cond
        g_dev = {'x': x, 'edge_index': graph['edge_index'].to(device=dev, dtype=torch.long), 'edge_attr': graph['edge_attr'].to(device=dev, dtype=torch.float32)}
        h = self.node_proj(x).unsqueeze(0)
        h = torch.cat([self.cls.expand(1, -1, -1), h], dim=1)
        attn_mask = make_sparse_attn_mask_with_bias(g_dev, self.rbf, self.rbf_mlp)
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        cond = self.readout(h[:, 0, :]).squeeze(0)
        return (cond, h.squeeze(0)) if return_nodes else cond

class TokenPocketCrossAttn(nn.Module):

    def __init__(self, d_model: int, d_pocket: int, n_heads: int=4, dropout: float=0.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_pocket, d_model, bias=False)
        self.v = nn.Linear(d_pocket, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln_q = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, pocket_nodes: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        q = self.q(self.ln_q(h))
        k = self.k(pocket_nodes)
        v = self.v(pocket_nodes)
        out, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return out

class EpitopeConditionedFRCDRModel(ProteinMLMContrastModel):

    def __init__(self, *, encoder, tokenizer, config=None):
        epi_config = config or EpiConfig()
        epi_config.validate(int(encoder.config.hidden_size))
        super().__init__(encoder=encoder, tokenizer=tokenizer, config=epi_config.backbone)
        self.epi_config = epi_config
        d_model = self.encoder.config.hidden_size
        self.pocket = PocketGraphTransformer(node_dim=self.epi_config.node_dim, hidden=self.epi_config.pocket_hidden, layers=self.epi_config.pocket_layers, num_heads=self.epi_config.pocket_heads, rbf_num_k=self.epi_config.pocket_rbf_k)
        self.cross = TokenPocketCrossAttn(d_model=d_model, d_pocket=self.epi_config.pocket_hidden, n_heads=self.epi_config.cross_heads)
        self.film_mlp = nn.Sequential(nn.Linear(2 * d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, 2 * d_model))
        with torch.no_grad():
            last = self.film_mlp[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            last.bias[:d_model].fill_(1.0)

    def _encode_graph_batch(self, graphs: list[dict[str, torch.Tensor]], dev: torch.device):
        node_list = []
        lengths = []
        for graph in graphs:
            _, nodes = self.pocket(graph, return_nodes=True)
            node_list.append(nodes)
            lengths.append(nodes.size(0))
        batch_size = len(node_list)
        max_len = max(lengths) if lengths else 0
        dtype = node_list[0].dtype if node_list else torch.float32
        pocket_nodes = torch.zeros((batch_size, max_len, self.epi_config.pocket_hidden), device=dev, dtype=dtype)
        key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=dev)
        for i, nodes in enumerate(node_list):
            n = nodes.size(0)
            if n > 0:
                pocket_nodes[i, :n, :] = nodes.to(device=dev, dtype=dtype)
                key_padding_mask[i, :n] = False
        return (pocket_nodes, key_padding_mask)

    def _condition_tokens(self, h: torch.Tensor, pocket_nodes: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        ctx = self.cross(h, pocket_nodes, key_padding_mask=key_padding_mask)
        gb = self.film_mlp(torch.cat([h, ctx], dim=-1))
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * h + beta

    def forward(self, input_ids, attention_mask, mask_pos, graphs):
        dev = input_ids.device
        pocket_nodes, key_padding_mask = self._encode_graph_batch(graphs, dev)
        masked_ids = self._apply_real_mask_ids(input_ids, attention_mask, mask_pos)
        h_masked = self._embed_ids(masked_ids, attention_mask)
        h_masked = self._run_rdt(h_masked, attention_mask)
        h_masked_cond = self._condition_tokens(h_masked, pocket_nodes, key_padding_mask)
        seq_logits = self.lm_head(h_masked_cond)
        w_masked, bias_masked = self._mask_soft(attention_mask, mask_pos)
        anchor_vec = self._pool_core(h_masked_cond, w_masked, self.config.pooling_type, score_bias=bias_masked)
        anchor = F.normalize(self.proj(anchor_vec), dim=1, eps=1e-08)
        with torch.no_grad():
            h_clean = self._embed_ids(input_ids, attention_mask)
            h_clean = self._run_rdt(h_clean, attention_mask)
            h_clean_cond = self._condition_tokens(h_clean, pocket_nodes, key_padding_mask)
            w_clean, bias_clean = self._mask_soft(attention_mask, mask_pos)
            pos_vec = self._pool_core(h_clean_cond, w_clean, self.config.pooling_type, score_bias=bias_clean)
            positive = F.normalize(self.proj(pos_vec), dim=1, eps=1e-08)
        return (seq_logits, anchor, positive)
