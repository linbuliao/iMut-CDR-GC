"""Model-only FR backbone extracted from the hash-pinned executed v1 source.

No trainer import, data access, device selection, environment mutation or RNG
seeding occurs at import. Configuration replaces research module globals.
"""
from __future__ import annotations
from contextlib import nullcontext
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import BackboneConfig

class RMSNorm(nn.Module):
    """Small RMSNorm implementation so AttnRes keys follow the paper more closely."""

    def __init__(self, dim, eps=1e-06):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        orig_dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x_float * rms * self.weight.float()
        return y.to(orig_dtype)

class SimpleTransformerBlock(nn.Module):
    """Lightweight PreNorm Transformer block used for explicit RDT Prelude/Coda."""

    def __init__(self, d_model, num_heads, ffn_mult=4, dropout=0.1, init_scale=0.1):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model))
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x, attn_mask=None):
        key_padding_mask = attn_mask == 0 if attn_mask is not None else None
        z = self.ln_attn(x)
        attn_out, _ = self.self_attn(z, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_scale * self.attn_dropout(attn_out)
        ffn_out = self.ffn(self.ln_ffn(x))
        x = x + self.ffn_scale * self.ffn_dropout(ffn_out)
        if attn_mask is not None:
            x = x.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        return x

class SharedRDTBlock(nn.Module):
    """Shared recurrent-depth Transformer block with full AttnRes over update sources.

    This block implements the RDT loop update
        h_{t+1} = A * h_t + B * e + Transformer(h_t, e)
    while making the Transformer input closer to Full Attention Residuals:
        - history_sources stores e and individual residual/update outputs, not hidden states.
        - the attention and FFN sublayers each use a loop/sublayer-specific pseudo-query.
        - keys/values are the previous residual/update sources with RMSNorm on keys.

    In other words, this is no longer "hidden-state history attention". It is an
    adaptation of paper-style depth-wise AttnRes to a looped/recurrent block.
    """

    def __init__(self, d_model, num_heads, ffn_mult=4, dropout=0.1, init_scale=0.1, attn_residual=True, gate_hidden=256, max_steps=4, use_loop_emb=True):
        super().__init__()
        self.attn_residual = attn_residual
        self.max_steps = int(max_steps)
        self.use_loop_emb = bool(use_loop_emb)
        self.ln_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model))
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.inject_h_logit = nn.Parameter(torch.tensor(2.0))
        self.inject_e_logit = nn.Parameter(torch.tensor(-2.0))
        self.depth_key_norm = RMSNorm(d_model)
        self.depth_queries = nn.Parameter(torch.zeros(max(1, self.max_steps), 2, d_model))
        self.attnres_gamma_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        if self.use_loop_emb:
            self.loop_emb = nn.Embedding(max(1, self.max_steps), d_model)
            nn.init.zeros_(self.loop_emb.weight)
        else:
            self.loop_emb = None

    def _step_index(self, step_idx: int) -> int:
        return min(max(int(step_idx), 0), self.depth_queries.shape[0] - 1)

    def _loop_bias(self, step_idx: int, dtype, device):
        if self.loop_emb is None:
            return None
        idx = torch.tensor([self._step_index(step_idx)], device=device, dtype=torch.long)
        return self.loop_emb(idx).view(1, 1, -1).to(dtype=dtype)

    def _full_attn_residual(self, sources, step_idx: int, sublayer_idx: int, attn_mask=None):
        """Full depth-wise softmax attention over previous update sources.

        sources: list of [B, L, D], where sources[0] is encoded input e and
                 later entries are individual attention/FFN update outputs.
        returns: [B, L, D]
        """
        if len(sources) == 1:
            mixed = sources[0]
        else:
            stacked = torch.stack(sources, dim=0)
            keys = self.depth_key_norm(stacked)
            q = self.depth_queries[self._step_index(step_idx), int(sublayer_idx)]
            q = q.to(dtype=keys.dtype, device=keys.device)
            scores = torch.einsum('sbld,d->sbl', keys.float(), q.float())
            alpha = torch.softmax(scores, dim=0).to(dtype=stacked.dtype)
            mixed = (stacked * alpha.unsqueeze(-1)).sum(dim=0)
        if attn_mask is not None:
            mixed = mixed.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        return mixed

    def forward(self, x, e, history_sources, step_idx: int, attn_mask=None):
        key_padding_mask = attn_mask == 0 if attn_mask is not None else None
        loop_bias = self._loop_bias(step_idx, dtype=x.dtype, device=x.device)
        gamma = torch.sigmoid(self.attnres_gamma_logit).to(dtype=x.dtype, device=x.device)
        if self.attn_residual:
            attn_mem = self._full_attn_residual(history_sources, step_idx, 0, attn_mask=attn_mask)
            attn_input = x + gamma * (attn_mem - x)
        else:
            attn_input = x
        if loop_bias is not None:
            attn_input = attn_input + loop_bias
        z = self.ln_attn(attn_input)
        attn_out, _ = self.self_attn(z, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        attn_update = self.attn_scale * self.attn_dropout(attn_out)
        if attn_mask is not None:
            attn_update = attn_update.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        if self.attn_residual:
            ffn_sources = history_sources + [attn_update]
            ffn_mem = self._full_attn_residual(ffn_sources, step_idx, 1, attn_mask=attn_mask)
            ffn_base = x + attn_update
            ffn_input = ffn_base + gamma * (ffn_mem - ffn_base)
        else:
            ffn_input = x + attn_update
        if loop_bias is not None:
            ffn_input = ffn_input + loop_bias
        ffn_out = self.ffn(self.ln_ffn(ffn_input))
        ffn_update = self.ffn_scale * self.ffn_dropout(ffn_out)
        if attn_mask is not None:
            ffn_update = ffn_update.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        core_update = attn_update + ffn_update
        a = torch.sigmoid(self.inject_h_logit)
        b = torch.sigmoid(self.inject_e_logit)
        x_next = a * x + b * e + core_update
        if attn_mask is not None:
            x_next = x_next.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        return (x_next, [attn_update, ffn_update])

class ProteinMLMContrastModel(nn.Module):

    def __init__(self, *, encoder, tokenizer, config=None):
        super().__init__()
        self.config = config or BackboneConfig()
        self.config.validate(int(encoder.config.hidden_size))
        self.tokenizer = tokenizer
        self.encoder = encoder
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = False
        if hasattr(self.encoder, 'contact_head') and self.encoder.contact_head is not None:
            for p in self.encoder.contact_head.parameters():
                p.requires_grad = False
        if not self.config.freeze_encoder and self.config.gradient_checkpointing:
            try:
                self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
        elif self.config.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.model_max_len = getattr(self.encoder.config, 'max_position_embeddings', None)
        self.mask_token_str = getattr(self.tokenizer, 'mask_token', None) or '<mask>'
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.mask_token_str)
        pass
        aa_tokens = list('ACDEFGHIKLMNPQRSTVWY')
        vocab_size = len(self.tokenizer)
        aa_mask = torch.zeros(vocab_size, dtype=torch.bool)
        for aa in aa_tokens:
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if isinstance(tid, int) and tid >= 0:
                aa_mask[tid] = True
        self.register_buffer('aa_mask', aa_mask, persistent=False)
        d_model = self.encoder.config.hidden_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)
        if self.config.tie_weights:
            token_embedding = self._find_token_embedding(self.encoder)
            if token_embedding is not None and token_embedding.weight.shape[0] == vocab_size and (token_embedding.weight.shape[1] == d_model):
                self.lm_head.weight = token_embedding.weight
                pass
                if self.config.freeze_encoder:
                    pass
            else:
                pass
        self.proj = nn.Linear(d_model, self.config.projection_dim, bias=False)
        self.pool_q = nn.Linear(d_model, 1, bias=False)
        self.soft_mask_bias = nn.Parameter(torch.tensor(self.config.soft_mask_bias_init, dtype=torch.float32))
        self.use_rdt = self.config.use_rdt
        self.rdt_steps = int(self.config.rdt_steps)
        if self.use_rdt and self.rdt_steps > 0:
            self.rdt_prelude = nn.ModuleList([SimpleTransformerBlock(d_model=d_model, num_heads=self.config.rdt_heads, ffn_mult=self.config.rdt_ffn_mult, dropout=self.config.rdt_dropout, init_scale=self.config.rdt_init_scale) for _ in range(max(0, self.config.rdt_prelude_layers))])
            self.rdt_block = SharedRDTBlock(d_model=d_model, num_heads=self.config.rdt_heads, ffn_mult=self.config.rdt_ffn_mult, dropout=self.config.rdt_dropout, init_scale=self.config.rdt_init_scale, attn_residual=self.config.attention_residual, gate_hidden=self.config.attention_residual_hidden, max_steps=self.rdt_steps, use_loop_emb=self.config.rdt_loop_embedding)
            self.rdt_coda = nn.ModuleList([SimpleTransformerBlock(d_model=d_model, num_heads=self.config.rdt_heads, ffn_mult=self.config.rdt_ffn_mult, dropout=self.config.rdt_dropout, init_scale=self.config.rdt_init_scale) for _ in range(max(0, self.config.rdt_coda_layers))])
            self.rdt_final_ln = nn.LayerNorm(d_model) if self.config.rdt_final_ln else nn.Identity()
            pass
        else:
            self.rdt_prelude = nn.ModuleList()
            self.rdt_block = None
            self.rdt_coda = nn.ModuleList()
            self.rdt_final_ln = nn.Identity()
        pass

    @staticmethod
    def _find_token_embedding(encoder):
        candidates = [('embeddings', 'word_embeddings'), ('embed_tokens',), ('encoder', 'embed_tokens')]
        for path in candidates:
            m = encoder
            ok = True
            for name in path:
                if not hasattr(m, name):
                    ok = False
                    break
                m = getattr(m, name)
            if ok and isinstance(m, nn.Embedding):
                return m
        return None

    def _embed_ids(self, input_ids, attention_mask):
        with self.inference_context():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state

    def _apply_real_mask_ids(self, input_ids, attention_mask, mask_pos):
        masked_ids = input_ids.clone()
        B, L = masked_ids.shape
        for i, pos_list in enumerate(mask_pos):
            valid = [p for p in pos_list if 0 <= p < L and attention_mask[i, p] > 0]
            if valid:
                masked_ids[i, torch.tensor(valid, device=masked_ids.device, dtype=torch.long)] = self.mask_token_id
        return masked_ids

    def _pool_core(self, H, mask_binary, mode='mean', score_bias=None):
        if mode == 'attn':
            scores = self.pool_q(H).squeeze(-1)
            if score_bias is not None:
                scores = scores + score_bias.to(dtype=scores.dtype, device=scores.device)
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask_binary == 0, neg_inf)
            row_all_masked = mask_binary.sum(dim=1, keepdim=True) == 0
            if row_all_masked.any():
                scores[row_all_masked.expand_as(scores)] = 0
            w = scores.softmax(dim=-1)
            return (H * w.unsqueeze(-1)).sum(1)
        else:
            w = mask_binary.float()
            denom = w.sum(1, keepdim=True).clamp_min(1e-08)
            w = w / denom
            return (H * w.unsqueeze(-1)).sum(1)

    def _mask_all(self, attn_mask):
        return attn_mask

    def _mask_context_only(self, attn_mask, mask_pos):
        B, L = attn_mask.shape
        w = attn_mask.clone()
        for i, pos_list in enumerate(mask_pos):
            for p in pos_list:
                if 0 <= p < L:
                    w[i, p] = 0
        return w

    def _mask_masked_only(self, attn_mask, mask_pos):
        B, L = attn_mask.shape
        w = torch.zeros_like(attn_mask)
        for i, pos_list in enumerate(mask_pos):
            for p in pos_list:
                if 0 <= p < L:
                    w[i, p] = 1
        empty = w.sum(1) == 0
        if empty.any():
            w[empty] = attn_mask[empty]
        return w

    def _mask_soft(self, attn_mask, mask_pos):
        B, L = attn_mask.shape
        w = attn_mask.clone()
        score_bias = torch.zeros_like(attn_mask, dtype=torch.float32)
        for i, pos_list in enumerate(mask_pos):
            for p in pos_list:
                if 0 <= p < L and attn_mask[i, p] > 0:
                    score_bias[i, p] = self.soft_mask_bias
        return (w, score_bias)

    def _mean_pool(self, H, attn_mask):
        w = attn_mask.float().unsqueeze(-1)
        return (H * w).sum(1) / w.sum(1).clamp_min(1e-08)

    def _run_rdt(self, H, attn_mask):
        if self.rdt_block is None or self.rdt_steps <= 0:
            return H
        if attn_mask is not None:
            H = H.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        e = H
        for block in self.rdt_prelude:
            e = block(e, attn_mask)
        if attn_mask is not None:
            e = e.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        x = e
        history_sources = [e]
        for step_idx in range(self.rdt_steps):
            x, new_sources = self.rdt_block(x, e, history_sources, step_idx, attn_mask)
            history_sources.extend(new_sources)
        for block in self.rdt_coda:
            x = block(x, attn_mask)
        x = self.rdt_final_ln(x)
        if attn_mask is not None:
            x = x.masked_fill(attn_mask.unsqueeze(-1) == 0, 0.0)
        return x

    def forward(self, input_ids, attention_mask, mask_pos):
        masked_ids = self._apply_real_mask_ids(input_ids, attention_mask, mask_pos)
        H_masked = self._embed_ids(masked_ids, attention_mask)
        attn_mask = attention_mask
        H_masked = self._run_rdt(H_masked, attn_mask)
        seq_logits = self.lm_head(H_masked)
        w_masked, bias_masked = self._mask_soft(attn_mask, mask_pos)
        anchor_vec = self._pool_core(H_masked, w_masked, self.config.pooling_type, score_bias=bias_masked)
        anchor = F.normalize(self.proj(anchor_vec), dim=1)
        if self.config.positive_teacher:
            with torch.no_grad():
                H_clean = self._embed_ids(input_ids, attention_mask)
                H_clean = self._run_rdt(H_clean, attention_mask)
                w_clean, bias_clean = self._mask_soft(attention_mask, mask_pos)
                pos_vec = self._pool_core(H_clean, w_clean, self.config.pooling_type, score_bias=bias_clean)
                positive = F.normalize(self.proj(pos_vec), dim=1)
        else:
            with torch.no_grad():
                H_clean = self._embed_ids(input_ids, attention_mask)
            H_clean = self._run_rdt(H_clean, attention_mask)
            w_clean, bias_clean = self._mask_soft(attention_mask, mask_pos)
            pos_vec = self._pool_core(H_clean, w_clean, self.config.pooling_type, score_bias=bias_clean)
            positive = F.normalize(self.proj(pos_vec), dim=1)
        return (seq_logits, anchor, positive)

    def inference_context(self):
        return nullcontext()
