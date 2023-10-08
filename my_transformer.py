# coding: UTF-8

import math
import torch
import torch.nn as nn
import types
from config import Config
from torch.nn import functional as F

class MySelfAttention(nn.Module):
    def __init__(self, hyparams):
        super().__init__()
        assert hyparams.n_embd % hyparams.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(hyparams.n_embd, 3 * hyparams.n_embd)
        # output projection
        self.c_proj = nn.Linear(hyparams.n_embd, hyparams.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(hyparams.attn_pdrop)
        self.resid_dropout = nn.Dropout(hyparams.resid_pdrop)
        self.n_head = hyparams.n_head
        self.n_embd = hyparams.n_embd

    def forward(self, x, attention_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MyEncoderBlock(nn.Module):
    def __init__(self, hyparams):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hyparams.n_embd)
        self.attn = MySelfAttention(hyparams)
        self.ln_2 = nn.LayerNorm(hyparams.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(hyparams.n_embd, 4 * hyparams.n_embd),
            c_proj  = nn.Linear(4 * hyparams.n_embd, hyparams.n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(hyparams.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class MyEncoderTransformer(nn.Module):
    @staticmethod
    def get_default_hyparams(vocab_size:int, num_labels:int):
        hyparams = types.SimpleNamespace()
        hyparams.n_layer = 12
        hyparams.n_head = 12
        hyparams.n_embd =  768
        hyparams.vocab_size = vocab_size #21128
        hyparams.num_labels = num_labels
        hyparams.max_seq_len = 1024 # We can larger this params to allow longer sequence when inference
        hyparams.embd_pdrop = 0.1 # dropout hyperparameters
        hyparams.resid_pdrop = 0.1 # dropout hyperparameters
        hyparams.attn_pdrop = 0.1 # dropout hyperparameters
        return hyparams

    def __init__(self, hyparams):
        super().__init__()
        self.hyparams = hyparams
        Config.getDefaultConfig().logger.info("MyEncoderTransformer hyparams:\n" + str(hyparams))
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(hyparams.vocab_size, hyparams.n_embd),
            wpe = nn.Embedding(hyparams.max_seq_len, hyparams.n_embd),
            drop = nn.Dropout(hyparams.embd_pdrop),
            h = nn.ModuleList([MyEncoderBlock(hyparams) for _ in range(hyparams.n_layer)]),
            ln_f = nn.LayerNorm(hyparams.n_embd),
        ))
        # output probability distribution of number of classifications 
        self.dence = nn.Linear(hyparams.n_embd, hyparams.n_embd, bias=True)
        self.dence_activation = nn.Tanh()
        self.dence_dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(hyparams.n_embd, hyparams.num_labels, bias=True)

        # init all weights, and apply a special scaled init to the residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * hyparams.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        Config.getDefaultConfig().logger.info("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.hyparams.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.hyparams.max_seq_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, attention_mask)
        x = self.transformer.ln_f(x)

        # only use the first token to make classification, x.shape is (b, t, vocab_size)
        logits = self.lm_head(self.dence_dropout(self.dence_activation(self.dence(x[:, 0, :]))))

        # if we are given some desired labels also calculate the loss
        loss = None
        if labels is not None:
            # import pdb # pip install pdbpp (aka pdb++)
            # pdb.set_trace()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        return {"loss":loss, "logits":logits}