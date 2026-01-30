from models.masked_attention import trunc_normal_, Transformer
import torch
from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
import random

class multimodal_predictor(nn.Module):
    def __init__(self, device, num_heads=8, num_classes=[2], embed_dim=512, depth=10, dropout=0.1):
        super(multimodal_predictor, self).__init__()
        self.depth = depth
        self.device = device
        self.num_heads = num_heads
        print('Multimodal predictor num_heads:', num_heads)
        self.embedding_ultrasound = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embedding_mammogram = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embedding_mri = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in num_classes])
        self.fusion = Transformer(dim = embed_dim, depth = depth, heads = num_heads, dim_head = 64, mlp_dim = int(4 * embed_dim), dropout = dropout)
        self.num_classes = num_classes
        self.linears = nn.ModuleList([nn.Linear(embed_dim, nc) for nc in num_classes])
        self.emb_dropout = nn.Dropout(dropout)
        for token in self.tokens:
            trunc_normal_(token, std=.02)
        trunc_normal_(self.embedding_ultrasound, std=.02)
        trunc_normal_(self.embedding_mammogram, std=.02)
        trunc_normal_(self.embedding_mri, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_mask(self, has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri):
        batch_size = len(has_ultrasound)
        num_tokens_ultrasound, num_tokens_mammogram, num_tokens_mri = (
            f_ultrasound.shape[1], f_mammogram.shape[1], f_mri.shape[1]
        )
        attn_mask_cls_token = torch.zeros(batch_size, len(self.num_classes), device=self.device, dtype=torch.bool)
        attn_mask_ultrasound = torch.tensor(
            [[False] * num_tokens_ultrasound if x else [True] * num_tokens_ultrasound for x in has_ultrasound],
            device=self.device, dtype=torch.bool
        )
        attn_mask_mammogram = torch.tensor(
            [[False] * num_tokens_mammogram if x else [True] * num_tokens_mammogram for x in has_mammogram],
            device=self.device, dtype=torch.bool
        )
        attn_mask_mri = torch.tensor(
            [[False] * num_tokens_mri if x else [True] * num_tokens_mri for x in has_mri],
            device=self.device, dtype=torch.bool
        )
        attn_mask = torch.cat((attn_mask_cls_token, attn_mask_ultrasound,
                               attn_mask_mammogram, attn_mask_mri), dim=1)

        seq_len = attn_mask.shape[1]
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(-1)  # shape: [B, 1, L, 1]
        attn_mask = attn_mask.repeat(1, self.num_heads, 1, seq_len)  # [B, H, L, L]
        attn_mask = rearrange(attn_mask, 'b c h w -> b c w h')
        return attn_mask

    def forward(self, has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri, noise_std=0.1):

        if noise_std > 0:
            f_ultrasound = f_ultrasound + torch.randn_like(f_ultrasound) * noise_std
            f_mammogram = f_mammogram + torch.randn_like(f_mammogram) * noise_std
            f_mri = f_mri + torch.randn_like(f_mri) * noise_std

        attn_mask = self.get_mask(has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri)
        batch_size = f_mri.shape[0]
        patch_num_ultrasound, patch_num_mammogram, patch_num_mri = f_ultrasound.shape[1], f_mammogram.shape[1], f_mri.shape[1]
        embedding_ultrasound = repeat(self.embedding_ultrasound, '1 1 d -> b p d', b=batch_size, p=patch_num_ultrasound)
        embedding_mammogram = repeat(self.embedding_mammogram, '1 1 d -> b p d', b=batch_size, p=patch_num_mammogram)
        embedding_mri = repeat(self.embedding_mri, '1 1 d -> b p d', b=batch_size, p=patch_num_mri)

        f_ultrasound = embedding_ultrasound + f_ultrasound
        f_mammogram = embedding_mammogram + f_mammogram
        f_mri = embedding_mri + f_mri

        f = torch.cat((f_ultrasound, f_mammogram, f_mri), dim=1)
        tokens = [token.expand(batch_size, -1, -1) for token in self.tokens]
        tokens.append(f)
        f = torch.cat(tokens, dim=1)
        f = self.emb_dropout(f)

        f, attention = self.fusion(f, attn_mask)
        predictions = [linear(f[:, i, :]) for i, linear in enumerate(self.linears)]
        return f, predictions

    def forward_with_grad(self, has_ultrasound, has_mammogram, has_mri,
                          f_ultrasound, f_mammogram, f_mri):

        batch_size = len(has_ultrasound)
        embed_dim = self.embedding_ultrasound.shape[-1]
        
        if f_ultrasound is None or f_ultrasound.shape[1] == 0:
            f_ultrasound = torch.zeros(batch_size, 0, embed_dim, device=self.device)
        if f_mammogram is None or f_mammogram.shape[1] == 0:
            f_mammogram = torch.zeros(batch_size, 0, embed_dim, device=self.device)
        if f_mri is None or f_mri.shape[1] == 0:
            f_mri = torch.zeros(batch_size, 0, embed_dim, device=self.device)
        attn_mask = self.get_mask(has_ultrasound, has_mammogram, has_mri, f_ultrasound, f_mammogram, f_mri)

        patch_num_ultrasound, patch_num_mammogram, patch_num_mri = (f_ultrasound.shape[1], f_mammogram.shape[1], f_mri.shape[1])
        embedding_ultrasound = repeat(self.embedding_ultrasound, '1 1 d -> b p d',
                                      b=batch_size, p=patch_num_ultrasound)
        embedding_mammogram = repeat(self.embedding_mammogram, '1 1 d -> b p d',
                                     b=batch_size, p=patch_num_mammogram)
        embedding_mri = repeat(self.embedding_mri, '1 1 d -> b p d',
                               b=batch_size, p=patch_num_mri)

        f_ultrasound = f_ultrasound + embedding_ultrasound
        f_mammogram = f_mammogram + embedding_mammogram
        f_mri = f_mri + embedding_mri

        f = torch.cat((f_ultrasound, f_mammogram, f_mri), dim=1)
        tokens = [token.expand(batch_size, -1, -1) for token in self.tokens]
        tokens.append(f)
        f = torch.cat(tokens, dim=1)
        f.retain_grad()

        f_out, attention = self.fusion(f, attn_mask)
        preds = [linear(f_out[:, i, :]) for i, linear in enumerate(self.linears)]

        pred = preds[0]
        class_idx = pred.argmax(dim=1)
        target_logit = pred.gather(1, class_idx.view(-1, 1)).squeeze().sum()
        self.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=False)

        token_importance = torch.abs((f * f.grad).sum(dim=-1))[:, len(self.tokens):]

        imp_ultrasound = token_importance[:, :patch_num_ultrasound].sum(dim=1, keepdim=True)
        imp_mammogram = token_importance[:, patch_num_ultrasound:
                                            patch_num_ultrasound + patch_num_mammogram].sum(dim=1, keepdim=True)
        imp_mri = token_importance[:, patch_num_ultrasound + patch_num_mammogram:].sum(dim=1, keepdim=True)

        sums = torch.cat([imp_ultrasound, imp_mammogram, imp_mri], dim=1)
        token_importance = sums / sums.sum(dim=1, keepdim=True)
        return f_out, preds, attention, token_importance


