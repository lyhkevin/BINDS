from models.ResNet_3D import ResNet, BasicBlock
from models.vit import trunc_normal_, Transformer
import torch
from torch import nn
from einops import rearrange, repeat

class mri_encoder(nn.Module):
    def __init__(self, modalities, depth_encoder, depth_fusion, pretrain_path=None, backbone='ResNet_3D',
                 num_classes=[2], embed_dim=384, embed_dim_pathology=768, multi_modal=False, alignment=False, dropout=0.1):
        super(mri_encoder, self).__init__()
        self.alignment = alignment
        self.depth_encoder = depth_encoder
        self.depth_fusion = depth_fusion
        self.multi_modal = multi_modal
        self.modalities = modalities
        self.encoders = nn.ModuleDict()
        self.embeddings = nn.ParameterDict()
        self.num_patches = 36

        print('MRI backbone:', backbone)
        for modality in modalities:
            self.encoders[modality] = ResNet(BasicBlock, layers=[2, 2, 1, 1], block_inplanes=[64, 128, 256, embed_dim])
        for modality in modalities:
            self.embeddings[modality] = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.fusion = Transformer(dim=embed_dim, depth=self.depth_fusion, heads=6, dim_head=64,
                      mlp_dim=int(4 * embed_dim), dropout=dropout, alignment=alignment, num_registers=32)
        self.tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in num_classes])

        for token in self.tokens:
            trunc_normal_(token, std=.02)
        for modality in modalities:
            trunc_normal_(self.embeddings[modality], std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

        self.linears = nn.ModuleList([nn.Linear(embed_dim, nc) for nc in num_classes])
        self.linears_alignment = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim_pathology) for _ in range(depth_fusion)]
        )
        self.emb_dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)

        if pretrain_path != None:
            print('load pretrained model from', pretrain_path)
            for modality in modalities:
                state_dict = torch.load(pretrain_path)['state_dict']
                self.encoders[modality].load_state_dict(state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        features = [token.expand(batch_size, -1, -1) for token in self.tokens]

        for i, modality in enumerate(self.modalities):
            modality_data = rearrange(x[:, i], 'b f c h w -> b c f h w')
            f = self.encoders[modality](modality_data)
            embedding = repeat(self.embeddings[modality], '1 1 d -> b p d', b=batch_size, p=self.num_patches)
            pos_embed = self.pos_embedding.repeat(batch_size, 1, 1)
            f = f + embedding + pos_embed
            f = self.emb_dropout(f)
            features.append(f)
        f = torch.cat(features, dim=1)

        if self.alignment == True:
            f, features, attention = self.fusion(f)
            features = [tensor[:, 1] for tensor in features]
            aligned_features = []
            for feature, linear_layer in zip(features, self.linears_alignment):
                feature = linear_layer(feature)
                aligned_features.append(feature)
            features = aligned_features
        else:
            f, attention = self.fusion(f)

        predictions = [linear(f[:, i, :]) for i, linear in enumerate(self.linears)]

        if self.multi_modal == True:
            return f, predictions, attention

        if self.alignment == True:
            return predictions, features
        else:
            return predictions

    def forward_with_grad(self, x):
        batch_size = x.shape[0]
        features = [token.expand(batch_size, -1, -1) for token in self.tokens]
        for i, modality in enumerate(self.modalities):
            modality_data = rearrange(x[:, i], 'b f c h w -> b c f h w')
            f = self.encoders[modality](modality_data)
            embedding = repeat(self.embeddings[modality], '1 1 d -> b p d', b=batch_size, p=self.num_patches)
            pos_embed = self.pos_embedding.repeat(batch_size, 1, 1)
            f = f + embedding + pos_embed
            features.append(f)
        f = torch.cat(features, dim=1)
        f.retain_grad()

        f_out, attention = self.fusion(f)

        preds = [linear(f_out[:, i, :]) for i, linear in enumerate(self.linears)]
        pred = preds[0]
        class_idx = pred.argmax(dim=1)
        target_logit = pred.gather(1, class_idx.view(-1, 1)).squeeze().sum()
        self.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=False)
        token_importance = torch.abs((f * f.grad).sum(dim=-1))
        token_importance = token_importance[:, len(self.tokens):]
        total_len = token_importance.shape[-1]
        quarter = total_len // 4
        sums = []
        for i in range(4):
            part = token_importance[:, i * quarter:(i + 1) * quarter]
            sums.append(part.sum(dim=1, keepdim=True))
        sums = torch.cat(sums, dim=1)
        token_importance = sums / sums.sum(dim=1, keepdim=True)
        max_idx = token_importance.argmax(dim=1)
        counts = [(max_idx == i).sum().item() for i in range(4)]
        return f_out, preds, attention, token_importance