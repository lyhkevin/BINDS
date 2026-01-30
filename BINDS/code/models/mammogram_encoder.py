from models.vit import trunc_normal_, Transformer
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, convnext_tiny, densenet121, vit_b_16
from einops import rearrange, repeat
import torch.nn.functional as F

class mammogram_encoder(nn.Module):
    def __init__(self, depth_encoder, depth_fusion, backbone='ResNet', num_classes=[2], img_size=224,
                 patch_size=16, embed_dim=384, embed_dim_pathology=768, multi_modal=False, alignment=False, dropout=0.1):
        super(mammogram_encoder, self).__init__()
        self.alignment = alignment
        self.depth_encoder = depth_encoder
        self.depth_fusion = depth_fusion
        self.multi_modal = multi_modal
        self.backbone = backbone

        print('Mammogram backbone:', backbone)
        if backbone == 'ResNet':
            self.encoder = resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.patch_num = 49
            embed_dim = 512
        if backbone == 'ConvNeXt':
            self.encoder = convnext_tiny(pretrained=True)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.patch_num = 49
            embed_dim = 768
        if backbone == 'DenseNet':
            self.encoder = densenet121(pretrained=True)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            self.patch_num = 49
            embed_dim = 1024
        if backbone == 'ViT':
            self.encoder = vit_b_16(pretrained=True)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
            self.patch_num = 196
            embed_dim = 768

        self.fusion = Transformer(dim=embed_dim, depth=self.depth_fusion, heads=6, dim_head=64,
                   mlp_dim=int(4 * embed_dim), dropout=dropout, alignment=alignment, num_registers=16)
        self.num_patches = (img_size // patch_size) ** 2
        self.tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in num_classes])
        self.embedding_1 = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.embedding_2 = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num, embed_dim))

        for token in self.tokens:
            trunc_normal_(token, std=.02)
        trunc_normal_(self.embedding_1, std=.02)
        trunc_normal_(self.embedding_2, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

        self.linears = nn.ModuleList([nn.Linear(embed_dim, nc) for nc in num_classes])
        self.linears_alignment = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim_pathology) for _ in range(depth_fusion)]
        )
        self.emb_dropout = nn.Dropout(0.1)
        
        self.fusion.apply(self._init_weights) 
        self.linears.apply(self._init_weights)
        if hasattr(self, 'linears_alignment'):
            self.linears_alignment.apply(self._init_weights)

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
        view1, view2 = x[:, 0], x[:, 1]
        f_view1, f_view2 = self.encoder(view1), self.encoder(view2)
        f_view1 = rearrange(f_view1, 'b d h w -> b (h w) d')
        f_view2 = rearrange(f_view2, 'b d h w -> b (h w) d')

        patch_num = f_view1.shape[1]
        embedding_1 = repeat(self.embedding_1, '1 1 d -> b p d', b=batch_size, p=patch_num)
        embedding_2 = repeat(self.embedding_2, '1 1 d -> b p d', b=batch_size, p=patch_num)
        f_view1 = f_view1 + embedding_1
        f_view2 = f_view2 + embedding_2
        pos_embed = self.pos_embedding.repeat(batch_size, 1, 1)
        f_view1 = f_view1 + pos_embed
        f_view2 = f_view2 + pos_embed

        f = torch.cat((f_view1, f_view2), dim=1)
        f = self.emb_dropout(f)
        tokens = [token.expand(batch_size, -1, -1) for token in self.tokens]
        tokens.append(f)
        f = torch.cat(tokens, dim=1)

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
        view1, view2 = x[:, 0], x[:, 1]
        f_view1, f_view2 = self.encoder(view1), self.encoder(view2)
        f_view1 = rearrange(f_view1, 'b d h w -> b (h w) d')
        f_view2 = rearrange(f_view2, 'b d h w -> b (h w) d')

        # norm_emb1 = torch.norm(self.embedding_1).item()
        # norm_emb2 = torch.norm(self.embedding_2).item()
        # norm_pos_embed = torch.norm(self.pos_embedding).item()
        # norm_f_view1 = torch.norm(f_view1).item()
        # norm_f_view2 = torch.norm(f_view2).item()
        # print(f"Embedding_1 norm: {norm_emb1:.4f}, Embedding_2 norm: {norm_emb2:.4f}")
        # print(f"Embedding_1 norm_pos_embed: {norm_pos_embed:.4f}")
        # print(norm_f_view1, norm_f_view2)

        patch_num = f_view1.shape[1]
        embedding_1 = repeat(self.embedding_1, '1 1 d -> b p d', b=batch_size, p=patch_num)
        embedding_2 = repeat(self.embedding_2, '1 1 d -> b p d', b=batch_size, p=patch_num)
        f_view1 = f_view1 + embedding_1
        f_view2 = f_view2 + embedding_2
        pos_embed = self.pos_embedding.repeat(batch_size, 1, 1)
        f_view1 = f_view1 + pos_embed
        f_view2 = f_view2 + pos_embed

        f = torch.cat((f_view1, f_view2), dim=1)
        tokens = [token.expand(batch_size, -1, -1) for token in self.tokens]
        tokens.append(f)
        f = torch.cat(tokens, dim=1)
        f.retain_grad()

        f_out, attention = self.fusion(f)
        preds = [linear(f_out[:, i, :]) for i, linear in enumerate(self.linears)]

        pred = preds[0]
        class_idx = pred.argmax(dim=1)
        target_logit = pred.gather(1, class_idx.view(-1, 1)).squeeze().sum()
        self.zero_grad(set_to_none=True)
        target_logit.backward(retain_graph=False)
        token_importance = torch.abs((f * f.grad).sum(dim=-1))[:, 2:]
        half = token_importance.shape[-1] // 2
        sum_1 = token_importance[:, :half].sum(dim=1, keepdim=True)
        sum_2 = token_importance[:, half:].sum(dim=1, keepdim=True)
        sums = torch.cat([sum_1, sum_2], dim=1)
        token_importance = sums / sums.sum(dim=1, keepdim=True)
        count_first = (token_importance[:, 0] > token_importance[:, 1]).sum().item()
        count_second = (token_importance[:, 1] > token_importance[:, 0]).sum().item()
        return f_out, preds, attention, token_importance