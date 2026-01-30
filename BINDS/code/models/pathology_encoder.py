from einops import rearrange
from models.dinov2 import vit_base_dinov2
from models.vit import *

class Pathology_encoder(nn.Module):
    def __init__(self, scale, depth_encoder, depth_fusion, device, backbone='ViT', embed_dim=384, num_classes=[2], alignment=True, pretrain_path=None):
        super(Pathology_encoder, self).__init__()
        
        self.depth_encoder = depth_encoder
        self.depth_fusion = depth_fusion
        self.backbone = backbone
        
        self.small_encoder = vit_base_dinov2(img_size=224, patch_size=14, block_chunks=0, init_values=1)
        self.medium_encoder = vit_base_dinov2(img_size=224, patch_size=14, block_chunks=0, init_values=1)
        self.large_encoder = vit_base_dinov2(img_size=224, patch_size=14, block_chunks=0, init_values=1)
        embed_dim = 768
        
        self.fusion = Transformer(dim=embed_dim, depth=self.depth_fusion, heads=6, dim_head=64,
                                  mlp_dim=int(4 * embed_dim), dropout=0., alignment=alignment)
        self.norm = nn.LayerNorm(embed_dim)
        self.scale = scale
        self.device = device
        self.tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, embed_dim)) for _ in num_classes])
        self.linears = nn.ModuleList([nn.Linear(embed_dim, nc) for nc in num_classes])
        self.encoders = {'small': self.small_encoder, 'medium': self.medium_encoder, 'large': self.large_encoder}

        self.embeddings = nn.ParameterDict()
        for scale in self.scale:
            self.embeddings[scale] = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.apply(self._init_weights)
        
        if pretrain_path is not None:
            print('load dinov2')
            state_dict = torch.load(pretrain_path, map_location="cpu")['teacher']
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.small_encoder.load_state_dict(state_dict, strict=False)
            self.medium_encoder.load_state_dict(state_dict, strict=False)
            self.large_encoder.load_state_dict(state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, small_imgs, medium_imgs, large_imgs):
        batch_size, patch_num = small_imgs.shape[0], small_imgs.shape[1]
        multi_scale_imgs = {'small': small_imgs, 'medium': medium_imgs, 'large': large_imgs}
        features = [token.expand(batch_size, -1, -1) for token in self.tokens]

        for scale in self.scale:
            imgs = multi_scale_imgs[scale]
            imgs = rearrange(imgs, 'b p c h w -> (b p) c h w')
            with torch.no_grad():
                f = self.encoders[scale](imgs)
            f = rearrange(f, '(b p) d -> b p d', b=batch_size, p=patch_num)
            embedding = repeat(self.embeddings[scale], '1 1 d -> b p d', b=batch_size, p=patch_num)
            f = f + embedding
            features.append(f)

        f = torch.cat(features, dim=1)
        f, features, _ = self.fusion(f)
        features = [tensor[:, 0] for tensor in features]
        predictions = [self.linears[i](f[:, i, :]) for i in range(len(self.tokens))]
        return predictions, features