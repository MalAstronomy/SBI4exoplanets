import vit_pytorch.vit_original as ViT
import torch
from torchsummary import summary
from torch import nn


class ViT_modified(nn.Module):
    def __init__(self,  image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        image_height, image_width = ViT.pair(image_size)
        patch_height, patch_width = ViT.pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            ViT.Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = ViT.Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim+13),
                nn.Linear(dim+13, num_classes)
            ) 

    def forward(self, image):
            
        img_log = (torch.log(img[:,:,:,13:])-torch.log(torch.exp(torch.Tensor([-18.]))))/torch.log(torch.exp(torch.Tensor([6.])))
        x = self.to_patch_embedding(img_log)
        b, n, _ = x.shape

        cls_tokens = ViT.repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        params = torch.squeeze(img[:,:,:,:13], dim=1)
        params = torch.squeeze(params, dim = 1)
        
        x = torch.cat((params,x),dim=1)
        
        return self.mlp_head(x)
