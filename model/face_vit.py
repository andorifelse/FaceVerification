"""CVLFace ViT-Base architecture, adapted to native PyTorch SDPA.

Architecture/state names follow Minchul Kim's CVLface (MIT license):
https://github.com/mk-minchul/CVLface/tree/master/cvlface/research/recognition/code/run_v1/models/vit
See third_party/CVLface-LICENSE and third_party/README.md.
No remote Python code is loaded.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(self, image_size=112, patch_size=8, dim=512):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Conv2d(3, dim, patch_size, stride=patch_size)

    def forward(self, x):
        if x.shape[-2:] != (self.image_size,self.image_size) or x.shape[1] != 3:
            raise ValueError(f'FaceViT requires RGB {self.image_size}x{self.image_size} input')
        return self.proj(x).flatten(2).transpose(1,2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim,dim*3,bias=False)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        batch,tokens,dim = x.shape
        q,k,v = self.qkv(x).reshape(batch,tokens,3,self.num_heads,dim//self.num_heads).permute(2,0,3,1,4).unbind(0)
        x = F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.proj(x.transpose(1,2).reshape(batch,tokens,dim))


class Mlp(nn.Module):
    def __init__(self,dim,ratio):
        super().__init__()
        self.fc1 = nn.Linear(dim,int(dim*ratio))
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(int(dim*ratio),dim)

    def forward(self,x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self,dim,heads,ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim,heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim,ratio)

    def forward(self,x):
        x = x+self.attn(self.norm1(x))
        return x+self.mlp(self.norm2(x))


class FaceViT(nn.Module):
    def __init__(self,embedding_size=512,dim=512,depth=24,heads=16,ratio=3.,image_size=112,patch_size=8,metric_projection=False):
        super().__init__()
        patches = (image_size//patch_size)**2
        self.patch_embed = PatchEmbed(image_size,patch_size,dim)
        self.pos_embed = nn.Parameter(torch.zeros(1,patches,dim))
        self.blocks = nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.feature = nn.Sequential(
            nn.Linear(dim*patches,dim,bias=False),nn.BatchNorm1d(dim,eps=2e-5),
            nn.Linear(dim,embedding_size,bias=False),nn.BatchNorm1d(embedding_size,eps=2e-5))
        self.frozen_prefix = 0
        self.metric_projection=nn.Linear(embedding_size,embedding_size,bias=False) if metric_projection else nn.Identity()
        if metric_projection:
            nn.init.eye_(self.metric_projection.weight)

    def set_trainable_blocks(self,count):
        if not 0 <= count <= len(self.blocks):
            raise ValueError('Invalid number of trainable Transformer blocks')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.frozen_prefix = len(self.blocks)-count
        for block in self.blocks[self.frozen_prefix:]:
            for parameter in block.parameters():parameter.requires_grad_(True)
        for module in [self.norm,self.feature]:
            for parameter in module.parameters():parameter.requires_grad_(True)
        # Freeze BN parameters/statistics, preserving pretrained feature scaling.
        for module in self.modules():
            if isinstance(module,nn.BatchNorm1d):
                module.eval()
                for parameter in module.parameters():parameter.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module,nn.BatchNorm1d):module.eval()
        return self

    def forward_tokens(self,x):
        x = self.patch_embed(x)+self.pos_embed
        for block in self.blocks:x = block(x)
        return x

    def forward(self,x):
        if self.training and self.frozen_prefix:
            with torch.no_grad():
                x = self.patch_embed(x)+self.pos_embed
                for block in self.blocks[:self.frozen_prefix]:x = block(x)
            for block in self.blocks[self.frozen_prefix:]:x = block(x)
        else:
            x = self.forward_tokens(x)
        # Match CVLFace's float32 final LayerNorm behavior under mixed precision.
        return self.metric_projection(self.feature(self.norm(x.float()).flatten(1)))
