"""A small encoder retaining the proposed RoPE/Pre-LN/GeGLU building blocks."""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    vocab: int = 8192
    layers: int = 6
    hidden: int = 256
    heads: int = 4
    intermediate: int = 688
    window: int = 64
    global_every: int = 3
    dropout: float = .1
    rope_theta: float = 10000.
    chunk: int = 256

def rope(x, theta=10000.):
    # [batch, heads, sequence, dimension]; rotate adjacent pairs.
    inv = theta ** (-torch.arange(0,x.shape[-1],2,device=x.device,dtype=torch.float32)/x.shape[-1])
    angles = torch.arange(x.shape[-2],device=x.device,dtype=torch.float32)[:,None]*inv[None,:]
    c,s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even,odd = x[...,0::2],x[...,1::2]
    return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)

def attention(q,k,v,valid,window=None,chunk=256):
    """Exactly window tokens at interior positions, offsets [-w/2,w/2)."""
    n=q.shape[-2]; outputs=[]
    for start in range(0,n,chunk):
        end=min(n,start+chunk)
        lo=0 if window is None else max(0,start-window//2)
        hi=n if window is None else min(n,end+(window-1)//2)
        mask=valid[:,None,None,lo:hi].bool()
        if window is not None:
            qi=torch.arange(start,end,device=q.device)[:,None]
            ki=torch.arange(lo,hi,device=q.device)[None,:]
            mask=mask & (ki>=qi-window//2)[None,None] & (ki<qi+window//2)[None,None]
        outputs.append(F.scaled_dot_product_attention(q[:,:,start:end],k[:,:,lo:hi],v[:,:,lo:hi],attn_mask=mask))
    return torch.cat(outputs,dim=-2)

class Block(nn.Module):
    def __init__(self,c,index):
        super().__init__(); self.c=c
        self.window=None if (index+1)%c.global_every==0 else c.window
        self.ln1=nn.LayerNorm(c.hidden);self.qkv=nn.Linear(c.hidden,3*c.hidden,bias=False)
        self.out=nn.Linear(c.hidden,c.hidden,bias=False);self.ln2=nn.LayerNorm(c.hidden)
        self.up=nn.Linear(c.hidden,2*c.intermediate,bias=False)
        self.down=nn.Linear(c.intermediate,c.hidden,bias=False);self.drop=nn.Dropout(c.dropout)
    def forward(self,x,valid):
        b,n,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).view(b,n,3,self.c.heads,d//self.c.heads).permute(2,0,3,1,4).unbind(0)
        a=attention(rope(q,self.c.rope_theta),rope(k,self.c.rope_theta),v,valid,self.window,self.c.chunk)
        x=x+self.drop(self.out(a.transpose(1,2).reshape(b,n,d)))
        a,g=self.up(self.ln2(x)).chunk(2,dim=-1)
        return x+self.drop(self.down(a*F.gelu(g)))

class Encoder(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.c=c or Config();c=self.c
        self.embedding=nn.Embedding(c.vocab,c.hidden,padding_idx=0)
        self.blocks=nn.ModuleList([Block(c,i) for i in range(c.layers)])
        self.final_ln=nn.LayerNorm(c.hidden)
        self.mlm_dense=nn.Linear(c.hidden,c.hidden,bias=False);self.mlm_ln=nn.LayerNorm(c.hidden)
        self.mlm_bias=nn.Parameter(torch.zeros(c.vocab));self.apply(self.init_weights)
    @staticmethod
    def init_weights(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def forward(self,ids,valid):
        x=self.embedding(ids)
        for block in self.blocks:x=block(x,valid)
        return self.final_ln(x)
    def logits(self,hidden):
        return F.linear(self.mlm_ln(F.gelu(self.mlm_dense(hidden))),self.embedding.weight,self.mlm_bias)
    def mlm(self,ids,valid,labels):
        h=self(ids,valid);sel=labels.ne(-100)
        return F.cross_entropy(self.logits(h[sel]).float(),labels[sel])

def mean_pool(hidden,ids,valid):
    mask=valid.bool() & ids.gt(4)
    return (hidden*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)

def estimated_flops(c,batch,length):
    # Executed dense chunk matmuls, not an ideal sparse-attention lower bound.
    # Training proxy: three times forward; ignores softmax/norm/dropout/backward details.
    d,f,L=c.hidden,c.intermediate,c.layers
    attn=0
    for i in range(L):
        for start in range(0,length,c.chunk):
            end=min(length,start+c.chunk)
            lo=0 if (i+1)%c.global_every==0 else max(0,start-c.window//2)
            hi=length if (i+1)%c.global_every==0 else min(length,end+(c.window-1)//2)
            attn+=(end-start)*(hi-lo)
    forward=batch*(L*length*(8*d*d+6*d*f)+4*d*attn)
    # sparse MLM predictions, 15% selected, tied projection and dense transform
    forward+=batch*length*.15*(2*d*d+2*d*c.vocab)
    return 3*forward
