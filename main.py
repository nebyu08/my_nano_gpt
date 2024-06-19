import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size:int=1024
    vocab_size:int =50257
    n_layer:int =12
    n_head:int =12
    n_embd:int =768


class CasualSelfAttention(nn.Module):
    def __init_(self,config):
        super().__init__()
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        
        self.n_emb=config.n_emd
        self.n_head=config.n_head

        #creating a register buffer
        self.register_buffer("bias",torch.tril(torch.ones(1,1,config.block_size,config.block_size)))

    def forward(self,x):
        #the input is 3D has batch size,number of sequences and dimensionality 

        B,T,C=x.size()  #batch size,sequence lenght and dimensionality

        #the input is probably batched text  so its 2D
        qkv=self.c_attn(x)   

        #lets split it into 3 
        q,k,v=qkv.split(self.n_embd,dim=1)
        #now lets reshape out querys,keys and values for multi-head purposes
        q=q.view(B,T,self.n_head,C//self.n_head).reshape(1,2)
        k=k.view(B,T,self.n_head,C//self.n_head).reshape(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).reshape(1,2)

        attn=(q @ k.transpose(-1,-2))*(1/(math.sqrt(k.size(-1))))   #the shape is B,nh,T,T
        attn.masked_fill(self.bias[:,:,T:T]==0,float(-"inf"))
        attn=F.softmax(attn,dim=-1)
        y=attn@v   #B,nh,T,T * B,nh,T,hs  ==> B,nh,T,hs
        y.transpose(1,2).contiguous().view(B,T,C)

        #the output projection becomes
        y=self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self,config):
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)

    def forward(self,x):
        x=self.h1(x)
        x=self.gelu(x)
        x=self.h2(x)
        return x

class Block(nn.Module):
    """this inside the transformer architecture
    that is  little bit differnt than the one on the original paper.
    """
    def __init__(self,config):
        self.ln_1=nn.LayerNorm(config.n_embd) 
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.layer_n1(x))
        x=x+self.MLP(self.layer_n2(x))

class GPT:
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.tranformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),  #this is the weight token embedder
            wpe=nn.Embedding(config.block_size,config.n_embd),               #wight position embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embed)   #this is the layer normalization after layers
        )
        )
        self.lm_head=torch.Linear(config.n_embed,config.vocab_size,bias=False)   
        
    

