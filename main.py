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
        assert config.n_emb//config.n_head==0,"problem with the way shape of the n_emb and n_head couldn't be divided"
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
        
    @classmethod
    def from_pretrained(model_type):
        """this loads model weights from pretrained hugging face 

        Args:
            model_type (string): tell us about the type of model to load from hugging face
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        #lets get the hyperparameters for the specified model
        model_config= {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        #since the model type is gpt2 the following hyperparameters
        model_config['vocab_size']=50257
        model_config['block_size']=1024
        
        #lets make it regular config
        config=GPTConfig(**model_config)
        #define the model
        model=GPT(config)

        #now lets clear some parameters that can't be loaded 
        sd=model.state_dict()
        #keys of the model
        sd_keys=sd.keys()
    
        #lets remove keys we dont want
        sd_keys=[k for k in sd_keys if not k.endwith('.attn.bias')]

        #lets load the model form hugging face
        from transformers import GPT2LMHeadModel
        model_hf=GPT2LMHeadModel.from_pretrained(config)
        sd_hf=model_hf.state_dict()
        sd_keys_hf=sd_hf.keys()

        #the keys we want are
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        
        #lets specify values that are transposed
        transpose=['attn.c_proj.weight','attn.c_attn.weight','mlp.c_fc.weight','mlp.c_proj.weight']
    
        assert len(sd_keys_hf)==len(sd_keys),f'the length of the model({len(sd_keys)}) and the blank model({sd_keys_hf}) are difference'
        #lets copy the parameters
        for k in sd_keys_hf:
            if any(k.endswith(i) for i in transpose):
                #am not still sure why but we need to check for the inverse
                assert k.shape[::-1]==k.shape
                with torch.no_grad:
                    sd_keys[k].copy_(sd_keys_hf[k].t())
            else:
                #those that are not conv1D
                assert sd_keys[k].shape==sd_keys_hf[k].shape
                with torch.no_grad:
                    sd_keys[k].copy_(sd_keys_hf[k])

        return model
        
    

#lets load the GPT2  model
model=GPT.from_pretrained('gpt2')
print("it worked yayayyayay")