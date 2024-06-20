import math
import tiktoken
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
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 ,"problem with the way shape of the n_emb and n_head couldn't be divided"
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        
        self.n_emb=config.n_embd
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
        super().__init__()
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
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd) 
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.MLP(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),  #this is the weight token embedder
            wpe=nn.Embedding(config.block_size,config.n_embd),               #wight position embedding
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd)   #this is the layer normalization after layers
        )
        )
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)   

    @classmethod
    def from_pretrained(cls,model_type):
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
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')]

        #lets load the model form hugging face
        from transformers import GPT2LMHeadModel
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
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
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                #those that are not conv1D
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self,idx):
        #the idx is 2D batch by tokens(max block size)
        B,T=idx.size()
        assert T<=self.config.block_size,f"the given sequence length {T} must decrease cause the block size is {self.config.block_size}"
        
        #pass it through the transformers

        #embedding part
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)  #shape is T
        pos_emb=self.transformer.wpe(pos)   #shape is t * n_embd.......this is because this shape is equal for every row of input 
        tok_emb=self.transformer.wte(idx)   #shape is b * t * n_embd
        
        #get the tok and post embd
        x=tok_emb+pos_emb

        #the casual self attention
        for block in self.transformer.h:
            x=block(x)
        
        x=self.transformer.ln_f(x)

        logits=self.lm_head(x)  #(B,T,vocab_size)
        return logits
    


#lets load the GPT2  model
model=GPT.from_pretrained('gpt2')
model.to("cuda")
print("it worked yayayyayay")

#lets generate some texts here
num_return_sequence=5
generated_next_tokens=30

encoder=tiktoken.get_encoding('gpt2')
text="hello i am a language model and "

tokens=encoder.encode(text)
tokens=torch.tensor(tokens,dtype=torch.long).unsqueeze(dim=0).repeat(num_return_sequence,1)   #the shape is (B * T)

#move it to the GPU
x=tokens.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.shape[1]<generated_next_tokens:
    with torch.no_grad():
            logits=model(x)
            logits=logits[:,-1:]   # (B * 1 * vocab size)
            
            #convert into probability distr
            probs=F.softmax(logits,dim=-1)

            logits_values,logits_indices=torch.topk(probs,50,dim=-1)  #extract the top 50 (B* 50)
            most_prob=torch.multinomial(logits_values,1,dim=-1)  # we got the highest indice values shape is [B *1]

            idx=torch.gather(most_prob,-1,most_prob) #[B*1]
            x=torch.cat((x,idx),1)

#print the values here
for i in range(num_return_sequence):
    tokens=x[i:generated_next_tokens].tolist()
    decode=encoder.decode(tokens)
    print(decode)