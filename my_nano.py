import math
import tiktoken
import torch
import torch.nn as nn
import time
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
        self.c_proj.nanogpt_interconnect=1
        
        self.n_embd=config.n_embd
        self.n_head=config.n_head

        #creating a register buffer
        self.register_buffer("bias",torch.tril(torch.ones(1,1,config.block_size,config.block_size)))

    def forward(self,x):
        #the input is 3D has batch size,number of sequences and dimensionality 

        B,T,C=x.size()  #batch size,sequence lenght and dimensionality

        #the input is probably batched text  so its 2D
        qkv=self.c_attn(x)   

        #lets split it into 3 
        q,k,v=qkv.split(self.n_embd,dim=2)


        #now lets reshape out querys,keys and values for multi-head purposes
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        # attn=(q @ k.transpose(-1,-2))*(1/(math.sqrt(k.size(-1))))   #the shape is B,nh,T,T
        # attn.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        # attn=F.softmax(attn,dim=-1)

        
        # y=attn@v   #B,nh,T,T * B,nh,T,hs  ==> B,nh,T,hs

        #lets use flash attention here
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y=y.transpose(1,2).contiguous().view(B,T,C)
        #the output projection becomes
        y=self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.nanogpt_interconnect=1

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
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
        x=x+self.mlp(self.ln_2(x))
        return x

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

        #weight sharing scheme between the wte and lm_h of the model..
        self.transformer.wte.weight=self.lm_head.weight #we are making the wte point to the lm head we saved alot of paramters by doing this

        self.apply(self._weight_init)
    
    def _weight_init(self,model):
        std=0.2

        if isinstance(model,nn.Linear):
            if hasattr(model,"nanogpt_interconnect"):
                std*=(2*self.config.n_layer)**-0.5

            torch.nn.init.normal_(model.weight,mean=0,std=std)
            if model.bias is not None:
                torch.nn.init.zeros_(model.bias)

        if isinstance(model,nn.Embedding):
            torch.nn.init.normal_(model.weight,mean=0,std=0.02)



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

    def configure_optimizers(self,weight_deacy,learning_rate,device_dtype):
        params_dict={pn:p for pn,p in self.named_parameters()}  #extracting params in general
        params_dict={pn:p  for pn,p in params_dict.items() if p.requires_grad()}

        #params that are 2d are decay params and 1d are non decay params
        decay_params=[p for pn,p in params_dict.items() if p.dim>=2]
        non_decay_params=[p for pn,p in params_dict.items() if p.dim<2]

        optim_groups=[
            {"params":decay_params   , "weight_decay":weight_decay}, #this is for the params that have weights to be decayed
            {"params":non_decay_params  ,"weight_deacy":0.0}  #they dont have weights that needs to be decayed
        ]

        num_decay_params=sum(p.numel() for p in decay_params)
        num_non_decay_params=sum(p.numel() for p in non_decay_params) 

        #lets count the number of decay params and there parameters
        print(f"the number of parameters are {len(decay_params)} and the parameters are {num_decay_params}")
        print(f"the numbe of parameters are {len(non_decay_params)} amd the parameters are {num_non_decay_params}")
        
        optimizer=torch.AdamW(optim_groups,)

    def forward(self,idx,targets=None):
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
        loss=None
        
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.shape[-1]),targets.view(-1))
        
        return logits,loss


class DataLoaderLite:
    def __init__(self,B,T,data_path):
        self.B=B
        self.T=T
        with open(data_path) as f:
            data=f.read()
        
        encoder=tiktoken.get_encoding("gpt2")
        
        self.tokens=torch.tensor(encoder.encode(data))

        print(f"the number of tokens is {len(self.tokens)}")
        print(f"we expect the following amount of batches:{len(self.tokens)/(self.B*self.T)}")

        self.current_batch=0
    
    def next_batch(self):
       B,T=self.B,self.T
       buf=self.tokens[self.current_batch:self.current_batch+B*T+1]
       x=buf[:-1].view(B,T)
       y=buf[1:].view(B,T)

       self.current_batch+=B*T

       if self.current_batch+(B*T+1)>len(self.tokens):
            self.current_batch=0

       return x,y

        
#model.to("cuda")
print("it worked yayayyayay")

encoder=tiktoken.get_encoding('gpt2')

#lets generate some texts here
# num_return_sequence=5
# generated_next_tokens=30

# encoder=tiktoken.get_encoding('gpt2')
# text="hello i am a language model and "

# tokens=encoder.encode(text)
# tokens=torch.tensor(tokens,dtype=torch.long).unsqueeze(dim=0).repeat(num_return_sequence,1)   #the shape is (B * T)

# #move it to the GPU
# #x=tokens.to("cuda")
# x=tokens   #...this is for CPU


# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# model.eval()

# while x.shape[1]<generated_next_tokens:
#     with torch.no_grad():
#             logits=model(x)
#             logits=logits[:,-1,:]   # (B * 1 * vocab size)
            
#             #convert into probability distr
#             probs=F.softmax(logits,dim=-1)

#             logits_values,logits_indices=torch.topk(probs,50,dim=-1)  #extract the top 50 (B* 50)
#             most_prob=torch.multinomial(logits_values,1)  # we got the highest indice values shape is [B *1]

#             idx=torch.gather(logits_indices,-1,most_prob) #[B*1]
#             x=torch.cat((x,idx),1)

# #print the values here
# for i in range(num_return_sequence):
#     tokens=x[i,:generated_next_tokens].tolist()
#     decode=encoder.decode(tokens)
#     print(f"> {decode} ")


# with open('C:/Users/nebiy/Documents/Dataset/input.txt') as t:
#     data=t.read()

# #lets make our own inputs and the next charachter output
# B,T=4,32

# tokens=torch.tensor(encoder.encode(data))

# buff=tokens[:B*T+1]

# x=buff[:-1].view(B,T)
# y=buff[1:].view(B,T)

# logits,loss=model(x,y)

#print(f"loss: {loss}")

#lets optimize the model


#lets load the GPT2  model
# model=GPT(GPTConfig())

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(device)

# train_dataloader=DataLoaderLite(B=16,T=1024)

# print(f"the device is: {device}")

# optimizer=torch.optim.AdamW(model.parameters())

# for i in range(50):
#     t0=time.time()  #this is the initial time
#     optimizer.zero_grad()
#     x,y=train_dataloader.next_batch()
#     x,y=x.to(device),y.to(device)

#     logits,loss=model(x,y)
#     loss.backward()
#     t1=time.time()
#     optimizer.step()

#     torch.cuda.synchronize()
#     tdif=(t1-t0)*1000
#     print(F"the iteration:{i},the loss: {loss}: time:{tdif} ms")
     