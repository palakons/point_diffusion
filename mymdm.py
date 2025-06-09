import torch as th
import torch.nn as nn
import numpy as np
import math

class MDM(nn.Module):
    def __init__(self, 
                 max_pc_len=128, 
                 in_channels=3, 
                 out_channels=3,
                 num_heads=6, 
                 ff_size=2048,
                 model_channels=512,
                 num_layers=3,
                 condition_dim=2,
                 dropout=0.1,
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.in_channels == self.out_channels
        self.model_channels = model_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.condition_dim = condition_dim
        self.dropout = dropout

        print(f"MDM: in_channels {self.in_channels}, out_channels {self.out_channels}, "
              f"model_channels {self.model_channels}, num_heads {self.num_heads}, "
              f"ff_size {self.ff_size}, num_layers {self.num_layers}, "
              f"condition_dim {self.condition_dim}, dropout {self.dropout}, ")
        
        # Input Embedding (Processing the input e.g., project into the latent before pass through the trasnformer layer)
        self.input_process = nn.Linear(self.in_channels, self.model_channels)
        # Condition Embedding (Processing the conditions) 
        self.cond_proj_layer = nn.Linear(self.condition_dim, self.model_channels)
        self.cond_time_combined = nn.Conv1d(in_channels=max_pc_len, 
                                            out_channels=1, 
                                            kernel_size=1)
        # Positional Embedding
        self.pos_encoder = PositionalEncoding(self.model_channels, self.dropout)
        # Timestep Embedding
        self.time_embed = TimestepEmbedder(latent_dim=self.model_channels, 
                                           sequence_pos_encoder=self.pos_encoder
                                           )
        
        # Networks
        # print(self.model_channels, self.num_heads, self.ff_size, self.dropout)
        mdm_layer = nn.TransformerEncoderLayer(d_model=self.model_channels,
                                              nhead=self.num_heads,
                                              dim_feedforward=self.ff_size,
                                              dropout=self.dropout,
                                              batch_first=True,
                                            )
        self.mdm = nn.TransformerEncoder(mdm_layer, num_layers=self.num_layers)
        # Output layer
        self.output_process = nn.Linear(self.model_channels, self.out_channels)
        
    def forward(self, x, timesteps):


        """
        x_org: [batch_size,  nframes , nfeats], denoted x_t in the paper (input)
        x: [batch_size, nfeats, nframes], denoted x_t in the paper (input)
        timesteps: [batch_size, nframes] (input)
        """
        # Condition Embedding
        # cond_emb = kwargs['cond']
        cond_emb = x[:, 0, self.in_channels :]   # [bs, condition_dim]
        # cond_emb *=0 #fortesting
        # print("cond_emb : ", cond_emb.shape) #torch.Size([1, 2])
        cond_emb_proj = self.cond_proj_layer(cond_emb)   # [bs,  model_channels]
        # Time Embedding
        # print("cond_emb_proj : ", cond_emb_proj.shape) #torch.Size([1, 128])
        t_emb = self.time_embed(timesteps)
        # print("After emb time : ", t_emb.shape)
        # emb = emb.unsqueeze(dim=1)  #NOTE: [bs, d] -> [bs, 1, d] since we need to add the #n frames dim for timesteps
        # print("Expand T-dim of emb time : ", emb.shape)
        
        emb = cond_emb_proj.unsqueeze(1) + t_emb  # [bs, 1, d] ????
        # print("timecomb emb : ", emb.shape) #torch.Size([1, 1, 512])
        # print("x : ", x.shape) #torch.Size([1, 128, 67])
        x=x[:,:,:self.in_channels]
        # print("2x : ", x.shape)#torch.Size([1, 128, 3])
        x = self.input_process(x)
        # print("x : ", x.shape)
        # print("emb : ", emb.shape)
        # print("sparse_emb : ", sparse_emb.shape)
        
        xseq = th.cat((emb, x), dim=1) #NOTE: [bs, nframes, d] -> [bs, nframes+1, d]
        # print("xseq : ", xseq.shape)
        xseq = self.pos_encoder(xseq)
        # print("xseq : ", xseq.shape)
        output = self.mdm(xseq)[:, 1:, ...]
        # print("output : ", output.shape)
        
        output = self.output_process(output)
        # print("output : ", output.shape) #torch.Size([1, 128, 3])
        return output
        # return {"output":output}
 
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        # print("timesteps : ", timesteps)
        self.sequence_pos_encoder.pe = self.sequence_pos_encoder.pe.to(timesteps.device)
        # print("pe shape : ", self.sequence_pos_encoder.pe.shape)
        # print("timesteps shape : ", self.sequence_pos_encoder.pe[timesteps].shape)
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps])
 
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = th.zeros(max_len, d_model)
#         position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
#         div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = th.sin(position * div_term)
#         pe[:, 1::2] = th.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)

#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # not used in the final model
#         # x = x + self.pe[:x.shape[1]].permute(1, 0, 2)   # since pe is max_len, 1, d_model, we need to permute it to 1, max_len, d_model
#         x = x + self.pe[:x.shape[0], :]
#         return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = th.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = th.sin(position * div_term)
        pe[:, 0, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

        # position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        # div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # pe = th.zeros(max_len, d_model)
        # pe[:, 0::2] = th.sin(position * div_term)
        # pe[:, 1::2] = th.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        
        # self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x.permute(1, 0, 2)   # B x T x D -> T x B x D
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)
        return self.dropout(x)
    

        
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # print(timesteps.shape)
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    # print("emb out : ", embedding.shape)
    return embedding