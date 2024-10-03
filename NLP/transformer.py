import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int, vocab_size):
        super().__init__()
        # d_model: dimension of model default 512
        self.d_model = d_model
        self.vocab_size = vocab_size
        # 构建embeding，vocab_size为词典大小，d_model为词向量维度
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self, x):
        # 首先经过初始化(Xavier初始化)的nn.Embedding.weight矩阵,所以我们需要乘以一个根号d_model把Embdeding调整到N(0,1)
        return self.embedding(x)* math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int,dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # 初始化一个矩阵，矩阵(max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)
        # 生成一个位置编码矩阵 (max_len, 1)
        position = torch.arange(0, max_len).unsqueeze(1).float()  
        # 分母(1,d_model/2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/ d_model ))
        # 生成位置编码矩阵
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch_size数据(1, max_len, d_model)
        self.pe = self.pe.unsqueeze(0)  
        # 它们不会在训练过程中更新，但会作为模型的一部分保存和加载 
        self.register_buffer('pe', self.pe)
        
    def forward(self, x):
        x=x + self.pe[:, :x.shape[1],:].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,  eps:float=1e-6):
        super().__init__()
        self.eps = eps
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads:int, d_model:int, dropout:float=0.1):
        super().__init__()
        self.d_model = d_model
        assert d_model % heads == 0,"d_model必须是heads的整数倍"
        self.d_k = d_model / heads
        self.h = heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        max_len = q.size(1)
        # perform linear operation and split into h heads
        k = self.W_K(k).view(batch_size, max_len, self.h, self.d_k)
        q = self.W_Q(q).view(batch_size, max_len, self.h, self.d_k)
        v = self.W_V(v).view(batch_size, max_len, self.h, self.d_k)
        # transpose to get dimensions bs * h * seq_len * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        output,scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_out(concat)
        
        return output
    @staticmethod
    def attention( q, k, v, d_k, mask=None, dropout=None):
        scores = (q @ k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        # (bs, h, seq_len, seq_len)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = scores@v
        return output,scores
    
class ResidualConnection(nn.Module):
    def __init__(self,  dropout:float=0.1):
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model:int, heads:int, d_ff:int, dropout:float=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(heads, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual =nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, mask):
        x = self.residual[0](x, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual[1](x, lambda x: self.feed_forward(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm=LayerNormalization() 
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)