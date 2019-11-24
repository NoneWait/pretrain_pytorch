"""
Date: 2019/11/19 14:39
"""

import math

import torch
from torch import nn


"""
transformer 代码库

一个transformer 主要是可以分为一下几个部分

- embedding layers
- self-attention layers
- ffn layers 
"""


def gelu(x):
    r"""
    高斯误差线性激活函数（Gaussian Error Linear Unit）
    https://arxiv.org/abs/1606.08415
    1. 激活函数负责非线性
    2. gaussian 负责提供随机正则

    理论上就是对输入乘以一个0,1组成的mask，而该mask的生成是依据输入
    而随机概率生成的。

    假设输入是X，mask为m，则m服从伯努利分布(离散的0-1分布)，而X则
    是服从标准正太分布（由于神经元的输入趋向于正太分布），这种设定
    使得输入x减小的时候，输入会有一个更高的概率被dropout掉。

    $$
    gelu(x)=xP(X<=x)=\Phi(x)
    $$

    erf: 误差函数（高斯误差函数）
    :param x:
    :return:
    """
    return x*0.5*(1.0+torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x*torch.sigmoid(x)


def gelu_new(x):
    return 0.5*x*(1+torch.tanh(math.sqrt(2 / math.pi)*(x+0.044715*torch.pow(x, 3))))


ACT2FN = {"gelu": gelu, 'relu': torch.nn.functional.relu,
          'swish': swish, 'gelu_new': gelu_new}


class TransformerEmbeddings(nn.Module):
    r"""
    """
    def __init__(self, config):
        super(TransformerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()  # [bs, seq_len]

        seq_length = input_shape[1]
        device = input_ids.devices

        if position_ids is None:
            # 采用自动学习
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)    # [bs, seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        return embeddings


class SelfAttention(nn.Module):
    r"""
    需要考虑的是：
    1. 怎么实现多头呢？
        每个head会有一个转换矩阵$W \in  R^{d \times h_s}$, 其中$h_s$是每个head
        产生的size，而$d$则是原来的hidden_size，那么这样我们其实可以直接用一个
        大的矩阵$W \in R^{d \times h_s*head_num}$来取代循环实现。
    2. 计算注意力得分

    """
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # [bs, head_nums, seq_len, h]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # [bs, head_nums, seq_len, h]
        context_layer = torch.matmul(attention_probs, value_layer)

        # [bs, seq_len, head_nums, h]
        # 这里的contiguous是将tensor变成在内存中连续分布的形式（这里是由于permute或者transpose
        # 会变得不连续，而view操作是在连续内存变量上进行的）
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [bs, seq_len, head_nums*h]
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class SelfOutput(nn.Module):
    def __init__(self, config):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)   # 这为什么要再加个W？
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    """
    Feed Forward 部分
    """
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = config.hidden_act



class TEncoder(nn.Module):
    def __init__(self):
        super(TEncoder, self).__init__()


