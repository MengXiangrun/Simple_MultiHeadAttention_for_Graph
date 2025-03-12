import torch_geometric
import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_dim=-1, out_dim=-1, bias=True, weight_initializer='kaiming_uniform', bias_initializer=None):
        super().__init__()
        if out_dim <= 0:
            assert False, f'out_dim <= 0'
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=in_dim,
                                                out_channels=out_dim,
                                                weight_initializer=weight_initializer,
                                                bias=bias,
                                                bias_initializer=bias_initializer)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 emb_dim,
                 num_head,
                 dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_head = num_head
        self.dropout = dropout
        self.head_dim = self.emb_dim // self.num_head
        assert self.head_dim * self.num_head == self.emb_dim

        self.q_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.k_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.v_linear = Linear(self.emb_dim, self.emb_dim, bias=False)
        self.out_linear = Linear(self.emb_dim, self.emb_dim, bias=True)

    def forward(self,
                source_emb,
                target_emb,
                attention_mask=None):
        # batch first:
        # source_emb (batch_size, num_source_token, token_dim)
        # target_emb (batch_size, num_target_token, token_dim)
        # sequence_length = num_token = num_source_token = num_target_token

        # key_padding_mask
        # (source sequence length)
        # (batch size, source sequence length)

        num_head = self.num_head
        head_dim = self.head_dim

        # (batch_size, num_target_token, token_dim)
        # -> (num_target_token, batch_size, target_emb_dim)
        # -> (num_target_token, batch_size, num_head, head_dim)
        # -> (num_target_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_target_token, head_dim)
        q_emb = target_emb.transpose(1, 0)
        num_target_token, batch_size, target_emb_dim = q_emb.shape
        q_emb = self.q_linear(q_emb)
        q_emb = q_emb.view(num_target_token, batch_size, num_head, head_dim)
        q_emb = q_emb.view(num_target_token, batch_size * num_head, head_dim)
        q_emb = q_emb.transpose(0, 1)

        # (num_batch, num_source_token, token_dim)
        # -> (num_source_token, batch_size, source_emb_dim)
        # -> (num_source_token, batch_size, num_head, head_dim)
        # -> (num_source_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        # -> (batch_size * num_head, head_dim, num_source_token)
        k_emb = source_emb.transpose(1, 0)
        num_source_token, batch_size, source_emb_dim = k_emb.shape
        k_emb = self.k_linear(k_emb)
        k_emb = k_emb.view(num_source_token, batch_size, num_head, head_dim)
        k_emb = k_emb.view(num_source_token, batch_size * num_head, head_dim)
        k_emb = k_emb.transpose(0, 1)
        k_emb_transpose = k_emb.transpose(-2, -1)

        # (num_batch, num_source_token, token_dim)
        # -> (num_source_token, batch_size, source_emb_dim)
        # -> (num_source_token, batch_size, num_head, head_dim)
        # -> (num_source_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        v_emb = source_emb.transpose(1, 0)
        num_source_token, batch_size, source_emb_dim = v_emb.shape
        v_emb = self.v_linear(v_emb)
        v_emb = v_emb.view(num_source_token, batch_size, num_head, head_dim)
        v_emb = v_emb.view(num_source_token, batch_size * num_head, head_dim)
        v_emb = v_emb.transpose(0, 1)

        # attention_mask
        # (batch_size * num_head, num_target_token, num_source_token)
        if attention_mask is not None:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.baddbmm(attention_mask, q_emb, k_emb_transpose)
        else:
            q_emb = q_emb / math.sqrt(float(head_dim))  # scaling
            attention = torch.bmm(q_emb, k_emb.transpose(-2, -1))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = torch.nn.functional.dropout(input=attention, p=self.dropout, training=self.training)

        # (batch_size * num_head, num_target_token, head_dim)
        # -> (num_target_token, batch_size * num_head, head_dim)
        # -> (num_target_token, batch_size, num_head, head_dim)
        # -> (num_target_token, batch_size, target_emb_dim)
        # -> (num_target_token, batch_size, out_emb_dim)
        out_emb = torch.bmm(attention, v_emb)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, num_head, head_dim)
        out_emb = out_emb.view(num_target_token, batch_size, target_emb_dim)
        out_emb = self.out_linear(out_emb)

        # (batch_size * num_head, num_target_token, num_source_token)
        # -> (batch_size, num_head, num_target_token, num_source_token)
        attention = attention.view(batch_size, num_head, num_target_token, num_source_token)

        return out_emb, attention

    def get_mask(self,
                 batch_size,
                 num_head,
                 num_source_token,
                 num_target_token):
        # Padding Mask (batch_size, num_source_token)
        padding_mask = torch.zeros(batch_size, num_source_token)  # 示例数据

        # Causal Mask (num_target_token, num_target_token)
        causal_mask = torch.tril(torch.ones(num_target_token, num_target_token))  # 示例数据

        # 扩展 Padding Mask 到 (batch_size, 1, 1, num_source_token)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

        # 扩展 Causal Mask 到 (1, 1, num_target_token, num_target_token)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 合并掩码 (batch_size, 1, num_target_token, num_source_token)
        attention_mask = padding_mask + causal_mask

        # 扩展到 (batch_size * num_head, num_target_token, num_source_token)
        attention_mask = attention_mask.expand(-1, num_head, -1, -1)
        attention_mask = attention_mask.reshape(batch_size * num_head, num_target_token, num_source_token)

        return attention_mask
