import torch_geometric
import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.linear = torch_geometric.nn.Linear(in_channels=-1,
                                                out_channels=self.out_dim,
                                                weight_initializer='kaiming_uniform',
                                                bias=True,
                                                bias_initializer=None)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)

class GraphMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_dim, num_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        assert hidden_dim % num_head == 0
        self.head_dim = hidden_dim // num_head
        self.q_linear = Linear(hidden_dim)
        self.k_linear = Linear(hidden_dim)
        self.v_linear = Linear(hidden_dim)
        self.out_linear = Linear(hidden_dim)

    def forward(self, node_emb) -> torch.Tensor:
        # node emb shape(num_node, hidden_dim)
        num_node = node_emb.shape[0]

        q_emb = self.q_linear(node_emb)
        k_emb = self.k_linear(node_emb)
        v_emb = self.v_linear(node_emb)

        q_emb = q_emb.view(num_node, self.num_head, self.head_dim) # (num_node, num_head, head_dim)
        k_emb = k_emb.view(num_node, self.num_head, self.head_dim) # (num_node, num_head, head_dim)
        v_emb = v_emb.view(num_node, self.num_head, self.head_dim) # (num_node, num_head, head_dim)

        q_emb = q_emb.transpose(0, 1) # (num_head, num_node, head_dim)
        k_emb = k_emb.transpose(0, 1) # (num_head, num_node, head_dim)
        v_emb = v_emb.transpose(0, 1) # (num_head, num_node, head_dim)

        k_emb = k_emb.transpose(1, 2) # (num_head, head_dim, num_node)

        attention = torch.bmm(q_emb, k_emb) / math.sqrt(self.head_dim)  # (num_head, num_node, num_node)
        attention = torch.nn.functional.softmax(attention, dim=-1)

        # (num_head, num_node, num_node) @ (num_head, num_node, head_dim) = (num_head, num_node, head_dim)
        out_node_emb = torch.bmm(attention, v_emb)

        out_node_emb = out_node_emb.transpose(0, 1).contiguous() # (num_node, num_head, head_dim)
        out_node_emb = out_node_emb.view(out_node_emb, self.hidden_dim)  # (num_node, hidden_dim)

        return out_node_emb
