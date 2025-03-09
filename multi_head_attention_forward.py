    def multi_head_attention_forward(self,
                                     q_emb: Tensor,
                                     k_emb: Tensor,
                                     v_emb: Tensor,
                                     num_head: int,
                                     dropout: float,
                                     key_padding_mask: Optional[Tensor] = None,
                                     attention_mask: Optional[Tensor] = None):
        # batch first:
        # source_emb (num_batch, num_source_token, token_dim)
        # target_emb (num_batch, num_target_token, token_dim)
        # sequence_length = num_token = num_source_token = num_target_token

        # q_emb
        # (target sequence length, embedding dimension)
        # (target sequence length, batch size, embedding dimension)

        # k_emb
        # (source sequence length, embedding dimension)
        # (source sequence length, batch size, embedding dimension)

        # v_emb
        # (source sequence length, embedding dimension)
        # (source sequence length, batch size, embedding dimension)

        # key_padding_mask
        # (source sequence length)
        # (batch size, source sequence length)

        # attention_mask
        # (target sequence length, source sequence length)
        # (the batch size * num_heads, target sequence length, source sequence length)

        # out_emb
        # (target sequence length, embedding dimension)
        # (target sequence length, batch size, embedding dimension)

        # attention
        # (target sequence length, source sequence length)
        # (batch size, target sequence length, source sequence length)

        num_target_token, batch_size, target_emb_dim = q_emb.shape
        num_source_token, batch_size, source_emb_dim = k_emb.shape
        num_source_token, batch_size, source_emb_dim = v_emb.shape

        head_dim = target_emb_dim // num_head
        correct_3d_size = (batch_size * num_head, num_target_token, num_source_token)

        # (num_target_token, batch_size, target_emb_dim)
        # -> (num_target_token, batch_size, num_head, head_dim)
        # -> (num_target_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_target_token, head_dim)
        num_target_token, batch_size, target_emb_dim = q_emb.shape
        q_emb = self.q_linear(q_emb)
        q_emb = q_emb.view(num_target_token, batch_size, num_head, head_dim)
        q_emb = q_emb.view(num_target_token, batch_size * num_head, head_dim)
        q_emb = q_emb.transpose(0, 1)

        # (num_source_token, batch_size, source_emb_dim)
        # -> (num_source_token, batch_size, num_head, head_dim)
        # -> (num_source_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        # -> (batch_size * num_head, head_dim, num_source_token)
        num_source_token, batch_size, source_emb_dim = k_emb.shape
        k_emb = self.k_linear(k_emb)
        k_emb = k_emb.view(num_source_token, batch_size, num_head, head_dim)
        k_emb = k_emb.view(num_source_token, batch_size * num_head, head_dim)
        k_emb = k_emb.transpose(0, 1)
        k_emb_transpose = k_emb.transpose(-2, -1)

        # (num_source_token, batch_size, source_emb_dim)
        # -> (num_source_token, batch_size, num_head, head_dim)
        # -> (num_source_token, batch_size * num_head, head_dim)
        # -> (batch_size * num_head, num_source_token, head_dim)
        num_source_token, batch_size, source_emb_dim = v_emb.shape
        v_emb = self.v_linear(v_emb)
        v_emb = v_emb.view(num_source_token, batch_size, num_head, head_dim)
        v_emb = v_emb.view(num_source_token, batch_size * num_head, head_dim)
        v_emb = v_emb.transpose(0, 1)

        # key_padding_mask
        # (batch size, num_source_token)
        # -> (batch size, 1, 1, num_source_token)
        # -> (batch size, num_head, 1, num_source_token)
        # -> (batch size * num_head, 1, num_source_token)
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, num_source_token)
        key_padding_mask = key_padding_mask.expand(-1, num_head, -1, -1)
        key_padding_mask = key_padding_mask.reshape(batch_size * num_head, 1, num_source_token)

        # attention_mask
        # (batch_size * num_head, num_target_token, num_source_token)
        if attention_mask is None:
            attention_mask = key_padding_mask
        else:
            attention_mask = attention_mask + key_padding_mask

        head_dim = q_emb.shape[2]
        q_emb = q_emb * math.sqrt(1.0 / float(head_dim))

        if attention_mask is not None:
            attention = torch.baddbmm(attention_mask, q_emb, k_emb_transpose)
        else:
            attention = torch.bmm(q_emb, k_emb.transpose(-2, -1))

        attention = torch.nn.functional.softmax(attention, dim=-1)
        attention = torch.nn.functional.dropout(input=attention, p=dropout, training=self.training)

        # (batch_size * num_head, num_target_token, head_dim)
        # -> (num_target_token, batch_size * num_head, head_dim)
        # -> (num_target_token, batch_size, num_head, head_dim)
        # -> (num_target_token, batch_size, target_emb_dim)
        # -> (num_target_token * batch_size, target_emb_dim)
        # -> (num_target_token, batch_size, target_emb_dim)
        out_emb = torch.bmm(attention, v_emb)
        out_emb = out_emb.transpose(dim0=0, dim1=1).contiguous()
        out_emb = out_emb.view(num_target_token, batch_size, num_head, head_dim)
        out_emb = out_emb.view(num_target_token, batch_size, target_emb_dim)
        out_emb = self.out_linear(out_emb)
        out_emb = out_emb.view(num_target_token, batch_size, out_emb.size(1))

        # (batch_size * num_head, num_target_token, num_source_token)
        # -> (batch_size, num_head, num_target_token, num_source_token)
        attention = attention.view(batch_size, num_head, num_target_token, num_source_token)

        return out_emb, attention
