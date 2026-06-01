
import paddle
import torch


def naive_retention(q, k, v):
    orig_type = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    _, n_heads, seq_len, d_head = q.shape
    s = (1 - paddle.to_tensor(data=2.0, dtype=torch.float).
            pow(-5.0 - paddle.to_tensor(data=range(n_heads), dtype=torch.float))).log2()
    n = paddle.to_tensor(data=range(seq_len), dtype=torch.float)
    n = 2.0 ** ((n.unsqueeze(-1) - n) * s.view(-1, 1, 1)) * n.unsqueeze(-1).ge(n)
    s = torch.einsum('bhqd,bhkd,hqk->bhqk', q * d_head ** -0.5, k, n.to(q.dtype))
    o = torch.einsum('bhqk,bhkd->bhqd', s, v)
    return o.to(orig_type)
