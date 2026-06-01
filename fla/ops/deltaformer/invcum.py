# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
import paddle
import torch


def forward(u, w):
    return paddle.linalg.triangular_solve(
        x=w.float(),
        y=u.float(),
        upper=False,
        unitriangular=True,
    ).to(u.dtype)


def forward_inplace(u, w):
    u.copy_(forward(u, w))


def backward_x(do, w):
    return paddle.linalg.triangular_solve(
        x=w.tril(-1).mH.float(),
        y=do.float(),
        upper=True,
        unitriangular=True,
    ).to(do.dtype)


def backward(do, w, x):
    du = paddle.linalg.triangular_solve(
        x=w.tril(-1).mH.float(),
        y=do.float(),
        upper=True,
        unitriangular=True,
    ).to(do.dtype)
    perm_0 = list(range(x.ndim))
    perm_0[-1], perm_0[-2] = perm_0[-2], perm_0[-1]
    dw = torch.bmm(-du, x.transpose(perm=perm_0).conj())
    dw = dw.tril(-1)
    return du, dw
