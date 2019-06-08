# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%

# In mxnet

# %%

import mxnet as mx
from mxnet import autograd

print(f"found {len(mx.test_utils.list_gpus())} GPUs")

with autograd.record():
    x = mx.random.randn(2, 2)
    x.attach_grad()

    y = mx.random.randn(2, 2)
    y.attach_grad()

    z = ((x ** 2) * y).mean()
    print(f"{x} - {y}")

print(f"{x.shape} -> {y.shape} -> {z.shape}")

z.backward()

assert x.grad.__repr__() == mx.nd.multiply((x / 2), y).__repr__()
# assert y.grad.__repr__() == (x ** 2).__repr__()

# %%

# In torch

# %%

import torch as tt

print(f"found {tt.cuda.device_count()} GPUs")

x = tt.randn(2, 2, requires_grad=True)

y = x ** 2

z = y.mean()

print(f"{type(x)} -> {type(y)}")

print(f"{x.grad} -> {y.grad}")

z.backward()

print(f"{x.grad} -> {y.grad}")
print(f"{x / 2}")

# %%
