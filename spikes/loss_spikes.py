# %%
from torch import nn
import torch

from torch.nn import functional as ff

# %%
loss = nn.CrossEntropyLoss()
input = torch.randn((3, 5), requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input)
print(target)
print(output)

# %%

batchN = 4

lossFn = nn.NLLLoss(reduction="none")
feature = torch.rand((batchN, 3, 5, 5), dtype=torch.float32)
dist = ff.softmax(feature, dim=1)

target = torch.ones((batchN, 5, 5), dtype=torch.long)

loss = lossFn(dist, target)
loss

# %%

batchN = 4

lossFn = nn.NLLLoss(reduction="none")

feature = torch.zeros((batchN, 5), dtype=torch.float32)
feature[:, 0] = torch.ones(batchN) * 4

print(feature)

dist = ff.softmax(feature, dim=1)

print(dist)

target = torch.zeros((batchN), dtype=torch.long)

loss = lossFn(dist, target)
loss

# %%
