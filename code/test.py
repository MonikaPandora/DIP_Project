import torch


a = torch.tensor(list(range(4 * 3 * 8 * 8))).resize(4, 3, 8, 8)
b = torch.tensor(list(range(4 * 3 * 8 * 8))).resize(4, 3, 8, 8)

c = torch.stack((a, b), dim=1)
print(c.shape)
print(torch.split(c, dim=1, split_size_or_sections=1)[0].shape)

print(c.view(4, -1, 8, 8).shape)
