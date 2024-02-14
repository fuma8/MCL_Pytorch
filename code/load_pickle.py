import torch
tmp = torch.load("/home/kimishima/MCL_pytorch/code/checkpoint/MCL_resisc45.pickle")
print(tmp["acc"])
print(tmp["auc"])