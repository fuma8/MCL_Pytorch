import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def PCA(data, d, device=device):
    N, H, W = data.shape
    data_mean = torch.zeros((1, H, W)).to(device)
    data -= data_mean
    data = torch.reshape(data, (N, H*W))
    cov = torch.mm(data.T, data)
    U, _, _ = torch.linalg.svd(cov)
    projection = U[:, :d]
    return projection

class VectorSensing(nn.Module):
    def __init__(self, nb_measurement, W, regularizer=None, constraint=None, device = device):
        super(VectorSensing, self).__init__()
        self.d = nb_measurement
        self.constraint = constraint
        self.regularizer = regularizer
        self.W = W
        self.E1 = nn.Parameter(self.W[:,0,:]).to(device)
        self.E2 = nn.Parameter(self.W[:,1,:]).to(device)
        self.E3 = nn.Parameter(self.W[:,2,:]).to(device)

    def forward(self, x):
        in_shape = x.shape
        x = x.view(-1, in_shape[2] * in_shape[3], 3)

        R = x[:, :, 0]
        G = x[:, :, 1]
        B = x[:, :, 2]

        R_encode = torch.mm(R, self.E1)
        G_encode = torch.mm(G, self.E2)
        B_encode = torch.mm(B, self.E3)

        R_decode = torch.mm(R_encode, self.E1.T)
        G_decode = torch.mm(G_encode, self.E2.T)
        B_decode = torch.mm(B_encode, self.E3.T)

        R_decode = R_decode.view(-1, in_shape[2], in_shape[3], 1)
        G_decode = G_decode.view(-1, in_shape[2], in_shape[3], 1)
        B_decode = B_decode.view(-1, in_shape[2], in_shape[3], 1)

        y = torch.cat((R_decode, G_decode, B_decode), dim=-1)
        y = torch.permute(y, (0, 3, 1, 2))

        return y