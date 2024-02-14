import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def flatten(data, mode):
  ndim = data.ndim
  new_axes = list(range(mode, ndim)) + list(range(mode))
  data = data.permute(new_axes)
  old_shape = data.shape
  data = data.reshape(old_shape[0], -1)
  return data, old_shape

def nmodeproduct(data, projection, mode):
  data, old_shape = flatten(data, mode)
  data = torch.mm(projection.T, data)
  new_shape = list(old_shape)
  new_shape[0] = projection.shape[-1]
  data = torch.reshape(data, new_shape)
  new_axes = list(range(data.ndim))
  new_axes = list(new_axes[-mode:]) + list(new_axes[:-mode])
  data = data.permute(new_axes)
  return data

def MSE(a,b):
  return torch.mean((a.flatten()-b.flatten())**2)

def HOSVD(data, h, w, d, centering=False, iteration=100, threshold=1e-4, regularization=0.0, device=device):
  N, _, H, W = data.shape
  W1 = torch.rand(H, h).to(device)
  W2 = torch.rand(W, w).to(device)
  W3 = torch.rand(3, d).to(device)
  mean_tensor = torch.zeros(data.shape[1:])
  mean_tensor = torch.unsqueeze(mean_tensor, dim=0)
  mean_tensor = mean_tensor.to(device)
  data -= mean_tensor
  for i in range(iteration):
    data_tmp = nmodeproduct(data, W2, 3)
    data_tmp = nmodeproduct(data_tmp, W3, 1)
    data_tmp, _ = flatten(data, 2)
    I = torch.eye(H).to(device)
    cov = torch.mm(data_tmp, data_tmp.T) + regularization*I
    U, _, _ = torch.linalg.svd(cov)
    W1_new = U[:, :h]

    data_tmp = nmodeproduct(data, W1_new, 2)
    data_tmp = nmodeproduct(data_tmp, W3, 1)
    data_tmp, _ = flatten(data, 3)
    I = torch.eye(W).to(device)
    cov = torch.mm(data_tmp, data_tmp.T) + regularization*I
    U, _, _ = torch.linalg.svd(cov)
    W2_new = U[:, :w]

    data_tmp = nmodeproduct(data, W1_new, 2)
    data_tmp = nmodeproduct(data_tmp, W2_new, 3)
    data_tmp, _ = flatten(data, 1)
    I = torch.eye(3).to(device)
    cov = torch.mm(data_tmp, data_tmp.T) + regularization*I
    U, _, _ = torch.linalg.svd(cov)
    W3_new = U[:, :d]

    data_tmp = nmodeproduct(data, W1, 2)
    data_tmp = nmodeproduct(data_tmp, W2, 3)
    data_tmp = nmodeproduct(data_tmp, W3, 1)

    data_tmp = nmodeproduct(data_tmp, W1.T, 2)
    data_tmp = nmodeproduct(data_tmp, W2.T, 3)
    data_tmp = nmodeproduct(data_tmp, W3.T, 1)

    print('Residual error: %.4f' % MSE(data_tmp, data))
    projection_error = MSE(W1, W1_new) + MSE(W2, W2_new) + MSE(W3, W3_new)
    print('Projection error: %.4f' % projection_error)

    W1 = W1_new
    W2 = W2_new
    W3 = W3_new

    if projection_error  < threshold:
        break

  return W1, W2, W3

def mode1_product(data, projection):
  data_shape = data.shape
  data = torch.permute(data, (2, 3, 1, 0))
  data = torch.reshape(data, (data_shape[2], -1))
  data = torch.mm(projection.T, data)
  data = torch.reshape(data, (projection.shape[1], data_shape[3], data_shape[1], -1))
  data = torch.permute(data, (3, 2, 0, 1))
  return data

def mode2_product(data, projection):
  data_shape = data.shape
  data = torch.permute(data, (3, 1, 2, 0))
  data = torch.reshape(data, (data_shape[3], -1))
  data = torch.mm(projection.T, data)
  data = torch.reshape(data, (projection.shape[1], data_shape[1], data_shape[2], -1))
  data = torch.permute(data, (3, 1, 2, 0))
  return data

def mode3_product(data, projection):
  data_shape = data.shape
  data = torch.permute(data, (1, 2, 3, 0))
  data = torch.reshape(data, (data_shape[1], -1))
  data = torch.mm(projection.T, data)
  data = torch.reshape(data, (projection.shape[1], data_shape[2], data_shape[3], -1))
  data = torch.permute(data, (3, 0, 1, 2))
  return data

class TensorSensing(nn.Module):
    def __init__(self, measurement_shape, W1, W2, W3, W1T, W2T, W3T, linear_sensing=False, separate_decoder=True, regularizer=None, constraint=None, device = device):
        super(TensorSensing, self).__init__()

        self.h = measurement_shape[0]
        self.w = measurement_shape[1]
        self.d = measurement_shape[2]
        self.W1 = W1
        self.W2 = W2
        self.W3 = W3
        self.W1T = W1T
        self.W2T = W2T
        self.W3T = W3T
        self.linear_sensing = linear_sensing
        self.separate_decoder = separate_decoder
        self.constraint = constraint
        self.regularizer = regularizer
        self.P1_encode = nn.Parameter(self.W1).to(device)
        self.P2_encode = nn.Parameter(self.W2).to(device)
        self.P3_encode = nn.Parameter(self.W3).to(device)
        if self.separate_decoder:
            self.P1_decode = nn.Parameter(self.W1T).to(device)
            self.P2_decode = nn.Parameter(self.W2T).to(device)
            self.P3_decode = nn.Parameter(self.W3T).to(device)

    def forward(self, x):
      encode = mode1_product(x, self.P1_encode)
      encode = mode2_product(encode, self.P2_encode)
      encode = mode3_product(encode, self.P3_encode)
      if not self.linear_sensing:
        encode = F.relu(encode)
      if self.separate_decoder:
          decode = mode1_product(encode, self.P1_decode)
          decode = mode2_product(decode, self.P2_decode)
          decode = mode3_product(decode, self.P3_decode)
      else:
          decode = mode1_product(encode, self.P1_encode.T)
          decode = mode2_product(decode, self.P2_encode.T)
          decode = mode3_product(decode, self.P3_encode.T)

      return decode