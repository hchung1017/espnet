
from __future__ import print_function, division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

class FLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_k=0, bias=True):
      super(FLinear, self).__init__()
      self.in_features = in_features
      self.out_features = out_features

      if rank_k > 0:
        self.k = rank_k
      else:
        self.k = int((out_features*in_features) / (out_features+in_features))

      self.U = torch.nn.Parameter(torch.Tensor(out_features, self.k))
      self.S = torch.nn.Parameter(torch.Tensor(self.k))
      self.V = torch.nn.Parameter(torch.Tensor(self.k, in_features))

      if bias:
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
      else:
        self.register_parameter('bias', None)
      self.reset_parameters()

    def reset_parameters_org(self):
      torch.nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

      stdv = 1. / math.sqrt(self.S.size(0))
      self.S.data.uniform_(-stdv, stdv)

      torch.nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))

      if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters(self):
      weight=torch.Tensor(self.out_features, self.in_features)
      torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
      U, S, V = torch.svd(weight)
      r = self.k
      U = U[:, :r]
      S = S[:r]
      V = V[:, :r]
      self.U = torch.nn.Parameter(U)
      self.S = torch.nn.Parameter(S)
      self.V = torch.nn.Parameter(V.t())
      if self.bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
      #self.weight = torch.mm(self.U, torch.t(torch.mul(F.relu(self.S),torch.t(self.V))))
      self.weight = torch.mm(self.U, torch.t(torch.mul(self.S,torch.t(self.V))))
      return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
      return '{}x{}->{}x{}x{}'.format(
        self.in_features, self.out_features,
        self.U.shape, self.S.shape, self.V.shape
      )   


    def pruning(self, thr=0.2, mink=1, verbose=True):
      sorted, indices = torch.sort(torch.abs(self.S.data),descending=True)

      if verbose:
        print("data    : ", self.S.data)
        print("sorted  : ", sorted)
        print("indices : ", indices)

#      self.S.data[torch.abs(F.relu(self.S.data)) < thr] = 0.0
      self.S.data[torch.abs(self.S.data) < thr] = 0.0
      nzeros = torch.nonzero(self.S.data).view(-1)
      nzeros = nzeros if len(nzeros) > 0 else indices[0:mink]
      
      self.nzeros = nzeros

      if verbose:
        print( "shrink from : ", self.U.data.shape, " : ", self.S.data.shape, " : ", self.V.data.shape )
      self.U.data = self.U.data[:, nzeros]
      self.S.data = self.S.data[nzeros]
      self.V.data = self.V.data[nzeros,:]
      
      self.U = torch.nn.Parameter(self.U.data)
      self.S = torch.nn.Parameter(self.S.data)
      self.V = torch.nn.Parameter(self.V.data)

      if verbose:
        print( "shrink to   : ", self.U.shape, " : ", self.S.shape, " : ", self.V.shape )

def test():      
  N, D_in, D_out = 4, 16, 16
  x = torch.randn(N, D_in)
  y = torch.randn(N, D_out)
  model = FLinear(D_in, D_out)
  loss_fn = torch.nn.MSELoss(reduction='sum')
#  learning_rate = 1e-1
  learning_rate = 1.0

  for t in range(5):
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.pruning()

    print("loss===", loss)
    for mparam, oparam in zip(model.parameters(), optimizer.param_groups[0]['params']) :
      print("param-shape", mparam.data.shape, ", grad-shape", oparam.grad.data.shape)

if __name__ == '__main__':

  torch.manual_seed(1)
  test()
