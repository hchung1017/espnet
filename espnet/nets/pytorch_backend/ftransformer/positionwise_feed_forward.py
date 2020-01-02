import torch
from espnet.nets.pytorch_backend.ftransformer.flinear import FLinear


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate, low_rank=False,
                 init_rank_k=0):
        super(PositionwiseFeedForward, self).__init__()
        if low_rank:
          self.w_1 = FLinear(idim, hidden_units, init_rank_k)
          self.w_2 = FLinear(hidden_units, idim, init_rank_k)
        else:
          self.w_1 = torch.nn.Linear(idim, hidden_units)
          self.w_2 = torch.nn.Linear(hidden_units, idim)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
