# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

import numpy as np
from itertools import permutations

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from espnet.nets.pytorch_backend.ftransformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.eend.diarization_dataset import KaldiDiarizationDataset
from espnet.nets.pytorch_backend.ftransformer.flinear import FLinear

class TransformerDiarization(nn.Module):
  def __init__(self,
                  odim,
                  idim,
                  attention_dim=256,
                  attention_heads=1,
                  linear_units=2048,
                  num_blocks=6,
                  dropout_rate=0.1,
                  low_rank=False):
    super(TransformerDiarization, self).__init__()

    self.enc = Encoder( idim, 
                        attention_dim,
                        attention_heads,
                        linear_units,
                        num_blocks,
                        dropout_rate,
                        low_rank=low_rank)

    if low_rank :
      self.linear = FLinear(attention_dim, odim)
    else:
      self.linear = torch.nn.Linear(attention_dim, odim)

  def forward(self, xs, activation=None):
    ilens = [x.shape[0] for x in xs]
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=-1)
    pad_shape = xs.shape
    src_mask = (~make_pad_mask(ilens)).to(xs_pad.device).unsqueeze(-2)
    hs_pad, hs_mask = self.enc.forward(xs_pad, src_mask)
    ys_pad = self.linear(hs_pad)
    return ys_pad
