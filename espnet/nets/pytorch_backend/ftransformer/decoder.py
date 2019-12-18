import torch

from espnet.nets.pytorch_backend.ftransformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.ftransformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.ftransformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.ftransformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.ftransformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.ftransformer.repeat import repeat
from espnet.nets.pytorch_backend.ftransformer.flinear import FLinear


class Decoder(torch.nn.Module):
    """Transfomer decoder module

    :param int odim: output dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate for attention
    :param str or torch.nn.Module input_layer: input layer type
    :param bool use_output_layer: whether to use output layer
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, odim,
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 self_attention_dropout_rate=0.0,
                 src_attention_dropout_rate=0.0,
                 input_layer="embed",
                 use_output_layer=True,
                 pos_enc_class=PositionalEncoding,
                 normalize_before=True,
                 concat_after=False,
                 low_rank=False):
        super(Decoder, self).__init__()
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "linear":
          if low_rank :
            self.embed = torch.nn.Sequential(
                FLinear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
          else:
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before
        self.decoders = repeat(
            num_blocks,
            lambda: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate, low_rank),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate, low_rank),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate, low_rank),
                dropout_rate,
                normalize_before,
                concat_after
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            if low_rank:
              self.output_layer = FLinear(attention_dim, odim)
            else:
              self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """forward decoder

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out) if input_layer == "embed"
                                 input tensor (batch, maxlen_out, #mels) in the other cases
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask, uint8  (batch, maxlen_in)
        :return x: decoded token score before softmax (batch, maxlen_out, token) if use_output_layer is True,
                   final block outputs (batch, maxlen_out, attention_dim) in the other cases
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def recognize(self, tgt, tgt_mask, memory):
        """recognize one step

        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask, uint8  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: torch.Tensor
        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(x, tgt_mask, memory, None)
        if self.normalize_before:
            x_ = self.after_norm(x[:, -1])
        else:
            x_ = x[:, -1]
        if self.output_layer is not None:
            return torch.log_softmax(self.output_layer(x_), dim=-1)
        else:
            return x_

    def pruning(self, thr=0.05, mink=1, verbose=False):
      for decoder in self.decoders :
        decoder.self_attn.linear_q.pruning(thr, mink, verbose)
        decoder.self_attn.linear_k.pruning(thr, mink, verbose)
        decoder.self_attn.linear_v.pruning(thr, mink, verbose)
        decoder.self_attn.linear_out.pruning(thr, mink, verbose)
        decoder.src_attn.linear_q.pruning(thr, mink, verbose)
        decoder.src_attn.linear_k.pruning(thr, mink, verbose)
        decoder.src_attn.linear_v.pruning(thr, mink, verbose)
        decoder.src_attn.linear_out.pruning(thr, mink, verbose)
        decoder.feed_forward.w_1.pruning(thr, mink, verbose)
        decoder.feed_forward.w_2.pruning(thr, mink, verbose)
