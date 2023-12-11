import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    The dot products of the queries and keys are scaled by the inverse square root of their dimension ( 1/sqrt(d_k) ).
    This avoids the saturation of the softmax function due to large dot products and preserves gradient flow.

    Parameters
    ----------
    d_k : int
        The dimension of the queries and keys.

    """

    def __init__(self, d_k: int):
        super().__init__()

        self.d_k = d_k

    def forward(self, queries, keys, values):
        """Compute the attention scores.

        Parameters
        ----------
        queries : torch.Tensor
            The queries, shape (batch, tokens, d_k).
        keys : torch.Tensor
            The keys, shape (batch, tokens, d_k).
        values : torch.Tensor
            The values, shape (batch, tokens, d_v).

        Returns
        -------
        torch.Tensor
            Computed attention scores, shape (batch, tokens, d_v).

        """

        dot = torch.bmm(queries, keys.transpose(1, 2))
        # Scale dot product
        dot = dot / (self.d_k ** (1/2))
        # Normalize
        attention_weights = F.softmax(dot, dim=2)

        return torch.bmm(attention_weights, values)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Attention as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    As in the paper, this implementation use the same dimension for queries/keys (d_k) and values (d_v).
    This dimension is equal to the head dimension, set as emb // heads.

    Parameters
    ----------
    emb : int
        Embedding dimension of the input (d_model).
    heads : int
        Number of attention heads (h), used to set the head dimension (d_k) as emb // heads.

    """

    def __init__(self, emb: int, heads: int) -> None:
        super().__init__()

        assert emb % heads == 0, (
            f"Embedding dimension must be divisible by the number of heads, Found emb: {emb}, heads: {heads}."
        )

        self.emb = emb
        self.heads = heads
        self.head_dim = emb // heads

        self.to_queries = nn.Linear(emb, emb, bias=False)
        self.to_keys = nn.Linear(emb, emb, bias=False)
        self.to_values = nn.Linear(emb, emb, bias=False)

        self.attention = ScaledDotProductAttention(self.head_dim)

        self.merge_heads = nn.Linear(emb, emb, bias=False)

    def forward(self, x):
        """Compute the multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, tokens, embedding).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, tokens, embedding).

        """

        b, t, _ = x.size()  # batch, tokens, embedding
        h = self.heads
        d_k = self.head_dim

        # Compute Q, K, V for all heads
        queries = self.to_queries(x).view(b, t, h, d_k)
        keys = self.to_keys(x).view(b, t, h, d_k)
        values = self.to_values(x).view(b, t, h, d_k)
        # Q, K, V: (batch, tokens, heads, head_dim)

        # We combine batch and heads for batch matrix multiplication
        # Computations will be done independently for each head, in parallel
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, d_k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, d_k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, d_k)
        # Q, K, V: (batch * heads, tokens, head_dim)

        attention_scores = self.attention(
            queries=queries, keys=keys, values=values)
        # Attention scores: (batch * heads, tokens, head_dim)

        concat_heads = attention_scores.view(b, h, t, d_k).transpose(
            1, 2).contiguous().view(b, t, h * d_k)
        # Concatenated heads: (batch, tokens, heads * head_dim = embedding)

        return self.merge_heads(concat_heads)


class Transformer(nn.Module):
    """
    TODO: You should implement the Transformer model from scratch here. You can
    use elementary PyTorch layers such as: nn.Linear, nn.Embedding, nn.ReLU, nn.Softmax, etc.
    DO NOT use pre-implemented layers for the architecture, positional encoding or self-attention,
    such as nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention, etc.
    """


def train_classifier(train_inputs):
    # TODO: Implement the training loop for the Transformer model.
    raise NotImplementedError(
        "You should implement `train_classifier` in transformer.py"
    )
