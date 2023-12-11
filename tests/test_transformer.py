import torch
import pytest
from nlp_engineer_assignment.transformer import MultiHeadSelfAttention, ScaledDotProductAttention


def test_scaled_dot_product_attention_output_shape():
    batch_size, tokens, d_k, d_v = 64, 20, 32, 16
    attention = ScaledDotProductAttention(d_k=d_k)
    queries = torch.rand(batch_size, tokens, d_k)
    keys = torch.rand(batch_size, tokens, d_k)
    values = torch.rand(batch_size, tokens, d_v)

    output = attention(queries, keys, values)

    assert output.shape == (
        batch_size, tokens, d_v), "Output shape is incorrect"


def test_multihead_attention_output_shape():
    batch_size, tokens, emb_size, heads = 64, 20, 512, 8
    model = MultiHeadSelfAttention(emb=emb_size, heads=heads)
    x = torch.rand(batch_size, tokens, emb_size)

    out = model(x)

    assert out.shape == (batch_size, tokens,
                         emb_size), "Output shape is incorrect"


def test_multi_head_attention_assertion():
    emb, heads = 128, 10

    with pytest.raises(AssertionError) as excinfo:
        _ = MultiHeadSelfAttention(emb=emb, heads=heads)

    assert "Embedding dimension must be divisible by the number of heads" in str(
        excinfo.value)
