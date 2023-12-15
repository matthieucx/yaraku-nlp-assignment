import torch
import pytest
from nlp_engineer_assignment.transformer import MultiHeadSelfAttention, ScaledDotProductAttention, \
    TransformerEncoderLayer, BasicLayerNorm, TransformerEmbeddings, TransformerTokenClassification, \
    evaluate_classifier
from nlp_engineer_assignment.dataset import TokenClassificationDataset


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


def test_transformer_encoder_layer_output_shape():
    batch_size, tokens, emb, heads, dim_ff = 8, 20, 64, 2, 256
    model = TransformerEncoderLayer(emb, heads, dim_ff)

    x = torch.rand(batch_size, tokens, emb)

    output = model(x)
    assert output.shape == (batch_size, tokens,
                            emb), "Output shape is incorrect"


def test_basic_layer_norm_output_shape():
    batch_size, tokens, emb_size = 64, 20, 512
    layer_norm = BasicLayerNorm(emb_size)
    x = torch.rand(batch_size, tokens, emb_size)

    output = layer_norm(x)

    assert output.shape == (batch_size, tokens,
                            emb_size), "Output shape is incorrect"


def test_basic_layer_norm_forward():

    torch.manual_seed(0)

    batch_size, tokens, emb_size = 2, 2, 3
    layer_norm = BasicLayerNorm(normalized_shape=emb_size)

    x = torch.rand(batch_size, tokens, emb_size)

    x_mean = x.mean(dim=-1, keepdim=True)
    x_std = x.std(dim=-1, keepdim=True, unbiased=False)
    expected = (x - x_mean) / (x_std + layer_norm.eps)
    expected = expected * layer_norm.gain + layer_norm.bias

    output = layer_norm(x)

    assert torch.allclose(
        output, expected), f"Expected: {expected}, but got: {output}"


def test_basic_layer_norm_zeros():

    torch.manual_seed(0)

    batch_size, tokens, emb_size = 2, 2, 3
    layer_norm = BasicLayerNorm(normalized_shape=emb_size)

    x = torch.zeros(batch_size, tokens, emb_size)

    output = layer_norm(x)

    assert torch.allclose(
        output, x), f"Expected: {x}, but got: {output}"


def test_transformer_embeddings_output_shape():
    batch_size, tokens, emb, vocab_size = 8, 20, 64, 1000
    model = TransformerEmbeddings(
        vocab_size=vocab_size, emb=emb, n_tokens=tokens)

    x = torch.randint(0, vocab_size, (batch_size, tokens))

    output = model(x)
    assert output.shape == (
        batch_size, tokens, emb), "Output shape is incorrect"


def test_transformer_token_classification_output_shape():
    batch_size, tokens, emb,  = 8, 20, 64
    heads, dim_ff, vocab_size, n_classes = 2, 256, 1000, 10

    model = TransformerTokenClassification(
        depth=2,
        emb=emb,
        heads=heads,
        dim_ff=dim_ff,
        vocab_size=vocab_size,
        n_tokens=tokens,
        n_classes=n_classes
    )

    x = torch.randint(0, vocab_size, (batch_size, tokens))

    output = model(x)
    assert output.shape == (
        batch_size, tokens, n_classes), "Output shape is incorrect"


def test_evaluate_classifier_empty_dataset():
    clf_model = TransformerTokenClassification(
        depth=1,
        emb=1,
        heads=1,
        dim_ff=1,
        vocab_size=1,
        n_tokens=1,
        n_classes=1
    )
    test_inputs = []
    test_dataset = TokenClassificationDataset(test_inputs, vocabs=["a"])

    pred = evaluate_classifier(model=clf_model, test_dataset=test_dataset)

    assert pred.shape == torch.tensor(
        []).shape and pred.numel() == 0, "Returned tensor should be empty"
