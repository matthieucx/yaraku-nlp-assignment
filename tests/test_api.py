"""Test the API endpoints.

This script tests the API endpoints using the FastAPI test client.

Validation of input data is done by FastAPI automatically and is not tested here.

Example:
    $ pytest tests/test_api.py
"""

from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from nlp_engineer_assignment.api import PredictionResponse, app
from nlp_engineer_assignment.transformer import TransformerTokenClassification

client = TestClient(app)

N_TOKENS = 20
MODEL_RETURN_VALUE = torch.tensor(
    [[[10, 0, 0]] * N_TOKENS]  # (batch_size, n_tokens, n_classes)
)


@pytest.fixture(autouse=True, scope="module")
def mock_settings():
    with patch('nlp_engineer_assignment.api.Settings') as mock_settings_cls:
        mock_settings_cls.return_value.artifacts_dir = "/dummy/dir"
        mock_settings_cls.return_value.clf_model_name = "dummy_model"
        mock_settings_cls.return_value.clf_model_class_name = TransformerTokenClassification.__name__
        yield


@pytest.fixture(autouse=True, scope="module")
def mock_model():
    with patch("nlp_engineer_assignment.transformer.TransformerTokenClassification") as mock_transformer, \
            patch('nlp_engineer_assignment.api.load_model') as mock_loader:

        # Mock the TransformerTokenClassification behavior
        model_instance = mock_transformer.return_value
        model_instance.n_tokens = N_TOKENS
        model_instance.eval.return_value = None
        model_instance.return_value = MODEL_RETURN_VALUE

        # Mock the load_model function
        mock_vocabs_mapping = {'a': 1}
        mock_loader.return_value = (model_instance, mock_vocabs_mapping, {})

        yield


@pytest.fixture
def mock_tokenize():
    with patch("nlp_engineer_assignment.utils.tokenize") as mock:
        yield mock


def test_predict_empty_input(
    mock_tokenize,
):
    input_text_empty = {"text": ""}
    mock_tokenize.return_value = [""]
    response = client.post("/predict", json=input_text_empty)
    expected_empty = f"Input text must be tokenizable to exactly {N_TOKENS} tokens."

    assert response.status_code == 422
    assert expected_empty in response.text


@pytest.mark.parametrize("input_text, token_count", [
    ("a" * (N_TOKENS + 1), N_TOKENS),  # Input too long
    ("a" * (N_TOKENS - 1), N_TOKENS),  # Input too short
])
def test_predict_non_fixed_size_input(
    mock_tokenize,
    input_text,
    token_count,
):
    mock_tokenize.return_value = ["a"] * token_count
    response = client.post("/predict", json={"text": input_text})

    assert response.status_code == 422
    assert f"Input text must be tokenizable to exactly {N_TOKENS} tokens." in response.text


def test_predict_regular_input(
    mock_tokenize,
):
    input_text = {"text": "a" * N_TOKENS}
    mock_tokenize.return_value = ["a"] * N_TOKENS
    response = client.post("/predict", json=input_text)

    expected_preds = MODEL_RETURN_VALUE.argmax(dim=-1).squeeze().tolist()
    expected_string = "".join([str(pred) for pred in expected_preds])
    expected_json = PredictionResponse(prediction=expected_string).model_dump()

    assert response.status_code == 200
    assert len(response.json()["prediction"]) == N_TOKENS
    assert response.json() == expected_json
