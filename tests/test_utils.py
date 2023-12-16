from unittest.mock import Mock, patch
import json
import os
import numpy as np
import pytest

from nlp_engineer_assignment.utils import count_letters, score, tokenize, save_artifacts, check_model_files, \
    load_model, load_hparams


def test_count_letters():
    assert np.array_equal(count_letters("hello"), np.array([0, 0, 0, 1, 0]))
    assert np.array_equal(count_letters("world"), np.array([0, 0, 0, 0, 0]))
    assert np.array_equal(
        count_letters("hello hello"),
        np.array([0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1])
    )


def test_score():
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 1, 1, 0, 1]])) == 1.0
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 1, 0, 0, 1]])) == 0.6
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[0, 0, 0, 0, 0]])) == 0.4
    assert score(np.array([[0, 1, 1, 0, 1]]),
                 np.array([[1, 0, 0, 1, 0]])) == 0.0


def test_tokenize_regular_input():
    string = "this is a test"
    vocabs = {"this", "is", "a", "test", " "}
    expected = ["this", " ", "is", " ", "a", " ", "test"]
    assert tokenize(string, vocabs) == expected


def test_tokenize_unk_at_first_position():
    string = "unknown this is a test"
    vocabs = {"this", "is", "a", "test",  " "}
    expected = ["<UNK>", " ", "this", " ", "is", " ", "a", " ", "test"]
    assert tokenize(string, vocabs) == expected


def test_tokenize_limit_token_length():
    string = "thisisatest"
    vocabs = {"this", "is", "a", "test", "th", "is", "te", "st"}
    expected = ["th", "is", "is", "a", "te", "st"]
    assert tokenize(string, vocabs, max_token_length=3) == expected


def test_tokenize_empty_vocab_error():
    string = "this is a test"
    empty_vocabs = set()
    with pytest.raises(ValueError) as excinfo:
        tokenize(string, empty_vocabs)
    assert "Vocabulary must be non-empty." in str(excinfo.value)


def test_tokenize_empty_string():
    vocabs = {"this", "is", "a", "test", " "}
    expected = []
    assert tokenize("", vocabs) == expected


def test_tokenize_warn_unks(caplog):
    string = "unknown token test"
    vocabs = {"token", "test", " "}
    tokenize(string, vocabs, verbose=True)

    assert "Found unknown token" in caplog.text


def test_tokenize_handle_vocab_types():
    string = "This is a test"
    vocab_list = ["Th", "te", "i", "s", "a", "t", "e"]
    vocab_set = {"Th", "te", "i", "s", "a", "t", "e"}
    vocab_dict = {"Th": 0, "te": 1, "i": 2, "s": 3, "a": 4, "t": 5, "e": 6}
    tokenize_list = tokenize(string, vocab_list)
    tokenize_set = tokenize(string, vocab_set)
    tokenize_dict = tokenize(string, vocab_dict)

    assert tokenize_list == tokenize_set == tokenize_dict, \
        f"tokenize_list: {tokenize_list}, " \
        f"tokenize_set: {tokenize_set}, " \
        f"tokenize_dict: {tokenize_dict}"


def test_check_model_files_exists(tmp_path):
    model_name = "test_model"
    # Create the three files
    for file_name in [f"{model_name}_state.pt", f"{model_name}_vocabs_mapping.json", f"{model_name}_params.json"]:
        (tmp_path / file_name).touch()

    assert check_model_files(str(tmp_path), model_name) is True


def test_check_model_files_missing(tmp_path):
    model_name = "test_model"
    # Some files missing
    (tmp_path / f"{model_name}_state.pt").touch()

    assert check_model_files(str(tmp_path), model_name) is False


def test_save_artifacts(tmp_path):

    model_name = "test_model"
    mock_model = Mock()
    mock_model.state_dict.return_value = {"dummy": "state"}
    model_params = {"param1": "value1"}
    vocab_mapping = {"vocab1": "mapping1"}

    save_artifacts(
        model_name=model_name,
        model=mock_model,
        model_params=model_params,
        vocabulary_mapping=vocab_mapping,
        artifacts_dir=str(tmp_path))

    # Ensure the files are created
    assert os.path.exists(tmp_path / f"{model_name}_state.pt")
    assert os.path.exists(tmp_path / f"{model_name}_vocabs_mapping.json")
    assert os.path.exists(tmp_path / f"{model_name}_params.json")

    # Ensure the files contain the correct content
    with open(tmp_path / f"{model_name}_vocabs_mapping.json", "r") as f:
        content = json.load(f)
    assert content == vocab_mapping

    with open(tmp_path / f"{model_name}_params.json", "r") as f:
        content = json.load(f)
    assert content == model_params


def test_load_model_success(tmp_path):
    model_name = "test_model"
    artifacts_dir = str(tmp_path)

    mock_state_dict = {"weights": [1, 2, 3]}
    dummy_vocabs_mapping = {"a": 1}
    dummy_model_params = {"model": {"param1": "value1"}}

    # Create mock files
    with open(tmp_path / f"{model_name}_vocabs_mapping.json", 'w') as f:
        json.dump(dummy_vocabs_mapping, f)
    with open(tmp_path / f"{model_name}_params.json", 'w') as f:
        json.dump(dummy_model_params, f)

    mock_model_class = Mock()
    mock_model_instance = Mock()
    mock_model_class.return_value = mock_model_instance
    # Mock external functions
    with patch("nlp_engineer_assignment.utils.check_model_files", return_value=True), \
            patch("torch.load", return_value=mock_state_dict):

        model, vocabs_mapping, model_params = load_model(
            model_name=model_name,
            artifacts_dir=artifacts_dir,
            model_class=mock_model_class
        )

    assert model == mock_model_instance
    assert vocabs_mapping == dummy_vocabs_mapping
    assert model_params == dummy_model_params


def test_initialize_hyperparameters_exists(tmp_path):

    hparams_name = "dummy_file.json"
    hparams_path = tmp_path / hparams_name
    hparams_data = {"param1": 10, "param2": 20}
    with open(hparams_path, 'w') as f:
        json.dump(hparams_data, f)

    hparams = load_hparams(str(tmp_path), hparams_name)

    assert hparams == hparams_data


def test_initialize_hyperparameters_not_exists(tmp_path):

    hparams_name = "dummy_file.json"
    hparams = load_hparams(str(tmp_path), hparams_name)

    assert hparams is None
