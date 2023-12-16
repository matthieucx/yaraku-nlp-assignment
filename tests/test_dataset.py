from unittest import mock

import numpy as np
import pytest

from nlp_engineer_assignment import TokenClassificationDataset, count_letters, tokenize


@pytest.fixture(autouse=True, scope="function")
def mock_tokenize():
    with mock.patch("nlp_engineer_assignment.tokenize") as mock_tokenizer:
        mock_tokenizer.side_effect = lambda string: string.split()
        yield mock_tokenizer


@pytest.fixture(autouse=True, scope="function")
def mock_count_letters():
    with mock.patch("nlp_engineer_assignment.count_letters") as mock_counter:
        mock_counter.side_effect = lambda line: [
            1] * len(line.split())
        yield mock_counter


# Sample data
text_data = ["hello world", "testing dataset"]
vocabs_mapping = {"<UNK>": 0, "hello": 1,
                  "world": 2, "testing": 3, "dataset": 4}


@pytest.fixture
def token_classification_dataset():
    return TokenClassificationDataset(text_data, vocabs_mapping=vocabs_mapping)


def test_vocab_mapping():
    vocab_list = list(vocabs_mapping.keys())
    vocab_mapped = TokenClassificationDataset(
        text_data, vocabs=vocab_list).vocabs_mapping
    assert vocab_mapped == vocabs_mapping, "The vocabs_mapping is incorrect."


def test_correct_line_returned(token_classification_dataset):
    for idx, line in enumerate(text_data):
        dataset_item = token_classification_dataset[idx]
        assert dataset_item['text'] == line, "The returned line does not match the expected line."


def test_correct_target_sequence(token_classification_dataset):
    for idx, line in enumerate(text_data):
        dataset_item = token_classification_dataset[idx]
        expected_target_seq = count_letters(line)

        assert np.array_equal(
            dataset_item['target_seq'], expected_target_seq), "The target sequence is incorrect."


def test_correct_indices(token_classification_dataset):
    for idx, line in enumerate(text_data):
        dataset_item = token_classification_dataset[idx]
        tokenized_line = tokenize(line, vocabs_mapping)
        expected_indices = [vocabs_mapping[word] for word in tokenized_line]

        assert np.array_equal(
            dataset_item['indices'], expected_indices), "The indices do not match the expected values."


def test_error_both_vocabs_and_vocabs_mapping_provided():
    text_data = ["sample text"]
    vocabs = ["sample", "text"]
    vocabs_mapping = {"sample": 1, "text": 2}

    with pytest.raises(TypeError, match="Exactly one of vocabs or vocabs_mapping must be specified."):
        TokenClassificationDataset(text_data, vocabs, vocabs_mapping)


def test_error_no_vocabs_provided():
    text_data = ["sample text"]

    with pytest.raises(TypeError, match="Exactly one of vocabs or vocabs_mapping must be specified."):
        TokenClassificationDataset(text_data)


def test_error_duplicate_indices_in_vocabs_mapping():
    text_data = ["sample text"]
    vocabs_mapping = {"sample": 1, "text": 1}

    with pytest.raises(ValueError, match="Duplicate indices in vocabs_mapping."):
        TokenClassificationDataset(text_data, vocabs_mapping=vocabs_mapping)


def test_error_empty_vocabs_list_provided():
    empty_vocabs = []
    text_data = ["sample text"]

    with pytest.raises(ValueError, match="vocabs must be non-empty."):
        TokenClassificationDataset(text_data, vocabs=empty_vocabs)


def test_error_empty_vocabs_mapping_provided():
    empty_vocabs_mapping = {}
    text_data = ["sample text"]

    with pytest.raises(ValueError, match="vocabs_mapping must be non-empty."):
        TokenClassificationDataset(
            text_data, vocabs_mapping=empty_vocabs_mapping)
