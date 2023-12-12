import numpy as np
import pytest

from nlp_engineer_assignment.utils import count_letters, score, tokenize


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
    assert "Vocabulary and string must be non-empty." in str(excinfo.value)


def test_tokenize_empty_string_error():
    vocabs = {"this", "is", "a", "test", " "}
    with pytest.raises(ValueError) as excinfo:
        tokenize("", vocabs)
    assert "Vocabulary and string must be non-empty." in str(excinfo.value)


def test_tokenize_warn_unks(caplog):
    string = "unknown token test"
    vocabs = {"token", "test", " "}
    tokenize(string, vocabs, verbose=True)

    assert "Found unknown token" in caplog.text
