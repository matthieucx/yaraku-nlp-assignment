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
