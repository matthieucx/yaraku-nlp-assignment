import numpy as np
from loguru import logger


def count_letters(text: str) -> np.array:
    """
    Count the number of times each letter appears in the text up to that point

    Parameters
    ----------
    text : str
        The text to count the letters in

    Returns
    -------
    np.array
        A vector of counts, one for each letter in the text
    """
    output = np.zeros(len(text), dtype=np.int32)
    for i in range(0, len(text)):
        output[i] = min(2, len([c for c in text[0:i] if c == text[i]]))

    return output


def print_line():
    """
    Print a line of dashes
    """
    print("-" * 80)


def read_inputs(path: str) -> list:
    """
    Read the inputs from a file

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    list
        A list of strings, one for each line in the file
    """
    lines = [line[:-1] for line in open(path, mode="r")]
    print(f"{len(lines)} lines read")
    print_line()
    return lines


def score(
    golds: np.array,
    predictions: np.array
) -> float:
    """
    Compute the accuracy of the predictions

    Parameters
    ----------
    golds : np.array
        Ground truth labels
    predictions : np.array
        Predicted labels

    Returns
    -------
    float
        Accuracy of the predictions
    """
    return float(np.sum(golds == predictions)) / len(golds.flatten())


def tokenize(
        string: str,
        vocabs: list[str] | set[str] | dict[str, int],
        max_token_length: int = None,
        verbose: bool = False
) -> list[str]:
    """Tokenize a string using a vocabulary list.

    Tokenize by matching the longest possible substring at each position.
    Substrings of up to `max_token_length` are considered.
    Substrings not found in the vocabulary are replaced by `<UNK>`.

    More efficient implementations could rely on a Trie structure for prefix search.

    Parameters
    ----------
    string : str
        Input string to tokenize.
    vocabs : list[str] | set[str] | dict[str, int]
        Vocabulary to use for tokenization.
        Hashed to a set for efficient lookup.
        Should be passed as a set or dict if tokenizing a large number of strings.
    max_token_length : int, optional
        Maximum length of a token, by default None.
        Set to the length of the longest string in the vocabulary if not provided.
        Should be provided if tokenizing a large number of strings.

    Returns
    -------
    list[str]
        A list of the tokens found in the string.

    """
    if not vocabs:
        raise ValueError("Vocabulary must be non-empty.")

    max_token_length = len(max(vocabs, key=len)
                           ) if max_token_length is None else max_token_length
    vocabs_hashed = set(vocabs) if not isinstance(
        vocabs, (set, dict)) else vocabs

    tokens = []
    current_index = 0

    while current_index < len(string):
        for len_substring in range(max_token_length, 0, -1):
            if current_index + len_substring > len(string):
                continue

            substring = string[current_index:current_index + len_substring]

            if substring in vocabs_hashed:
                tokens.append(substring)
                current_index += len_substring
                break

        # Else executes if the for loop was completed normally (i.e., no break)
        # Here, if there was no match
        else:
            # Avoid adding consecutive UNK tokens
            if not tokens or tokens[-1] != "<UNK>":
                tokens.append("<UNK>")
                if verbose:
                    logger.info(
                        "Found unknown token at position {pos} in string '{string}'", pos=current_index, string=string)
            current_index += 1

    return tokens
