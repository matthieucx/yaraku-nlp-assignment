import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    logger.info(f"{len(lines)} lines read from {path}")
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


def check_model_files(artifacts_dir, model_name):
    """
    Check if the three necessary model files exist.

    Parameters:
    ----------
    model_name: str
        The name of the model.
    artifacts_dir: str
        The directory where the model files are stored.

    Returns:
    -------
    bool
        True if all three files exist, False otherwise.

    """
    expected_files = [
        f"{model_name}_state.pt",
        f"{model_name}_vocabs_mapping.json",
        f"{model_name}_params.json"
    ]

    files_exist = all(
        os.path.exists(os.path.join(artifacts_dir, f))
        for f in expected_files
    )

    return files_exist


def _save_to_file(path: str, data: plt.Figure | Any):
    """Save data to disk.

    Parameters:
    ----------
    path: str
        The path to the file.
    data: plt.Figure | any
        The data to save. Can be a matplotlib figure or any serializable object.

    """

    if isinstance(data, dict):
        if not path.endswith(".json"):
            path += ".json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    elif isinstance(data, plt.Figure):
        data.savefig(path)

    else:
        raise ValueError("Unsupported data type")


def save_artifacts(model_name: str,
                   model: torch.nn.Module,
                   model_params: dict,
                   vocabulary_mapping: dict,
                   artifacts_dir: str,
                   additional_artifacts: dict = None):
    """Save the model, model parameters, and vocabulary mapping to disk.

    Will save training curves and other artifacts if provided.

    Parameters:
    ----------
    model_name: str
        Name of the model, used as a prefix for the files.
    model: torch.nn.Module
        The actual model.
    model_params: dict
        The parameters used to initialize the model.
    vocabulary_mapping: dict
        The vocabulary mapping used to tokenize the training data.
    artifacts_dir: str
        The directory where the model files will be saved.

    """

    os.makedirs(artifacts_dir, exist_ok=True)

    model_state_path = os.path.join(
        artifacts_dir,
        f"{model_name}_state.pt"
    )
    vocabs_mapping_path = os.path.join(
        artifacts_dir,
        f"{model_name}_vocabs_mapping.json"
    )
    model_params_path = os.path.join(
        artifacts_dir,
        f"{model_name}_params.json"
    )

    torch.save(model.state_dict(), model_state_path)
    _save_to_file(vocabs_mapping_path, vocabulary_mapping)
    _save_to_file(model_params_path, model_params)

    if additional_artifacts is not None:
        for name, data in additional_artifacts.items():
            path = os.path.join(artifacts_dir, f"{model_name}_{name}")
            _save_to_file(path, data)


def load_model(model_name: str, artifacts_dir: str, model_class: torch.nn.Module):
    """Load the model, model parameters, and vocabulary mapping from disk.

    Parameters:
    ----------
    model_name: str
        Name of the model, used as a prefix for the files.
    artifacts_dir: str
        The directory where the model files are stored.
    model_class: torch.nn.Module
        The class of the model.

    Returns:
    -------
    model: torch.nn.Module
        The actual model.
    vocabs_mapping: dict
        The vocabulary mapping used to tokenize the training data.
    model_params: dict
        The parameters used to initialize the model.

    """

    model_found = check_model_files(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
    )
    if not model_found:
        raise FileNotFoundError(f"Files not found for model {model_name}")

    model_state_path = os.path.join(
        artifacts_dir,
        f"{model_name}_state.pt"
    )
    vocabs_mapping_path = os.path.join(
        artifacts_dir,
        f"{model_name}_vocabs_mapping.json"
    )
    model_params_path = os.path.join(
        artifacts_dir,
        f"{model_name}_params.json"
    )

    with open(vocabs_mapping_path, 'r') as f:
        vocabs_mapping = json.load(f)

    with open(model_params_path, 'r') as f:
        model_params = json.load(f)

    model = model_class(**model_params["model"])
    model.load_state_dict(torch.load(model_state_path))

    return model, vocabs_mapping, model_params


def load_hparams(artifacts_dir, hparams_name):
    """Load the hyperparameters from disk.

    Parameters:
    ----------
    artifacts_dir: str
        The directory where the hyperparameters file is stored.
    hparams_name: str
        The name of the hyperparameters file.

    Returns:
    -------
    dict
        The hyperparameters, or None if the file was not found.

    """

    hparams_path = os.path.join(artifacts_dir, hparams_name)
    if os.path.exists(hparams_path):

        with open(hparams_path, 'r') as f:
            hparams = json.load(f)

        logger.info(f"Found {hparams_name} in {artifacts_dir}")
        return hparams

    logger.info(f"Did not find {hparams_name} in {artifacts_dir}")
    return None
