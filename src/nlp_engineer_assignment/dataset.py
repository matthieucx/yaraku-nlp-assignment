from typing import Any

import torch
from torch.utils.data import Dataset

from .utils import count_letters, tokenize


class TokenClassificationDataset(Dataset):
    """Dataset for Yaraku's NLP Engineer Assignment token classification task.

    We assume lines can always be tokenized to a fixed number of tokens.
    Otherwise, we would need to pad sequences to fit the TransformerTokenClassification model.

    Parameters
    ----------
    text_data : list
        List of strings, one for each line in the file.
    vocabs : list
        List of tokens to use as vocabulary.
        Cannot be specified together with vocabs_mapping.
    vocabs_mapping : dict[str, int]
        Tokens in the vocabulary, mapped to indices.
        Cannot be specified together with vocabs.

    Attributes
    ----------
    data : list
        List of strings, one for each line in the file.
    vocabs_mapping : dict[str, int]
        Tokens in the vocabulary, mapped to indices.
        Allow efficient lookup during tokenization.
    max_token_length : int
        Maximum length of the tokens in the vocabulary.
    n_classes : int
        Number of classes in the classification task.
        Depends on the function used to create gold labels.

    """

    def __init__(self, text_data: list, vocabs: list | None = None, vocabs_mapping: dict[str, int] | None = None):

        if (vocabs_mapping is None) == (vocabs is None):
            raise TypeError(
                "Exactly one of vocabs or vocabs_mapping must be specified.")

        self.data = text_data
        self.n_classes = 3

        if vocabs_mapping is not None:
            if len(vocabs_mapping) == 0:
                raise ValueError("vocabs_mapping must be non-empty.")
            if len(vocabs_mapping) != len(set(vocabs_mapping.values())):
                raise ValueError("Duplicate indices in vocabs_mapping.")

            self.vocabs_mapping = vocabs_mapping

        else:
            assert vocabs is not None  # Explicitly assert for MyPy
            if len(vocabs) == 0:
                raise ValueError("vocabs must be non-empty.")
            # Map tokens to indices based on first appearance, with special tokens first
            self.vocabs_mapping = {'<UNK>': 0}
            next_idx = len(self.vocabs_mapping)

            seen_tokens = set(self.vocabs_mapping)
            for token in vocabs:
                if token not in seen_tokens:
                    seen_tokens.add(token)
                    self.vocabs_mapping[token] = next_idx
                    next_idx += 1

        # Pre-compute the maximum token length for efficient tokenization
        self.max_token_length = len(max(self.vocabs_mapping, key=len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single data point (input, target, text).

        Parameters
        ----------
        idx : int
            Index of the data point to retrieve.

        Returns
        -------
        sample : dict[str, Any]
            A dictionary containing the indices of the tokens in the input,
            the target labels per token, and the original text.

        """

        line = self.data[idx]
        tokenized = tokenize(string=line, vocabs=self.vocabs_mapping,
                             max_token_length=self.max_token_length)
        indices = [self.vocabs_mapping[token] for token in tokenized]
        target_seq = count_letters(line)

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        target_seq_tensor = torch.tensor(target_seq, dtype=torch.long)

        return {"indices": indices_tensor, "target_seq": target_seq_tensor, "text": line}
