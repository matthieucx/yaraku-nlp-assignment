import numpy as np
import os
import uvicorn
from loguru import logger
from rich.logging import RichHandler
import torch

from nlp_engineer_assignment import count_letters, print_line, read_inputs, \
    score, train_classifier, TokenClassificationDataset, evaluate_classifier


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_inputs = read_inputs(
        os.path.join(cur_dir, "data", "train.txt")
    )
    train_dataset = TokenClassificationDataset(train_inputs, vocabs)

    model, _, _ = train_classifier(
        train_dataset, save="default_model", hparams={"epochs": 2})

    ###
    # Test
    ###

    test_inputs = read_inputs(
        os.path.join(cur_dir, "data", "test.txt")
    )
    test_dataset = TokenClassificationDataset(
        test_inputs, vocabs_mapping=train_dataset.vocabs_mapping
    )

    pred = evaluate_classifier(model=model, test_dataset=test_dataset)
    pred_np = pred.numpy()
    golds = np.stack([count_letters(text) for text in test_inputs])

    # Log the first five inputs, golds, and predictions for analysis
    sample_idx = np.random.randint(0, len(test_inputs))
    logger.info("Sample input: {}", sample_idx)
    sample = test_dataset[sample_idx]
    logger.info("Input: {}", sample["text"])
    logger.info("Gold: {}", sample["target_seq"].tolist())
    logger.info("Pred: {}", pred_np[sample_idx].tolist())
    print_line()

    logger.info("Test Accuracy: {:.2f}%", 100.0 * score(golds, pred_np))
    print_line()


if __name__ == "__main__":

    SEED = 777
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Format is handled by rich
    # Allows compatibility with progress bars
    logger.configure(
        handlers=[{"sink": RichHandler(markup=True),
                   "format": "{message}",
                   "level": "DEBUG"}],
    )
    with logger.catch():
        train_model()
        uvicorn.run(
            "nlp_engineer_assignment.api:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            workers=1
        )
