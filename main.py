import numpy as np
import os
import uvicorn
from loguru import logger
from rich.logging import RichHandler
import torch
import json

from nlp_engineer_assignment import print_line, read_inputs, \
    score, train_classifier, TokenClassificationDataset, evaluate_classifier, optimize_classifier, \
    load_model, TransformerTokenClassification, save_artifacts, load_hparams


def main(seed: int = 777):
    """Main function for the assignment.

    Runs the following steps:
    1. Load the model if it exists
    2a. If the model does not exist, check if optimal hyperparameters exist
        2b. If optimal hyperparameters do not exist, run an optimization procedure
    3. Train the model using the optimal hyperparameters

    The model is saved in the artifacts directory.

    Parameters:
    ----------
    seed : int
        The random seed to use. Default is 777.
        It is propagated to Optuna, PyTorch, and NumPy.

    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(cur_dir, "artifacts")
    data_dir = os.path.join(cur_dir, "data")

    os.makedirs(artifacts_dir, exist_ok=True)

    model_name = "optimal_model"

    try:
        model, vocabs_mapping, _ = load_model(
            artifacts_dir=artifacts_dir,
            model_name=model_name,
            model_class=TransformerTokenClassification
        )
        logger.info("Found {} in {}", model_name, artifacts_dir)

        return model, vocabs_mapping

    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Model files not found. Proceeding to train a new model.")

    hparams_name = "optimal_hparams.json"
    hparams = load_hparams(
        artifacts_dir=artifacts_dir,
        hparams_name=hparams_name
    )

    train_model(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        hparams=hparams,
        seed=seed
    )

    evaluate_model(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        model_name=model_name
    )


def train_model(
        data_dir: str,
        artifacts_dir: str,
        model_name: str,
        hparams: dict[str, any] = None,
        seed: int = 777):
    """Trains a model using the optimal hyperparameters and saves it.

    If the hyperparameters are not provided, they are optimized using Optuna.

    Parameters:
    ----------
    data_dir : str
        The directory where the data is stored.
    artifacts_dir : str
        The directory where the artifacts are stored.
    model_name : str
        The name of the model.
    hparams : dict[str, any]
        The hyperparameters to use. If not provided, they are optimized.
    seed : int
        The random seed to use. Default is 777.

    """

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']

    ###
    # Train
    ###

    train_data_path = os.path.join(data_dir, "train.txt")
    train_inputs = read_inputs(train_data_path)
    train_dataset = TokenClassificationDataset(train_inputs, vocabs)

    if hparams is None:

        logger.info("Searching for optimal parameters...")

        hparams = optimize_classifier(
            train_dataset, n_trials=30, seed=seed)
        with open(os.path.join(artifacts_dir, "optimal_hparams.json"), "w") as f:
            json.dump(hparams, f, indent=4)

        logger.info("Found and saved optimal parameters")
        print_line()

    hparams["epochs"] = 10
    logger.info("Training model using parameters found...")

    model, artifacts = train_classifier(
        train_dataset=train_dataset,
        hparams=hparams
    )

    additional_artifacts = {
        k: v for k, v in artifacts.items()
        if k not in ["model_params", "vocabs_mapping"]
    }

    save_artifacts(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        model=model,
        model_params=artifacts["model_params"],
        vocabulary_mapping=artifacts["vocabs_mapping"],
        additional_artifacts=additional_artifacts
    )


def evaluate_model(
        data_dir: str,
        artifacts_dir: str,
        model_name: str):
    """Evaluates the model on the test set.

    Results are logged to the console.

    Parameters:
    ----------
    data_dir : str
        The directory where the data is stored.
    artifacts_dir : str
        The directory where the artifacts are stored.
    model_name : str
        The name of the model.

    """

    model, vocabs_mapping, _ = load_model(
        artifacts_dir=artifacts_dir,
        model_name=model_name,
        model_class=TransformerTokenClassification
    )

    test_data_path = os.path.join(data_dir, "test.txt")
    test_inputs = read_inputs(test_data_path)
    test_dataset = TokenClassificationDataset(
        text_data=test_inputs,
        vocabs_mapping=vocabs_mapping
    )

    logger.info("Testing model...")

    pred = evaluate_classifier(
        model=model,
        test_dataset=test_dataset
    )
    pred_np = pred.numpy()
    golds = np.array(
        [sample["target_seq"] for sample in test_dataset]
    )

    # Display a sample input and its prediction
    sample_idx = np.random.randint(0, len(test_inputs))
    logger.info("Sample input: {}", sample_idx)
    sample = test_dataset[sample_idx]
    logger.info("Input: {}", sample["text"])
    logger.info("Gold: {}", sample["target_seq"].tolist())
    logger.info("Pred: {}", pred_np[sample_idx].tolist())

    logger.info("Test Accuracy: {:.2f}%", 100.0 * score(golds, pred_np))


def set_logger(level: str = "INFO"):
    """Sets the logger configuration.

    Sets the loguru configuration to use a RichHandler.
    Allows for compatibility with Rich tools such as the progress bar.

    Parameters:
    ----------
    level : str
        The logging level to use. Default is "INFO".

    """
    rich_handler = {
        "sink": RichHandler(markup=True),
        "format": "{message}",
        "level": level,
    }
    logger.configure(
        handlers=[rich_handler],
    )


if __name__ == "__main__":

    SEED = 777
    set_logger(level="INFO")

    with logger.catch():
        main(seed=SEED)
        exit(0)
        uvicorn.run(
            "nlp_engineer_assignment.api:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            workers=1
        )
