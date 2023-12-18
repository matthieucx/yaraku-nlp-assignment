# NLP Engineer Assignment

This repository contains my submission for the coding assignment for the NLP Engineer position at Yaraku, Inc.

## Tasks completed

1) Transformer-based token classification model

- Multi-head attention layer
- Positional embeddings
- Layer normalization
- Transformer encoder
- Transformer-based token classification model

2) Model training and evaluation

- Input data tokenization and encoding
- Model training and evaluation
- Hyperparameter tuning

3) Model serving

- Model serving through a FastAPI app

The trained model achieves a reasonable performance on the test data, with over 99% accuracy with the provided scoring function. Tuned hyperparameters can be found in the `artifacts/` folder.

## Workflow

Given the scope of this project, and to facilitate reviewing its content after the assignment has been submitted, it has been kept on a single branch. Commit messages are descriptive and should allow for a clear understanding of the development process.

Some notes on the technical infrastructure used for this project can be found in the `technical_decisions.md` file.

## Assignment assumptions

While the assignment's boundaries were clearly defined, I tried to keep the code as generic and reusable as possible. This can be seen in the `nlp_engineer_assignment.utils.tokenize()` function, among others.

The following simplifications were made:

- Models are designed to work with a fixed-size input. Padding and masking are not implemented.
- Model training does not require a GPU, given the assignment requirements. The model code should be device-agnostic, but the training code does not take GPUs into consideration.

While this does not apply to the provided data, substrings that cannot be tokenized are mapped to a single `<UNK>` token, enabling the model to work with unseen data. As masking is not implemented, all ignored substrings are thus treated as a single regular token.

## Requirements

- Python (completed and tested with version `3.10.13`)
- Poetry (completed and tested with version `1.6.1`)
- (Optional) Docker

## Setup

1. Start by cloning the repository into your local environment.
2. Install poetry in your local environment by running: `pip install poetry`
3. Create the virtual environment for the project by running: `poetry install`
4. Initialize the virtual environment by running: `poetry shell`
5. Run the entrypoint script with: `python main.py`

This last step will:

- Train a transformer model on the assignment problem and train data, using tuned hyperparameters
- Evaluate it on the test data
- Set up a FastAPI app to serve the model

Navigate to http://localhost:8000/docs to access the Swagger UI and test the API.

## Development tooling

- pytest is used for unit testing
- autopep8 is used for autoformatting
- flake8 is used for linting
- isort is used for sorting imports

All 4 of them can be run using your IDE, which is recommended. They can also be invoked from the CLI, using `poetry run <tool>`.

Autoformatting and import sorting are executed as pre-commit hooks, ensuring a consistent style throughout the codebase.

### Pre-commit

To install pre-commit hooks, run `poetry run pre-commit install`.

Pre-commit hooks will be run on staged files when committing. You can run them whenever you want with `poetry run pre-commit run`.

In addition to autoformatting and import sorting, pre-commit hooks will also check for:
- Trailing whitespaces
- End of file newline
- Known typos in the codebase
- Up-to-date `poetry.lock` file (in sync with `pyproject.toml`)

## Docker

A Dockerfile is included in the repository to ensure that it works across different environments. You can consult how to work with Dockerfiles at the [Docker documentation](https://docs.docker.com).
