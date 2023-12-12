# NLP Engineer Assignment

This repository contains my submission for the coding assignment for the NLP Engineer position at Yaraku, Inc.

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

- Train a transformer model on the assignment problem and train data
- Evaluate it on the test data
- Set up an API for inference using that model

Navigate to http://localhost:8000/docs to access the Swagger UI and test the API.

## Development tooling

- pytest is used for unit testing
- autopep8 is used for autoformatting
- flake8 is used for linting

All 3 of them can be run using your IDE, which is recommended. They can also be invoked from the CLI, using `poetry run <tool>`.

Autoformatting is executed as a pre-commit hook, ensuring a consistent style throughout the codebase.

### Pre-commit

To install pre-commit hooks, run `poetry run pre-commit install`.

Pre-commit hooks will be run on staged files when committing. You can run them whenever you want with `poetry run pre-commit run`.

## Docker

A Dockerfile is included in the repository to ensure that it works across different environments. You can consult how to work with Dockerfiles at the [Docker documentation](https://docs.docker.com).
