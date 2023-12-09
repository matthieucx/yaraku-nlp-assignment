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
- autopep8 is used for autoformating
- flake8 is used for linting

All 3 of them can be run using your IDE, which is recommended.

### Pre-commit

To install pre-commits, run `pre-commit install`.

Autoformating ensures a consistent style throughout the codebase. It makes sense as a pre-commit because:

- All committed code will be formatted, streamlining parsing and reviewing
- Autoformating does not prevent commits

Pre-commits should not prevent commits as this hinders workflows. Linting and testing are blocking, requiring errors to be fixed. They are better used in a CI pipeline.

Files modified by a pre-commit hook are not staged automatically, on purpose ([see this post by the tool creator](https://stackoverflow.com/questions/64309766/prettier-using-pre-commit-com-does-not-re-stage-changes/64309843#64309843)).

## Docker

A Dockerfile is included in the repository to ensure that it works across different environments. You can consult how to work with Dockerfiles at the [Docker documentation](https://docs.docker.com).
