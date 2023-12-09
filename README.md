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

## Docker

A Dockerfile is included in the repository to ensure that it works across different environments. You can consult how to work with Dockerfiles at the [Docker documentation](https://docs.docker.com).
