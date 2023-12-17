
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Type

import torch.nn as nn
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import RedirectResponse
from typing_extensions import Annotated

from nlp_engineer_assignment import TransformerTokenClassification, load_model, predict_text, set_logger

MODEL_CLASS_MAP = {
    TransformerTokenClassification.__name__: TransformerTokenClassification,
}


class TextRequest(BaseModel):
    text: str


class Settings(BaseSettings):
    artifacts_dir: str
    clf_model_name: str
    clf_model_class_name: Annotated[
        str, "'__name__' of the model class to use"]


def get_model_class(settings: Settings) -> Type[nn.Module]:
    clf_model_class_name = settings.clf_model_class_name

    if clf_model_class_name in MODEL_CLASS_MAP:
        return MODEL_CLASS_MAP[clf_model_class_name]

    raise ValueError(f"Unknown model class: {clf_model_class_name}")


@lru_cache
def get_settings():
    return Settings()


@lru_cache
def get_model():

    settings = get_settings()
    clf_model_class = get_model_class(settings)

    model, vocabs_mapping, _ = load_model(
        artifacts_dir=settings.artifacts_dir,
        model_name=settings.clf_model_name,
        model_class=clf_model_class,
    )
    return model, vocabs_mapping


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize FastAPI and add variables

    """

    set_logger(level="DEBUG")
    logger.info("Initializing FastAPI app")

    settings = get_settings()
    clf_model_name = settings.clf_model_name
    # Preload the model
    get_model()
    logger.info("Loaded model: '{}'", clf_model_name)

    yield

    logger.info("Shutting down FastAPI app")


app = FastAPI(
    title="NLP Engineer Assignment",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def index():
    """
    Redirects to the OpenAPI Swagger UI
    """
    return RedirectResponse(url="/docs")


@app.get('/info')
async def info(
    settings: Annotated[
        Settings, Depends(get_settings)],
    model_resources: Annotated[
        tuple[nn.Module, dict[str, int]], Depends(get_model),],
):
    """Returns information about the model, what it does and how to use it for inference.

    """

    n_tokens = model_resources[0].n_tokens
    model_description = "Predicts the number of occurrences of each letter in the text up to that point."
    format_requirements = (f"Text should be tokenizable to exactly {n_tokens} tokens. "
                           "Unknown substrings are mapped to a single <UNK> token.")
    return {
        "model_class": settings.clf_model_class_name,
        "model_name": settings.clf_model_name,
        "model_version": "1.0.0",
        "model_type": "token classification",
        "model_description": model_description,
        "how_to_call": "POST /predict",
        "input": {
            "text": "string",
            "format_requirements": format_requirements
        },
        "output": {
            "prediction": "string"
        }

    }


@app.post("/predict")
def predict(
    input_data: TextRequest,
    model_resources: Annotated[
        tuple[nn.Module, dict[str, int]], Depends(get_model),
    ],
):
    """
    Predict the number of occurrences of each letter in the text up to that point.

    Parameters
    ----------
    input_data : TextRequest
        The text to predict on.
        Must be tokenizable to a fixed number of tokens compatible with the model, using its vocabulary.
        Unknown substrings are mapped to **a single** <UNK> token.

    Notes
    -----
    The method `evaluate_classifier` from `nlp_engineer_assignment` is not used.
    That method instantiate a dataset and a dataloader.
    As we expect a single line per request, this is not necessary.

    A new route could be added, or this one modified, if requirements evolve.

    """

    model, vocabs_mapping = model_resources

    text = input_data.text

    try:
        preds = predict_text(
            text=text,
            model=model,
            vocabs_mapping=vocabs_mapping
        )
    except ValueError as e:
        logger.error(e)
        logger.error("Invalid input: '{}'", text)
        raise HTTPException(status_code=422, detail=str(e))

    predictions_string = "".join(str(p) for p in preds)

    return {"prediction": predictions_string}
