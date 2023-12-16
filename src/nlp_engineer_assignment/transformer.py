from loguru import logger
from rich.progress import Progress
import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from .dataset import TokenClassificationDataset
from .utils import score
from itertools import chain


class BasicLayerNorm(nn.Module):
    """Layer Normalization, as defined in `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_.

    This implementation is simplified to only normalize the last dimension.
    This is suitable for the Transformer model ; each token is normalized independently.

    In Layer Normalization, the inputs to a layer are normalized.
    This is done for each sample in a batch.
    It allows for more stable training and accelerates convergence.

    After normalization, a trainable gain and bias is applied to the input.

    Parameters
    ----------
    normalized_shape : int
        The dimensions of the elements to normalize.
        Here, the implementation is simplified to only normalize the last dimension.
    eps : float
        Small value added to the variance for numerical stability.

    Attributes
    ----------
    gain : torch.Parameter
        Trainable gain.
    bias : torch.Parameter
        Trainable bias.
    eps : float
        Small value added to the variance for numerical stability.

    """

    def __init__(self, normalized_shape: int, eps=1e-5) -> None:
        super().__init__()

        self.gain = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.eps = eps

    def forward(self, x):
        """Compute the layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (*, embedding).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (*, embedding).
        """

        x_mean = x.mean(axis=-1, keepdims=True)
        x_std = x.std(axis=-1, keepdims=True, unbiased=False)
        x_normalized = (x - x_mean) / (x_std + self.eps)

        return self.gain * x_normalized + self.bias


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    The dot products of the queries and keys are scaled by the inverse square root of their dimension ( 1/sqrt(d_k) ).
    This avoids the saturation of the softmax function due to large dot products and preserves gradient flow.

    Parameters
    ----------
    d_k : int
        The dimension of the queries and keys.

    """

    def __init__(self, d_k: int):
        super().__init__()

        self.d_k = d_k

    def forward(self, queries, keys, values):
        """Compute the attention scores.

        Parameters
        ----------
        queries : torch.Tensor
            The queries, shape (batch, tokens, d_k).
        keys : torch.Tensor
            The keys, shape (batch, tokens, d_k).
        values : torch.Tensor
            The values, shape (batch, tokens, d_v).

        Returns
        -------
        torch.Tensor
            Computed attention scores, shape (batch, tokens, d_v).

        """

        dot = torch.bmm(queries, keys.transpose(1, 2))
        # Scale dot product
        dot = dot / (self.d_k ** (1/2))
        # Normalize
        attention_weights = F.softmax(dot, dim=2)

        return torch.bmm(attention_weights, values)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Attention as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    As in the paper, this implementation use the same dimension for queries/keys (d_k) and values (d_v).
    This dimension is equal to the head dimension, set as emb // heads.

    Parameters
    ----------
    emb : int
        Embedding dimension of the input (d_model).
    heads : int
        Number of attention heads (h), used to set the head dimension (d_k) as emb // heads.

    """

    def __init__(self, emb: int, heads: int) -> None:
        super().__init__()

        assert emb % heads == 0, (
            f"Embedding dimension must be divisible by the number of heads, Found emb: {emb}, heads: {heads}."
        )

        self.emb = emb
        self.heads = heads
        self.head_dim = emb // heads

        self.to_queries = nn.Linear(emb, emb, bias=False)
        self.to_keys = nn.Linear(emb, emb, bias=False)
        self.to_values = nn.Linear(emb, emb, bias=False)

        self.attention = ScaledDotProductAttention(self.head_dim)

        self.merge_heads = nn.Linear(emb, emb, bias=False)

    def forward(self, x):
        """Compute the multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, tokens, embedding).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, tokens, embedding).

        """

        b, t, _ = x.size()  # batch, tokens, embedding
        h = self.heads
        d_k = self.head_dim

        # Compute Q, K, V for all heads
        queries = self.to_queries(x).view(b, t, h, d_k)
        keys = self.to_keys(x).view(b, t, h, d_k)
        values = self.to_values(x).view(b, t, h, d_k)
        # Q, K, V: (batch, tokens, heads, head_dim)

        # We combine batch and heads for batch matrix multiplication
        # Computations will be done independently for each head, in parallel
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, d_k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, d_k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, d_k)
        # Q, K, V: (batch * heads, tokens, head_dim)

        attention_scores = self.attention(
            queries=queries, keys=keys, values=values)
        # Attention scores: (batch * heads, tokens, head_dim)

        concat_heads = attention_scores.view(b, h, t, d_k).transpose(
            1, 2).contiguous().view(b, t, h * d_k)
        # Concatenated heads: (batch, tokens, heads * head_dim = embedding)

        return self.merge_heads(concat_heads)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    The layer is made up of two sub-layers:
    - a multi-head attention mechanism
    - a position-wise (applied to each token) fully connected feed-forward network, source of non-linearity

    After each sub-layer, residual connections help preserve the gradient flow.
    Layer normalization stabilizes and speeds up the training process. Both allow for deeper networks.

    Regularization is achieved through dropout on the output of each sub-layer, before the sum with the residuals.

    Parameters
    ----------
    emb : int
        Embedding dimension of the input.
    heads : int
        Number of attention heads.
    dim_ff : int
        Dimension of the feedforward layer (d_ff). Set to 4 * emb in the paper.
        dim_ff is typically bigger than emb, allowing for a more expressive transformation.
    dropout_rate : float
        Dropout rate.

    """

    def __init__(self, emb: int, heads: int, dim_ff: int = None, dropout_rate: float = 0.1):
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.dim_ff = dim_ff if dim_ff is not None else 4 * emb

        # Sub-layer 1: Multi-head attention
        self.multi_attention = MultiHeadSelfAttention(emb=emb, heads=heads)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_1 = BasicLayerNorm(normalized_shape=emb)

        # Sub-layer 2: Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(emb, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, emb)
        )
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = BasicLayerNorm(normalized_shape=emb)

    def forward(self, x):
        """Compute a forward pass through the Transformer encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, tokens, embedding).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, tokens, embedding).

        """

        attended = self.multi_attention(x)
        x = self.dropout_1(attended) + x
        x = self.layer_norm_1(x)

        ff_out = self.ff(x)
        x = self.dropout_2(ff_out) + x
        x = self.layer_norm_2(x)

        return x


class TransformerEmbeddings(nn.Module):
    """Embeddings for a Transformer model.

    This implementation assumes a fixed sequence length, and does not use padding.

    This class implements the embedding layer and positional encoding.
    Embeddings provide the model with information about the meaning of each token.
    Positional encoding is achieved with positional embeddings, as used for `BERT <https://arxiv.org/abs/1810.04805>`_.
    Positional embeddings provide the model with information about the relative position of tokens in the sequence.
    This is crucial for the given task, as the model needs to understand cooccurrences of the same token.

    Parameters
    ----------
    vocab_size : int
        Number of unique tokens in the vocabulary.
    emb : int
        Embedding dimension of the input.
    n_tokens : int
        Number of tokens in each sequence.
    dropout_rate : float
        Dropout rate.
    """

    def __init__(self, vocab_size: int, emb: int, n_tokens: int, dropout_rate: float = 0.1):
        super().__init__()

        self.tok_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=emb)
        self.pos_embedding = nn.Embedding(
            num_embeddings=n_tokens, embedding_dim=emb)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Compute the embeddings.

        Each token is mapped to an embedding vector.
        Each possible position is mapped to a positional embedding vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, tokens). Each value is an index in the vocabulary.

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, tokens, embedding). Embeddings for each token.

        """

        position_indices = torch.arange(
            x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        tokens = self.tok_embedding(x)
        positions = self.pos_embedding(position_indices)

        return self.dropout(tokens + positions)


class TransformerTokenClassification(nn.Module):
    """Transformer encoder with specialized layer for token classification.

    This implementation assumes a fixed sequence length, and does not use masking or padding.

    The model is a Transformer encoder, as defined in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    Positional encoding is achieved with positional embeddings, as used for `BERT <https://arxiv.org/abs/1810.04805>`_.
    Positional embeddings provide the model with information about the relative position of tokens in the sequence.
    This is crucial for the given task, as the model needs to understand cooccurrences of the same token.

    It is specialized for token classification tasks, through the use of a linear layer.
    Each token in the output is labeled in a single forward pass.

    Parameters
    ----------
    depth : int
        Number of encoder layers.
    emb : int
        Embedding dimension of the input.
    heads : int
        Number of attention heads.
    dim_ff : int
        Dimension of the feedforward layer (d_ff). Set to 4 * emb in the paper.
    vocab_size : int
        Number of unique tokens in the vocabulary.
    n_tokens : int
        Number of tokens in each sequence.
    n_classes : int
        Number of classes to predict.
    dropout_rate : float
        Dropout rate.

    """

    def __init__(
            self,
            depth: int,
            emb: int,
            heads: int,
            dim_ff: int,
            vocab_size: int,
            n_tokens: int,
            n_classes: int,
            dropout_rate: float = 0.1
    ):
        super().__init__()

        self.emb = emb
        self.heads = heads
        self.dim_ff = dim_ff

        self.embedding = TransformerEmbeddings(
            vocab_size=vocab_size, emb=emb, n_tokens=n_tokens, dropout_rate=dropout_rate)

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderLayer(emb=emb, heads=heads, dim_ff=dim_ff) for _ in range(depth)]
        )

        # Compute logits for each class, for each token
        self.to_classes = nn.Linear(emb, n_classes)

    def forward(self, x):
        """

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, tokens). Each value is an index in the vocabulary.

        Returns
        -------
        torch.Tensor
            Output tensor, shape (batch, tokens, n_classes). Logits for each class, per token.

        """

        x = self.embedding(x)
        x = self.transformer_encoder(x)

        class_logits = self.to_classes(x)

        return class_logits


def train_epoch(model: TransformerTokenClassification,
                dataloader: DataLoader[TokenClassificationDataset],
                optimizer,
                criterion,) -> float:
    """Train the model using the training dataloader for one epoch.

    Parameters
    ----------
    model : TransformerTokenClassification
        Model to train.
    dataloader : DataLoader[TokenClassificationDataset]
        Training dataloader.
    optimizer
        Optimizer.
    criterion
        Loss function.

    Returns
    -------
    batch_losses : list[float]
        Training loss for each batch in the epoch.
    batch_accuracies : list[float]
        Training accuracy for each batch in the epoch.

    """

    model.train()
    batch_losses = []
    batch_accuracies = []

    for batch in dataloader:

        indices = batch["indices"]
        targets = batch["target_seq"]
        curr_batch_size, curr_n_tokens = indices.size()

        optimizer.zero_grad()

        logits = model(indices)
        targets_loss = targets.view(curr_batch_size * curr_n_tokens)
        logits_loss = logits.view(curr_batch_size * curr_n_tokens, -1)
        loss = criterion(logits_loss, targets_loss)

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

        _, pred = logits.max(dim=-1)
        batch_accuracies.append(100 * score(targets.numpy(), pred.numpy()))

    return batch_losses, batch_accuracies


def validate_epoch(model: TransformerTokenClassification,
                   dataloader: DataLoader[TokenClassificationDataset],
                   criterion) -> float:
    """Validate the model using the validation dataloader for one epoch.

    Parameters
    ----------
    model : TransformerTokenClassification
        Model to validate.
    dataloader : DataLoader[TokenClassificationDataset]
        Validation dataloader.
    criterion
        Loss function.

    Returns
    -------
    batch_losses : list[float]
        Validation loss for each batch in the epoch.
    batch_accuracies : list[float]
        Validation accuracy for each batch in the epoch.

    """

    model.eval()
    batch_losses = []
    batch_accuracies = []

    with torch.no_grad():
        for batch in dataloader:

            indices = batch["indices"]
            targets = batch["target_seq"]
            curr_batch_size, curr_n_tokens = indices.size()

            logits = model(indices)
            targets_loss = targets.view(curr_batch_size * curr_n_tokens)
            logits_loss = logits.view(curr_batch_size * curr_n_tokens, -1)
            loss = criterion(logits_loss, targets_loss)

            batch_losses.append(loss.item())

            _, pred = logits.max(dim=-1)
            batch_accuracies.append(100 * score(targets.numpy(), pred.numpy()))

    return batch_losses, batch_accuracies


def evaluate_classifier(model: TransformerTokenClassification,
                        test_dataset: TokenClassificationDataset,
                        batch_size: int = 256) -> torch.Tensor:
    """Returns the predictions of the model on the test set, for further evaluation.

    Parameters
    ----------
    model : TransformerTokenClassification
        Model to test.
    test_dataset : TokenClassificationDataset
        Test dataset.
    batch_size : int, optional
        Batch size.

    Returns
    -------
    torch.Tensor
        Predictions of the model on the test set, shape (num_samples, n_classes).

    """

    model.eval()

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if len(test_dataset) == 0:
        return torch.tensor([], dtype=torch.long)

    predictions = []

    with torch.no_grad():
        for batch in dataloader:

            indices = batch["indices"]
            pred = model(indices)
            pred = pred.argmax(dim=-1)
            predictions.append(pred)

    num_samples = len(test_dataset)
    n_tokens = test_dataset[0]["indices"].numel()

    # We have a fixed sequence length, we can concatenate the predictions and view as below
    predictions = torch.cat(predictions)

    return predictions.view(num_samples, n_tokens)


def train_classifier(
    train_dataset: TokenClassificationDataset,
    hparams: dict = None,
) -> (TransformerTokenClassification, dict[str, any]):
    """Train a TransformerTokenClassification model.

    Parameters
    ----------
    train_dataset : TokenClassificationDataset
        Training dataset.
    hparams : dict, optional
        Hyperparameters for the model and training process.

    Returns
    -------
    model: TransformerTokenClassification
        Trained model.
    artifacts : dict
        Dictionary of artifacts from the model training process.
        Includes: model parameters, vocabulary mapping, training curves, training metrics, validation metrics.

    Raises
    ------
    ValueError
        If the training dataset is empty.

    """
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")

    ###
    # Model parameters
    ###

    # Default hyperparameters if not specified
    if hparams is None:
        hparams = {}

    hparams['epochs'] = hparams.get('epochs', 8)
    hparams['batch_size'] = hparams.get('batch_size', 256)
    hparams['learning_rate'] = hparams.get('learning_rate', 5e-3)
    hparams['depth'] = hparams.get('depth', 2)
    hparams['emb'] = hparams.get('emb', 64)
    hparams['heads'] = hparams.get('heads', 4)
    hparams['dim_ff'] = hparams.get('dim_ff', 256)
    hparams['dropout_rate'] = hparams.get('dropout_rate', 0.1)

    n_tokens = train_dataset[0]["indices"].numel()
    n_classes = train_dataset.n_classes

    ###
    # Setup
    ###

    train_size = int(0.9 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train, val = random_split(train_dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train, batch_size=hparams["batch_size"], shuffle=True)
    val_dataloader = DataLoader(
        val, batch_size=hparams["batch_size"], shuffle=True)

    vocab_size = len(train_dataset.vocabs_mapping)

    model = TransformerTokenClassification(
        depth=hparams['depth'],
        emb=hparams['emb'],
        heads=hparams['heads'],
        dim_ff=hparams['dim_ff'],
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        n_classes=n_classes,
        dropout_rate=hparams['dropout_rate']
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hparams['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Train and validate model
    logger.info("Starting training")

    with Progress() as progress:
        task_train = progress.add_task(
            "[green]Training...[/]", total=hparams['epochs']
        )

        for epoch in range(hparams['epochs']):

            batch_train_losses, batch_train_accuracy = train_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion
            )
            train_losses.append(batch_train_losses)
            train_accuracies.append(batch_train_accuracy)

            batch_val_losses, batch_val_accuracy = validate_epoch(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion
            )
            val_losses.append(batch_val_losses)
            val_accuracies.append(batch_val_accuracy)

            progress.advance(task_train)
            logger.debug(
                "Epoch {}/{}: Train loss: {:.4f}, Train acc: {:.2f} / Val loss: {:.4f}, Val acc: {:.2f}",
                epoch+1, hparams['epochs'],
                sum(batch_train_losses)/len(train_dataloader),
                sum(batch_train_accuracy)/len(train_dataloader),
                sum(batch_val_losses)/len(val_dataloader),
                sum(batch_val_accuracy)/len(val_dataloader)
            )

    logger.info("Finished training")

    model_params = {
        "training": {
            "epochs": hparams['epochs'],
            "batch_size": hparams['batch_size'],
            "learning_rate": hparams['learning_rate']
        },
        "model": {
            "depth": hparams['depth'],
            "emb": hparams['emb'],
            "heads": hparams['heads'],
            "dim_ff": hparams['dim_ff'],
            "dropout_rate": hparams['dropout_rate'],
            "n_classes": n_classes,
            "n_tokens": n_tokens,
            "vocab_size": vocab_size,
        }
    }

    # Flatten lists of lists
    fig = plot_training_curves(
        train_losses=list(chain.from_iterable(train_losses)),
        val_losses=list(chain.from_iterable(val_losses)),
        train_metric=list(chain.from_iterable(train_accuracies)),
        val_metric=list(chain.from_iterable(val_accuracies)),
        metric_name="Accuracy", loss_name="Cross-Entropy",
        epochs=len(train_losses)
    )

    train_metrics = {
        "loss": train_losses,
        "accuracy": train_accuracies
    }

    val_metrics = {
        "loss": val_losses,
        "accuracy": val_accuracies
    }

    artifacts = {
        "model_params": model_params,
        "vocabs_mapping": train_dataset.vocabs_mapping,
        "training_curves": fig,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics
    }

    return model, artifacts


def plot_training_curves(train_losses, val_losses, train_metric, val_metric, metric_name="Performance metric",
                         loss_name="Loss function", epochs=None):
    """Plot the training/validation loss and performance metric curves.

    Parameters
    ----------
    train_losses : list[float]
        Training loss for each batch in each epoch.
    val_losses : list[float]
        Validation loss for each batch in each epoch.
    train_metric : list[float]
        Training metric for each batch in each epoch.
    val_metric : list[float]
        Validation metric for each batch in each epoch.
    metric_name : str, optional
        Name of the performance metric.
    metric_name : str
        Name of the performance metric.
    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 12))

    if epochs is None:
        epochs = 1
        logger.warning(
            "Epochs not specified, using 1 epoch for the training curves"
        )

    epochs_train = np.linspace(0, epochs, len(train_losses))
    epochs_val = np.linspace(0, epochs, len(val_losses))

    # Plotting training and validation loss
    ax1.plot(epochs_train, train_losses,
             label=f"Training {loss_name.lower()} loss")
    ax1.plot(epochs_val, val_losses,
             label=f"Validation {loss_name.lower()} loss", color="orange")

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel(f"{loss_name} loss")
    ax1.legend(loc='upper right')
    ax1.set_title(
        f"Training and validation {loss_name.lower()} loss per epoch")

    # Plotting performance metric
    ax2.plot(epochs_train, train_metric,
             label=f"Training {metric_name.lower()}")
    ax2.plot(epochs_val, val_metric,
             label=f"Validation {metric_name.lower()}", color="orange")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(f"{metric_name}")
    ax2.legend(loc='lower right')
    ax2.set_title(f"Training and validation {metric_name.lower()} per epoch")

    plt.tight_layout()

    return fig


def objective(trial, train_dataset: TokenClassificationDataset):
    """Objective function for Optuna.

    The metric to minimize is the average validation loss for the last epoch.
    To speed up the optimization, the number of epochs is small.
    Minimizing on loss from the last epoch avoid overly favoring large learning rates.
    For the assignment, 5 epochs is enough to approach the final loss.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial.
    train_dataset : TokenClassificationDataset
        Training dataset.

    Returns
    -------
    float
        Average validation loss for the last epoch.
        Used by the Optuna study as the metric to minimize.

    """

    # Set ranges of hyperparameters to sample from
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256, 512])
    learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    depth = trial.suggest_int('num_layers', 1, 2)
    emb = trial.suggest_categorical('emb_size', [32, 64, 96, 128])
    heads = trial.suggest_categorical('heads', [1, 2, 4])
    dim_ff = trial.suggest_categorical('dim_ff', [128, 256, 512])
    dropout_rate = trial.suggest_float('dropout_rate', 0.001, 0.3, log=True)

    epochs = 5

    hparams = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'depth': depth,
        'emb': emb,
        'heads': heads,
        'dim_ff': dim_ff,
        'dropout_rate': dropout_rate
    }

    _, artifacts = train_classifier(
        train_dataset=train_dataset,
        hparams=hparams
    )

    last_epoch_losses = artifacts["val_metrics"]["loss"][-1]

    return sum(last_epoch_losses) / len(last_epoch_losses)


def optimize_classifier(
        train_dataset: TokenClassificationDataset,
        seed: int = 777,
        n_trials: int = 20,
        verbose: bool = False) -> dict:
    """Optimize the hyperparameters of a TransformerTokenClassification model.

    Parameters
    ----------
    train_dataset : TokenClassificationDataset
        Training dataset.
    seed : int, optional
        Used to set the seed of the Optuna sampler.
    n_trials : int, optional
        Number of trials for the Optuna study.
        The TPESampler is used, with 10 warmup trials or half the number of trials, whichever is smaller.
    verbose : bool, optional
        Whether to display Optuna logs.

    Returns
    -------
    best_params : dict
        The best hyperparameters found by Optuna.

    """

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("Starting hyperparameter optimization")

    study = optuna.create_study(
        study_name="Tune TransformerTokenClassifier",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=min(10, n_trials // 2)
        ),
    )

    study.optimize(
        lambda trial: objective(trial, train_dataset),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    logger.info("Finished hyperparameter optimization")

    best_params = study.best_trial.params

    logger.info("Best trial: {}", best_params)

    return best_params
