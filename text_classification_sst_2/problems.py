from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from utils import problem


@problem.tag("hw4-B", start_line=1)
def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a batch of data given a list of N sequences and labels. Sequences are stacked into a single tensor
    of shape (N, max_sequence_length), where max_sequence_length is the maximum length of any sequence in the
    batch. Sequences shorter than this length should be filled up with 0's. Also returns a tensor of shape (N, 1)
    containing the label of each sequence.

    Args:
        batch ([type]): A list of size N, where each element is a tuple containing a sequence LongTensor and a single item
            LongTensor containing the true label of the sequence.

    Returns:
        Tuple[torch.LongTensor, torch.LongTensor]: A tuple containing two tensors (both LongTensor).
            The first tensor has shape (N, max_sequence_length) and contains all sequences.
            Sequences shorter than max_sequence_length are padded with 0s at the end.
            The second tensor has shape (N, 1) and contains all labels.
    """

    from torch.nn.utils.rnn import pad_sequence

    feature_list, label_list = [], []
    for x,y in batch:
        feature_list.append(x)
        label_list.append(y)
    feat_ts = pad_sequence(feature_list, batch_first=True, padding_value=0)
    label_ts = torch.tensor(label_list, dtype=torch.long)
    return feat_ts, label_ts


class RNNBinaryClassificationModel(nn.Module):
    @problem.tag("hw4-B", start_line=10)
    def __init__(self, embedding_matrix: torch.Tensor, rnn_type: str):
        """Create a model with either RNN, LSTM or GRU layer followed by a linear layer.

        Args:
            embedding_matrix (torch.Tensor): Weights for embedding layer.
                Used in starter code, nothing you should worry about.
            rnn_type (str): Either "RNN", "LSTM" or "GRU". Defines what kind of a layer should be used.
        """
        super().__init__()

        vocab_size = embedding_matrix.shape[0]
        embedding_dim = embedding_matrix.shape[1]

        # Construct embedding layer and initialize with given embedding matrix. Do not modify this code.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.embedding.weight.data = embedding_matrix

        # My codes
        if rnn_type == 'LSTM':
            self.model = nn.LSTM(embedding_dim, 64, 2, batch_first=True)
        if rnn_type == 'RNN':
            self.model = nn.RNN(embedding_dim, 64, 2, batch_first=True)
        if rnn_type == 'GRU':
            self.model = nn.GRU(embedding_dim, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.loss_fun = nn.BCEWithLogitsLoss()

    @problem.tag("hw4-B")
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Takes in a batch of data of shape (N, max_sequence_length). Returns a tensor of shape (N, 1), where each
        element corresponds to the prediction for the corresponding sequence.

        Args:
            inputs (torch.Tensor): FloatTensor of shape (N, max_sequence_length) containing N sequences to make predictions for.

        Returns:
            torch.Tensor: FloatTensor of predictions for each sequence of shape (N, 1).
        """
        x = self.embedding(inputs)
        x = self.model(x)
        x = x[0][:,-1]
        x = self.fc(x)
        return x

    @problem.tag("hw4-B")
    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the binary cross-entropy loss.

        Args:
            logits (torch.Tensor): FloatTensor - Raw predictions from the model of shape (N, 1)
            targets (torch.Tensor): LongTensor - True labels of shape (N, 1)

        Returns:
            torch.Tensor: Binary cross entropy loss between logits and targets as a single item FloatTensor.
        """
        return self.loss_fun(logits.flatten(), targets.float())

    @problem.tag("hw4-B")
    def accuracy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy, i.e number of correct predictions / N.

        Args:
            logits (torch.Tensor): FloatTensor - Raw predictions from the model of shape (N, 1)
            targets (torch.Tensor): LongTensor - True labels of shape (N, 1)

        Returns:
            torch.Tensor: Accuracy as a scalar FloatTensor.
        """
        pred = torch.sigmoid(logits).flatten().cpu()
        targets = targets.cpu().float()
        ypred = torch.zeros(len(logits)).float()
        ypred[pred>0.5] = 1
        return torch.mean((ypred == targets).float())


@problem.tag("hw4-B", start_line=4)
def get_parameters() -> Dict:
    """Returns parameters for training a model. Is should have 4 entries, with these specific keys:

    {
        "TRAINING_BATCH_SIZE": TRAINING_BATCH_SIZE,  # type: int
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,  # type: int
        "NUM_EPOCHS": NUM_EPOCHS,  # type: int
        "LEARNING_RATE": LEARNING_RATE,  # type: float
    }

    Returns:
        Dict: Dictionary, as described above.
            (Feel free to copy dict above, and define TRAINING_BATCH_SIZE and LEARNING_RATE)
    """
    # Batch size for validation, this only affects performance.
    VAL_BATCH_SIZE = 128

    # Training parameters
    return {
        "TRAINING_BATCH_SIZE": 128,
        "VAL_BATCH_SIZE": VAL_BATCH_SIZE,
        "NUM_EPOCHS": 12,
        "LEARNING_RATE": 1e-4,
    }