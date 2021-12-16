"""A model for plausibility classification."""
from typing import Dict

import torch
import transformers


class PlausibilityClassifier(torch.nn.Module):
    """Classifier that predicts the plausibility of a filler in context."""

    def __init__(self, bert: transformers.BertModel, output_dim: int, dropout: float):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()["hidden_size"]
        self.output_dim = output_dim
        self.out = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, batch: Dict):
        raise NotImplementedError


class SimplePlausibilityClassifier(PlausibilityClassifier):
    """Simple classifier that uses the hidden state of [CLS] as the sentence embedding."""

    def __init__(self, bert: transformers.BertModel, output_dim: int, dropout: float):
        super().__init__(bert=bert, output_dim=output_dim, dropout=dropout)

    def forward(self, batch: Dict):
        bert_output = self.bert(**batch["text"])
        # pooler output: last hidden state for [CLS]
        embedding = self.dropout(bert_output.pooler_output)
        return self.out(embedding)


class StartMarkerPlausibilityClassifier(PlausibilityClassifier):
    """Classifier that uses the hidden state of the filler start marker as the sentence embedding."""

    def __init__(self, bert, output_dim, dropout: float):
        super().__init__(bert=bert, output_dim=output_dim, dropout=dropout)

    def forward(self, batch: Dict):
        bert_output = self.bert(**batch["text"])

        # 1-dim tensor with ints in range from 0 to the batch size
        tensor_range = torch.arange(len(batch["identifier"]))
        embedding = bert_output.last_hidden_state[
            tensor_range, batch["filler_start_index"]
        ]
        embedding = self.dropout(embedding)
        return self.out(embedding)
