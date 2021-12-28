"""A model for plausibility classification."""
from typing import Dict

import torch
import transformers


class PlausibilityClassifier(torch.nn.Module):
    """Classifier that predicts the plausibility of a filler in context."""

    def __init__(self, bert: transformers.BertModel, output_dim: int, dropout: float):
        super().__init__()
        self.bert = bert
        self.embedding_dim = bert.config.to_dict()["hidden_size"]
        self.output_dim = output_dim
        self.out = torch.nn.Linear(self.embedding_dim, output_dim)
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


class DualInputPlausibilityClassifier(PlausibilityClassifier):
    """Simple classifier that uses the hidden state of [CLS] as the sentence embedding."""

    def __init__(self, bert: transformers.BertModel, output_dim: int, dropout: float):
        super().__init__(bert=bert, output_dim=output_dim, dropout=dropout)
        self.embedding_dim = 2 * bert.config.to_dict()["hidden_size"]
        #self.relu = torch.nn.ReLU()
        #self.linear1 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        #self.linear2 = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

        self.sequential = torch.nn.Sequential(
                            torch.nn.Linear(self.embedding_dim, self.embedding_dim), 
                            torch.nn.ReLU(), 
                            torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                            torch.nn.ReLU(), 
        )
        self.out = torch.nn.Linear(self.embedding_dim, output_dim)

    def forward(self, batch: Dict):
        text_embedding = self.bert(**batch["text"])
        filler_embedding = self.bert(**batch["filler"])

        # pooler output: last hidden state for [CLS]
        text_embedding = text_embedding.pooler_output
        filler_embedding = filler_embedding.pooler_output

        embedding = torch.cat((text_embedding, filler_embedding), dim=-1)
        sequential_output = self.sequential(embedding)
        sequential_output = self.dropout(sequential_output)
        return self.out(sequential_output)


class TwistedPlausibilityClassifier(PlausibilityClassifier):
    """Simple classifier that uses the hidden state of [CLS] as the sentence embedding."""

    def __init__(self, bert: transformers.BertModel, output_dim: int, dropout: float):
        super().__init__(bert=bert, output_dim=output_dim, dropout=dropout)
        self.embedding_dim = 2 * bert.config.to_dict()["hidden_size"]

        self.sequential = torch.nn.Sequential(
                            torch.nn.Linear(self.embedding_dim, self.embedding_dim), 
                            torch.nn.ReLU(), 
                            # torch.nn.Linear(self.embedding_dim, self.embedding_dim),
                            # torch.nn.ReLU(), 
        )
        self.out = torch.nn.Linear(self.embedding_dim, output_dim)

    def forward(self, batch: Dict):
        text_embedding = self.bert(**batch["text"])

        # pooler output: last hidden state for [CLS]
        cls_embedding = text_embedding.pooler_output
        cls_embedding = self.dropout(cls_embedding)

        tensor_range = torch.arange(len(batch["identifier"]))
        start_index_embedding = text_embedding.last_hidden_state[
            tensor_range, batch["filler_start_index"]
        ]

        embedding = torch.cat((cls_embedding, start_index_embedding), dim=-1)
        # sequential_output = self.sequential(embedding)
        # sequential_output = self.dropout(sequential_output)
        return self.out(embedding)
