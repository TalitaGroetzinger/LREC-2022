"""A model for plausibility classification."""
from typing import Dict

import torch
import transformers


class PlausibilityClassifier(torch.nn.Module):
    def __init__(self, bert: transformers.BertModel, output_dim: int):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()["hidden_size"]
        self.output_dim = output_dim
        self.out = torch.nn.Linear(embedding_dim, output_dim)

    def forward(self, batch: Dict):
        raise NotImplementedError


class SimplePlausibilityClassifier(PlausibilityClassifier):
    def __init__(self, bert: transformers.BertModel, output_dim: int):
        super().__init__(bert=bert, output_dim=output_dim)

    def forward(self, batch: Dict):
        bert_output = self.bert(**batch["text"])
        # pooler output: last hidden state for [CLS]
        return self.out(bert_output.pooler_output)


class StartMarkerPlausibilityClassifier(PlausibilityClassifier):
    def __init__(self, bert, output_dim):
        super().__init__(bert=bert, output_dim=output_dim)

    def forward(self, batch: Dict):
        bert_output = self.bert(**batch["text"])

        # 1-dim tensor with ints in range from 0 to the batch size
        tensor_range = torch.arange(batch["text"].size()[0])
        filler_start_embedding = bert_output.last_hidden_state[
            tensor_range, batch["filler_start_index"]
        ]
        return self.out(filler_start_embedding)
