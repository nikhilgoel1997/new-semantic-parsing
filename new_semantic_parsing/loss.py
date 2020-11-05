# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Loss functions used in the model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, weights=None):
        super().__init__()
        self.eps = eps
        self.weights = weights

        self._weights_normalized = None
        if weights is not None:
            self._weights_normalized = weights / weights.mean()

        if self.weights is not None and torch.any(weights < 0):
            raise ValueError(f"weights should be all positive, got {weights} instead.")

    def forward(self, preds, target, mask, example_weights=None):
        """
        Args:
            preds: torch.FloatTensor of shape (batch_size * seq_len, vocab_size)
            target: torch.LongTensor of shape (batch_size * seq_len,)
            mask: torch.FloatTensor of shape (batch_size * seq_len,) with 0 at masked tokens
            example_weights: torch.FloatTensor of shape (batch_size * seq_len,)
        """
        if mask is None:
            mask = torch.ones_like(target)
        if example_weights is None:
            example_weights = torch.ones(preds.shape[0])

        log_probs = F.log_softmax(preds, dim=-1)

        one_hot = torch.zeros_like(preds).scatter(1, target.view(-1, 1), 1)
        smoothing = self.eps / (preds.size(-1) - 1)

        # main class = 1.0 - eps - 1/(num-classes - 1) + 1/(num-classes - 1) = 1.0 - eps
        # other classes = 1/(num-classes - 1)
        # total sum = 1
        one_hot = (1 - self.eps - smoothing) * one_hot + smoothing

        if self.weights is None:
            loss = -(one_hot * log_probs).sum(dim=-1)
        else:
            loss = -(one_hot * log_probs * self._weights_normalized).sum(dim=-1)

        loss *= mask
        loss = loss.sum() / torch.sum(mask)  # divide by the total number of tokens

        return loss
