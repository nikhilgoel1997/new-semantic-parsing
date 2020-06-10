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
import numpy as np
import torch
import transformers

from new_semantic_parsing.dataclasses import InputDataClass


class PointerDataset(torch.utils.data.Dataset):
    def __init__(self, source_tensors, target_tensors=None, target_pointer_masks=None):
        """
        :param source_tensors: list of tensors, input ids
        :param target_tensors: list of tensors, labels
        :param target_pointer_masks: list of tensors, mask showing pointer locations in labels
        """
        self.source_tensors = source_tensors
        self.target_tensors = target_tensors
        self.target_pointer_masks = target_pointer_masks

        self.torchified = isinstance(source_tensors[0], torch.Tensor)
        if target_tensors is not None:
            self.torchified = self.torchified and isinstance(target_tensors[0], torch.Tensor)
        if target_pointer_masks is not None:
            self.torchified = self.torchified and isinstance(target_pointer_masks[0], torch.Tensor)

    def __len__(self):
        return len(self.source_tensors_dict)

    def __getitem__(self, item):
        if self.target_tensors is None:
            return InputDataClass(
                input_ids=self.source_tensors[item],
            )

        target_pointer_mask = None
        if self.target_pointer_masks is not None:
            target_pointer_mask = self.target_pointer_masks[item]

        return InputDataClass(
            input_ids=self.source_tensors[item],
            decoder_input_ids=self.target_tensors[item],
            target_pointer_mask=target_pointer_mask,
        )

    def torchify(self):
        if self.torchified:
            return

        self.source_tensors = [torch.LongTensor(t) for t in self.source_tensors]
        self.target_tensors = [torch.LongTensor(t) for t in self.source_tensors]
        if self.target_pointer_masks is not None:
            self.target_pointer_masks = [torch.FloatTensor(t) for t in self.target_pointer_masks]

        self.torchified = True


class PaddedDataCollator(transformers.DataCollator):
    """This data collator assumes that all examples are padded to the same length"""
    def collate_batch(self, examples):
        batch = dict()

        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            batch[k] = torch.stack([getattr(ex, k) for ex in examples])

        return batch


class Seq2SeqDataCollator(transformers.DataCollator):
    """Pads tensors to the maximum length in batch.
    Length is different for encoder and decoder inputs.

    Decoder inputs should have prefix `decoder_`
    Encoder inputs do not have prefix
    """
    def collate_batch(self, examples):
        batch = dict()

        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            batch[k] = torch.stack([getattr(ex, k) for ex in examples])

        return batch

    # def pad_tensor*



def get_vocab_top_schema(text):
    schema_tokens = {'[', ']', 'IN:', 'SL:'}

    text = text.replace('[', '')
    text = text.replace(']', '')

    for token in text.split(' '):
        if token[:3] in ['IN:', 'SL:']:
            schema_tokens.add(token[3:])
    return schema_tokens


def compute_metrics(eval_prediction: transformers.EvalPrediction):
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    accuracy = np.mean(predictions.reshape(-1) == eval_prediction.label_ids.reshape(-1))
    exact_match = np.mean(np.all(predictions == eval_prediction.label_ids, axis=1))

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
    }


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    import numpy
    numpy.random.seed(seed)
    import random
    random.seed(seed)

