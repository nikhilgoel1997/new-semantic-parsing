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
import os
import json
from pathlib import Path

import torch
import transformers
import numpy as np

from new_semantic_parsing.dataclasses import SchemaItem, InputDataClass


class TopSchemaTokenizer:
    """
    Used for TOP schema tokenization

    encodes schema into token_ids from schema_vocab
    and words into position-based ids

    word_id = tokenizer.vocab_size + position

    [CLS] token is ignored for position calculation
    """
    def __init__(self, schema_vocab, src_text_tokenizer):
        """
        :param schema_vocab: iterable with all schema tokens (not source text tokens)
        :param src_text_tokenizer: transformers.PreTrainedTokenizer object
        """
        self.pad_token = '[PAD]'

        self._vocab = schema_vocab
        self._itos = [self.pad_token] + sorted(schema_vocab)
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self.pad_token_id = self._stoi[self.pad_token]

        self.src_tokenizer = src_text_tokenizer

    @property
    def vocab_size(self):
        return len(self._itos)

    def encode(self, schema_text, source_ids, max_length=None, pad_to_max_length=False):
        return self.encode_plus(schema_text, source_ids, max_length, pad_to_max_length).ids

    def encode_plus(self, schema_text, source_ids, max_length=None, pad_to_max_length=False) -> SchemaItem:
        schema_tokens = self.tokenize(schema_text)

        if max_length is not None:
            schema_tokens = schema_tokens[:max_length]

        if pad_to_max_length:
            delta = max_length - len(schema_tokens)
            if delta > 0:
                schema_tokens += [self.pad_token] * delta

        item = self.convert_tokens_to_ids(schema_tokens, source_ids)

        return item

    def batch_encode_plus(
        self,
        batch_schema_text,
        batch_source_ids,
        max_length=None,
        pad_to_max_length=True,
        return_tensors=None,
        device='cpu',
    ) -> InputDataClass:
        assert pad_to_max_length, "Not padding to max length is not supported"

        batch_size = len(batch_schema_text)
        batch_items = []

        for schema_text, source_ids in zip(batch_schema_text, batch_source_ids):
            item = self.encode_plus(schema_text, source_ids, max_length)
            batch_items.append(item)

        if max_length is None:
            max_length = max(len(t) for t in batch_items)

        batch_ids_padded = np.full([batch_size, max_length], self.pad_token_id, dtype=np.int32)
        batch_schema_masks = np.zeros([batch_size, max_length])
        padding_masks = np.ones([batch_size, max_length])
        for j, item in enumerate(batch_items):
            ids = item.ids
            schema_mask = item.pointer_mask

            difference = max_length - len(item)
            if difference > 0:
                ids = ids + [self.pad_token_id] * difference
                schema_mask = schema_mask + [0] * difference
                padding_masks[j, -difference:] = 0.

            if difference < 0:
                ids = ids[:max_length]
                schema_mask = schema_mask[:max_length]

            batch_ids_padded[j] = ids
            batch_schema_masks[j] = schema_mask

        if return_tensors is None:
            return InputDataClass(
                input_ids=batch_ids_padded,
                attention_mask=padding_masks,
                decoder_pointer_mask=batch_schema_masks,
            )

        if return_tensors == 'pt':
            return InputDataClass(
                input_ids=torch.LongTensor(batch_ids_padded, device=device),
                attention_mask=torch.FloatTensor(padding_masks, device=device),
                decoder_pointer_mask=torch.FloatTensor(batch_schema_masks, device=device),
            )

        raise ValueError('`return_tensors` can be eigher None or "pt"')

    def convert_tokens_to_ids(self, schema_tokens, src_token_ids) -> SchemaItem:
        """
        :param schema_tokens: string
        :param src_token_ids: list or numpy array of integers
        :return: list of integers - a mix of token ids and position ids
            position id = position + vocab_size
        """
        schema_ids = []
        pointer_mask = []

        # points to a first token corresponding to a word
        has_cls = (
            self.src_tokenizer.cls_token is not None and
            self.src_tokenizer.cls_token_id in src_token_ids
        )
        src_tokens_pointer = int(has_cls)

        for i, token in enumerate(schema_tokens):
            token_follows_schema = token in {'[', ']', 'IN:', 'SL:'} or schema_tokens[i-1] in {'IN:', 'SL:'}
            if token in self._vocab and token_follows_schema:
                # The reason for second condition are cases when a word from a text exacly equal to the schema word
                # e.g. "IS THERE A PATH"
                # PATH is in a schema vocabulary, but not a schema word

                pointer_mask.append(0)
                schema_ids.append(self._stoi[token])
                continue

            subtokens = self.src_tokenizer.encode(token, add_special_tokens=False)

            for subtoken in subtokens:
                assert subtoken == src_token_ids[src_tokens_pointer]
                pointer_mask.append(1)
                schema_ids.append(self.vocab_size + src_tokens_pointer)
                src_tokens_pointer += 1

        return SchemaItem(schema_ids, pointer_mask)

    def save(self, path, encoder_model_type=None):
        """
        Save schema tokenizer and text tokenizer
        Optionally, save pre-trained encoder model type - this is a workaround for Transformers #4197
        """
        _path = Path(path)
        os.makedirs(_path)

        with open(_path / 'schema_vocab.txt', 'w') as f:
            f.write('\n'.join(self._vocab))

        self.src_tokenizer.save_pretrained(path)

        if encoder_model_type is not None:
            with open(_path / 'config.json', 'w') as f:
                json.dump({'model_type': encoder_model_type}, f)

    @classmethod
    def load(cls, path: str):
        if isinstance(path, Path):
            raise ValueError('AutoTokenizer.from_pretrained does not support Path')
        with open(Path(path)/'schema_vocab.txt') as f:
            schema_vocab = set(f.read().strip('\n').split('\n'))

        text_tokenizer = transformers.AutoTokenizer.from_pretrained(path)

        return cls(schema_vocab, text_tokenizer)

    @staticmethod
    def tokenize(text):
        # TODO: make a faster regex version
        tokenized = ''
        for char in text:
            if char in ['[', ']']:
                char = ' ' + char + ' '
            if char in [':']:
                char = char + ' '
            tokenized += char
        tokens = tokenized.strip(' ').split(' ')
        tokens = [t for t in tokens if t != '']
        return tokens
