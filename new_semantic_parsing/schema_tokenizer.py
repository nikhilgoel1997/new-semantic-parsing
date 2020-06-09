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
import torch
import numpy as np


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
        self._itos = [self.pad_token] + list(schema_vocab)
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self.pad_token_id = self._stoi[self.pad_token]

        self._src_tokenizer = src_text_tokenizer

    @property
    def vocab_size(self):
        return len(self._itos)

    def encode(self, schema_text, source_ids, max_length=None, pad_to_max_length=False):
        schema_tokens = self.tokenize(schema_text)

        if max_length is not None:
            schema_tokens = schema_tokens[:max_length]

        if pad_to_max_length:
            delta = max_length - len(schema_tokens)
            if delta > 0:
                schema_tokens += [self.pad_token] * delta

        ids = self.convert_tokens_to_ids(schema_tokens, source_ids)

        return ids

    def batch_encode_plus(
        self,
        batch_schema_text,
        batch_source_ids,
        max_length=None,
        pad_to_max_length=True,
        return_tensors=None,
        device='cpu',
    ):
        assert pad_to_max_length, "Not padding to max length is not supported"

        batch_size = len(batch_schema_text)
        batch_ids = []

        for schema_text, source_ids in zip(batch_schema_text, batch_source_ids):
            ids = self.encode(schema_text, source_ids, max_length)
            batch_ids.append(ids)

        if max_length is None:
            max_length = max(len(t) for t in batch_ids)

        batch_ids_padded = np.ones([batch_size, max_length]) * self.pad_token_id
        padding_masks = np.ones([batch_size, max_length])
        for j, ids in enumerate(batch_ids):
            difference = max_length - len(ids)
            if difference > 0:
                ids = ids + [self.pad_token_id] * difference
                padding_masks[j, -difference:] = 0.

            if difference < 0:
                ids = ids[:max_length]

            batch_ids_padded[j] = ids

        if return_tensors is None:
            return {'input_ids': batch_ids_padded, 'attention_mask': padding_masks}

        if return_tensors == 'pt':
            return {'input_ids': torch.LongTensor(batch_ids_padded, device=device),
                    'attention_mask': torch.FloatTensor(padding_masks, device=device)}

        raise ValueError('`return_tensors` can be eigher None or "pt"')

    def convert_tokens_to_ids(self, schema_tokens, src_token_ids):
        """
        :param schema_tokens: string
        :param src_token_ids: list or numpy array of integers
        :return: list of integers - a mix of token ids and position ids
            position id = position + vocab_size
        """
        schema_ids = []
        # points to a first token corresponding to a word
        has_cls = (
            self._src_tokenizer.cls_token is not None and
            self._src_tokenizer.cls_token_id in src_token_ids
        )
        src_tokens_pointer = int(has_cls)

        for token in schema_tokens:
            if token in self._vocab:
                schema_ids.append(self._stoi[token])
                continue

            subtokens = self._src_tokenizer.encode(token, add_special_tokens=False)

            for subtoken in subtokens:
                assert subtoken == src_token_ids[src_tokens_pointer]
                schema_ids.append(self.vocab_size + src_tokens_pointer)
                src_tokens_pointer += 1

        return schema_ids

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
