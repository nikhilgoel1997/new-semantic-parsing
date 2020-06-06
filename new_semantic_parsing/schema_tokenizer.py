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

    def convert_tokens_to_ids(self, schema_tokens, src_token_ids):
        schema_ids = []
        # points to a first token corresponding to a word
        has_cls = (
            self._src_tokenizer.cls_token is not None
            and src_token_ids[0] == self._src_tokenizer.cls_token_id
        )
        src_tokens_pointer = int(has_cls)

        for token in schema_tokens:
            if token in self._vocab:
                schema_ids.append(self._stoi[token])
                continue

            subtokens = self._src_tokenizer.encode(token, add_special_tokens=False)

            for subtoken in subtokens:
                assert subtoken == src_token_ids[src_tokens_pointer]
                schema_ids.append(self.vocab_size + src_tokens_pointer - int(has_cls))
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
