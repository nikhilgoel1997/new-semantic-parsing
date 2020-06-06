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

import unittest
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer


class TransformersTokenizerMock:
    cls_token = '[CLS]'
    cls_token_id = 101

    def encode(self, x, add_special_tokens=False):
        subtokens = x.split(',')
        return [int(t[3:]) for t in subtokens]


class TopSchemaTokenizerTest(unittest.TestCase):

    def test_tokenize(self):
        """
        Test cases are examples from TOP dataset arxiv.org/abs/1810.07942
        """
        schema_str = '[IN:INTENT1 tok1 tok2 tok3 [SL:SLOT1 tok4 tok5 ] ]'
        schema_tok = '[ IN: INTENT1 tok1 tok2 tok3 [ SL: SLOT1 tok4 tok5 ] ]'.split(' ')

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = ('[IN:GET_EVENT Any [SL:CATEGORY_EVENT festivals ] '
                      '[SL:DATE_TIME this weekend ] ]')

        schema_tok = ('[ IN: GET_EVENT Any [ SL: CATEGORY_EVENT festivals ] '
                      '[ SL: DATE_TIME this weekend ] ]').split(' ')

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

        schema_str = ("[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
                      "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
                      "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
                      "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]")

        schema_tok = ("[ IN: GET_ESTIMATED_ARRIVAL What time will I arrive at "
                      "[ SL: DESTINATION [ IN: GET_LOCATION_HOME [ SL: CONTACT_RELATED "
                      "my ] [ SL: TYPE_RELATION Mom ] 's house ] ] if I leave "
                      "[ SL: DATE_TIME_DEPARTURE in five minutes ] ? ]")
        schema_tok = schema_tok.split(' ')

        res = TopSchemaTokenizer.tokenize(schema_str)
        self.assertSequenceEqual(res, schema_tok)

    def test_encode_nocls(self):
        vocab = ['[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1']
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        expected_ids = [1, 3, 4, 7, 8, 9, 1, 5, 6, 10, 11, 2, 2]

        res = tokenizer.encode(schema_str, source_tokens)

        self.assertSequenceEqual(res, expected_ids)

    def test_encode_cls(self):
        vocab = ['[', ']', 'IN:', 'INTENT1', 'SL:', 'SLOT1']
        src_tokenizer = TransformersTokenizerMock()

        tokenizer = TopSchemaTokenizer(vocab, src_tokenizer)

        schema_str = '[IN:INTENT1 tok6,tok2 tok31 [SL:SLOT1 tok42 tok5 ] ]'
        source_tokens = [TransformersTokenizerMock.cls_token_id, 6, 2, 31, 42, 5]
        # note that TransformersTokenizerMock splits tok6,tok2 into two subtokens
        expected_ids = [1, 3, 4, 7, 8, 9, 1, 5, 6, 10, 11, 2, 2]

        res = tokenizer.encode(schema_str, source_tokens)

        self.assertSequenceEqual(res, expected_ids)
