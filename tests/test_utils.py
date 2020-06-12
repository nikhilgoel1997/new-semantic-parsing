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

from new_semantic_parsing import utils


class TopSchemaGetVocabularyTest(unittest.TestCase):
    def test_get_vocab(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        schema_str = ("[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
                      "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
                      "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
                      "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]")
        schema_voc = {
            '[',
            ']',
            'IN:',
            'SL:',
            'CONTACT_RELATED',
            'DATE_TIME_DEPARTURE',
            'DESTINATION',
            'GET_ESTIMATED_ARRIVAL',
            'GET_LOCATION_HOME',
            'TYPE_RELATION',
        }

        res = utils.get_vocab_top_schema(schema_str)
        self.assertSetEqual(res, schema_voc)


class TestGetModelType(unittest.TestCase):
    def test_model_type(self):
        model_type = utils.get_model_type('distilbert-base-uncased')
        self.assertEqual(model_type, 'distilbert')
