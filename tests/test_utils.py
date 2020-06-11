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
import random
import unittest
import torch

from new_semantic_parsing import utils
from new_semantic_parsing.dataclasses import InputDataClass


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


class PointerDatasetTest(unittest.TestCase):
    def test_getitem(self):
        utils.set_seed(29)

        src_tensors = [torch.randint(0, 10, size=(random.randint(5, 13),), dtype=torch.int64)
                       for _ in range(10)]
        tgt_tensors = [torch.randint(0, 20, size=(random.randint(5, 15),), dtype=torch.int64)
                       for _ in range(10)]

        dataset = utils.PointerDataset(src_tensors, tgt_tensors, None)

        item = dataset[0]

        self.assertIsInstance(item, InputDataClass)
        self.assertIsInstance(item.input_ids, torch.LongTensor)
        self.assertIsInstance(item.decoder_input_ids, torch.LongTensor)
        self.assertIsInstance(item.labels, torch.LongTensor)

    def test_getitem_inference(self):
        utils.set_seed(29)

        src_tensors = [torch.randint(0, 10, size=(random.randint(5, 13),), dtype=torch.int64)
                       for _ in range(10)]

        dataset = utils.PointerDataset(src_tensors)

        item = dataset[0]

        self.assertIsInstance(item, InputDataClass)
        self.assertIsInstance(item.input_ids, torch.LongTensor)

        self.assertIsNone(item.decoder_input_ids)
        self.assertIsNone(item.labels)

    def test_len(self):
        dataset = utils.PointerDataset([None, None], [None, None], None)
        self.assertEqual(len(dataset), 2)


class Seq2SeqDataCollatorDataset(unittest.TestCase):
    def test_collate_batch_shapes(self):
        utils.set_seed(29)

        bs = 3
        e_pad = 0
        d_pad = 1
        d_mask = 2
        src_tensors = [torch.tensor([2, 5, 4, 4, 2]),
                       torch.tensor([1, 8, 2, 8, 5, 4, 2, 2, 5, 7]),
                       torch.tensor([4, 6, 4, 2, 1, 2])]
        tgt_tensors = [torch.tensor([6, 7, 8, 7, 2, 2, 4, 8, 5]),
                       torch.tensor([5, 2, 2, 8, 7, 3, 5, 4, 2, 2, 1]),
                       torch.tensor([8, 2, 2, 3, 5, 2, 2, 2, 3, 8, 4, 6, 7, 8])]
        tgt_masks = [(tgt_tensors[i] == d_mask).type(torch.FloatTensor) for i in range(3)]

        assert d_mask in tgt_tensors[0]

        examples = [
            InputDataClass(
                input_ids=src_tensors[i],
                decoder_input_ids=tgt_tensors[i],
                decoder_pointer_mask=tgt_masks[i],
                labels=tgt_tensors[i],
            )
            for i in range(bs)
        ]

        collator = utils.Seq2SeqDataCollator(e_pad, d_pad)

        batch = collator.collate_batch(examples)

        self.assertEqual(batch['input_ids'].shape, (bs, 10))
        self.assertIsInstance(batch['input_ids'], torch.LongTensor)
        self.assertEqual(batch['input_ids'][0, -1], e_pad)

        self.assertEqual(batch['decoder_input_ids'].shape, (bs, 14))
        self.assertIsInstance(batch['decoder_input_ids'], torch.LongTensor)
        self.assertEqual(batch['decoder_input_ids'][0, -1], d_pad)

        self.assertEqual(batch['labels'].shape, (bs, 14))
        self.assertIsInstance(batch['labels'], torch.LongTensor)

        self.assertEqual(batch['decoder_pointer_mask'].shape, (bs, 14))
        self.assertIsInstance(batch['decoder_pointer_mask'], torch.FloatTensor)
        _mask = batch['decoder_pointer_mask']
        self.assertTrue(((_mask == 0) | (_mask == 1)).all())


class TestGetModelType(unittest.TestCase):
    def test_model_type(self):
        model_type = utils.get_model_type('distilbert-base-uncased')
        self.assertEqual(model_type, 'distilbert')
