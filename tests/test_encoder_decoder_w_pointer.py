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

import torch
import transformers
from new_semantic_parsing import EncoderDecoderWPointerModel


class EncoderDecoderWPointerTest(unittest.TestCase):
    def test_shape_on_random_data(self):
        torch.manual_seed(42)

        bs = 3
        src_len = 5
        tgt_len = 7

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=17,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=23,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        x_enc = torch.randint(0, encoder_config.vocab_size, size=(bs, src_len))
        x_dec = torch.randint(0, decoder_config.vocab_size, size=(bs, tgt_len))

        out = model(input_ids=x_enc, decoder_input_ids=x_dec)

        # different encoders return different number of outputs
        # e.g. BERT returns two, but DistillBERT only one
        self.assertGreaterEqual(len(out), 4)

        combined_logits = out[0]
        expected_shape = (bs, tgt_len, decoder_config.vocab_size - encoder_config.vocab_size + src_len)
        self.assertEqual(combined_logits.shape, expected_shape)

        decoder_hidden = out[1]
        expected_shape = (bs, tgt_len, decoder_config.hidden_size)
        self.assertEqual(decoder_hidden.shape, expected_shape)

        combined_logits = out[2]
        expected_shape = (bs, decoder_config.hidden_size)
        self.assertEqual(combined_logits.shape, expected_shape)

        encoder_hidden = out[3]
        expected_shape = (bs, src_len, encoder_config.hidden_size)
        self.assertEqual(encoder_hidden.shape, expected_shape)

    def test_shape_on_real_data(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        # e.g. '[CLS] Directions to Lowell [SEP]'
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2]])
        # e.g. '[IN:GET_DIRECTIONS Directions to [SL:DESTINATION Lowell]]'
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_attention_mask=mask)[0]

        expected_shape = (1, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_shape_on_real_data_batched(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        combined_logits = model(input_ids=src_seq,
                                decoder_input_ids=tgt_seq,
                                pointer_attention_mask=mask)[0]

        expected_shape = (2, tgt_seq.shape[1], tgt_vocab_size + src_seq.shape[1])
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_loss_computation(self):
        torch.manual_seed(42)
        src_vocab_size = 17
        tgt_vocab_size = 23

        encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=src_vocab_size,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=tgt_vocab_size + src_vocab_size,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        decoder = transformers.BertModel(decoder_config)

        model = EncoderDecoderWPointerModel(encoder, decoder)

        # similar to real data
        src_seq = torch.LongTensor([[1, 6, 12, 15, 2, 0, 0],
                                    [1, 6, 12, 15, 5, 3, 2]])
        tgt_seq = torch.LongTensor([[8, 6, 4, 10, 11, 8, 5, 1, 12,  7, 7, 0, 0],
                                    [8, 6, 4, 10, 11, 8, 5, 1, 12, 13, 14, 7, 7]])
        mask = torch.FloatTensor([[0, 1, 1, 1, 0, 0, 0],
                                  [0, 1, 1, 1, 1, 1, 0]])

        loss = model(input_ids=src_seq,
                     decoder_input_ids=tgt_seq,
                                pointer_attention_mask=mask,
                                lm_labels=tgt_seq)[0]

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss, 0)
