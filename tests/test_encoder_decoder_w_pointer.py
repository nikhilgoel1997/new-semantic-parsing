import unittest

import torch
import transformers
from new_semantic_parsing import EncoderDecoderWPointerModel


class EncoderDecoderWPointerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        self.bs = 3
        self.src_len = 5
        self.tgt_len = 7

        self.encoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=17,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        self.encoder = transformers.BertModel(self.encoder_config)

        self.decoder_config = transformers.BertConfig(
            hidden_size=11,
            intermediate_size=44,
            vocab_size=23,
            is_decoder=True,
            num_hidden_layers=1,
            num_attention_heads=1,
        )
        self.decoder = transformers.BertModel(self.decoder_config)

        model = EncoderDecoderWPointerModel(self.encoder, self.decoder)

        x_enc = torch.randint(0, self.encoder_config.vocab_size, size=(self.bs, self.src_len))
        x_dec = torch.randint(0, self.decoder_config.vocab_size, size=(self.bs, self.tgt_len))

        self.out = model(input_ids=x_enc, decoder_input_ids=x_dec)

    def test_output_size(self):
        # different encoders return different number of outputs
        # e.g. BERT returns two, but DistillBERT only one
        self.assertGreaterEqual(len(self.out), 4)

    def test_combined_logits_shape(self):
        combined_logits = self.out[0]
        expected_shape = (self.bs, self.tgt_len, self.decoder_config.vocab_size + self.src_len)
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_decoder_hidden_shape(self):
        decoder_hidden = self.out[1]
        expected_shape = (self.bs, self.tgt_len, self.decoder_config.hidden_size)
        self.assertEqual(decoder_hidden.shape, expected_shape)

    def test_combined_logits_shape(self):
        combined_logits = self.out[2]
        expected_shape = (self.bs, self.decoder_config.hidden_size)
        self.assertEqual(combined_logits.shape, expected_shape)

    def test_encoder_hidden_shape(self):
        encoder_hidden = self.out[3]
        expected_shape = (self.bs, self.src_len, self.encoder_config.hidden_size)
        self.assertEqual(encoder_hidden.shape, expected_shape)
