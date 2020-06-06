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
from copy import deepcopy

import torch
import torch.nn as nn
import transformers

from transformers import PreTrainedModel, BertModel


class EncoderDecoderWPointerModel(transformers.EncoderDecoderModel):
    """
    Encoder-decoder with pointer model as in arxiv.org/abs/2001.11458
    """
    def __init__(self, encoder, decoder, **kwargs):
        """
        :param encoder: transformers.PreTrainedModel
        :param decoder: transformers.BertModel, BertFor*Model are not supported
        """
        assert encoder.config.hidden_size == decoder.config.hidden_size
        super().__init__(encoder=encoder, decoder=decoder, **kwargs)

        head_config = deepcopy(decoder.config)
        # TODO: we have two types of padding - encoder and decoder output padding, it would be nice to combine them
        # this gives a schema vocab size
        head_config.vocab_size = decoder.config.vocab_size - encoder.config.vocab_size
        # Linear -> activation -> Linear -> LayerNorm
        # from config only .hidden_size, .hidden_act, .layer_norm_eps and .vocab_size are used
        self.lm_head = transformers.modeling_bert.BertLMPredictionHead(head_config)

        # One does not simply ties weights of embeddings with different vocabularies
        # # lm_head.decoder is just a linear layer
        # self._tie_or_clone_weights(self.lm_head.decoder,
        #                            self.decoder.get_input_embeddings())

        self.decoder_q_proj = nn.Linear(self.decoder.config.hidden_size,
                                        self.encoder.config.hidden_size,
                                        bias=False)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        head_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        decoder_inputs_embeds=None,
        masked_lm_labels=None,
        lm_labels=None,
        **kwargs,
    ):

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            **kwargs_decoder,
        )

        decoder_hidden_states = decoder_outputs[0]

        # compute pointer scores via attending from decoder hiddens to encoder hiddens
        # NOTE: this implementaion is computationally inefficient during inference

        query = self.decoder_q_proj(decoder_hidden_states)  # (bs, tgt_len, decoder_hidden)
        keys = encoder_hidden_states  # (bs, src_len, encoder_hidden)

        attention_scores = query @ keys.transpose(1, 2)  # (bs, tgt_len, src_len)

        # maybe add some kind of normalization between dec_logits?
        decoder_logits = self.lm_head(decoder_hidden_states)  # (bs, tgt_len, tgt_vocab_size)
        combined_logits = torch.cat([decoder_logits, attention_scores], dim=-1)

        return (combined_logits,) + decoder_outputs + encoder_outputs
