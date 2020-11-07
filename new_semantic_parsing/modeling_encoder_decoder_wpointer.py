# Copyright 2020 Google LLC
# Copyright 2018 The HuggingFace Inc. team.
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
"""Encoder-decoder with pointer model as in arxiv.org/abs/2001.11458"""
import sys
import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from new_semantic_parsing.configuration_encoder_decoder_wpointer import (
    EncoderDecoderWPointerConfig,
)
from new_semantic_parsing.loss import LabelSmoothedCrossEntropy

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


class EncoderDecoderWPointerModel(transformers.PreTrainedModel):
    config_class = EncoderDecoderWPointerConfig
    base_model_prefix = "encoder_decoder_wpointer"

    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None,
        max_src_len=None,
        dropout=None,
        model_args=None,
        **kwargs,
    ):
        """Initializes the model either from config or from encoder and decoder models.

        Args:
            config: EncoderDecoderWPointerConfig
            encoder: transformers.PreTrainedModel
            decoder: transformers.BertModel, BertFor*Model are not supported
            model_args: argparse arguments, architecture parameters
            max_src_len: maximum length of the encoder sequence, defines the maximum possible pointer index
        """
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"

        if config is None:
            config = self.config_class(
                encoder=encoder.config.to_dict(),
                decoder=decoder.config.to_dict(),
                max_src_len=max_src_len,
                dropout=dropout,
                model_args=model_args,
                **kwargs,
            )

        if not isinstance(config, self.config_class):
            raise ValueError(f"config: {config} has to be of the type {self.config_class}")

        super().__init__(config)

        self.encoder = encoder or transformers.AutoModel.from_config(config.encoder)
        self.decoder = decoder or transformers.AutoModel.from_config(config.decoder)

        if self.encoder.get_output_embeddings() is not None:
            raise RuntimeError(
                "The encoder should not have an LM Head. Please use a model without LM Head"
            )

        self.enc_dec_proj = None
        if self.encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_dec_proj = nn.Linear(
                self.encoder.config.hidden_size, self.decoder.config.hidden_size
            )

        # this gives a schema vocab size (without pointer embeddings)
        self.output_vocab_size = self.decoder.config.vocab_size - self.config.max_src_len

        head_config = deepcopy(self.decoder.config)
        head_config.vocab_size = self.output_vocab_size

        # Linear -> activation -> LayerNorm -> Linear
        # from config only .hidden_size, .hidden_act, .layer_norm_eps and .vocab_size are used
        self.lm_head = transformers.modeling_bert.BertLMPredictionHead(head_config)

        self.decoder_q_proj = nn.Linear(
            self.decoder.config.hidden_size,
            self.decoder.config.hidden_size,
            bias=self.config.use_pointer_bias,
        )

        # used in .generate to reset decoder vocab size value after generation
        self._actual_vocab_size = self.decoder.config.vocab_size

        self.label_smoothing_loss_layer = None
        if self.config.label_smoothing > 0:
            self.label_smoothing_loss_layer = LabelSmoothedCrossEntropy(
                eps=self.config.label_smoothing
            )

        # fine-tuning specific regularizations
        use_ewc = self.config.track_grad_square or self.config.weight_consolidation

        if self.config.weight_consolidation:
            logger.info("Fine-tuning ewc-ready model, setting config.track_grad_square=False")
            self.config.track_grad_square = False

        self._grad_squared_buffer_names = None
        if use_ewc:
            self._grad_squared_buffer_names = self._register_grad_squared_buffers()

        # used for weight consolidation and move norm during both training and finetuning
        self.initial_params = None
        if use_ewc or self.config.move_norm is not None:
            self.reset_initial_params()  # set initial_params

    def to(self, *args, **kwargs):
        if self.initial_params is not None:
            self.initial_params = {
                n: p.to(*args, **kwargs) for n, p in self.initial_params.items()
            }

        return super(EncoderDecoderWPointerModel, self).to(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = ...):
        super().load_state_dict(state_dict, strict)
        logger.info("Resetting model initial parameters.")
        self.reset_initial_params()

    def tie_weights(self):
        # no weights tying
        pass

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        """
        The only purpose of this function is to pass get_output_embeddings check in the .decode() method
        """
        return object()

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        pointer_mask = kwargs.get("pointer_mask", None)
        if pointer_mask is None:
            raise ValueError("pointer_mask should be specified")

        input_batch_size = input_ids.shape[0]
        pointer_mask_batch_size = pointer_mask.shape[0]

        if input_batch_size != pointer_mask_batch_size:
            # happends during beam search, when input_ids got copied num_beams times
            num_beams = input_batch_size // pointer_mask_batch_size
            assert input_batch_size % pointer_mask_batch_size == 0
            pointer_mask = pointer_mask.repeat_interleave(repeats=num_beams, dim=0)

        # first step
        if type(past) is tuple:
            encoder_outputs, _ = past
        else:
            encoder_outputs = (past,)

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)

        # decoder attention mask is not passed as it does not make sense to do it while generating
        return {
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "pointer_mask": pointer_mask,
        }

    def generate(self, input_ids, pointer_mask, bos_token_id, **kwargs) -> torch.LongTensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids should be of shape (batch_size, seq_len)")

        if kwargs.get("use_cache", False) is True:
            # it is not clear what exactly use_cache is supposed to do
            raise ValueError(
                "caching decoder outputs is not supported, "
                "encoder output is cached regardless of this option"
            )
        kwargs["use_cache"] = False

        if input_ids.shape != pointer_mask.shape:
            # this bug may be tricky to catch layer, so better to validate it here
            raise ValueError(
                "input_ids and pointer_mask shapes do not align:"
                f" input_ids.shape={input_ids.shape},"
                f" pointer_mask.shape={pointer_mask.shape}"
            )

        # Tricky hack with dynamic output size. Beam search is using vocab size for reshaping
        # e.g. .view(batch_size, num_beams, vocab_size) however, in the case of pointer network,
        # the number of logits depends on the input size (batch_size, tgt_seq_len, schema_vocab_size + src_seq_len).
        # To overcome this, we change self.decoder.vocab_size when generating.
        src_seq_len = input_ids.shape[1]
        self.config.decoder.vocab_size = self.output_vocab_size + src_seq_len

        generated = super().generate(
            input_ids=input_ids,
            pointer_mask=pointer_mask,
            bos_token_id=bos_token_id,
            **kwargs,
        )

        self.config.decoder.vocab_size = self._actual_vocab_size

        # cut the [BOS] token at the beginning (super().generate() adds it)
        assert generated.dim() == 2
        assert torch.all(generated[:, 0] == bos_token_id)
        generated = generated[:, 1:]

        return generated

    @classmethod
    def from_parameters(
        cls,
        layers,
        hidden,
        heads,
        src_vocab_size,
        tgt_vocab_size,
        max_src_len,
        decoder_layers=None,
        decoder_hidden=None,
        decoder_heads=None,
        encoder_pad_token_id=0,
        decoder_pad_token_id=None,
        dropout=0,
        move_norm=None,
        track_grad_square=False,
        weight_consolidation=None,
        model_args=None,
    ):
        """
        :param layers: number of layers for encoder and for decoder
        :param hidden: hidden size
        :param heads: number of attention heads
        :param src_vocab_size: source vocabulary size
        :param tgt_vocab_size: size of the target vocabulary (excluding pointers)
        :param encoder_pad_token_id: pad id to ignore in the encoder input
        :param decoder_pad_token_id: pad id to ignore in the decoder input, equal to encoder_pad_token_id by default
        :return: EncoderDecoderWPointerModel
        """
        encoder_config = transformers.BertConfig(
            hidden_size=hidden,
            intermediate_size=4 * hidden,
            vocab_size=src_vocab_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            pad_token_id=encoder_pad_token_id,
        )
        encoder = transformers.BertModel(encoder_config)

        decoder_pad_token_id = decoder_pad_token_id or encoder_pad_token_id
        decoder_hidden = decoder_hidden or hidden
        decoder_layers = decoder_layers or layers
        decoder_heads = decoder_heads or heads

        decoder_config = transformers.BertConfig(
            hidden_size=decoder_hidden,
            intermediate_size=4 * decoder_hidden,
            vocab_size=tgt_vocab_size + max_src_len,
            is_decoder=True,
            num_hidden_layers=decoder_layers,
            num_attention_heads=decoder_heads,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            pad_token_id=decoder_pad_token_id,
        )
        decoder = transformers.BertModel(decoder_config)

        return cls(
            encoder=encoder,
            decoder=decoder,
            max_src_len=max_src_len,
            model_args=model_args,
            move_norm=move_norm,
            dropout=dropout,
            track_grad_square=track_grad_square,
            weight_consolidation=weight_consolidation,
        )

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
        pointer_mask=None,
        decoder_pointer_mask=None,
        labels=None,
        **kwargs,
    ):
        """
        Note that all masks are FloatTensors and equal 1 for keep and 0 for mask

        :param input_ids: LongTensor of shape (batch_size, src_seq_len)
        :param inputs_embeds: alternative to input_ids, shape (batch_size, src_seq_len, embed_dim)
        :param attention_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the encoder
        :param head_mask: FloatTensor of shape (num_heads,) or (num_layers, num_heads)
        :param encoder_outputs: alternative to input_ids, tuple(last_hidden_state, hidden_states, attentions)
            last_hidden_state has shape (batch_size, src_seq_len, hidden)
        :param decoder_input_ids: LongTensor of shape (batch_size, tgt_seq_len)
        :param decoder_inputs_embeds: alternative to decoder_input_ids, shape (batch_size, tgt_seq_len, embed_dim
        :param decoder_attention_mask: FloatTensor of shape (batch_size, tgt_seq_lqn), padding mask for the encoder
            decoder input padding mask, causal mask is generated inside the decoder class
        :param decoder_head_mask: FloatTensor of shape (num_heads,) or (num_layers, num_heads)
        :param pointer_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the pointer
        :param labels: LongTensor of shape (batch_size, tgt_seq_len), typically equal decoder_input_ids
        :param kwargs:
        :return:
            tuple of
                loss (if labels are specified)
                combined_logits tensor of shape (batch_size, tgt_seq_len, tgt_vocab_size + src_vocab_size),
                decoder outputs,
                encoder outputs,
        """
        # fmt: off
        kwargs_encoder = {k: v for k, v in kwargs.items() if not k.startswith("decoder_")}
        kwargs_decoder = {k[len("decoder_"):]: v for k, v in kwargs.items() if k.startswith("decoder_")}
        # fmt: on

        # Encode
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                **kwargs_encoder,
            )

        encoder_hidden_states = encoder_outputs[0]
        assert encoder_hidden_states.shape[-1] == self.encoder.config.hidden_size

        # Project encoder representations into decoder-sized dimension
        if self.enc_dec_proj is not None:
            encoder_hidden_states = self.enc_dec_proj(encoder_hidden_states)

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
        assert (
            decoder_hidden_states.shape[-1] == self.decoder.config.hidden_size
        ), "decoder has classification head"

        # compute pointer scores via attending from decoder hiddens to encoder hiddens
        query = self.decoder_q_proj(decoder_hidden_states)  # (bs, tgt_len, decoder_hidden)
        keys = encoder_hidden_states  # (bs, src_len, decoder_hidden)

        # NOTE: this implementation is computationally inefficient during inference
        attention_scores = query @ keys.transpose(1, 2)  # (bs, tgt_len, src_len)
        attention_scores = F.dropout(
            attention_scores, p=self.config.dropout, training=self.training
        )

        # mask becomes 0 for all 1 (keep) positions and -1e4 in all 0 (mask) positions
        # NOTE: we can use this mask to additionaly guide the model
        # batch size is passed using attention_scores tensor instead of input_ids
        # because while generating, input_ids can be None (using cached encoder_outputs)
        pointer_mask = self._get_pointer_attention_mask(
            pointer_mask,
            attention_scores.shape[0],
            dtype=attention_scores.dtype,
            device=attention_scores.device,
        )

        attention_scores_shape = attention_scores.shape
        attention_scores = attention_scores + pointer_mask
        # attention_scores = attention_scores * attention_scores.shape[-1] ** -0.5
        assert attention_scores.shape == attention_scores_shape, "attention scores changed shape"

        # NOTE: maybe add some kind of normalization between dec_logits?
        decoder_hidden_states = F.dropout(
            decoder_hidden_states, p=self.config.dropout, training=self.training
        )
        decoder_logits = self.lm_head(decoder_hidden_states)  # (bs, tgt_len, tgt_vocab_size)
        combined_logits = torch.cat([decoder_logits, attention_scores], dim=-1)

        if labels is None:
            return (combined_logits,) + decoder_outputs + encoder_outputs

        loss = self._compute_loss(combined_logits, labels, decoder_attention_mask)

        return (loss, combined_logits) + decoder_outputs + encoder_outputs

    def _compute_loss(self, input, target, mask):
        input = input.view(-1, input.shape[-1])
        target = target.view(-1)
        if mask is not None:
            mask = mask.view(-1)

        if self.label_smoothing_loss_layer is None:
            loss = F.cross_entropy(
                input,
                target,
                ignore_index=self.decoder.embeddings.word_embeddings.padding_idx,
            )
        else:
            loss = self.label_smoothing_loss_layer(input, target, mask)

        if self.config.move_norm is not None:
            loss += self.config.move_norm * self.get_move_norm()

        if self.config.weight_consolidation is not None:
            loss += self.config.weight_consolidation * self._get_weight_consolidation()

        return loss

    def _get_pointer_attention_mask(
        self, pointer_attention_mask=None, batch_size=None, device=None, dtype=None
    ):
        """
        :param pointer_attention_mask: FloatTensor of shape (batch_size, src_seq_len), padding mask for the pointer
            0 for masking and 1 for no masking
        :param shape: alternative to pointer_attention_mask, tuple (batch_size, tgt_seq_len, src_seq_len)
        :param device: torch.device
        :return: FloatTensor of shape (batch_size, 1, 1)
            attention mask which equals -1e4 for src padding and special tokens and zero otherwise
        """
        device = device or self.device
        dtype = dtype or self.dtype

        if pointer_attention_mask is None:
            return torch.zeros([batch_size, 1, 1], device=device, dtype=dtype)

        # We use -1e4 for masking analogous to Transformers library
        # ideally, this number should depend on dtype and should be
        # bigger for float32 and smaller for float16 and bfloat16
        return ((1.0 - pointer_attention_mask) * -1e4).unsqueeze(1)

    def _reorder_cache(self, past, beam_idx):
        return past

    def freeze_encoder(self, freeze=True):
        value = not freeze

        for param in self.encoder.parameters():
            param.requires_grad = value

    def freeze_decoder(self, freeze=True):
        value = not freeze

        for param in self.decoder.parameters():
            param.requires_grad = value

        if self.enc_dec_proj is None:
            return

        for param in self.enc_dec_proj.parameters():
            param.requires_grad = value

    def freeze_head(self, freeze=True):
        value = not freeze

        for param in self.decoder_q_proj.parameters():
            param.requires_grad = value

        for name, param in self.lm_head.named_parameters():
            param.requires_grad = value

    def get_move_norm(self):
        norm = 0

        for n, p1 in self.named_parameters():
            p2 = self.initial_params[n]
            norm += torch.dist(p1, p2, p=self.config.move_norm_p)

        norm /= len(self.initial_params)
        return norm

    def reset_initial_params(self):
        self.initial_params = {n: p.detach().clone() for n, p in self.named_parameters()}

    def update_grad_squared(self, batch_size=1):
        """
        Updates grad_squared buffer, should be called between .backward() and .zero_grad().

        Used to update gradient moving average during training,
        for example in lightning_module.on_after_backward.

            grad_squared = 0.99 * grad_squared + 0.01 * grad^2 / batch_size

        This trick allows to efficiently estimate grad norms during training instead of
        forwarding full dataset in the end.
        """
        if next(self.parameters()).grad is None:
            raise RuntimeError("You should .backward before calling .update_grad_squared")

        if self._grad_squared_buffer_names is None:
            raise RuntimeError(
                ".grad_squared is None, but trying to update it. "
                "To initialize the model with grad_squared specify track_grad_square=True"
            )

        for name, param in self.named_parameters():
            _grad = param.grad if param.grad is not None else torch.zeros_like(param)
            buffer_name = self._grad_buffer_name(name)

            # read current squared grad value from the buffer
            _old_grad2 = getattr(self, buffer_name)
            _new_grad = 0.99 * _old_grad2 + 0.01 * _grad ** 2 / batch_size

            # write updated value to the buffer
            # e.g. self.grad_squared_my_param = _new_grad
            setattr(self, buffer_name, _new_grad)

        self.zero_grad()

    def _register_grad_squared_buffers(self):
        """Registers buffers to keep track for squares of gradients initialized as zeroes.

        Call this function before training the model if you want to later fine-tune the model with EWC.
        During training exponential mean of the grad squares is stored in these buffers.

        self.grad_squared_parameter_name = 0.99 * self.grad_squared_parameter_name + 0.01 * parameter.grad ** 2

        Returns:
            set(str), grad squared buffer names
        """
        grad_squared_names = set()

        for name, param in self.named_parameters():
            buffer_name = self._grad_buffer_name(name)
            self.register_buffer(buffer_name, torch.zeros_like(param))
            grad_squared_names.add(buffer_name)

        return grad_squared_names

    def _get_weight_consolidation(self):

        reg = 0
        n_params = 0
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue

            grad_squared = getattr(self, self._grad_buffer_name(n))
            delta = self.initial_params[n] - p
            reg += torch.sum(grad_squared * delta ** 2)
            n_params += p.numel()

        reg /= n_params

        return reg

    @staticmethod
    def _grad_buffer_name(param_name):
        return "grad_squared_" + param_name.replace(".", "_")
