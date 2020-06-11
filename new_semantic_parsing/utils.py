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
import numpy as np
import torch
import transformers

from new_semantic_parsing.dataclasses import InputDataClass


class PointerDataset(torch.utils.data.Dataset):
    def __init__(self, source_tensors, target_tensors=None, source_pointer_masks=None, target_pointer_masks=None):
        """
        :param source_tensors: list of tensors, input ids
        :param target_tensors: list of tensors, labels
        :param target_pointer_masks: list of tensors, mask showing pointer locations in labels
        """
        self.source_tensors = source_tensors
        self.target_tensors = target_tensors
        self.target_pointer_masks = target_pointer_masks
        self.source_pointer_masks = source_pointer_masks

        self.torchified = isinstance(source_tensors[0], torch.Tensor)
        if target_tensors is not None:
            self.torchified = self.torchified and isinstance(target_tensors[0], torch.Tensor)
        if source_pointer_masks is not None:
            self.torchified = self.torchified and isinstance(source_pointer_masks[0], torch.Tensor)
        if target_pointer_masks is not None:
            self.torchified = self.torchified and isinstance(target_pointer_masks[0], torch.Tensor)

    def __len__(self):
        return len(self.source_tensors)

    def __getitem__(self, item) -> InputDataClass:
        source_pointer_mask = None
        if self.source_pointer_masks is not None:
            source_pointer_mask = self.source_pointer_masks[item]

        if self.target_tensors is None:
            return InputDataClass(
                input_ids=self.source_tensors[item],
                pointer_mask=source_pointer_mask,
            )

        target_pointer_mask = None
        if self.target_pointer_masks is not None:
            target_pointer_mask = self.target_pointer_masks[item]

        return InputDataClass(
            input_ids=self.source_tensors[item],
            pointer_mask=source_pointer_mask,
            decoder_input_ids=self.target_tensors[item],
            decoder_pointer_mask=target_pointer_mask,
            labels=self.target_tensors[item],
        )

    def torchify(self):
        if self.torchified:
            return

        self.source_tensors = [torch.LongTensor(t) for t in self.source_tensors]

        if self.target_tensors is not None:
            self.target_tensors = [torch.LongTensor(t) for t in self.target_tensors]
        if self.source_pointer_masks is not None:
            self.source_pointer_masks = [torch.FloatTensor(t) for t in self.source_pointer_masks]
        if self.target_pointer_masks is not None:
            self.target_pointer_masks = [torch.FloatTensor(t) for t in self.target_pointer_masks]

        self.torchified = True


class PaddedDataCollator(transformers.DataCollator):
    """This data collator assumes that all examples are padded to the same length"""
    def collate_batch(self, examples):
        batch = dict()

        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            batch[k] = torch.stack([getattr(ex, k) for ex in examples])

        return batch


class Seq2SeqDataCollator(transformers.DataCollator):
    """Pads tensors to the maximum length in batch.
    Length is different for encoder and decoder inputs.

    Decoder inputs should have prefix `decoder_`
    `labels` considered a decoder field too
    All other tensors considered encoder inputs

    All values in the input DataClasses should be torch.Tensor or shape (seq_len, *)
    or None, None values are ignored

    All values corresponsing to the keys ending with `mask` are padded with zeroes
    """
    def __init__(self, pad_id, decoder_pad_id=None):
        self.encoder_pad_id = pad_id
        self.decoder_pad_id = decoder_pad_id or pad_id

        self._encoder_max_len = None
        self._decoder_max_len = None

    def collate_batch(self, examples):
        """
        :param examples: list of DataClass
        :return: dict with the DataClass fields
        """
        batch = dict()
        batch_size = len(examples)

        # iterate ofer the first example to get shapes
        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            is_decoder = self._is_decoder_field(k)

            maxlen = max(getattr(ex, k).shape[0] for ex in examples)
            self._shape_check(maxlen, is_decoder, k)

            batched_shape = (batch_size, maxlen, *v.shape[1:])
            batch[k] = torch.zeros(batched_shape, dtype=v.dtype, device=v.device)

            if k.endswith('mask'):
                continue

            batch[k].fill_(self.decoder_pad_id if is_decoder else self.encoder_pad_id)

        return batch

    @staticmethod
    def _is_decoder_field(field_name):
        return field_name.startswith('decoder_') or field_name == 'labels'

    def _shape_check(self, maxlen, is_decoder, key):
        """Data shape validation"""
        if is_decoder:
            if self._decoder_max_len is not None and self._decoder_max_len != maxlen:
                raise ValueError(f'decoder input tensors have different lengths ({key})')
            self._decoder_max_len = maxlen
        else:
            if self._encoder_max_len is not None and self._encoder_max_len != maxlen:
                raise ValueError(f'encoder input tensors have different lengths({key})')
            self._encoder_max_len = maxlen


def get_vocab_top_schema(text):
    schema_tokens = {'[', ']', 'IN:', 'SL:'}

    text = text.replace('[', '')
    text = text.replace(']', '')

    for token in text.split(' '):
        if token[:3] in ['IN:', 'SL:']:
            schema_tokens.add(token[3:])
    return schema_tokens


def get_src_pointer_mask(input_ids, tokenizer: transformers.PreTrainedTokenizer):
    """Compute mask which ignores padding and special tokens"""
    mask = np.ones(len(input_ids))
    if input_ids[0] == tokenizer.cls_token_id:
        mask[0] = 0
    for i, token_id in enumerate(input_ids):
        if token_id in (tokenizer.sep_token_id, tokenizer.pad_token_id):
            mask[i] = 0
    return mask


def compute_metrics(eval_prediction: transformers.EvalPrediction):
    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    accuracy = np.mean(predictions.reshape(-1) == eval_prediction.label_ids.reshape(-1))
    exact_match = np.mean(np.all(predictions == eval_prediction.label_ids, axis=1))

    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
    }


def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    import numpy
    numpy.random.seed(seed)
    import random
    random.seed(seed)


def get_lr(model: transformers.PreTrainedModel):
    """Get optimal learning rate according to the Scaling Laws
    https://arxiv.org/abs/2001.08361

    lr ~= 0.003239 - 0.0001395 log(n_non_embedding_params)
    """

    if hasattr(model, 'embeddings'):
        n_embedding_params = get_n_embed_params_trainable(model.embeddings)
    elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        n_embed_encoder = get_n_embed_params_trainable(model.encoder.embeddings)
        n_embed_decoder = get_n_embed_params_trainable(model.decoder.embeddings)
        n_embedding_params = n_embed_encoder + n_embed_decoder
    else:
        raise ValueError('Model object should have .embeddings or'
                         '.encoder.embeddings and .decoder.embeddings')

    n_non_embedding_params = model.num_parameters(only_trainable=True) - n_embedding_params
    return 0.003239 - 0.0001395 * np.log(n_non_embedding_params)


def get_n_embed_params_trainable(embeddings):
    """
    :param embeddings: BertEmbeddings
    :returns: number of trainable embedding parameters
    """
    n_params = 0

    tok = embeddings.word_embeddings
    if tok.training:
        n_params += tok.num_embeddings * tok.embedding_dim

    pos = embeddings.position_embeddings
    if pos.training:
        n_params += pos.num_embeddings * pos.embedding_dim

    typ = embeddings.token_type_embeddings
    if typ.training:
        n_params += typ.num_embeddings * typ.embedding_dim

    return n_params


def get_model_type(model_name):
    """Search for a largest substring from transformers.CONFIG_MAPPING"""
    candidate = ''

    for name in transformers.CONFIG_MAPPING:
        if name in model_name and len(name) > len(candidate):
            candidate = name

    if len(candidate) == 0:
        raise ValueError(f'{model_name} is not found in transformers.CONFIG_MAPPING')

    return candidate
