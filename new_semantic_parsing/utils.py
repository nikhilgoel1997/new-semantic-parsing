# Copyright 2020 Google LLC
# Copyright 2020 The HuggingFace Inc. team.
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
"""Utilities used across code.

Include fixing random seeds, metrics computation, learning rate selection, model loading, and prediction.
"""
import random
import torch
import numpy as np
import transformers


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_vocab_top_schema(text):
    schema_tokens = {"[", "]", "IN:", "SL:"}

    text = text.replace("[", "")
    text = text.replace("]", "")

    for token in text.split(" "):
        if token[:3] in ["IN:", "SL:"]:
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


def get_model_type(model_name):
    """Search for a largest substring from transformers.CONFIG_MAPPING"""
    candidate = ""

    for name in transformers.CONFIG_MAPPING:
        if name in model_name and len(name) > len(candidate):
            candidate = name

    if len(candidate) == 0:
        raise ValueError(f"{model_name} is not found in transformers.CONFIG_MAPPING")

    return candidate


def make_subset(dataset, subset_size):
    """Make torch Subset by randomly sampling indices from dataset

    Args:
        dataset: torch Dataset
        subset_size: float, 0 < subset_size < 1
    """
    if subset_size == 1:
        return dataset

    if not (0 < subset_size < 1):
        raise ValueError(subset_size)

    _subset_size = int(subset_size * len(dataset))
    _subset_ids = np.random.permutation(len(dataset))[:_subset_size]

    _subset = torch.utils.data.Subset(dataset, indices=_subset_ids)
    return _subset


def get_required_example_ids(schema_vocab, train_data):
    """Find a subset of train_data that contains all schema_vocab tokens.
    
    Args:
        schema_vocab: set of str, required schema tokens
        train_data: pd.DataFrame with field "schema"

    Returns:
        a set of train_data ids
    """
    required_schema_vocab = set()
    required_example_ids = set()

    for i, row in train_data.iterrows():
        add_this = False
        tokens_not_present = schema_vocab.difference(required_schema_vocab)
        
        # Add the example id to required_example_ids if the example
        # contains a schema token not present in the required_schema_vocab
        for token in tokens_not_present:
            if token in row.schema:
                add_this = True
                required_schema_vocab.add(token)

        if add_this:
            required_example_ids.add(i)

        if required_schema_vocab == schema_vocab:
            break
    else:
        raise RuntimeError("Full vocabulary was not found in the training set")

    return required_example_ids
