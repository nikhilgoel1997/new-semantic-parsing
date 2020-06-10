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
import os
import sys
import argparse
import logging
from pathlib import Path
from functools import reduce

import toml
import torch
import pandas as pd

import transformers
import tokenizers

from tqdm import tqdm

from new_semantic_parsing import TopSchemaTokenizer
from new_semantic_parsing.utils import get_vocab_top_schema
from new_semantic_parsing.utils import PointerDataset, set_seed


SAVE_FORMAT_VERSION = '0.1-nightly-Jun10'


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('train')


def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True,
                        help='path to TOP dataset directory')
    parser.add_argument('--text-tokenizer', required=True,
                        help='pratrained tokenizer name or path to a saved tokenizer')
    parser.add_argument('--save-dir', required=True,
                        help='directory to save preprocessed data')
    parser.add_argument('--seed', default=34)

    # NOTE: maybe store training args too?
    return parser.parse_args(args)


def make_dataset(filepath, text_tokenizer, schema_tokenizer):
    data = pd.read_table(filepath, names=['text', 'tokens', 'schema'])

    text_ids = [text_tokenizer.encode(text) for text in tqdm(data.tokens)]

    schema_ids = []
    pointer_masks = []

    for i, schema in tqdm(enumerate(data.schema), total=len(data)):

        item = schema_tokenizer.encode_plus(schema, text_ids[i])
        schema_ids.append(item.ids)
        pointer_masks.append(item.pointer_mask)

    dataset = PointerDataset(text_ids, schema_ids, pointer_masks)
    dataset.torchify()

    return dataset


if __name__ == '__main__':
    args = parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)

    if save_dir.exists():
        raise ValueError(f'save_dir {save_dir.as_posix()} already exists')

    # File structure:
    # that's text\tthat 's text\t[IN:UNSUPPORTED that 's text]
    train_data = pd.read_table(Path(args.data)/'train.tsv', names=['text', 'tokens', 'schema'])

    logger.info('Getting schema vocabulary')
    schema_vocab = reduce(set.union, map(get_vocab_top_schema, train_data.schema))
    logger.info(f'Schema vocabulary size: {len(schema_vocab)}')

    logger.info('Building tokenizers')
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_tokenizer)
    schema_tokenizer = TopSchemaTokenizer(schema_vocab, text_tokenizer)

    logger.info('Tokenizing train dataset')
    train_dataset = make_dataset(os.path.join(args.data, 'train.tsv'), text_tokenizer, schema_tokenizer)

    logger.info('Tokenizing validation and test datasets')
    valid_dataset = make_dataset(os.path.join(args.data, 'eval.tsv'), text_tokenizer, schema_tokenizer)
    test_dataset = make_dataset(os.path.join(args.data, 'test.tsv'), text_tokenizer, schema_tokenizer)

    logger.info(f'Saving everything to {save_dir.as_posix()}')
    os.makedirs(args.save_dir)

    with open(save_dir/'args.toml', 'w') as f:
        args_dict = {'version': SAVE_FORMAT_VERSION}
        args_dict.update(**vars(args))
        toml.dump(args_dict, f)

    # text tokenizer is saved along with schema_tokenizer
    schema_tokenizer.save(save_dir/'tokenizer')

    data_state = {
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
        'test_dataset': test_dataset,
        'version': SAVE_FORMAT_VERSION,
    }

    torch.save(data_state, save_dir/'data.pkl')
