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
"""Preprocess text data and save binary Dataset objects along with tokenizers to a directory."""

import os
import sys
import logging
import argparse
from functools import reduce
from os.path import join as path_join
from random import shuffle

import toml
import torch
import pandas as pd

import transformers

import new_semantic_parsing.data
from new_semantic_parsing import (
    utils,
    TopSchemaTokenizer,
    SAVE_FORMAT_VERSION,
)


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--data", required=True,
                        help="path to TOP dataset directory")
    parser.add_argument("--text-tokenizer", required=True,
                        help="pratrained tokenizer name or path to a saved tokenizer")
    parser.add_argument("--output-dir", required=True,
                        help="directory to save preprocessed data")
    parser.add_argument("--seed", default=34)

    # splitting parameters
    parser.add_argument("--split-class", default=None,
                        help="remove --split-ratio of the class from the training dataset and make a finetune_data; "
                             "do not perform split by default")
    parser.add_argument("--split-amount", default=None, type=float,
                        help="0 < --split-amount < 1, amount of data to remove from the training dataset")
    # fmt: on

    args = parser.parse_args(args)

    if args.split_amount is not None:
        if not 0.0 < args.split_amount < 1.0:
            raise ValueError("--split-amount should be between 0. and 1.")

    if args.split_class is not None:
        if args.split_amount is None:
            raise ValueError("--split-amount should be specified if --split-class is provided")

    return args


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    # File structure:
    # that's text\tthat 's text\t[IN:UNSUPPORTED that 's text]
    train_path = path_join(path_join(args.data, "train.tsv"))
    train_data = pd.read_table(train_path, names=["text", "tokens", "schema"])
    full_train_data_size = len(train_data)  # used to check the train/finetune split
    finetune_data, finetune_path = None, None

    schema_vocab = reduce(set.union, map(utils.get_vocab_top_schema, train_data.schema))

    if args.split_amount is not None:
        # NOTE: this is not train/eval split, this is train/finetune split
        # finetune part is not used by train script, but used by retrain script

        # Get a small set of examples that contains all classes from schema_vocab
        required_example_ids = utils.get_required_example_ids(schema_vocab, train_data)

        logger.info("Splitting the training dataset")
        split_ids = list(set(range(len(train_data))) - required_example_ids)

        if args.split_class is not None:
            split_ids = [
                i for i, schema in enumerate(train_data.schema) if args.split_class in schema
            ]
            logger.info(
                f"Moving {100 * args.split_amount}% of {args.split_class} into a finetuning subset"
            )

        take = int(len(split_ids) * args.split_amount)
        leave = len(split_ids) - take

        assert take > 0

        logger.info(f"Taking {take} examples and leaving {leave} examples")

        shuffle(split_ids)
        subset_ids = split_ids[:take]
        train_data_ids = list(set(range(len(train_data))) - set(subset_ids) | required_example_ids)

        finetune_data = train_data.iloc[subset_ids]
        train_data = train_data.iloc[train_data_ids]

        os.makedirs(args.output_dir)

        finetune_path = path_join(args.output_dir, "finetune.tsv")
        logger.info(f"Saving the finetune_data to {finetune_path}")
        finetune_data.to_csv(finetune_path, sep="\t", index=False, header=False)

        train_path = path_join(args.output_dir, "train.tsv")
        logger.info(f"Saving the modified training set to {train_path}")
        train_data.to_csv(train_path, sep="\t", index=False, header=False)

    logger.info("Getting schema vocabulary")

    if args.split_amount is not None:
        finetune_schema_vocab = reduce(
            set.union, map(utils.get_vocab_top_schema, finetune_data.schema)
        )
        vocab_delta = finetune_schema_vocab - schema_vocab
        if len(vocab_delta) > 0:
            logger.warning(
                f"Finetuning subset contains vocabulary elements not from the training subset"
            )
            logger.warning(f"New elements: {', '.join(vocab_delta)}")

    logger.info(f"Schema vocabulary size: {len(schema_vocab)}")

    logger.info("Building tokenizers")
    text_tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_tokenizer, use_fast=True)
    schema_tokenizer = TopSchemaTokenizer(schema_vocab, text_tokenizer)

    logger.info("Tokenizing train dataset")
    train_dataset = new_semantic_parsing.data.make_dataset(train_path, schema_tokenizer)

    logger.info("Tokenizing validation and test datasets")
    valid_dataset = new_semantic_parsing.data.make_dataset(
        path_join(args.data, "eval.tsv"), schema_tokenizer
    )
    test_dataset = new_semantic_parsing.data.make_dataset(
        path_join(args.data, "test.tsv"), schema_tokenizer
    )

    finetune_dataset = None
    if args.split_amount is not None:
        logger.info("Tokenizing finetune set")
        finetune_dataset = new_semantic_parsing.data.make_dataset(finetune_path, schema_tokenizer)

        logger.info(f"Original train set size: {full_train_data_size}")
        logger.info(f"Reduced  train set size: {len(train_dataset)}")
        logger.info(f"Finetune       set size: {len(finetune_dataset)}")

        train_finetune_data_size = len(train_dataset) + len(finetune_dataset)
        if train_finetune_data_size != full_train_data_size:
            raise RuntimeError(f"{train_finetune_data_size} != {full_train_data_size}")

    logger.info(f"Saving config, data and tokenizer to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    # text tokenizer is saved along with schema_tokenizer
    model_type = None
    if not os.path.exists(args.text_tokenizer):
        model_type = utils.get_model_type(args.text_tokenizer)

    schema_tokenizer.save(path_join(args.output_dir, "tokenizer"), encoder_model_type=model_type)

    data_state = {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "finetune_dataset": finetune_dataset,
        "version": SAVE_FORMAT_VERSION,
    }

    torch.save(data_state, path_join(args.output_dir, "data.pkl"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
