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
"""Reformat Snips into TOP format and save .tsv files to snips_path

Splits snips train data into train and valid. Valid size is 700 (same size as test).
Creates snips_path/train.tsv, snips_path/eval.tsv, and snips_path/test.tsv
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

from new_semantic_parsing import utils


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--snips-path', required=True,
                        help="path to snips repository containing 2017-06-custom-intent-engines")
    # fmt: on

    args = parser.parse_args()

    snips_train_pattern = os.path.join(
        args.snips_path, "2017-06-custom-intent-engines/*/train*full.json"
    )
    snips_test_pattern = os.path.join(
        args.snips_path, "2017-06-custom-intent-engines/*/validate*.json"
    )

    logger.info("Reformatting snips into TOP format")
    snips_trainval: pd.DataFrame = utils.make_snips_df(glob.glob(snips_train_pattern))
    snips_test: pd.DataFrame = utils.make_snips_df(glob.glob(snips_test_pattern))

    logger.info("Creating train/valid split")
    permutation = np.random.permutation(len(snips_trainval))
    train_subset_ids = permutation[700:]
    valid_subset_ids = permutation[:700]

    snips_train = snips_trainval.iloc[train_subset_ids]
    snips_valid = snips_trainval.iloc[valid_subset_ids]

    new_snips_path = os.path.join(args.snips_path, "top_format")
    train_path = os.path.join(new_snips_path, "train.tsv")
    valid_path = os.path.join(new_snips_path, "eval.tsv")
    test_path = os.path.join(new_snips_path, "test.tsv")

    logger.info(f"Saving preprocessed files to {new_snips_path}")
    os.makedirs(new_snips_path)

    snips_train.to_csv(train_path, sep="\t", index=False, header=False)
    snips_valid.to_csv(valid_path, sep="\t", index=False, header=False)
    snips_test.to_csv(test_path, sep="\t", index=False, header=False)

    logger.info("Done")
