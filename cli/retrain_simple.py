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
"""Finetune a trained model on a dataset.

Simpler version of retrain.py that only restores model weights.
Some of the cli arguments use saved train args as defaults.
"""

import os
import sys
import shutil
import pprint
import logging
from os.path import join as path_join

import toml
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers

import new_semantic_parsing as nsp

from new_semantic_parsing import utils, cli_utils

from cli import train_lightning as cli_train
from cli import retrain as cli_retrain


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_args(args):
    if args.lr is None and (args.encoder_lr is None or args.decoder_lr is None):
        raise ValueError("--lr or --encoder-lr and --decoder-lr should be specified")

    if args.new_classes is None:
        logger.warning("--new-classes is not specified for finetuning")


def set_default_args(args):
    train_args = cli_utils.load_saved_args(path_join(args.model_dir, "args.toml"))

    if args.batch_size is None:
        args.batch_size = train_args["batch_size"]

    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = train_args["gradient_accumulation_steps"]

    if args.max_grad_norm is None:
        args.max_grad_norm = train_args["max_grad_norm"]

    if args.label_smoothing is None:
        args.label_smoothing = train_args["label_smoothing"]

    if args.weight_decay is None:
        args.weight_decay = train_args["weight_decay"]

    args.warmup_steps = args.warmup_steps or 0

    return args


def main(args):
    utils.set_seed(args.seed)

    if os.path.exists(args.output_dir):
        raise ValueError(f"output_dir {args.output_dir} already exists")

    wandb_logger = pl.loggers.WandbLogger(project=args.wandb_project, tags=args.tags)
    wandb_logger.log_hyperparams(args)

    logger.info(f"Starting finetuning with args: \n{pprint.pformat(vars(args))}")

    logger.info("Loading tokenizers")
    schema_tokenizer = cli_retrain.load_tokenizer(args.model_dir, args.data_dir)

    logger.info("Loading data")
    train_dataset, eval_dataset = cli_retrain.load_data(
        path=path_join(args.data_dir, "data.pkl"),
        new_data_amount=args.new_data_amount,
        old_data_amount=args.old_data_amount,
        old_data_sampling_method=args.old_data_sampling_method,
        wandb_logger=wandb_logger,
    )
    train_args = cli_utils.load_saved_args(path_join(args.model_dir, "args.toml"))

    # NOTE: do not log metrics as hyperparameters
    wandb_logger.log_hyperparams(
        {"pretrain_" + k: v for k, v in train_args.items() if k != "metrics"}
    )

    wandb_logger.log_hyperparams({"num_total_data": len(train_dataset)})

    logger.info("Loading model")
    model = cli_retrain.load_model(
        args.model_dir, args.dropout, args.move_norm, args.move_norm_p, args.label_smoothing
    )

    logger.info("Preparing for training")

    max_tgt_len = train_args.get("max_tgt_len", train_dataset)

    lightning_module = cli_train.make_lightning_module(
        model, schema_tokenizer, train_dataset, eval_dataset, max_tgt_len, args, wandb_logger
    )
    trainer = cli_train.make_trainer(args, wandb_logger)

    # get evaluation metrics of the initial model
    pretrain_metrics = train_args["metrics"]
    _first_step_metrics = {
        "epoch": -1,
        "global_step": -1,
        **pretrain_metrics["means"],
        **pretrain_metrics["stdevs"],
    }
    wandb_logger.log_metrics(_first_step_metrics, step=-1)

    # --- FIT

    cli_utils.check_config(lightning_module, trainer, args, strict=False)
    if model.config.move_norm is not None:
        assert torch.allclose(model.get_move_norm(), torch.zeros(1, device=model.device))

    trainer.fit(lightning_module)

    cli_utils.check_config(lightning_module, trainer, args, strict=False)

    with open(path_join(args.output_dir, "args.toml"), "w") as f:
        args_dict = {"version": nsp.SAVE_FORMAT_VERSION, **vars(args)}
        toml.dump(args_dict, f)

    logger.info("Training finished!")
    final_metrics, description = cli_utils.evaluate_model(
        trainer.checkpoint_callback.last_checkpoint_path,
        schema_tokenizer,
        eval_dataset,
        prefix="eval",
        max_len=train_args.get("max_tgt_len", 68),  # 68 is max_tgt_len for TOP
    )

    logger.info(description)
    wandb_logger.log_metrics({**final_metrics["means"], **final_metrics["stdevs"]})

    # Compute RI and RD
    class_weights = eval_dataset.get_class_frequencies(schema_tokenizer)
    class_weights = {f"cls/eval_{cls}_tree_path_f1": p for cls, p in class_weights.items()}

    finetuning_metrics = cli_utils.evaluate_finetuning_procedure(
        pretrain_metrics, final_metrics, class_weights
    )
    wandb_logger.log_metrics(finetuning_metrics)

    wandb_logger.close()

    if args.clean_output:
        shutil.rmtree(args.output_dir)


if __name__ == "__main__":
    args = cli_retrain.parse_args()
    check_args(args)
    args = set_default_args(args)

    main(args)
