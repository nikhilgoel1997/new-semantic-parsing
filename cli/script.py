#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:05:58 2020

@author: rahul
"""
import sys
sys.path.append('/home/rahul/Downloads/new-semantic-parsing-master/new_semantic_parsing')
import os
import pandas as pd
import toml
import logging
from convert_data import preprocess_data
from tqdm.auto import tqdm
import new_semantic_parsing as nsp
from new_semantic_parsing.data import PointerDataset
from os.path import join as path_join
import torch
import argparse
import tempfile
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from new_semantic_parsing.dataclasses import InputDataClass, List, Tensor, PairItem
from new_semantic_parsing import utils, cli_utils, optimization
from new_semantic_parsing.utils import get_src_pointer_mask, make_subset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(os.path.basename(__file__))
logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(
    model_dir,
    dropout=None,
    move_norm=None,
    move_norm_p=None,
    label_smoothing=None,
    weight_consolidation=None,
):
    """Load a trained model and override some model properties if specified."""
    model_config = nsp.EncoderDecoderWPointerConfig.from_pretrained(model_dir)
    if dropout is not None:
        model_config.set_dropout(dropout)
    if move_norm is not None:
        model_config.move_norm = move_norm
    if move_norm_p is not None:
        model_config.move_norm_p = move_norm_p
    if label_smoothing is not None:
        model_config.label_smoothing = label_smoothing
    if weight_consolidation is not None:
        model_config.weight_consolidation = weight_consolidation

    model = nsp.EncoderDecoderWPointerModel.from_pretrained(model_dir, config=model_config)
    model.reset_initial_params()
    return model

def load_lightning_module(
    checkpoint_path: str,
    model: nsp.EncoderDecoderWPointerModel,
    train_dataset: nsp.PointerDataset,
    eval_dataset: nsp.PointerDataset,
    schema_tokenizer: nsp.TopSchemaTokenizer,
    args: argparse.Namespace,
    wandb_logger: pl.loggers.WandbLogger,
):
    """Loads lightning module with some of the parameters overwritten."""

    wandb_logger.log_hyperparams({"new_classes": " ".join(args.new_classes)})

    # Lightning loads all params which are not specified in .load_from_checkpoint
    # thus, some arguments are only provided if we want to override the loaded values
    module_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "log_every": args.log_every,
        "batch_size": args.batch_size,
    }
    module_kwargs = {k: v for k, v in module_kwargs.items() if v is not None}

    # always overwrite freezing schedule because global_step starts from zero
    module_kwargs["freezing_schedule"] = nsp.dataclasses.EncDecFreezingSchedule.from_args(args)

    lightning_module = nsp.PointerModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        valid_dataset = eval_dataset,
        monitor_classes=args.new_classes,
        **module_kwargs,
    )

    return lightning_module

def iterative_prediction(
    model: nsp.EncoderDecoderWPointerModel,
    dataloader,
    schema_tokenizer: nsp.TopSchemaTokenizer,
    max_len,
    num_beams,
    device="cpu",
    return_tokens=False,
):
    """Executes inference-time prediction loop.

    Returns:
        A tuple of two elements (predictions_ids, predictions_str)
            predictions_ids: list of np.arrays
            predictions_str: list of strings if return_tokens is False
                or list of lists of strings if return_tokens is True
    """
    model = model.to(device)

    predictions_ids = []
    predictions_str = []
    text_tokenizer = schema_tokenizer.src_tokenizer

    for batch in tqdm(dataloader, desc="generation"):
        prediction_batch: torch.LongTensor = model.generate(
            input_ids=batch["input_ids"].to(device),
            pointer_mask=batch["pointer_mask"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_length=max_len,
            num_beams=num_beams,
            pad_token_id=text_tokenizer.pad_token_id,
            bos_token_id=schema_tokenizer.bos_token_id,
            eos_token_id=schema_tokenizer.eos_token_id,
        )

        for i, prediction in enumerate(prediction_batch):
            prediction = [
                p for p in prediction.cpu().numpy() if p not in schema_tokenizer.special_ids
            ]
            predictions_ids.append(prediction)

            prediction_str: str = schema_tokenizer.decode(
                prediction,
                batch["input_ids"][i],
                skip_special_tokens=True,
                return_tokens=return_tokens,
            )
            predictions_str.append(prediction_str)

    return predictions_ids, predictions_str

def load_saved_args(path):
    with open(path) as f:
        train_args = toml.load(f)
        if train_args["version"] != nsp.SAVE_FORMAT_VERSION:
            logger.warning(
                "Binary data version differs from the current version. "
                "This may cause failing and unexpected behavior."
            )

        if "metrics" in train_args:
            # for some reason means and stdevs are saved as strings
            train_args["metrics"]["means"] = {
                k: float(v) for k, v in train_args["metrics"]["means"].items()
            }
            train_args["metrics"]["stdevs"] = {
                k: float(v) for k, v in train_args["metrics"]["stdevs"].items()
            }

    return train_args

def parse_args(args=None):
    """Parses cli arguments.

    This function is shared between retrain.py and retrain_simple.py
    """
    parser = argparse.ArgumentParser()

    # fmt: off

    # data
    parser.add_argument("--data-dir", required=True,
                        help="Path to preprocess.py --save-dir containing tokenizer, "
                             "data.pkl, and args.toml")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to store checkpoints and other output files")
    parser.add_argument("--eval-data-amount", default=1., type=float,
                        help="amount of validation set to use when training. "
                             "The final evaluation will use the full dataset.")
    parser.add_argument("--new-classes", default=None,
                        help="names of classes to track")

    parser.add_argument("--new-data-amount", default=1., type=float,
                        help="Amount of new data (finetune_set) to train on, 0 < amount <= 1")
    parser.add_argument("--old-data-amount", default=0., type=float,
                        help="Amount of old data (train_set) to train on, only values from {0, 1} are supported")
    parser.add_argument("--old-data-sampling-method", default="merge_subset",
                        help="how to sample from old data")
    parser.add_argument("--average-checkpoints", default=False, action="store_true")
    parser.add_argument("--new-model-weight", default=0.5, type=float)

    # model
    parser.add_argument("--model-dir", required=True,
                        help="Model directory containing 1) checkpoint loadable via "
                             "EncoderDecoderWPointerModel.from_pretrained and "
                             "2) tokenizer directory")

    # training
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--min-epochs", default=1, type=int)
    parser.add_argument("--max-steps", default=None, type=int)
    parser.add_argument("--min-steps", default=None, type=int)
    parser.add_argument("--early-stopping", default=None, type=int,
                        help="Early stopping patience. No early stopping by default.")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=None, type=float,
                        help="By default, checkpoint lr is used.")
    parser.add_argument("--encoder-lr", default=None, type=float,
                        help="Encoder learning rate, overrides --lr")
    parser.add_argument("--decoder-lr", default=None, type=float,
                        help="Decoder learning rate, overrides --lr")

    parser.add_argument("--weight-decay", default=None, type=float)
    parser.add_argument("--dropout", default=None, type=float,
                        help="Dropout amount for the encoder and decoder, by defalut checkpoint value is used")
    parser.add_argument("--warmup-steps", default=0, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--label-smoothing", default=None, type=float)

    # --- retrain-specific
    parser.add_argument("--move-norm", default=None, type=float,
                        help="Regularization coefficient for the distance between the initial and current network")
    parser.add_argument("--move-norm-p", default=2, type=int,
                        help="Parameter p of the L-p norm used in move-norm regularization")
    parser.add_argument("--no-opt-state", default=False, action="store_true",
                        help="Initialize optimizer state randomly instead of loading it from the trainer checkpoint")
    parser.add_argument("--no-lr-scheduler", default=False, action="store_true",
                        help="Keep learning rate constant instead of scheduling it. Only works with retrain_simple.")
    parser.add_argument("--weight-consolidation", default=None, type=float,
                        help="Weight consolidation regularization strength.")

    # --- freezing
    parser.add_argument("--freeze-encoder", default=None, type=int,
                        help="Step to freeze encoder")
    parser.add_argument("--unfreeze-encoder", default=None, type=int,
                        help="Step to unfreeze encoder")
    parser.add_argument("--freeze-decoder", default=None, type=int,
                        help="Step to freeze decoder")
    parser.add_argument("--unfreeze-decoder", default=None, type=int,
                        help="Step to unfreeze decoder")
    parser.add_argument("--freeze-head", default=None, type=int,
                        help="Step to freeze head")
    parser.add_argument("--unfreeze-head", default=None, type=int,
                        help="Step to unfreeze head")

    # misc
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--log-every", default=None, type=int)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--gpus", default=None, type=int,
                        help="Number of gpus to train the model on")
    parser.add_argument("--clean-output", default=False, action="store_true")
    parser.add_argument("--split-amount-finetune", default=None, type=float,
                        help="Only used for logging, amount of data that was removed from the training set")

    # fmt: on

    args = parser.parse_args(args)

    # set defaults for None fields

    if (args.encoder_lr is not None) ^ (args.decoder_lr is not None):
        raise ValueError("--encoder-lr and --decoder-lr should be both specified")

    if args.encoder_lr is None and args.lr is not None:
        args.encoder_lr = args.lr
        args.decoder_lr = args.lr

    if args.lr is None and args.encoder_lr is not None:
        args.lr = {"encoder_lr": args.encoder_lr, "decoder_lr": args.decoder_lr}

    args.wandb_project = args.wandb_project or "new_semantic_parsing"
    args.tags = args.tags.split(",") if args.tags else []  # list is required by wandb interface
    args.new_classes = args.new_classes.split(",") if args.new_classes else []

    if args.split_amount_finetune is not None:
        args.split_amount_train = 1.0 - args.split_amount_finetune

    if args.gpus is None:
        args.gpus = 1 if torch.cuda.is_available() else 0

    if args.output_dir is None:
        args.output_dir = os.path.join("output_dir", next(tempfile._get_candidate_names()))

    if not (0 < args.new_data_amount <= 1):
        raise ValueError(f"--new-data-amount should be between 0 and 1 (exclusive)")

    if not (0 <= args.old_data_amount <= 1):
        raise ValueError(f"--old-data-amount should be between 0 and 1 (inclusive)")

    if not hasattr(nsp.dataclasses.SamplingMethods, args.old_data_sampling_method):
        raise ValueError(args.old_data_sampling_method)

    return args
def load_tokenizer(model_dir, data_dir):
    tokenizer_path1 = model_dir
    tokenizer_path2 = path_join(data_dir, "tokenizer")

    if os.path.exists(path_join(tokenizer_path1, "schema_vocab.txt")):
        schema_tokenizer = nsp.TopSchemaTokenizer.load(tokenizer_path1)
    elif os.path.exists(tokenizer_path2):
        schema_tokenizer = nsp.TopSchemaTokenizer.load(tokenizer_path2)
    else:
        raise ValueError("Tokenizer is not found in both --model-dir and --data-dir.")

    return schema_tokenizer

def main(args):
    num_beams = 1

    names=["text", "tokens", "schema"]

    model = load_model(
        model_dir=args.model_dir,
        dropout=args.dropout,
        move_norm=args.move_norm,
        move_norm_p=args.move_norm_p,
        label_smoothing=args.label_smoothing,
        weight_consolidation=args.weight_consolidation,
    )

    wandb_logger = pl.loggers.WandbLogger(project=args.wandb_project, tags=args.tags)
 
    schema_tokenizer = load_tokenizer(args.model_dir, args.data_dir)
    wandb_logger.log_hyperparams(args)
    flag = 1
    train_args = load_saved_args(path_join(args.model_dir, "args.toml"))
    lightning_module = load_lightning_module(
        checkpoint_path=train_args["pl_checkpoint_path"],
        model=model,
        train_dataset=None,
        eval_dataset=None,
        schema_tokenizer=schema_tokenizer,
        args=args,
        wandb_logger=wandb_logger,
        )
    while(flag != '0'):
        data = pd.DataFrame(index=None, columns=names)
        data.loc[0] = train_data.loc[9]
        data.loc[0].text = input("input text: ")
        data.loc[0].tokens = data.loc[0].text
        data.loc[0].schema = data.loc[0].text
        max_len = None
        data = preprocess_data(data)
        
        text_ids: List[list] = [
            schema_tokenizer.encode_source(text) for text in tqdm(data.tokens, desc="tokenization")
            ]
        if max_len is not None:
            text_ids = [t[:max_len] for t in text_ids]
            
        text_pointer_masks: List[np.ndarray] = [
             get_src_pointer_mask(t, schema_tokenizer.src_tokenizer) for t in text_ids
             ]
            
        dataset = PointerDataset(source_tensors=text_ids, source_pointer_masks=text_pointer_masks)
        dataset.torchify()
        
        
        eval_dataset = dataset

        loader = DataLoader(
            eval_dataset,
            batch_size=lightning_module.batch_size,
            num_workers=8,
            pin_memory=True,
            collate_fn=lightning_module._collator.collate_batch,
            shuffle=False,
        )

        _, pred_tokens = iterative_prediction(
            lightning_module.model,
            loader,
            schema_tokenizer,
            max_len=train_args.get("max_tgt_len", 68),
            num_beams=num_beams,
            device="cuda" if torch.cuda.is_available() else "cpu",
            return_tokens=True,
            )   
        
        print("output = " + ' '.join(pred_tokens[0]))
        
        flag = (input("do you want to continue? (0/1) : "))
        
    wandb_logger.close()
    

args = parse_args()

main(args)
