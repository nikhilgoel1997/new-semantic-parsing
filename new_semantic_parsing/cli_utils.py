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
"""Utils only used in cli scripts"""

import os

import numpy as np

import torch
import wandb

from tqdm.auto import tqdm

from new_semantic_parsing import (
    metrics,
    EncoderDecoderWPointerModel,
    PointerModule,
    TopSchemaTokenizer,
)


def evaluate_model(
    checkpoint_path,
    schema_tokenizer: TopSchemaTokenizer,
    eval_dataset,
    prefix,
    n_rounds=5,
    subset_size=0.7,
    max_len=68,
    num_beams=1,
):
    """Compute metrics for each class n_rounds times and average results

    To reduce the amount of noise in the metrics,
    in every round evaluate on a random subset of eval_dataset.
    Final results contain both mean and standard deviation.

    Args:
        checkpoint_path: str, path to the pytorch lightning checkpoint
        schema_tokenizer: TopSchemaTokenizer
        eval_dataset: PointerDataset to evaluate on
        prefix: str, prefix for all metrics
        n_rounds: int, number of times to evaluate on eval_dataset subset
        subset_size: float, size of the subset for every round

    Returns:
        tuple (final_metrics, description)
            final_metrics is a dict with keys means, stdevs
            description is a string describing main metircs
    """

    best_model_dir = os.path.dirname(checkpoint_path)
    model = EncoderDecoderWPointerModel.from_pretrained(best_model_dir)
    model.eval()

    lightning_module = PointerModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        schema_tokenizer=schema_tokenizer,
        train_dataset=None,
        valid_dataset=eval_dataset,
    )

    pred_ids, _ = iterative_prediction(
        lightning_module.model,
        lightning_module.val_dataloader(),
        schema_tokenizer,
        max_len=max_len,
        num_beams=num_beams,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pred_tokens = [
        schema_tokenizer.decode(t, source_ids=s, return_tokens=True, skip_special_tokens=True)
        for t, s in zip(pred_ids, lightning_module.valid_dataset.source_tensors)
    ]

    true_ids = lightning_module.valid_dataset.target_tensors
    true_tokens = [
        schema_tokenizer.decode(t, source_ids=s, return_tokens=True, skip_special_tokens=True)
        for t, s in zip(true_ids, lightning_module.valid_dataset.source_tensors)
    ]

    subset_size_int = int(subset_size * len(pred_tokens))
    all_final_metrics = []
    for _ in tqdm(range(n_rounds), desc="Computing metrics"):
        permutation = np.random.permutation(len(pred_tokens))
        permutation = permutation[:subset_size_int]

        predictions_subset = [pred_tokens[i] for i in permutation]
        labels_subset = [true_tokens[i] for i in permutation]

        _final_metrics = metrics.get_metrics(
            predictions_subset,
            labels_subset,
            monitor_classes=schema_tokenizer.vocab,
            prefix=prefix,
            schema_tokenizer=schema_tokenizer,
            do_each=True,
        )
        all_final_metrics.append(_final_metrics)

    final_metrics = {
        "means": {},
        "stdevs": {},
    }

    metric_names = all_final_metrics[0].keys()
    for metric_name in metric_names:
        metric = [m[metric_name] for m in all_final_metrics]
        final_metrics["means"][metric_name] = np.mean(metric)
        final_metrics["stdevs"][metric_name + "_std"] = np.std(metric)

    description = _get_description(final_metrics)
    return final_metrics, description


def _get_description(final_metrics):
    metric_names = final_metrics["means"].keys()

    description = "\n"
    max_name_len = max(map(len, metric_names))
    for k in metric_names:
        if not k.endswith("_f1"):
            continue
        mean = round(float(final_metrics["means"][k]), 3)
        stdev = round(float(final_metrics["stdevs"][k + "_std"]), 3)

        space = " " * (max_name_len - len(k))
        description += f"{k} {space}: {mean} +- {2 * stdev}\n"

    return description


def check_config(pointer_module, trainer, args, strict=False):
    """Check that both module and trainer comply with args"""
    _cfg = pointer_module.model.config
    if args.dropout is not None:
        assert _cfg.dropout == args.dropout
        assert _cfg.encoder.hidden_dropout_prob == args.dropout
        assert _cfg.decoder.hidden_dropout_prob == args.dropout
        assert _cfg.encoder.attention_probs_dropout_prob == args.dropout
        assert _cfg.decoder.attention_probs_dropout_prob == args.dropout

    if getattr(args, "move_norm", None) is not None or strict:
        assert _cfg.move_norm == args.move_norm
        assert _cfg.move_norm_p == args.move_norm_p

    if args.label_smoothing is not None:
        assert _cfg.label_smoothing == args.label_smoothing

    if args.weight_decay is not None and (trainer.optimizers is not None or strict):
        for param_group in trainer.optimizers[0].param_groups:
            if not param_group["use_weight_decay"]:
                continue
            assert param_group["weight_decay"] == args.weight_decay


def iterative_prediction(model, dataloader, schema_tokenizer, max_len, num_beams, device="cpu"):
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
                prediction, batch["input_ids"][i], skip_special_tokens=True,
            )
            predictions_str.append(prediction_str)

    return predictions_ids, predictions_str


def get_outliers(pretrain_metrics, final_metrics, filter_by_name=None):
    """Get the names of metrics that are worse and better than the initial value with a high probability (0.97)

    97% confidence comes from the probability that both metrics will exceed/fall behind their standard deviation
    e.g. m1 > m1 + std1 with prob 1/6 and m2 < m2 - std2 with prob 1/6, then m2 < m1 with prob 1/6**2 ~= 0.03

    Args:
        pretrain_metrics: dict with keys "means" and "stdevs",
            "means" value is a subdictionary with metric names as keys,
            "stdevs" value is a subdictionary with metric_name + "_std" as keys
        final_metrics: dict with the structure as pretrain_metrics
        filter_by_name: str that should be in the metric name
    """
    negative_outliers, positive_outliers = [], []

    for metric_name, pretrain_mean in pretrain_metrics["means"].items():
        if filter_by_name and filter_by_name not in metric_name:
            continue

        pretrain_std = pretrain_metrics["stdevs"][metric_name + "_std"]
        final_mean = final_metrics["means"][metric_name]
        final_std = final_metrics["stdevs"][metric_name + "_std"]

        if pretrain_mean - pretrain_std > final_mean + final_std:
            negative_outliers.append(metric_name)

        if pretrain_mean + pretrain_std < final_mean - final_std:
            positive_outliers.append(metric_name)

    return negative_outliers, positive_outliers


def get_outliers_logs(outliers, pretrain_metrics, final_metrics, prefix, suffix):
    table = wandb.Table(
        columns=[
            "metric_name",
            "pretrain_mean",
            "pretrain_stdev",
            "final_mean",
            "final_stdev",
            "absolute_improvement",
            "relative_improvement",
        ]
    )

    abs_deltas = []
    rel_deltas = []
    digits = 4

    for name in outliers:
        abs_delta = final_metrics["means"][name] - pretrain_metrics["means"][name]
        rel_delta = abs_delta / pretrain_metrics["means"][name]
        abs_deltas.append(abs_delta)
        rel_deltas.append(rel_delta)

        table.add_data(
            name,
            round(pretrain_metrics["means"][name], digits),
            round(pretrain_metrics["stdevs"][name + "_std"], digits),
            round(final_metrics["means"][name], digits),
            round(final_metrics["stdevs"][name + "_std"], digits),
            round(abs_delta, digits),
            round(rel_delta, digits),
        )

    abs_delta_overall = round(sum(abs_deltas) / len(abs_deltas), digits)
    rel_delta_overall = round(sum(rel_deltas) / len(rel_deltas), digits)

    res = {
        f"{prefix}_outliers": table,
        f"absolute_{suffix}": abs_delta_overall,
        f"relative_{suffix}": rel_delta_overall,
    }
    return res
