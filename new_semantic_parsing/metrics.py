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
"""Metrics computation."""
from collections import Counter
from functools import reduce
from operator import add
from typing import List
from pprint import pformat

import numpy as np
import torch

from new_semantic_parsing.dataclasses import Seq2SeqEvalPrediciton


LBR = "["
RBR = "]"
IN = "IN:"
SL = "SL:"


class Tree:
    """TOP format tree object."""

    def __init__(self, entity, subtrees: List = None):
        """Make a tree node with value entity and subtrees.

        if subtrees is None, .subtrees attribute will be empty list

        Args:
            entity: intent/slot value, e.g. IN:INTENT
            subtrees: list of Tree objects
        """
        self.entity = entity
        self.subtrees = subtrees
        if subtrees is None:
            self.subtrees = []

        # for per-class metrics
        self._counts = Counter([entity])
        self._len = 1

        if len(self.subtrees) > 0:
            self._len += sum(map(len, self.subtrees))
            self._counts += reduce(add, (s._counts for s in self.subtrees))

        # for __repr__ and __eq__
        self._dict_repr = {self.entity: [s._dict_repr for s in self.subtrees]}

    def __repr__(self):
        return pformat(self._dict_repr)

    def __eq__(self, other):
        if isinstance(other, dict):
            return self._dict_repr == other
        if isinstance(other, Tree):
            return self._dict_repr == other._dict_repr
        raise ValueError(type(other))

    def __len__(self):
        return self._len

    def get_size(self, classes=None):
        """Get the number of nodes with node.entity in classes.

        Args:
            classes: optional, a iterable of classes to consider when computing the size

        Returns:
            if classes argument is not specified, returns the total number of nodes
            (same as __len__)
            if classes argument is specified, returns only the count of nodes corresponding to classes
        """
        if classes is None:
            return self.__len__()

        _size = sum(self._counts.get(c, 0) for c in classes)
        return _size

    @property
    def counts(self):
        return self._counts

    @classmethod
    def from_tokens(cls, tokens, return_index=False):
        """Builds a parsing tree for labeled bracketing score computation.

        The tree is build until the last ] symbol, everything after it is ignored

        Args:
            tokens: list of tokens
            return_index: used in recursion to provide toke index

        Returns:
            Tree object, if return_index == False
            tuple (Tree, index), if return_index == True

        Raises:
            ValueError, if tokens do not represent a valid tree
        """
        # every tree should start with
        # [ ENTITY_TYPE: ENTITY
        if len(tokens) < 3 or tokens[0] != LBR:
            raise ValueError(f"Tree starts with {tokens[:4]}")

        entity_type = tokens[1]

        # ignore invalid subtrees
        if entity_type not in [IN, SL]:
            raise ValueError(f"Tree starts with {tokens[:4]}")

        entity = entity_type + tokens[2]  # e.g. IN:INTENT

        subtrees = []
        slot_value_tokens = []

        i = 3
        while i < len(tokens):
            token = tokens[i]

            # ignore non-slot values
            # e.g. ignore "go stuff in" [IN:STUFF Do stuff]
            if entity_type == IN and token not in [LBR, RBR]:
                i += 1
                continue

            # LBR starts a new subtree
            if token == LBR:
                subtree, j = cls.from_tokens(tokens[i:], return_index=True)

                subtrees.append(subtree)
                i += j
                continue

            # RBR ends the tree, merge slot values into a single leaf if any
            # e.g. "stuff value" becomes a single leaf in [IN:GET_STUFF [SL:STUF_VALUE stuff value]]
            if token == RBR:
                if slot_value_tokens:
                    subtrees.append(Tree(" ".join(slot_value_tokens)))
                i += 1
                break

            if entity_type == SL:
                slot_value_tokens.append(token)
                i += 1
                continue

        tree = Tree(entity, subtrees)

        if return_index:
            return tree, i

        return tree


# Tree path scores


def get_tree_path_metrics(pred_tokens, true_tokens, monitor_classes, prefix):
    """Get metrics for all classes, for monitor classes and for monitor_classes[0].

    Apply prefix to all keys.

    Args:
        pred_tokens: List[List[str]]
        true_tokens: List[List[str]]
        monitor_classes: List[str]
        prefix: str, will be appended to all return dict keys

    Returns:
        dictionary with keys
            {prefix}_{score_name}
            {prefix}_new_classes_{score_name}
            {prefix}_{monitor_classes[0]}_{score_name}
        for each score_name - key from get_tree_path_scores output dictionary
    """

    tree_path_scores = get_tree_path_scores(pred_tokens=pred_tokens, true_tokens=true_tokens)
    tree_path_scores = {f"{prefix}_{k}": v for k, v in tree_path_scores.items()}

    tree_path_scores_new_cls = dict()
    tree_path_scores_new_cls_main = dict()
    if monitor_classes is not None:
        tree_path_scores_new_cls = get_tree_path_scores(
            pred_tokens=pred_tokens, true_tokens=true_tokens, classes=monitor_classes
        )
        tree_path_scores_new_cls = {
            f"{prefix}_new_classes_{k}": v for k, v in tree_path_scores_new_cls.items()
        }

        _main_class = monitor_classes[0]
        tree_path_scores_new_cls_main = get_tree_path_scores(
            pred_tokens=pred_tokens, true_tokens=true_tokens, classes=[_main_class]
        )
        tree_path_scores_new_cls_main = {
            f"{prefix}_{_main_class}_{k}": v for k, v in tree_path_scores_new_cls_main.items()
        }

    tree_metrics = {
        **tree_path_scores,
        **tree_path_scores_new_cls,
        **tree_path_scores_new_cls_main,
    }

    return tree_metrics


def get_tree_path_scores(pred_tokens, true_tokens, classes=None):
    """
    Args:
        pred_tokens: list of lists of tokens
        true_tokens: list of lists of tokens

    Returns:
        dictionary with keys
            predicted_paths
            expected_paths
            tree_path_precision
            tree_path_recall
            tree_path_f1
    """
    pred_paths_lst, true_paths_lst = [], []

    for pred, true in zip(pred_tokens, true_tokens):
        try:
            pred_paths = _get_paths_with_values(Tree.from_tokens(pred))
        except ValueError:
            pred_paths = dict()

        # we need to build true tree even if pred tree is not valid to compute correct n_expected
        true_paths = _get_paths_with_values(Tree.from_tokens(true))

        pred_paths_lst.append(pred_paths)
        true_paths_lst.append(true_paths)

    true_positives = 0
    n_predicted = 0
    n_expected = 0

    for pred_paths, true_paths in zip(pred_paths_lst, true_paths_lst):
        if classes is None:
            n_expected += len(true_paths)
            n_predicted += len(pred_paths)
        else:
            n_expected += len([p for p in true_paths if any(c in p for c in classes)])
            n_predicted += len([p for p in pred_paths if any(c in p for c in classes)])

        true_positives += _get_tree_path_matches(pred_paths, true_paths, classes)

    precision = 0
    if n_predicted > 0:
        precision = true_positives / n_predicted

    recall = 0
    if n_expected > 0:
        recall = true_positives / n_expected

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "predicted_paths": n_predicted,
        "expected_paths": n_expected,
        "tree_path_precision": precision,
        "tree_path_recall": recall,
        "tree_path_f1": f1,
    }


def _get_slot_paths(tree):
    slot_paths = Counter()

    for subtree in tree.subtrees:
        is_slot = SL in subtree.entity

        slot_name = subtree.entity[3:]

        slot_subpaths = _get_slot_paths(subtree)

        for slot_subname, freq in slot_subpaths.items():
            if is_slot:
                slot_paths[slot_name + "." + slot_subname] += 1
            else:
                slot_paths[slot_subname] += 1

        if len(slot_subpaths) == 0 and is_slot:
            slot_paths[slot_name] += 1

    return slot_paths


def _get_paths_with_values(tree):
    slot_paths = dict()

    for subtree in tree.subtrees:
        slot_subpaths = _get_paths_with_values(subtree)

        for slot_subname, value in slot_subpaths.items():
            if value is None:
                slot_paths[tree.entity] = slot_subname
            else:
                slot_paths[tree.entity + "." + slot_subname] = value

    if len(tree.subtrees) == 0:
        slot_paths = {tree.entity: None}

    return slot_paths


def _get_tree_path_matches(pred_tree_paths, true_tree_paths, classes=None):
    n_matches = 0
    for true_path, true_slot in true_tree_paths.items():
        path_value_match = true_path in pred_tree_paths and pred_tree_paths[true_path] == true_slot
        class_match = classes is None or any(c in true_path for c in classes)

        if path_value_match and class_match:
            n_matches += 1

    return n_matches


# Non-tree metrics: EM, Accuracy, first intent precision


def compute_metrics_from_batch(predictions, labels, masks, stop_tokens):
    """Compute metrics where all predictions, labels and masks are torch.tensor"""
    device = predictions.device

    # correct tokens which are not masked (masked tokens have mask == 0)
    n_correct = ((predictions == labels) & masks.bool()).sum()
    n_total = masks.sum()
    accuracy = n_correct / n_total.float()

    # trauncate until EOS token
    # for exact match we consider all tokens until EOS/PAD
    # this is closer to inference setup when generation stops after EOS/PAD

    def truncate(pred):
        i = 0
        for i, idx in enumerate(pred):
            if idx in stop_tokens:
                break
        return pred[:i]

    truncated_preds = [truncate(p) for p in predictions.detach().unbind()]
    truncated_labels = [truncate(l) for l in labels.detach().unbind()]

    exact_match = sum(
        (p.shape == l.shape and torch.all(p == l))
        for p, l in zip(truncated_preds, truncated_labels)
    )
    if exact_match == 0:
        exact_match = torch.tensor(0.0, device=device)

    exact_match = exact_match.float() / len(truncated_preds)

    # intent is the third token in the sequence
    # [ IN: INTENT ...
    # 0 1   2
    intent_tokens_preds = predictions[:, 2]
    intent_tokens_labels = labels[:, 2]

    first_intent_precision = torch.sum(intent_tokens_preds == intent_tokens_labels) / float(
        intent_tokens_preds.shape[0]
    )

    return {
        "accuracy": accuracy,
        "exact_match": exact_match,
        "first_intent_precision": first_intent_precision,
    }
