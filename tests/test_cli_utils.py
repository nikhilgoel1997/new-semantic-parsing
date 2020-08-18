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
import unittest
import torch

import new_semantic_parsing as nsp
from new_semantic_parsing import cli_utils


class CliUtilsTest(unittest.TestCase):
    def test_get_outliers(self):
        initial_metrics = {
            "means": {"metric1": 0.7, "metric2": 0.8, "metric3": 0.9},
            "stdevs": {"metric1_std": 0.05, "metric2_std": 0.03, "metric3_std": 0.05},
        }

        final_metrics = {
            "means": {"metric1": 0.5, "metric2": 0.9, "metric3": 0.95},
            "stdevs": {"metric1_std": 0.05, "metric2_std": 0.05, "metric3_std": 0.05},
        }

        negative, positive = cli_utils.get_outliers(initial_metrics, final_metrics)
        self.assertSequenceEqual(negative, ["metric1"])
        self.assertSequenceEqual(positive, ["metric2"])

    def test_get_outliers_sigma0(self):
        initial_metrics = {
            "means": {"metric1": 0.7, "metric2": 0.8, "metric3": 0.9},
            "stdevs": {"metric1_std": 0.05, "metric2_std": 0.03, "metric3_std": 0.05},
        }

        final_metrics = {
            "means": {"metric1": 0.5, "metric2": 0.9, "metric3": 0.95},
            "stdevs": {"metric1_std": 0.05, "metric2_std": 0.05, "metric3_std": 0.05},
        }

        negative, positive = cli_utils.get_outliers(initial_metrics, final_metrics, sigma=0)
        self.assertSequenceEqual(negative, ["metric1"])
        self.assertSequenceEqual(positive, ["metric2", "metric3"])

    def test_get_kfold_subsets(self):
        for k_folds in [3, 4]:
            with self.subTest(k_folds):
                x = [[f"tok{i}", f"tok{i}_2"] for i in range(100)]
                y = [[f"pred_tok{i}", f"tpred_ok{i}_2", f"pred_tok{i}_3"] for i in range(100)]

                folds = cli_utils._get_kfold_subsets(x, y, k_folds)

                for i, (subset_x, subset_y) in enumerate(folds, 1):
                    self.assertTrue(len(subset_x) == len(subset_y))

                self.assertEqual(i, k_folds)


class TestAverageModels(unittest.TestCase):
    def test_average_models(self):
        nsp.utils.set_seed(18)

        params = dict(
            layers=2, hidden=6, heads=3, src_vocab_size=17, tgt_vocab_size=11, max_src_len=13,
        )

        old_model = nsp.EncoderDecoderWPointerModel.from_parameters(**params)
        new_model = nsp.EncoderDecoderWPointerModel.from_parameters(**params)

        for (name1, param1), (name2, param2) in zip(
            old_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            if not torch.allclose(param1, param2):
                break
        else:
            assert False, "models are identical"

        averaged_model = cli_utils.average_models(old_model, new_model, 0.5)

        for (name1, param1), (name2, param2), (name3, param3) in zip(
            old_model.named_parameters(),
            new_model.named_parameters(),
            averaged_model.named_parameters(),
        ):
            assert name1 == name2 == name3

            self.assertTrue(torch.allclose(param3, (param1 + param2) / 2.0))
