# Copyright 2020 Vladislav Lialin
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
import torch.nn as nn

from new_semantic_parsing import buffers


class ParamsBufferHolderTest(unittest.TestCase):
    def test_init(self):
        model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        buffers_object = buffers.ParamsBufferHolder(model.named_parameters())

        self.assertTrue(len(buffers_object._buffer_names))

    def test_get(self):
        model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        buffers_object = buffers.ParamsBufferHolder(model.named_parameters())

        param_name = "0.weight"
        buffer_expected_value = model[0].weight

        buffer_value = buffers_object.get(param_name)
        self.assertTrue(torch.allclose(buffer_value, buffer_expected_value))

    def test_set(self):
        model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        buffers_object = buffers.ParamsBufferHolder(model.named_parameters())

        param_name = "0.weight"
        buffer_expected_value = torch.zeros_like(model[0].weight)

        buffers_object.set(param_name, buffer_expected_value)
        buffer_value = buffers_object.get(param_name)
        self.assertTrue(torch.allclose(buffer_value, buffer_expected_value))

    def test_getitem(self):
        model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        buffers_object = buffers.ParamsBufferHolder(model.named_parameters())

        param_name = "0.weight"
        buffer_expected_value = model[0].weight

        buffer_value = buffers_object[param_name]
        self.assertTrue(torch.allclose(buffer_value, buffer_expected_value))

    def test_items(self):
        model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3))
        buffers_object = buffers.ParamsBufferHolder(model.named_parameters())

        n_params = 0
        expected_n_params = len(buffers_object)

        for n, p in buffers_object.items():
            self.assertIsInstance(n, str)
            self.assertIsInstance(p, torch.Tensor)
            n_params += 1

        self.assertEqual(n_params, expected_n_params)
