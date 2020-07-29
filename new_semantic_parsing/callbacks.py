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
"""Lightning Callbacks used when training."""

import os
from pytorch_lightning import callbacks


class TransformersModelCheckpoint(callbacks.ModelCheckpoint):
    """Saves model and tokenizer in Transformers format when ModelCheckpoint does save.

    This way it is possible to simply load the model (without training hparameters)
    using transformers.from_pretrained. Also adds an attribute .last_checkpoint_path.
    """

    def on_train_start(self, trainer, pl_module):
        super(TransformersModelCheckpoint, self).on_train_start(trainer, pl_module)
        self._model = pl_module.model
        self._tokenizer = pl_module.schema_tokenizer

    def _save_model(self, filepath):
        super(TransformersModelCheckpoint, self)._save_model(filepath)
        self.last_checkpoint_path = filepath

        save_path = os.path.dirname(filepath)
        self._model.save_pretrained(save_path)
        self._tokenizer.save(save_path)
