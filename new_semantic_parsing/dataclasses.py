from typing import NewType, List, Union
from dataclasses import dataclass

import numpy as np
import torch


Tensor = NewType('Tensor', Union[List, np.ndarray, torch.Tensor])
LongTensor = NewType('LongTensor', Union[List, np.ndarray, torch.LongTensor])
FloatTensor = NewType('FloatTensor', Union[List, np.ndarray, torch.FloatTensor])


@dataclass
class InputDataClass:
    input_ids: LongTensor
    decoder_input_ids: LongTensor = None
    attention_mask: FloatTensor = None
    target_pointer_mask: FloatTensor = None
    decoder_attention_mask: FloatTensor = None
    labels: LongTensor = None


@dataclass
class SchemaItem:
    ids: List[int]
    pointer_mask: List[int]

    def __len__(self):
        return len(self.ids)
