import torch
import transformers

from new_semantic_parsing.dataclasses import InputDataClass


class Seq2SeqDataCollator(transformers.DataCollator):
    """Pads tensors to the maximum length in batch.
    Length is different for encoder and decoder inputs.

    Decoder inputs should have prefix `decoder_`
    `labels` considered a decoder field too
    All other tensors considered encoder inputs

    All values in the input DataClasses should be torch.Tensor or shape (seq_len, *)
    or None, None values are ignored

    All values corresponsing to the keys ending with `mask` are padded with zeroes
    """
    def __init__(self, pad_id, decoder_pad_id=None):
        self.encoder_pad_id = pad_id
        self.decoder_pad_id = decoder_pad_id or pad_id

    def collate_batch(self, examples):
        """
        :param examples: list of DataClass
        :return: dict with the DataClass fields
        """
        batch = dict()
        batch_size = len(examples)

        self._encoder_max_len = None
        self._decoder_max_len = None

        # iterate ofer the first example to get shapes
        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            is_decoder = self._is_decoder_field(k)

            maxlen = max(getattr(ex, k).shape[0] for ex in examples)
            self._shape_check(maxlen, is_decoder, k)

            batched_shape = (batch_size, maxlen, *v.shape[1:])
            batch[k] = torch.zeros(batched_shape, dtype=v.dtype, device=v.device)

            if k.endswith('mask'):
                continue

            batch[k].fill_(self.decoder_pad_id if is_decoder else self.encoder_pad_id)

        for i, example in enumerate(examples):
            for k, tensor in vars(example).items():
                if tensor is None: continue
                batch[k][i, :len(tensor)] = tensor

        return batch

    @staticmethod
    def _is_decoder_field(field_name):
        return field_name.startswith('decoder_') or field_name == 'labels'

    def _shape_check(self, maxlen, is_decoder, key):
        """Data shape validation"""
        if is_decoder:
            if self._decoder_max_len is not None and self._decoder_max_len != maxlen:
                raise ValueError(f'decoder input tensors have different lengths ({key})')
        else:
            if self._encoder_max_len is not None and self._encoder_max_len != maxlen:
                raise ValueError(f'encoder input tensors have different lengths({key})')


class PaddedDataCollator(transformers.DataCollator):
    """This data collator assumes that all examples are padded to the same length"""
    def collate_batch(self, examples):
        batch = dict()

        for k, v in vars(examples[0]).items():
            if v is None:
                continue
            batch[k] = torch.stack([getattr(ex, k) for ex in examples])

        return batch


class PointerDataset(torch.utils.data.Dataset):
    def __init__(self, source_tensors, target_tensors=None, source_pointer_masks=None, target_pointer_masks=None):
        """
        :param source_tensors: list of tensors, input ids
        :param target_tensors: list of tensors, labels
        :param target_pointer_masks: list of tensors, mask showing pointer locations in labels
        """
        self.source_tensors = source_tensors
        self.target_tensors = target_tensors
        self.target_pointer_masks = target_pointer_masks
        self.source_pointer_masks = source_pointer_masks

        self.torchified = isinstance(source_tensors[0], torch.Tensor)
        if target_tensors is not None:
            self.torchified = self.torchified and isinstance(target_tensors[0], torch.Tensor)
        if source_pointer_masks is not None:
            self.torchified = self.torchified and isinstance(source_pointer_masks[0], torch.Tensor)
        if target_pointer_masks is not None:
            self.torchified = self.torchified and isinstance(target_pointer_masks[0], torch.Tensor)

    def __len__(self):
        return len(self.source_tensors)

    def __getitem__(self, item) -> InputDataClass:
        source_pointer_mask = None
        if self.source_pointer_masks is not None:
            source_pointer_mask = self.source_pointer_masks[item]

        if self.target_tensors is None:
            return InputDataClass(
                input_ids=self.source_tensors[item],
                pointer_mask=source_pointer_mask,
            )

        target_pointer_mask = None
        if self.target_pointer_masks is not None:
            target_pointer_mask = self.target_pointer_masks[item][:-1]

        return InputDataClass(
            input_ids=self.source_tensors[item],
            pointer_mask=source_pointer_mask,
            decoder_input_ids=self.target_tensors[item][:-1],
            decoder_pointer_mask=target_pointer_mask,
            labels=self.target_tensors[item][1:],
        )

    def torchify(self):
        """Make all tensors torch.Tensor"""
        if self.torchified:
            return

        self.source_tensors = [torch.LongTensor(t) for t in self.source_tensors]

        if self.target_tensors is not None:
            self.target_tensors = [torch.LongTensor(t) for t in self.target_tensors]
        if self.source_pointer_masks is not None:
            self.source_pointer_masks = [torch.FloatTensor(t) for t in self.source_pointer_masks]
        if self.target_pointer_masks is not None:
            self.target_pointer_masks = [torch.FloatTensor(t) for t in self.target_pointer_masks]

        self.torchified = True

    def get_max_len(self):
        """Get maximum length of source sequences and target sequences in the dataset
        Returns a tuple (source_max_len, target_max_len)
        if target_tensors is None, target_max_len is also None
        """
        source_max_len = max(len(t) for t in self.source_tensors)
        if self.target_tensors is None:
            return source_max_len, None

        target_max_len = max(len(t) for t in self.target_tensors)
        return source_max_len, target_max_len