from __future__ import annotations

import dataclasses as dc
import typing as tp

import numpy as np

import grain
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from ._dataset_transforms import ApplyFirstFitPacking, BaseDatasetTransform, EnsureMapDataset, ToIterDataset


@dc.dataclass
class LLMBatch:
    """Batch container for next-token prediction setups."""

    inputs: Array
    targets: Array
    inputs_position: Array
    targets_position: Array
    inputs_segmentation: Array
    targets_segmentation: Array
    attention_mask: Array | None = None


def _stack_last_axis(values: Array, pad_to: int, pad_value: int) -> Array:
    values = np.asarray(values, dtype=np.int32)
    if values.shape[-1] > pad_to:
        values = values[..., :pad_to]
    pad_amount = pad_to - values.shape[-1]
    if pad_amount <= 0:
        return values
    pad_shape = list(values.shape)
    pad_shape[-1] = pad_amount
    padding = np.full(pad_shape, pad_value, dtype=values.dtype)
    return np.concatenate([values, padding], axis=-1)


def _infer_pad_id(tokenizer: PreTrainedTokenizerBase | None, default: int = 0) -> int:
    if tokenizer is None:
        return default
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        return int(tokenizer.unk_token_id)
    return default


def _infer_eos_id(tokenizer: PreTrainedTokenizerBase | None, default: int = 0) -> int:
    if tokenizer is None:
        return default
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return default


def _infer_bos_id(tokenizer: PreTrainedTokenizerBase | None, default: int) -> int:
    if tokenizer is None:
        return default
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    return default


@dc.dataclass
class TokenizeText(grain_transforms.Map):
    """Tokenize raw text coming from a column."""

    column: str
    tokenizer: PreTrainedTokenizerBase
    max_length: int | None = None
    truncation: bool = True

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        text = features[self.column]
        encoded = self.tokenizer(
            text,
            truncation=self.truncation,
            max_length=self.max_length,
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        tokens = np.asarray(encoded["input_ids"], dtype=np.int32)
        attn = np.asarray(encoded.get("attention_mask"), dtype=np.int32)
        if tokens.ndim > 1:
            tokens = tokens.squeeze(axis=0)
            attn = attn.squeeze(axis=0)
        return {"input_ids": tokens, "attention_mask": attn}


@dc.dataclass
class EnsureTokenIds(grain_transforms.Map):
    """Ensure numeric token ids are provided in the expected field."""

    column: str

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not found in element")
        token_ids = np.asarray(features[self.column], dtype=np.int32)
        return {"input_ids": token_ids}


@dc.dataclass
class PadToMaxLength(grain_transforms.Map):
    """Pad or trim input ids to a fixed length."""

    max_length: int
    pad_token_id: int

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        tokens = np.asarray(features["input_ids"], dtype=np.int32)
        original_length = tokens.shape[-1]
        tokens = _stack_last_axis(tokens, self.max_length, self.pad_token_id)

        attn = features.get("attention_mask")
        if attn is None:
            attn = np.zeros_like(tokens, dtype=np.int32)
            attn[..., : min(original_length, self.max_length)] = 1
        else:
            attn = np.asarray(attn, dtype=np.int32)
            attn = _stack_last_axis(attn, self.max_length, 0)
        return {
            "input_ids": tokens,
            "attention_mask": attn,
        }


@dc.dataclass
class MakeInputsTargets(grain_transforms.Map):
    """Duplicate input ids into inputs/targets dictionary."""

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        tokens = np.asarray(features["input_ids"], dtype=np.int32)
        out: dict[str, Array] = {
            "inputs": tokens.copy(),
            "targets": tokens.copy(),
        }
        if "attention_mask" in features:
            out["attention_mask"] = np.asarray(features["attention_mask"], dtype=np.int32)
        return out


@dc.dataclass
class ShiftTokensForNTD(grain_transforms.Map):
    """Shift inputs to create teacher-forced targets."""

    bos_token_id: int
    pad_token_id: int

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        targets = np.asarray(features["targets"], dtype=np.int32)
        inputs = np.asarray(features["inputs"], dtype=np.int32)
        if inputs.shape != targets.shape:
            raise ValueError("inputs and targets must have identical shapes before shifting")
        pad_shape = list(inputs.shape)
        pad_shape[-1] = 1
        prefix = np.full(pad_shape, self.bos_token_id, dtype=np.int32)
        shifted_inputs = np.concatenate([prefix, inputs[..., :-1]], axis=-1)
        features["inputs"] = shifted_inputs
        features["targets"] = targets

        mask = features.get("attention_mask")
        if mask is not None:
            mask = np.asarray(mask, dtype=np.int32)
            prefix_mask = np.ones(pad_shape, dtype=np.int32)
            features["attention_mask"] = np.concatenate([prefix_mask, mask[..., :-1]], axis=-1)
        else:
            prefix_mask = np.ones(pad_shape, dtype=np.int32)
            target_mask = (targets != self.pad_token_id).astype(np.int32)
            features["attention_mask"] = np.concatenate([prefix_mask, target_mask[..., :-1]], axis=-1)
        return features


@dc.dataclass
class InferSegmentsAndPositions(grain_transforms.Map):
    """Infer segmentation and positional annotations."""

    pad_token_id: int

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        inputs = np.asarray(features["inputs"], dtype=np.int32)
        targets = np.asarray(features["targets"], dtype=np.int32)

        if "input_segmentation" in features:
            features.setdefault("inputs_segmentation", features.pop("input_segmentation"))

        attention = features.get("attention_mask")
        if attention is not None:
            attention_mask = np.asarray(attention, dtype=np.int32)
        else:
            attention_mask = (inputs != self.pad_token_id).astype(np.int32)

        inputs_seg = np.asarray(features.get("inputs_segmentation", attention_mask), dtype=np.int32)
        targets_seg = np.asarray(features.get("targets_segmentation", inputs_seg), dtype=np.int32)

        seq_len = inputs.shape[-1]
        base_positions = np.arange(seq_len, dtype=np.int32)
        base_positions = np.broadcast_to(base_positions, inputs.shape)
        features.setdefault("inputs_position", base_positions * (inputs_seg > 0))
        features.setdefault("targets_position", base_positions * (targets_seg > 0))
        features["inputs_segmentation"] = inputs_seg
        features["targets_segmentation"] = targets_seg
        return features


@dc.dataclass
class ReformatPackedKeys(grain_transforms.Map):
    """Rename packed dataset keys to the canonical naming."""

    def map(self, features: dict[str, Array]) -> dict[str, Array]:
        renames = {
            "inputs_segment_ids": "inputs_segmentation",
            "targets_segment_ids": "targets_segmentation",
            "inputs_positions": "inputs_position",
            "targets_positions": "targets_position",
        }
        for old, new in renames.items():
            if old in features:
                features[new] = features.pop(old)
        return features


def next_token_prediction_transforms(
    *,
    dataset_type: str,
    column: str,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    is_tokenized: bool,
    add_bos: bool = True,
    add_eos: bool = True,
    packing: bool = False,
    pad_id: int | None = None,
    packing_bins: int | None = None,
) -> tuple[
    list[grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform],
    type[LLMBatch],
]:
    """Build the list of transforms required for next-token prediction."""

    pad_token_id = pad_id if pad_id is not None else _infer_pad_id(tokenizer)
    eos_id = _infer_eos_id(tokenizer, default=pad_token_id)
    bos_id = _infer_bos_id(tokenizer, default=eos_id if add_bos else pad_token_id)

    operations: list[
        grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform
    ] = [EnsureMapDataset(dataset_type=dataset_type)]

    if is_tokenized:
        operations.append(EnsureTokenIds(column=column))
    else:
        operations.append(TokenizeText(column=column, tokenizer=tokenizer, max_length=max_length))

    operations.append(MakeInputsTargets())

    if not packing:
        operations.append(PadToMaxLength(max_length=max_length, pad_token_id=pad_token_id))

    operations.append(ToIterDataset())

    if packing:
        length_struct = {"inputs": max_length, "targets": max_length}
        operations.append(ApplyFirstFitPacking(length_struct=length_struct, num_packing_bins=packing_bins))
        operations.append(ReformatPackedKeys())

    if add_eos:
        operations.append(ShiftTokensForNTD(bos_token_id=bos_id, pad_token_id=pad_token_id))

    operations.append(InferSegmentsAndPositions(pad_token_id=pad_token_id))

    return operations, LLMBatch
