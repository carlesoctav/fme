from __future__ import annotations

import dataclasses as dc
import typing as tp
from dataclasses import dataclass

import grain
import numpy as np
from grain import transforms as grain_transforms
from jaxtyping import Array
from transformers import PreTrainedTokenizerBase

from ._dataset_transforms import BaseDatasetTransform, EnsureMapDataset, ToIterDataset


@dataclass
class DataTransformsForMaskedLMGivenText(grain.transforms.RandomMap):
    tokenizer: PreTrainedTokenizerBase
    columns: str
    max_length: int
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int | None = None


    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def random_map(
        self,
        texts: dict[str, str],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:

        if self.columns not in texts:
            raise ValueError(f"Column {self.columns} not found in the input data.")

        tokenized = self.tokenizer(
            texts[self.columns],
            return_tensors="np",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenizers return shape (1, seq_len) for a single string; squeeze it.
        if tokenized["input_ids"].ndim == 2 and tokenized["input_ids"].shape[0] == 1:
            for k in list(tokenized.keys()):
                tokenized[k] = np.squeeze(tokenized[k], axis=0)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            tokenized["input_ids"].tolist(), already_has_special_tokens=True
        )
        special_tokens_mask = np.asarray(special_tokens_mask, dtype=bool)

        tokenized["input_ids"], labels = self.mask_tokens(
            tokenized["input_ids"],
            len(self.tokenizer),
            self.tokenizer.mask_token_id,
            special_tokens_mask=special_tokens_mask,
            rng=rng,
        )

        out: dict[str, np.ndarray] = {
            self.columns: {
            "input_ids": tokenized["input_ids"],
            "token_type_ids": tokenized.get(
                "token_type_ids", np.zeros_like(tokenized["input_ids"], dtype=np.int32)
            ),

            "attention_mask": tokenized["attention_mask"],
            },
            "labels": labels,
        }

        return out

    def mask_tokens(
        self,
        input_ids: np.ndarray,
        vocab_size: int,
        mask_token_id: int,
        special_tokens_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:

        if not self.mlm_probability or self.mlm_probability <= 0:
            labels = np.full_like(input_ids, fill_value=-100)
            return input_ids, labels

        masked_draw = rng.binomial(1, self.mlm_probability, size=input_ids.shape)
        masked_indices = masked_draw & ~special_tokens_mask

        mask_replace_draw = rng.binomial(1, self.mask_replace_prob, size=input_ids.shape)
        replace_with_mask_token_indices = masked_indices & mask_replace_draw

        denom = max(1e-12, 1.0 - self.mask_replace_prob)
        random_replace_prob = self.random_replace_prob / denom

        random_replace_prob = float(min(1.0, max(0.0, random_replace_prob)))
        random_replace_draw = rng.binomial(1, random_replace_prob, size=input_ids.shape)

        replace_with_random_indices = masked_indices & (~replace_with_mask_token_indices) & random_replace_draw

        random_token = rng.integers(
            low=0,
            high=vocab_size,
            size=input_ids.shape,
            dtype=input_ids.dtype,
        )

        output_ids = np.where(replace_with_mask_token_indices, mask_token_id, input_ids)
        output_ids = np.where(replace_with_random_indices, random_token, output_ids)

        labels = np.where(
            masked_indices,
            input_ids,
            np.asarray(-100, dtype=input_ids.dtype),
        )

        return output_ids, labels


@dc.dataclass
class MLMBatch:
    """Batch container for masked language modeling."""

    input_ids: Array
    attention_mask: Array
    token_type_ids: Array
    labels: Array


@dc.dataclass
class FormatMLMOutputs(grain_transforms.Map):
    """Flatten outputs from MLM random map transform."""

    column: str

    def map(self, features: dict[str, tp.Any]) -> dict[str, Array]:
        if self.column not in features:
            raise KeyError(f"Column {self.column!r} not present in features")
        column_data = features[self.column]
        labels = np.asarray(features["labels"], dtype=np.int32)

        input_ids = np.asarray(column_data.get("input_ids"), dtype=np.int32)
        attention_mask = column_data.get("attention_mask")
        token_type_ids = column_data.get("token_type_ids")

        if attention_mask is None:
            attention_mask = np.ones_like(input_ids, dtype=np.int32)
        else:
            attention_mask = np.asarray(attention_mask, dtype=np.int32)

        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids, dtype=np.int32)
        else:
            token_type_ids = np.asarray(token_type_ids, dtype=np.int32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


def masked_language_modeling_transforms(
    *,
    dataset_type: str,
    column: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    mlm_probability: float = 0.15,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
    pad_to_multiple_of: int | None = None,
) -> tuple[
    list[grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform],
    type[MLMBatch],
]:
    """Build transforms for creating MLM datasets."""

    random_transform = DataTransformsForMaskedLMGivenText(
        tokenizer=tokenizer,
        columns=column,
        max_length=max_length,
        mlm_probability=mlm_probability,
        mask_replace_prob=mask_replace_prob,
        random_replace_prob=random_replace_prob,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    operations: list[grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform] = [
        EnsureMapDataset(dataset_type=dataset_type),
        ToIterDataset(),
        random_transform,
        FormatMLMOutputs(column=column),
    ]

    return operations, MLMBatch

