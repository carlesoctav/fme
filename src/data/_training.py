from __future__ import annotations

import dataclasses as dc
import typing as tp
from collections.abc import Sequence

import grain
from grain import transforms as grain_transforms
import jax
import jax.tree_util as jtu
from jaxtyping import Array
from jax.sharding import Mesh

from ._dataset_transforms import (
    BaseDatasetTransform,
    BatchDataset,
    BatchRampUpDataset,
    EnsureMapDataset,
    _as_dataset_list,
)
from ._transforms import CollateToBatch
from .next_token_prediction import LLMBatch


Batch = tp.Any


@dc.dataclass
class MultiHostDataLoadIterator:
    """Simple iterator wrapper that materialises global arrays."""

    dataset: grain.IterDataset
    global_mesh: Mesh
    iterator_length: int | None = None
    reset_after_epoch: bool = False

    def __post_init__(self) -> None:
        self._iter: tp.Iterator[tp.Any] | None = None

    def __iter__(self) -> "MultiHostDataLoadIterator":  # pragma: no cover - trivial
        self._ensure_iterator()
        return self

    def __next__(self):  # pragma: no cover - trivial forwarding
        self._ensure_iterator()
        batch = next(self._iter)
        return jtu.tree_map(lambda x: _broadcast_to_mesh(x, self.global_mesh), batch)

    def _ensure_iterator(self) -> None:
        if self._iter is None:
            self._iter = iter(self.dataset)


def _broadcast_to_mesh(array: Array, mesh: Mesh) -> jax.Array:
    del mesh  # Placeholder until sharded support is required.
    return jax.device_put(array)


def _make_iterator(
    grain_datasets: grain.IterDataset | Sequence[grain.IterDataset],
    dataset_lengths: int | Sequence[int] | None,
    *,
    global_mesh: Mesh,
    global_batch_size: int,
    dataloading_host_count: int,
    dataset_weights: Sequence[float] | None = None,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    batch_class: type[LLMBatch] = LLMBatch,
    reset_after_epoch: bool = False,
    use_thread_prefetch: bool = False,
    batch_rampup_factors: dict[str, float] | None = None,
) -> MultiHostDataLoadIterator:
    datasets = _as_dataset_list(grain_datasets)
    if not all(isinstance(ds, grain.IterDataset) for ds in datasets):
        raise TypeError("All datasets must be instances of grain.IterDataset")
    iter_datasets = datasets
    mixed = grain.IterDataset.mix(iter_datasets, weights) if len(iter_datasets) != 1 else

    local_batch_size = global_batch_size // dataloading_host_count
    if batch_rampup_factors:
        batch_transform: BaseDatasetTransform = BatchRampUpDataset(
            batch_size=local_batch_size,
            rampup_factors=batch_rampup_factors,
            drop_remainder=drop_remainder,
        )
    else:
        batch_transform = BatchDataset(
            batch_size=local_batch_size,
            drop_remainder=drop_remainder,
        )
    mixed = batch_transform(mixed)
    mixed = mixed.map(CollateToBatch(batch_class=batch_class))

    if worker_count > 0 and not use_thread_prefetch:
        mp_options = grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=worker_buffer_size,
        )
        mixed = mixed.mp_prefetch(mp_options)

    iterator_length: int | None = None
    if dataset_lengths is not None and not isinstance(dataset_lengths, Sequence):
        iterator_length = int(dataset_lengths) // local_batch_size

    return MultiHostDataLoadIterator(
        mixed,
        global_mesh=global_mesh,
        iterator_length=iterator_length,
        reset_after_epoch=reset_after_epoch,
    )


def make_data_loader(
    datasets: Sequence[tp.Any],
    operations: Sequence[
        grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform
    ],
    *,
    global_mesh: Mesh | None = None,
    global_batch_size: int,
    dataloading_host_index: int,
    dataloading_host_count: int,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    dataset_weights: Sequence[float] | None = None,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    batch_class: type[Batch],
    reset_after_epoch: bool = False,
    use_thread_prefetch: bool = False,
    batch_rampup_factors: dict[str, float] | None = None,
) -> MultiHostDataLoadIterator:
    prepared: list[grain.MapDataset | grain.IterDataset] = []
    length_hints: list[int] = []
    for dataset in datasets:
        if not operations:
            raise ValueError("No operations provided for dataset preparation")

        first_op, *rest_ops = operations
        if not isinstance(first_op, EnsureMapDataset):
            raise ValueError("First operation must wrap dataset into a Grain MapDataset")

        ds = first_op(dataset)

        if shuffle:
            ds = ds.seed(seed + dataloading_host_index)
            ds = ds.shuffle()
        if num_epochs is not None:
            ds = ds.repeat(num_epochs)

        if dataloading_host_count > 1:
            ds = ds[dataloading_host_index::dataloading_host_count]

        for op in rest_ops:
            if isinstance(op, BaseDatasetTransform):
                ds = op(ds) 
            elif isinstance(op, grain_transforms.RandomMap):
                ds = ds.random_map(op) 
            else:
                ds = ds.map(op) 

        length_hint: int | None
        try:
            length_hint = len(ds) 
        except TypeError:
            length_hint = None

        if isinstance(ds, grain.MapDataset):
            ds = ds.to_iter_dataset()
        prepared.append(ds)
        if length_hint is not None:
            length_hints.append(length_hint)

    length_value: int | Sequence[int] | None = length_hints if length_hints else None

    return _make_iterator(
        prepared,
        dataset_lengths=length_value,
        global_mesh=global_mesh,
        global_batch_size=global_batch_size,
        dataloading_host_count=dataloading_host_count,
        dataset_weights=dataset_weights,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        drop_remainder=drop_remainder,
        batch_class=batch_class,
        reset_after_epoch=reset_after_epoch,
        use_thread_prefetch=use_thread_prefetch,
        batch_rampup_factors=batch_rampup_factors,
    )


__all__ = ["make_data_loader", "_make_iterator", "MultiHostDataLoadIterator"]
