from __future__ import annotations

import typing as tp
from collections.abc import Sequence
from contextlib import nullcontext

import logging
import time

import grain
import jax
import jax.tree_util as jtu
from grain import DatasetIterator, IterDataset, transforms as grain_transforms
from grain._src.python.dataset.transformations.prefetch import ThreadPrefetchIterDataset
from jax.sharding import Mesh, PartitionSpec

from dataclasses import replace as dc_replace

from ._dataset_transforms import (
    ApplyFirstFitPacking,
    BaseDatasetTransform,
    EnsureMapDataset,
)
from ._transforms import CollateToBatch
from .._wallclock import ProgramWallClock


Batch = tp.Any
_T = tp.TypeVar("_T")
_S = tp.TypeVar("_S")



class _DatasetIteratorWithInputSpec(DatasetIterator[_T]):

    _SLEEP_SECONDS = 2.0
    _MAX_ATTEMPTS = 5

    def __init__(
        self,
        parent: DatasetIterator[_S],
        pspec: PartitionSpec ,
        mesh: Mesh, 
    ):
        super().__init__(parent)
        self._pspec = pspec 
        self._mesh = mesh
        self._logger = logging.getLogger(__name__)

    def __next__(self) -> _T:
        attempts = 0
        last_error: Exception | None = None
        while True:
            try:
                local_values = next(self._parent)
                break
            except StopIteration:
                raise
            except Exception as err:  # pylint: disable=broad-except
                attempts += 1
                last_error = err
                if attempts >= self._MAX_ATTEMPTS:
                    break
                if self._logger.isEnabledFor(logging.WARNING):
                    self._logger.warning(
                        "Failed to fetch next batch (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempts,
                        self._MAX_ATTEMPTS,
                        err,
                        self._SLEEP_SECONDS,
                    )
                time.sleep(self._SLEEP_SECONDS)

        if last_error is not None and attempts >= self._MAX_ATTEMPTS:
            raise last_error
        with self._stats.record_self_time():
            return self._stats.record_output_spec(
                jtu.tree_map(self.array_from_local_process, local_values)
            )

    def array_from_local_process(self, local_values: _T) -> _T:
        return jax.make_array_from_process_local_data(
            sharding = jax.NamedSharding(self._mesh, self._pspec),
            local_data = local_values,
        )

    def get_state(self):
        return self._parent.get_state()

    def set_state(self, state):
        self._parent.set_state(state)


class IterDatasetWithInputSpec(IterDataset[_T]):
    @tp.overload
    def __init__(
        self,
        parent: IterDataset[_S],
        axis_names: str | tuple[str, ...],
        pspec: None = None,
        mesh: Mesh | None = None,
    ) -> None: ...

    @tp.overload
    def __init__(
        self,
        parent: IterDataset[_S],
        pspec: PartitionSpec,
        axis_names: None = None,
        mesh: Mesh | None = None,
    ) -> None: ...

    def __init__(
        self,
        parent: IterDataset[_S],
        axis_names: str | tuple[str, ...] | None = None,
        pspec: PartitionSpec | None = None,
        mesh: Mesh | None = None,
    ):
        super().__init__(parent)

        if (axis_names is None) == (pspec is None):
            raise ValueError("Exactly one of `axis_name` or `pspec` must be provided.")
        self._pspec = PartitionSpec(axis_names) if axis_names else pspec
        self._mesh = mesh

    def __iter__(self) -> IterDatasetWithInputSpec:
        parent_iter = self._parent.__iter__()
        return _DatasetIteratorWithInputSpec(
                parent_iter, 
                pspec = self._pspec,
                mesh = self._mesh
        )

def _maybe_check_size(
    mesh: Mesh | None,
    pspec: PartitionSpec | None,
    axis_names: str | tuple[str, ...] | None,
    global_batch_size: int,
) -> PartitionSpec:
    """
    Checks that global_batch_size is divisible by the product of mesh axis sizes.
    Returns a pspec or PartitionSpec(axis_names) 
    """
    if axis_names:
        if isinstance(axis_names, str):
            axis_names_tuple = (axis_names,)
        else:
            axis_names_tuple = tuple(axis_names)

        pspec_out = PartitionSpec(axis_names_tuple)
    elif pspec:
        axis_names_tuple = pspec[0]
        pspec_out = pspec
    else:
        raise ValueError("Either axis_names or pspec must be provided.")


    if mesh is None:
        return pspec_out

    axis_size = 1
    for axis in axis_names_tuple:
        axis_size *= mesh.shape.get(axis)
    if global_batch_size % axis_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by the "
            f"product of the sizes of the mesh axes {axis_names_tuple} ({axis_size})."
        )
    return pspec_out 

def make_iterator_with_inputspec(
    grain_datasets: grain.IterDataset | Sequence[grain.IterDataset],
    pspec: PartitionSpec | None = None,
    *,
    mesh: Mesh,
    global_batch_size: int,
    dataloading_host_count: int,
    dataset_weights: Sequence[float] | None = None,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    batch_class: type[Batch] | None = None,
    reset_after_epoch: bool = False,
    use_thread_prefetch: bool = False,
) -> IterDatasetWithInputSpec:

    if isinstance(grain_datasets, tp.Sequence):
        if not all(isinstance(ds, grain.IterDataset) for ds in grain_datasets):
            raise TypeError("All datasets must be instances of grain.IterDataset.")
        mixed = grain.IterDataset.mix(grain_datasets,  weights = dataset_weights) 
    else:
        mixed = grain_datasets 

    local_process_batch_size = global_batch_size // dataloading_host_count

    mixed = mixed.batch(batch_size = local_process_batch_size, drop_remainder = drop_remainder) 

    if batch_class:
        mixed = mixed.map(CollateToBatch(batch_class=batch_class))


    if use_thread_prefetch:
        mixed = ThreadPrefetchIterDataset(
            mixed,
            prefetch_buffer_size=int(worker_buffer_size * worker_count),
        )
    elif worker_count > 0 and not use_thread_prefetch:
        mp_options = grain.MultiprocessingOptions(
            num_workers=worker_count,
            per_worker_buffer_size=worker_buffer_size,
        )
        mixed = mixed.mp_prefetch(mp_options)



    return IterDatasetWithInputSpec(
        mixed,
        pspec = pspec,
        mesh = mesh
    )

def make_dataloader(
    datasets: Sequence[tp.Any],
    operations: Sequence[
        grain_transforms.Map | grain_transforms.RandomMap | BaseDatasetTransform
    ],
    global_batch_size: int,
    axis_names: str | tuple[str, ...] | None = None,
    pspec: PartitionSpec | None = None, 
    mesh: Mesh | None = None,
    num_epochs: int | None = None,
    *,
    dataset_weights: Sequence[float] | None = None,
    dataloading_host_index: int = jax.process_index(),
    dataloading_host_count: int = jax.process_count(),
    shuffle: bool = True,
    seed: int = 0,
    worker_count: int = 1,
    worker_buffer_size: int = 1,
    drop_remainder: bool = True,
    batch_class: type[Batch] | None = None,
    use_thread_prefetch: bool = False,
    wall_clock: ProgramWallClock | None = None,
) -> IterDatasetWithInputSpec:


    if dataloading_host_count <= 0:
        raise ValueError("dataloading_host_count must be positive")
    if global_batch_size % dataloading_host_count != 0:
        raise ValueError(
            "global_batch_size must be divisible by dataloading_host_count"
        )

    local_process_batch_size = global_batch_size // dataloading_host_count

    measurement = wall_clock.measure("dataloader.prepare", mode="setup") if wall_clock else nullcontext()

    prepared: list[grain.MapDataset | grain.IterDataset] = []
    if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
        datasets = (datasets,)
    else:
        datasets = tuple(datasets)

    pspec_out = _maybe_check_size(mesh ,pspec, axis_names, global_batch_size)

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
            #NOTES: pretty much all transformation just wrapping the dataset by another dataset class with new __iter__ and __next__ (the iterator part)
            # so by this we shouldnt differentiate between BaseDatasetTransform and grain transforms
            # need to think more about makeing single interface for all transformation
            if isinstance(op, BaseDatasetTransform):
                if (
                    isinstance(op, ApplyFirstFitPacking)
                    and op.num_packing_bins is None
                ):
                    op = dc_replace(op, num_packing_bins=local_process_batch_size)
                ds = op(ds)  
            elif isinstance(op, grain_transforms.RandomMap):
                if isinstance(ds, grain.MapDataset):
                    ds = ds.to_iter_dataset()
                ds = ds.random_map(op)  
            elif isinstance(op, grain_transforms.Map):
                ds = ds.map(op)  
            else:
                raise TypeError(f"Unsupported operation type: {type(op)}")

        if isinstance(ds, grain.MapDataset):
            ds = ds.to_iter_dataset()

        prepared.append(ds)

    with measurement:
        return make_iterator_with_inputspec(
            prepared,
            pspec=pspec_out,
            mesh=mesh,
            global_batch_size=global_batch_size,
            dataloading_host_count=dataloading_host_count,
            dataset_weights=dataset_weights,
            worker_count=worker_count,
            worker_buffer_size=worker_buffer_size,
            drop_remainder=drop_remainder,
            batch_class=batch_class,
            use_thread_prefetch=use_thread_prefetch,
        )
