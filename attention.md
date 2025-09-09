

bert.encoder.layer.0.attention.self.query: ("", "", "")
bert.encoder.layer.0.attention.self.query: ("", "", "")

this one is from torch
layer_tp_plan = {
    # by default ColwiseParallel input layouts is replicated
    # and RowwiseParallel output layouts is replicated
    "feed_foward.w1": ColwiseParallel(axis_name = self.model_name),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
    "*self.query": RowWiseParallel(),
}


def f(path, module):
    if pattern_match(self.query):
        return eq.tree_at(join(path), module, TPLinear(**module_config))


def make_parallel(module, pattern_matching)-> [where, replacement]:
    where = []

    def _f_with_path(path, m):
        for  k, v in pattern_matching.items()
            if match(k, path) :
                where.append(eval(path))) 
                replacement.append(v)


    return where repalcvement


the idea colwiseparallel is actually to make the linear from (x, y) into (n, x, y) where n, x//len(self.axis_nam)

and put an sharding over that array ("model", None, None)
same with the rowwiseparallel


module = eqx.tree_at(where, module, repacelement) 


reinit(module)

class RowLinear:
    self.weight:
    self.bias:
    
    def __init__(
        cls,
        linear: Linear
        tp_axis_name,
        tp_axis_size, 
        skip_communication, 
    ):
            row, column = Linear.weight.size
            new_row = row // tp_axis_name
            self.weight = Darray(linear.weight.resize(tp_axis_size, new_row, column), pspec = (tp_axis_name, None, None)) 
            self.bias = do the same 


    def __call__(self, x: Float[Array, " column"])-> Float[Array " TP column"]:
        
        with_sharding_constrant(x, (None))
        x = self.einsum("tnr,r"-> "tn" , weight, x)
        if skip_communcation:
            retunrn x
        else:
            return with_sharding_constraint(x, conxtraint)

        
class ColumnLinear:
    self.weight:
    self.bias:
    
    def __init__(
        cls,
        linear: Linear
        tp_axis_name,
        tp_axis_size, 
        skip_communcation
    ):
            row, column = Linear.weight.size
            new_column = column // tp_axis_name
            self.weight = Darray(value = linear.weight.resize(tp_axis_size, row, new_column), pspec = (tp_axis_name, NOne, None))
            we need to rescale the weight by tp_axis_name?
            self.bias = ...


    def __call__(self, x: Float[Array, " column/tp_axis_name"])-> Float[Array " TP column"]:
        with_sharding_constrant(x, (self.tp_axis_name))
         
        x = self.einsum("tnr,r"-> "tn" , weight, x)
        if skip_communication:
            return x
        with_sharding_constraint(x)


kinda like this, but since we're not manually using shard_map, no need to 

        
----------

use with abstract module

abstract_module = jax.filter_eval_shape(model)

pattern_matching = {
    "*self.query": partial(TPLinear, tp_axis_name = tp_axis_name)
}


def make_parallel(module, pattern_matching)-> [where, replacement]:
    where = []

    def _f_with_path(path, m):
        for  k, v in pattern_matching.items()
            if match(k, path) :
                where.append(eval(path))) 
                if m is abstrct module
                    replacement.append(jax.eval_shape(v(linear)))


    return eqx.tree_at(where, module, replacement)


    
abstract_modified_module = make_parallel(abstrct_module, pattern_match)

@jax.jit
def init(abtract_module):
    real_module = reinit(abtract_module)
    partition_spec = nn.get_parititon_sepc(real_module)
    with sharding_constraint(real_module, real_module)
    return real_module

with mesh:
     real_module = init(abstract_module)

-----------
use with module directly
module = Model()

module = make_parallel(module, pattern_match)

@jax.jit
def init_module():
    module = Model()
    module = make_paralell(module, pattern match)
    with sharindg_contartn(module, nn.get_parititon_sepc(module))
    return module


with mesh:
    module = init_module()

-----
fully_sharding


make_fully_shard(module, fsdp_axis_name, min_weight, fsdp_pattern_matching: list[str], remat_pattern_matching: list[str]):
    mostly the same as the make_parallel
    the problem is  we need a new Array where i can store partition_spec, metadata:




fully_shard(module)

and this fully_shard is actually just edit the pspec and find the bigg axis, and anotate the pspec
like from None -> (None , "fsdp")
or form ("model", NOne, NOne) -> ("model", None, "fsdp") #because last dim was divisble by len(fsdp_axis) and it's the largest axis taht has not been sharded yet 

with this function
def predict_shard_params_fn(model: M, axis_name: str, min_weight_size: int = 2**18) -> M:
    """
    Predict a PartitionSpec for each parameter leaf.

    Returns a tree with the same structure as `model` where every Darray leaf
    is replaced with `Darray(value=None, pspec=predicted_pspec)` and all other
    leaves are left unchanged. Intended to be run under a mesh with `axis_name`.
    """
    axis_size = jax.lax.psum(1, axis_name)

    def _shard(node):
        if isinstance(node, Darray):
            value, pspec = node.value, node.pspec
            if value is None:
                return node
            if pspec is None:
                pspec = (None,) * value.ndim
            if len(pspec) != value.ndim:
                raise ValueError("attribute names on Darray should have the same dimension with the attribute value")
            if axis_name in pspec:
                logging.warning(
                    f"Parameter {value.shape} with names {pspec} already sharded on axis {axis_name}."
                )
                return Darray(value=None, pspec=pspec)
            if value.size <= min_weight_size:
                logging.info(
                    f"Parameter {value.shape} with names {pspec} too small to shard, size {value.size} < {min_weight_size}."
                )
                return Darray(value=None, pspec=pspec)
            shape = value.shape
            idx = np.argsort(shape)[::-1]
            for i in idx:
                if shape[i] % axis_size == 0 and pspec[i] is None:
                    return Darray(value=None, pspec=pspec[:i] + (axis_name,) + pspec[i + 1 :])
            logging.warning(
                f"Could not shard {value.shape} with names {pspec} on axis {axis_name}, no suitable axis found"
            )
            return Darray(value=None, pspec=pspec)
        else:
            return node

    return jtu.tree_map(_shard, model, is_leaf=is_darray)

on remat, u need to create filter_remat
and do filter_remat(remat_where, module)

----
here's darray spec

@jtu.register_dataclass
@dataclass(slots = True) #think about this later
class Darray:
    value: jax.Array  | None
    pspec: str | tuple[str, ...] | None = field(metadata=dict(static=True), default = None)



-----

actually for the make_parallel can we make this more general and make this works even on others module?


RowParallelXXX:
    module:
    xxx:
    xxx:

if yes give me someplan


please improve my implementation of parallelism on _tp.py, iwant have a similar things like torch where we provide or tp_plan, and the function will shard the params, constraint the input and the output sharding

after that please add column_parallel and
prepare_input():
prepare_output():
prepare_input_output():

to just put the input, output annotatoin in the function without shard_the params (unlike the row_parallel, column_parallel)

also i dont understand the infer_pspec_from_placement, that handle the Replicate cases, please solve this,

to shard_a_param, we dont necessary shard the params, but more put a partitionspec on the Darray see @spmd.py (please make a general shard_params fucntion to shard param over certain axis and we need to make sure the shard is possible (by checking the divisbble of shape) and make sure the composibility is running correctly (like we shard the same tensor dim over 2 axis, this is possible is arr.size[t.dim] %(mesh[axis1].size * mesh[axis2].size))
dont change the style of mine, i want most of this transormation as a function instead of class :D

here's the torch counterpart:

# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import Placement


__all__ = [
    "ParallelStyle",
    "RowwiseParallel",
    "SequenceParallel",
    "ColwiseParallel",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
]


class ParallelStyle(ABC):
    """
    The parallel style contract defines how the module or submodule should be parallelized.

    It only defines the ``apply`` method for ``parallelize_module`` to use, this allows maximum
    flexibility for different kind of style implementations.
    """

    src_data_rank: Optional[int] = 0

    @abstractmethod
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module: ...


class ColwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a column-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it together with RowwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be replicated.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is sharded on the last dimension.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Colwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w1" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w1" Linear will be converted to Replicated DTensor
        >>> # and the output of "w1" will return :class:`torch.Tensor` that shards on the last dim.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel()})
        >>> ...

    .. note:: By default ``ColwiseParallel`` output is sharded on the last dimension if the ``output_layouts`` not
        specified, if there're operators that require specific tensor shape (i.e. before the paired ``RowwiseParallel``),
        keep in mind that if the output is sharded the operator might need to be adjusted to the sharded size.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # TODO: figure out dynamo support for instance method and switch this to instance method

        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank
                )
            )
            module.register_parameter(name, dist_param)

    def _partition_embedding_fn(self, name, module, device_mesh):
        # colwise shard embedding.weight is straight forward as Shard(1)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(1)], src_data_rank=self.src_data_rank
                )
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
        else:
            raise NotImplementedError(
                "ColwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class RowwiseParallel(ParallelStyle):
    """
    Partition a compatible nn.Module in a row-wise fashion. Currently supports nn.Linear and nn.Embedding.
    Users can compose it with ColwiseParallel to achieve the sharding of more complicated modules.
    (i.e. MLP, Attention)

    Keyword Args:
        input_layouts (Placement, optional):
            The DTensor layout of input tensor for the nn.Module, this is used to annotate the input tensor to
            become a DTensor. If not specified, we assume the input tensor to be sharded on the last dimension.
        output_layouts (Placement, optional):
            The DTensor layout of the output for the nn.Module, this is used to ensure the output of the nn.Module
            with the user desired layout. If not specified, the output tensor is replicated.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: True.
    Returns:
        A :class:`ParallelStyle` object that represents Rowwise sharding of the nn.Module.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "w2" nn.Linear submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "w2" Linear will be converted to DTensor that shards on the last dim
        >>> # and the output of "w2" will return a replicated :class:`torch.Tensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"w2": RowwiseParallel()}),
        >>> ...
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    [Shard(1)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        if getattr(module, "bias", None) is not None:
            # The Linear module has bias
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        [Replicate()],
                        src_data_rank=self.src_data_rank,
                    )
                ),
            )

    def _partition_embedding_fn(self, name, module, device_mesh):
        # rowwise shard embedding.weight is Shard(0)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank
                )
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.Linear):
            partition_fn = self._partition_linear_fn
            # rowwise linear runtime sharding requires input tensor shard on last dim
            self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)
        elif isinstance(module, nn.Embedding):
            partition_fn = self._partition_embedding_fn
            # rowwise embedding runtime sharding requires input tensor replicated
            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel currently only support nn.Linear and nn.Embedding!"
            )

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class SequenceParallel(ParallelStyle):
    """
    SequenceParallel replicates a compatible ``nn.Module`` parameters and runs the sharded computation with
    input sharded on the sequence dimension. This currently supports ``nn.LayerNorm``, ``nn.Dropout``, and the
    `RMSNorm python implementation <https://github.com/facebookresearch/llama/blob/main/llama/model.py#L34>`__

    This style implements the operation that is described in the paper
    `Reducing Activation Recomputation in Large Transformer Models <https://arxiv.org/abs/2205.05198>`__

    If the input passed in to this ``nn.Module`` is a :class:`torch.Tensor`, it assumes that the input is already sharded
    on the sequence dimension and converts the input to a :class:`DTensor` sharded on the sequence dimension. If the input
    passed in to this ``nn.Module`` is already a :class:`DTensor` but is not sharded on the sequence dimension, it would
    redistribute the input to be sharded on the sequence dimension.

    The output of the ``nn.Module`` will be sharded on the sequence dimension.

    Keyword Args:
        sequence_dim (int, optional):
            The sequence dimension of the input tensor for the ``nn.Module``, this is used to annotate the input tensor to
            become a DTensor that is sharded on the sequence dimension, default: 1.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module output, default: False.
    Returns:
        A :class:`ParallelStyle` object that represents Sequence Parallel of the ``nn.Module``.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> m = Model(...)  # m is a nn.Module that contains a "norm" nn.LayerNorm submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # By default, the input of the "norm" will be converted to DTensor that shards on the sequence dim
        >>> # and the output of "norm" will return a sharded on sequence dimension :class:`DTensor`.
        >>>
        >>> sharded_mod = parallelize_module(m, tp_mesh, {"norm": SequenceParallel()}),
        >>> ...

    .. note:: SequenceParallel style assumes ones initialization if there are weights in the nn.Module (i.e.
        ``nn.LayerNorm`` or ``RMSNorm``, and they by default have ones initialization). If you have custom
        inits for the weights on those modules, you need to broadcast the weights before/after parallelizing
        to ensure that they are replicated.
    """

    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def _replicate_module_fn(
        self, name: str, module: nn.Module, device_mesh: DeviceMesh
    ):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False)
            )
            module.register_parameter(p_name, replicated_param)

    @staticmethod
    def _prepare_input_fn(sequence_sharding, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        if isinstance(input_tensor, DTensor):
            # if the passed in input DTensor is not sharded on the sequence dim, we need to redistribute it
            if input_tensor.placements != sequence_sharding:
                input_tensor = input_tensor.redistribute(
                    placements=sequence_sharding, async_op=True
                )
            return input_tensor
        elif isinstance(input_tensor, torch.Tensor):
            # assume the input passed in already sharded on the sequence dim and create the DTensor
            return DTensor.from_local(
                input_tensor, device_mesh, sequence_sharding, run_check=False
            )
        else:
            raise ValueError(
                f"expecting input of {mod} to be a torch.Tensor or DTensor, but got {input_tensor}"
            )

    @staticmethod
    def _prepare_output_fn(use_local_output, mod, outputs, device_mesh):
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),
            partial(self._prepare_output_fn, self.use_local_output),
        )

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        if len(self.sequence_sharding) == 1:
            tmpstr += f"sequence_dim={self.sequence_sharding[0].dim}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInput(ParallelStyle):
    """
    Configure the nn.Module's inputs to convert the input tensors of the nn.Module to DTensors at runtime according to
    ``input_layouts``, and perform layout redistribution according to the ``desired_input_layouts``.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated to Sharded DTensor
        >>> # and then redistributed to Replicated DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...)
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_output: bool = False,
    ):
        self.input_layouts = (
            (input_layouts,) if isinstance(input_layouts, Placement) else input_layouts
        )
        self.desired_input_layouts = (
            (desired_input_layouts,)
            if isinstance(desired_input_layouts, Placement)
            else desired_input_layouts
        )
        self.use_local_output = use_local_output
        if self.input_layouts is not None:
            assert self.desired_input_layouts is not None, (
                "desired module inputs should not be None!"
            )
            assert len(self.input_layouts) == len(self.desired_input_layouts), (
                "input_layouts and desired_input_layouts should have same length!"
            )
        self.with_kwargs = input_kwarg_layouts is not None
        self.input_kwarg_layouts = input_kwarg_layouts or {}
        self.desired_input_kwarg_layouts = desired_input_kwarg_layouts or {}
        if self.with_kwargs:
            assert len(self.input_kwarg_layouts) == len(
                self.desired_input_kwarg_layouts
            ), (
                "input_kwarg_layouts and desired_input_kwarg_layouts should have same length!"
            )

    def _prepare_input_arg(
        self,
        input: Any,
        mesh: DeviceMesh,
        input_layout: Optional[Placement],
        desired_layout: Optional[Placement],
    ):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(input, torch.Tensor), (
                    "expecting input to be a torch.Tensor!"
                )
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _prepare_input_fn(self, inputs, device_mesh):
        if self.input_layouts is None:
            return inputs
        prepared_inputs = []
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if len(inputs) != len(self.input_layouts):
            raise ValueError("module inputs and input_layouts should have same length!")

        assert self.desired_input_layouts is not None, (
            "desired module inputs should not be None!"
        )
        for inp, input_layout, desired_layout in zip(
            inputs, self.input_layouts, self.desired_input_layouts
        ):
            prepared_inputs.append(
                self._prepare_input_arg(inp, device_mesh, input_layout, desired_layout)
            )
        return tuple(prepared_inputs)

    def _prepare_input_kwarg_fn(self, inputs, kwarg_inputs, device_mesh):
        prepared_arg_inputs = self._prepare_input_fn(inputs, device_mesh)
        prepared_kwarg_inputs = {}
        for kwarg_key in kwarg_inputs.keys():
            kwarg_val = kwarg_inputs[kwarg_key]
            input_layout = self.input_kwarg_layouts.get(kwarg_key)
            desired_input_layout = self.desired_input_kwarg_layouts.get(kwarg_key)

            prepared_kwarg_inputs[kwarg_key] = self._prepare_input_arg(
                kwarg_val, device_mesh, input_layout, desired_input_layout
            )

        return (prepared_arg_inputs, prepared_kwarg_inputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if self.with_kwargs:
            module.register_forward_pre_hook(
                lambda _, inputs, kwargs: self._prepare_input_kwarg_fn(
                    inputs, kwargs, device_mesh
                ),
                with_kwargs=True,
            )  # type: ignore[misc]
        else:
            module.register_forward_pre_hook(
                lambda _, inputs: self._prepare_input_fn(inputs, device_mesh)
            )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.input_layouts}, "
        tmpstr += f"desired_input_layouts={self.desired_input_layouts}, "
        tmpstr += f"input_kwarg_layouts={self.input_kwarg_layouts}, "
        tmpstr += f"desired_input_kwarg_layouts={self.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleOutput(ParallelStyle):
    """
    Configure the nn.Module's outputs to convert the output tensors of the nn.Module to DTensors at runtime according to
    ``output_layouts``, and perform layout redistribution according to the ``desired_output_layouts``.

    Keyword Args:
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A ParallelStyle object that prepares the sharding layouts of the nn.Module's outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the output of the TransformerBlock will be converted to Replicated DTensor
        >>> # and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan = PrepareModuleOutput(
        >>>         output_layouts=Replicate(),
        >>>         desired_output_layouts=Shard(0)
        >>>     )
        >>> )
    """

    def __init__(
        self,
        *,
        output_layouts: Union[Placement, tuple[Placement]],
        desired_output_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.output_layouts = (
            (output_layouts,)
            if isinstance(output_layouts, Placement)
            else output_layouts
        )
        self.desired_output_layouts = (
            (desired_output_layouts,)
            if isinstance(desired_output_layouts, Placement)
            else desired_output_layouts
        )
        self.use_local_output = use_local_output
        assert len(self.output_layouts) == len(self.desired_output_layouts), (
            "output_layouts and desired_output_layouts should have same length!"
        )

    def _prepare_out_fn(self, outputs, device_mesh):
        prepared_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_layouts):
            raise ValueError(
                "module outputs and output_layouts should have same length!"
            )
        for out, out_layout, desired_out_layout in zip(
            outputs, self.output_layouts, self.desired_output_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert out.placements[0] == out_layout
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,), run_check=False
                    )

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        else:
            return tuple(prepared_outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        module.register_forward_hook(
            lambda _, inputs, outputs: self._prepare_out_fn(outputs, device_mesh)
        )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"output_layouts={self.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr


class PrepareModuleInputOutput(ParallelStyle):
    """
    Configure the nn.Module's inputs (and outputs) to convert the input tensors (and output tensors, respectively) of the nn.Module
    to DTensors at runtime according to ``input_layouts`` (and output_layouts, respectively), and perform layout redistribution
    according to the ``desired_input_layouts`` (and ``desired_output_layouts``, respectively). This is a combination of
    :class:`PrepareModuleInput` and :class:`PrepareModuleOutput`.

    Keyword Args:
        input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The DTensor layouts of input tensors for the nn.Module, this is used to convert the input tensors to
            DTensors. If some inputs are not torch.Tensor or no need to convert to DTensors, ``None`` need to be specified
            as a placeholder. default: None.
        desired_input_layouts (Union[Placement, Tuple[Optional[Placement]]]):
            The desired DTensor layout of input tensors for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. This argument needs to have the same length with ``input_layouts``. default: None.
        input_kwarg_layouts (Dict[str, Placement]):
            The DTensor layouts of input kwargs for the nn.Module, this is used to convert the input kwarg tensors to DTensors.
            default: None
        desired_input_kwarg_layouts: (Dict[str, Placement]):
            The desired DTensor layout of input kwargs for the nn.Module, this is used to ensure the inputs of the nn.Module
            have the desired DTensor layouts. default: None.
        use_local_input (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module inputs, default: False.
        output_layouts (Union[Placement, Tuple[Placement]]):
            The DTensor layouts of output tensors for the nn.Module, this is used to convert the output tensors to
            DTensors if they are :class:`torch.Tensor`. If some outputs are not torch.Tensor or no need to convert to DTensors,
            ``None`` need to be specified as a placeholder.
        desired_output_layouts (Union[Placement, Tuple[Placement]]):
            The desired DTensor layouts of output tensors for the nn.Module, this is used to ensure the outputs of the nn.Module
            have the desired DTensor layouts.
        use_local_output (bool, optional):
            Whether to use local :class:`torch.Tensor` instead of :class:`DTensor` for the module outputs, default: True.
    Returns:
        A :class:`ParallelStyle` object that prepares the sharding layouts of the nn.Module's inputs and outputs.

    Example::
        >>> # xdoctest: +SKIP(failing)
        >>> from torch.distributed.tensor.parallel import parallelize_module, PrepareModuleInputOutput
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> ...
        >>> block = TransformerBlock(...)  # block is a nn.Module that contains an "attn" Attention submodule
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>>
        >>> # According to the style specified below, the first input of attn will be annotated as Sharded DTensor
        >>> # and then redistributed to Replicated DTensor, and the output of the TransformerBlock will be annotated
        >>> # as Replicated DTensor and then redistributed to Sharded DTensor.
        >>> parallelize_module(
        >>>     block, # this can be a submodule or module
        >>>     tp_mesh,
        >>>     parallelize_plan={
        >>>         "attn": PrepareModuleInputOutput(
        >>>             input_layouts=(Shard(0), None, None, ...),
        >>>             desired_input_layouts=(Replicate(), None, None, ...),
        >>>             output_layouts=Replicate(),
        >>>             desired_output_layouts=Shard(0),
        >>>         ),
        >>>     }
        >>> )
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Union[Placement, tuple[Optional[Placement]]]] = None,
        desired_input_layouts: Optional[
            Union[Placement, tuple[Optional[Placement]]]
        ] = None,
        input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        desired_input_kwarg_layouts: Optional[dict[str, Placement]] = None,
        use_local_input: bool = False,
        output_layouts: Union[Placement, tuple[Placement]],
        desired_output_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.prepare_module_input = PrepareModuleInput(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_input,
        )
        self.prepare_module_output = PrepareModuleOutput(
            output_layouts=output_layouts,
            desired_output_layouts=desired_output_layouts,
            use_local_output=use_local_output,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)

        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"input_layouts={self.prepare_module_input.input_layouts}, "
        tmpstr += (
            f"desired_input_layouts={self.prepare_module_input.desired_input_layouts}, "
        )
        tmpstr += (
            f"input_kwarg_layouts={self.prepare_module_input.input_kwarg_layouts}, "
        )
        tmpstr += f"desired_input_kwarg_layouts={self.prepare_module_input.desired_input_kwarg_layouts}, "
        tmpstr += f"use_local_input={self.prepare_module_input.use_local_output}, "
        tmpstr += f"output_layouts={self.prepare_module_output.output_layouts}, "
        tmpstr += f"desired_output_layouts={self.prepare_module_output.desired_output_layouts}, "
        tmpstr += f"use_local_output={self.prepare_module_output.use_local_output}"
        tmpstr += ")"
        return tmpstr


# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import inspect
import warnings
from collections.abc import Sequence
from typing import Any, Callable, cast, Optional
from typing_extensions import deprecated

import torch
import torch.distributed.tensor._dispatch as op_dispatch
import torch.distributed.tensor._random as random
import torch.nn as nn
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.tensor._collective_utils import check_tensor_meta, mesh_broadcast
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._redistribute import (
    Redistribute,
    redistribute_local_tensor,
)
from torch.distributed.tensor._utils import (
    compute_global_tensor_info,
    compute_local_shape_and_global_offset,
    normalize_to_torch_size,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

aten = torch.ops.aten


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure DTensor to work
# together with torch.Tensor within the autograd engine. This
# allows DTensor to only exist on part of the module hierarchy.
#
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DTensor params, the following forward/backward should work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input (from_local) -> Sharded Module B -> DTensor output
#           -> torch.Tensor output (to_local) -> Module C
#
# So from_local/to_local must be Autograd functions.
#
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input: "DTensor",
        grad_placements: Optional[Sequence[Placement]],
    ):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        _, tensor_stride = compute_global_tensor_info(
            grad_output, mesh, dtensor_spec.placements
        )
        tensor_stride = tuple(tensor_stride)
        grad_placements = grad_placements or dtensor_spec.placements
        grad_spec = DTensorSpec(
            mesh,
            grad_placements,
            tensor_meta=TensorMeta(
                shape=dtensor_meta.shape,
                stride=tensor_stride,
                dtype=dtensor_meta.dtype,
            ),
        )

        return (
            DTensor(
                grad_output,
                grad_spec,
                requires_grad=grad_output.requires_grad,
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        run_check: bool,
        shape: Optional[torch.Size] = None,
        stride: Optional[tuple[int, ...]] = None,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh

        if shape and stride:
            tensor_shape, tensor_stride = shape, stride
        elif not shape and not stride:
            # if it's not by default run_check, we assume user is certain that each
            # rank has the same tensor shape, and we just use that to calculate the
            # global shape
            global_shape, global_stride = compute_global_tensor_info(
                input, device_mesh, placements
            )
            tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)
        else:
            raise RuntimeError(
                f"Found shape:{shape}, stride:{stride}.",
                "Please pass both shape and stride at the same time.",
            )

        if device_mesh.get_coordinate() is None:
            # if the global rank is not participating in the device mesh, we
            # simply set the local tensor to an empty tensor
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            # TODO: support uneven sharding when global shape/stride not passed, by
            # building the global TensorMeta during check_tensor_meta
            check_shape_stride = not shape and not stride
            check_tensor_meta(input, check_shape_stride=check_shape_stride)
            # TODO: See if we need to make this run_check logic
            # have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    mesh_broadcast(input, device_mesh, mesh_dim=idx)

        dist_spec = DTensorSpec(
            device_mesh,
            placements,
            tensor_meta=TensorMeta(
                tensor_shape,
                tensor_stride,
                input.dtype,
            ),
        )

        # We want a fresh Tensor object that shares memory with the input tensor
        dist_tensor = DTensor(
            input.view_as(input),
            dist_spec,
            # requires_grad of the dist tensor depends on if input
            # requires_grad or not
            requires_grad=input.requires_grad,
        )
        return dist_tensor

    @staticmethod
    def backward(ctx, grad_output: "DTensor"):  # type: ignore[override]
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        if grad_output.placements != previous_placement:
            current_spec = grad_output._spec
            target_spec = DTensorSpec(
                previous_device_mesh,
                previous_placement,
                tensor_meta=grad_output._spec.tensor_meta,
            )
            local_tensor = grad_output._local_tensor
            output = redistribute_local_tensor(
                local_tensor, current_spec, target_spec, is_backward=True
            )
            # TODO: return the redistributed local tensor directly without
            # differentiable backward. see if this make sense for all cases.
            return output, None, None, None, None, None

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return grad_output.to_local(), None, None, None, None, None


class DTensor(torch.Tensor):
    """
    ``DTensor`` (Distributed Tensor) is a subclass of ``torch.Tensor`` that provides single-device like
    abstraction to program with multi-device ``torch.Tensor``. It describes the distributed tensor sharding
    layout (DTensor Layout) through the :class:`DeviceMesh` and following types of :class:`Placement`:

    * :class:`Shard`: Tensor sharded on the tensor dimension ``dim`` on the devices of the ``DeviceMesh`` dimension
    * :class:`Replicate`: Tensor replicated on the devices of the ``DeviceMesh`` dimension
    * :class:`Partial`: Tensor is pending reduction on the devices of the ``DeviceMesh`` dimension

    When calling PyTorch operators, ``DTensor`` overrides the PyTorch operators to perform sharded computation and issue
    communications whenever necessary. Along with the operator computation, ``DTensor`` will transform or propagate the
    placements (DTensor Layout) properly (based on the operator semantic itself) and generate new ``DTensor`` outputs.

    To ensure numerical correctness of the ``DTensor`` sharded computation when calling PyTorch operators, ``DTensor``
    requires every Tensor argument of the operator be DTensor.

    .. note:: Directly using the Tensor subclass constructor here is not the recommended way to create a ``DTensor``
        (i.e. it does not handle autograd correctly hence is not the public API). Please refer to the `create_dtensor`_
        section to see how to create a ``DTensor``.
    """

    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # _op_dispatcher instance as a class attribute to handle runtime dispatching logic
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "DTensor":
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).

        .. note:: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using ``DTensor.from_local``, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using ``distribute_tensor``.
        """
        if local_tensor.requires_grad and not requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        assert spec.tensor_meta is not None, "TensorMeta should not be None!"
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        r._spec = spec
        r._local_tensor = local_tensor
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(self, *args, **kwargs):
        super().__init__()

    # pyre-fixme[14]: `__repr__` overrides method defined in `DTensor` inconsistently.
    # pyre-fixme[3]: Return type must be annotated.
    def __repr__(self):  # type: ignore[override]
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        unflatten_spec = DTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
        )
        return DTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    def __coerce_tangent_metadata__(self):
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        return self.redistribute(device_mesh=self.device_mesh, placements=placements)

    def __coerce_same_metadata_as_tangent__(self, flatten_spec, expected_type=None):
        if expected_type is not None:
            return None

        (spec, _) = flatten_spec  # Result of tensor_flatten()
        return self.redistribute(
            device_mesh=self.device_mesh,
            placements=spec.placements,
        )

    @classmethod
    @torch._disable_dynamo
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        return DTensor._op_dispatcher.dispatch(
            func,
            args,
            kwargs or {},
        )

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        run_check: bool = False,
        shape: Optional[torch.Size] = None,
        stride: Optional[tuple[int, ...]] = None,
    ) -> "DTensor":
        """
        Create a :class:`DTensor` from a local torch.Tensor on each rank
        according to the ``device_mesh`` and ``placements`` specified.

        Args:
            local_tensor (torch.Tensor): local torch.Tensor on each rank.
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                tensor, if not specified, must be called under a DeviceMesh
                context manager, default: None
            placements (List[:class:`Placement`], optional): the placements that
                describes how to place the local torch.Tensor on DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.

        Keyword args:
            run_check (bool, optional): at a cost of extra communications, perform
                sanity check across ranks to check each local tensor's meta information
                to ensure correctness. If have :class:`Replicate` in ``placements``, the
                data on first rank of the device mesh dimension will be broadcasted
                to other ranks. default: False
            shape (torch.Size, optional): A List of int which specifies the size of
                DTensor which build on top of `local_tensor`. Note this needs to be
                provided if the shape of ``local_tensor`` are different across the ranks.
                If not provided, ``shape`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None
            stride (tuple, optional): A List of int which specifies the stride of DTensor.
                If not provided, ``stride`` will be computed assuming the given distributed
                tensor is evenly sharded across ranks. default: None

        Returns:
            A :class:`DTensor` object

        .. note:: When ``run_check=False``, it is the user's responsibility to ensure the
            local tensor passed in is correct across ranks (i.e. the tensor is sharded for
            the ``Shard(dim)`` placement or replicated for the ``Replicate()`` placement).
            If not, the behavior of the created DTensor is undefined.

        .. note:: ``from_local`` is differentiable, the `requires_grad` of the created
            `DTensor` object will depend on if `local_tensor` requires_grad or not.
        """
        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            run_check,
            shape,
            stride,
        )

    def to_local(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Get the local tensor of this DTensor on its current rank. For sharding it returns
        a local shard of the logical tensor view, for replication it returns the replica on
        its current rank.

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the Tensor returned from this
                function.
                `to_local` converts DTensor to local tensor and the returned local tensor
                might not be used as the original DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original DTensor layout.
                If not specified, we will assume the gradient layout remains the same
                as the original DTensor and use that for gradient computation.

        Returns:
            A :class:`torch.Tensor` or ``AsyncCollectiveTensor`` object. it represents the
            local tensor on its current rank. When an ``AsyncCollectiveTensor`` object is returned,
            it means the local tensor is not ready yet (i.e. communication is not finished). In this
            case, user needs to call ``wait`` to wait the local tensor to be ready.

        .. note:: ``to_local`` is differentiable, the ``requires_grad`` of the local tensor returned
            will depend on if the `DTensor` requires_grad or not.
        """
        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None and not isinstance(grad_placements, tuple):
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(
            self, grad_placements
        )  # pyre-ignore[16]: autograd func

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        async_op: bool = False,
        forward_dtype: Optional[torch.dtype] = None,
        backward_dtype: Optional[torch.dtype] = None,
    ) -> "DTensor":
        """
        ``redistribute`` performs necessary collective operations that redistribute the current
        DTensor from its current placements to a new placements, or from its current DeviceMesh
        to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
        specifying a Replicate placement for each dimension of the DeviceMesh.

        When redistributing from current to the new placements on one device mesh dimension, we
        will perform the following operations including communication collective or local operation:

        1. ``Shard(dim)`` -> ``Replicate()``: ``all_gather``
        2. ``Shard(src_dim)`` -> ``Shard(dst_dim)``: ``all_to_all``
        3. ``Replicate()`` -> ``Shard(dim)``: local chunking (i.e. ``torch.chunk``)
        4. ``Partial()`` -> ``Replicate()``: ``all_reduce``
        5. ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter``


        ``redistribute`` would correctly figure out the necessary redistribute steps for DTensors
        that are created either on 1-D or N-D DeviceMesh.

        Args:
            device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
                DTensor. If not specified, it would use the current DTensor's DeviceMesh.
                default: None
            placements (List[:class:`Placement`], optional): the new placements that
                describes how to place the DTensor into the DeviceMesh, must
                have the same number of elements as ``device_mesh.ndim``.
                default: replicate on all mesh dimensions

        Keyword args:
            async_op (bool, optional): whether to perform the DTensor redistribute operation
                asynchronously or not. Default: False
            forward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``forward_dtype`` before redistributing the local tensor in its forward.
                The result DTensor will be in ``forward_dtype`` Default: None.
            backward_dtype (torch.dtype, optional): the local tensor datatype can be converted to
                ``backward_dtype`` before redistributing the local tensor in its backward.
                The result DTensor gradient would be converted back to the current DTensor dtype. Default: None

        Returns:
            A :class:`DTensor` object

        .. note:: ``redistribute`` is differentiable, which means user do not need to worry about
            the backward formula of the redistribute operation.

        .. note:: ``redistribute`` currently only supports redistributing DTensor on the same DeviceMesh,
            Please file an issue if you need to redistribute DTensor to different DeviceMesh.
        """
        # NOTE: This redistribute API currently only supports out
        # of place redistribution, i.e. it always create a new
        # DTensor object and leave the original one unchanged.

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return Redistribute.apply(
            self, device_mesh, placements, async_op, forward_dtype, backward_dtype
        )

    def full_tensor(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntatic sugar of the following code:

        ``dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()``

        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.

        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.

        .. note:: ``full_tensor`` is differentiable.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        return _ToTorchTensor.apply(redist_res, grad_placements)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: ``device_mesh`` is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: ``placements`` is a read-only property, it can not be set.
        """
        return self._spec.placements

    def __create_write_items__(self, fqn: str, object: Any):
        from torch.distributed.checkpoint.planner_helpers import (
            _create_write_items_for_dtensor,
        )

        if hasattr(self._local_tensor, "__create_write_items__"):
            return self._local_tensor.__create_write_items__(fqn, object)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_write_items_for_dtensor(fqn, object)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __create_chunk_list__(self):
        """
        Return a list of ChunkStorageMetadata, which is a dataclass that describes the size/offset of the local shard/replica
        on current rank. For DTensor, each rank will have a single local shard/replica, so the returned list usually only
        has one element.

        This dunder method is primariy used for distributed checkpoint purpose.

        Returns:
            A List[:class:`ChunkStorageMetadata`] object that represents the shard size/offset on the current rank.
        """
        from torch.distributed.checkpoint.planner_helpers import (
            _create_chunk_from_dtensor,
        )

        if hasattr(self._local_tensor, "__create_chunk_list__"):
            return self._local_tensor.__create_chunk_list__()  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return [_create_chunk_from_dtensor(self)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __get_tensor_shard__(self, index):
        if hasattr(self._local_tensor, "__get_tensor_shard__"):
            return self._local_tensor.__get_tensor_shard__(index)  # type: ignore[attr-defined]
        elif isinstance(self._local_tensor, torch.Tensor):
            return self.to_local()
        else:
            raise RuntimeError("Unsupported tensor type!")


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    *,
    src_data_rank: Optional[int] = 0,
) -> DTensor:
    """
    Distribute a leaf ``torch.Tensor`` (i.e. nn.Parameter/buffers) to the ``device_mesh`` according
    to the ``placements`` specified. The rank of ``device_mesh`` and ``placements`` must be the
    same. The ``tensor`` to distribute is the logical or "global" tensor, and the API would use
    the ``tensor`` from first rank of the DeviceMesh dimension as the source of truth to preserve
    the single-device semantic. If you want to construct a DTensor in the middle of the Autograd
    computation, please use :meth:`DTensor.from_local` instead.

    Args:
        tensor (torch.Tensor): torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use ``torch.chunk``
            semantic to shard the tensor and scatter the shards. The uneven sharding
            behavior is experimental and subject to change.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as ``device_mesh.ndim``. If not specified, we will
            by default replicate the tensor across the ``device_mesh`` from the
            first rank of each dimension of the `device_mesh`.

    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is
            used by :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks.
            By default, we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve
            the single-device semantic. If passing ``None`` explicitly, :meth:`distribute_tensor` simply uses
            its local data instead of trying to preserve the single-device semantic via scatter/broadcast.
            Default: 0

    Returns:
        A :class:`DTensor` or ``XLAShardedTensor`` object.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_tensor``
        return `XLAShardedTensor` instead. see `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # call PyTorch/XLA SPMD for `xla` backend type device mesh.
            # This returns XLAShardedTensor
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_tensor,
            )

            return xla_distribute_tensor(tensor, device_mesh, placements)  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    if not tensor.is_leaf:
        raise RuntimeError(
            "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
        )

    # convert tensor to the corresponding device type if it's not in that device type
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    # set default placements to replicated if not specified
    if placements is None:
        placements = [Replicate() for _ in range(device_mesh.ndim)]

    if len(placements) != device_mesh.ndim:
        raise ValueError(
            f"`placements` must have the same length as `device_mesh.ndim`! "
            f"Found placements length: {len(placements)}, and device_mesh.ndim: {device_mesh.ndim}."
        )
    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we need to check:
        # 1. if the we can further shard this DTensor if the two device mesh belong to
        #   the same parenet mesh and further sharding is possible.
        # 2. check if device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != tuple(placements):
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor.detach()

    # TODO(xilun): address sharding order
    # distribute the tensor according to the placements.
    placements = list(placements)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            placement = cast(Shard, placement)
            if placement.dim < 0:
                # normalize shard placement dim
                placement = Shard(placement.dim + tensor.ndim)
                placements[idx] = placement
            local_tensor = placement._shard_tensor(
                local_tensor, device_mesh, idx, src_data_rank
            )
        elif placement.is_replicate():
            placement = cast(Replicate, placement)
            local_tensor = placement._replicate_tensor(
                local_tensor, device_mesh, idx, src_data_rank
            )
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )
    placements = tuple(placements)

    assert local_tensor is not None, "distributing a tensor should not be None"
    # detach the local tensor passed to DTensor since after the construction
    # of DTensor, autograd would work on top of DTensor instead of local tensor
    spec = DTensorSpec(
        mesh=device_mesh,
        placements=placements,
        tensor_meta=TensorMeta(
            shape=tensor.size(),
            stride=tensor.stride(),
            dtype=tensor.dtype,
        ),
    )
    return DTensor(
        local_tensor.requires_grad_(tensor.requires_grad),
        spec,
        requires_grad=tensor.requires_grad,
    )


@deprecated("Please use `distribute_tensor` with `src_data_rank=None` instead.")
def _shard_tensor(
    full_tensor: torch.Tensor,
    placements: Sequence[Shard],
    device_mesh: Optional[DeviceMesh] = None,
) -> "DTensor":
    """
    Locally shards a full tensor based on indicated sharding arrangement, and
    returns a DTensor containing the local shard.

    .. warning:: This is a private API that is subject to change. It skips the
        communication otherwise required by `distribute_tensor`. It is only
        applicable to cases where all ranks have the same `full_tensor`. For
        example, in distributed inference all ranks load from the same
        checkpoint. This API will not check for data equality between ranks, it
        is thus user's responsibility to ensure the `full_tensor` is the same
        across ranks.

    Args:
        full_tensor (torch.Tensor): the full tensor to be sharded.
        placements (Sequence[:class:`Shard`]): the placements that
            describes how to place the local tensor on DeviceMesh.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            DTensor.  Must have same dimension as the number of placements.
            If not specified, would be retrieve from current context.

    Returns:
        A :class:`DTensor` object with the shard as its local tensor.

    Examples:
        >>> # xdoctest: +SKIP("need world_size and rank")
        >>> device_mesh = dist.init_device_mesh("cuda", (world_size,))
        >>> full_tensor = torch.arange(world_size, device=f"cuda:{rank}")
        >>> dtensor = _shard_tensor(full_tensor, [Shard(1)], device_mesh)
    """
    return distribute_tensor(full_tensor, device_mesh, placements, src_data_rank=None)


def distribute_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
    output_fn: Optional[Callable[[nn.Module, Any, DeviceMesh], None]] = None,
) -> nn.Module:
    """
    This function expose three functions to control the parameters/inputs/outputs of the module:

    1. To perform sharding on the module before runtime execution by specifying the
    ``partition_fn`` (i.e. allow user to convert Module parameters to :class:`DTensor`
    parameters according to the `partition_fn` specified).
    2. To control the inputs or outputs of the module during runtime execution by
    specifying the ``input_fn`` and ``output_fn``. (i.e. convert the input to
    :class:`DTensor`, convert the output back to ``torch.Tensor``)

    Args:
        module (:class:`nn.Module`): user module to be partitioned.
        device_mesh (:class:`DeviceMesh`): the device mesh to place the module.
        partition_fn (Callable): the function to partition parameters (i.e. shard certain
            parameters across the ``device_mesh``). If ``partition_fn`` is not specified,
            by default we replicate all module parameters of ``module`` across the mesh.
        input_fn (Callable): specify the input distribution, i.e. could control how the
            input of the module is sharded. ``input_fn`` will be installed as a module
            ``forward_pre_hook`` (pre forward hook).
        output_fn (Callable): specify the output distribution, i.e. could control how the
            output is sharded, or convert it back to torch.Tensor. ``output_fn`` will be
            installed as a module ``forward_hook`` (post forward hook).

    Returns:
        A module that contains parameters/buffers that are all ``DTensor`` s.

    .. note::
        When initialize the DeviceMesh with the ``xla`` device_type, ``distribute_module``
        return nn.Module with PyTorch/XLA SPMD annotated parameters. See
        `this issue <https://github.com/pytorch/pytorch/issues/92909>`__
        for more details. The XLA integration is experimental and subject to change.

    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_module")

    already_distributed = getattr(module, "_distribute_module_applied", False)
    if already_distributed:
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )

    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type
    if device_type == "xla":
        try:
            # This function annotates all module parameters for auto-partitioning with
            # PyTorch/XLA SPMD or explicitly partition to :class:`XLAShardedTensor` parameters
            # according to the `partition_fn` specified.
            from torch_xla.distributed.spmd import (  # type:ignore[import]
                xla_distribute_module,
            )

            return xla_distribute_module(
                module, device_mesh, partition_fn, input_fn, output_fn
            )  # type:ignore[return-value]
        except ImportError as e:
            msg = "To use DTensor API with xla, you must install the torch_xla package!"
            raise ImportError(msg) from e

    def replicate_module_params_buffers(m: nn.Module, mesh: DeviceMesh) -> None:
        # This function loop over the immediate module parameters and
        # buffers, replicate all non DTensor params/buffers to DTensor
        # parameters/buffers, if they have not been partitioned in the
        # partition_fn, we can't easily use `module._apply` here
        # because we don't know what happened inside partition_fn as
        # user could do anything, i.e. install hooks, and we want to
        # preserve those.
        full_replicate = [Replicate()] * mesh.ndim
        for key, param in m._parameters.items():
            if param is not None and not isinstance(param, DTensor):
                m.register_parameter(
                    key,
                    nn.Parameter(distribute_tensor(param.data, mesh, full_replicate)),
                )
        for key, buffer in m._buffers.items():
            if buffer is not None and not isinstance(buffer, DTensor):
                m._buffers[key] = distribute_tensor(buffer, mesh, full_replicate)

    if partition_fn is None:
        # if partition_fn not specified, we by default replicate
        # all module params/buffers
        for name, submod in module.named_modules():
            replicate_module_params_buffers(submod, device_mesh)
    else:
        # apply partition_fun to submodules
        for name, submod in module.named_modules():
            partition_fn(name, submod, device_mesh)
            replicate_module_params_buffers(submod, device_mesh)

    # register input_fn as module forward pre hook
    if input_fn is not None:
        # check the input_fn signature
        num_args = len(inspect.signature(input_fn).parameters)
        if num_args == 2:
            # input_fn only takes in inputs and device mesh
            warnings.warn(
                "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
                "please use input_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_pre_hook(
                lambda _, inputs: input_fn(inputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            # input_fn takes in module, inputs, device mesh
            module.register_forward_pre_hook(
                lambda mod, inputs: input_fn(mod, inputs, device_mesh)
            )
        else:
            raise ValueError(
                f"input_fn should take in 3 arguments, but got {num_args} arguments!"
            )
    # register output_fn as module forward hook
    if output_fn is not None:
        num_args = len(inspect.signature(output_fn).parameters)
        if num_args == 2:
            # output_fn only takes in outputs and device mesh
            warnings.warn(
                "Deprecating output_fn that takes two arguments (inputs, device_mesh), "
                "please use output_fn that takes in (module, inputs, device_mesh) instead!",
                FutureWarning,
                stacklevel=2,
            )
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh)  # type: ignore[call-arg]
            )
        elif num_args == 3:
            module.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh)
            )
        else:
            raise ValueError(
                f"output_fn should take in 3 arguments, but got {num_args} arguments!"
            )

    module._distribute_module_applied = True  # type: ignore[assignment]
    return module


# Below are tensor factory function APIs, which are used to create a DTensor directly. We need
# to make separate factory function APIs because tensor subclass could not override the tensor
# factory methods, and we need user to call the factory functions with user intended device_mesh
# and placements to create a proper DTensor.


def _dtensor_init_helper(  # type: ignore[no-untyped-def]
    init_op,
    size: torch.Size,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    **kwargs,
) -> DTensor:
    # if device_mesh is None, use the one from mesh resources
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    kwargs["device"] = device_mesh.device_type

    # set default placements to replicated if not specified
    placements = placements or tuple(Replicate() for _ in range(device_mesh.ndim))

    # check device_mesh againts placements
    assert device_mesh.ndim == len(placements), (
        "mesh dimension does not match the length of placements"
    )

    assert kwargs["layout"] == torch.strided, "layout value not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(size)

    # get local tensor shape
    local_shape, _ = compute_local_shape_and_global_offset(
        size, device_mesh, placements
    )

    # initialize the local tensor
    if init_op == torch.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    elif init_op == torch.rand or init_op == torch.randn:
        # this tensor meta is not used except `shape`
        dtype = kwargs.get("dtype", torch.get_default_dtype())

        tensor_meta = TensorMeta(size, (0,), dtype)
        spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=tensor_meta)

        if random.is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker(device_mesh)

        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    spec = DTensorSpec(
        device_mesh,
        tuple(placements),
        tensor_meta=TensorMeta(
            size,
            torch_stride,
            local_tensor.dtype,
        ),
    )

    return DTensor(
        local_tensor,
        spec,
        requires_grad=kwargs["requires_grad"],
    )


def ones(  # type: ignore[no-untyped-def]
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.ones,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(  # type: ignore[no-untyped-def]
    *size,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.empty,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(  # type: ignore[no-untyped-def]
    size,
    fill_value,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))
        fill_value(Scalar): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.full,
        torch_size,
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def rand(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a uniform distribution
    on the interval ``[0, 1)``. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.rand,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def randn(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a normal distribution
    with mean 0 and variance 1. The shape of the tensor is defined by the variable
    argument ``size``.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.randn,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def zeros(  # type: ignore[no-untyped-def]
    *size,
    requires_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (int...): a sequence of integers defining the shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
    Keyword args:
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    torch_size = normalize_to_torch_size(size)

    return _dtensor_init_helper(
        torch.zeros,
        torch_size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )



