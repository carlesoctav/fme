from __future__ import annotations

import logging
import typing as tp
from contextlib import nullcontext

import equinox as eqx
import jax
import jax.numpy as jnp
import transformers
from jax.sharding import Mesh
from jaxtyping import PRNGKeyArray

from ._darray import DArray


LOGGER = logging.getLogger(__name__)

_AbstractModule = tp.Any
_M = tp.TypeVar("_M", bound = eqx.Module | _AbstractModule)
_KeyResolverFn = tp.Callable[[str], str | None]


HuggingFaceModelType = tp.TypeVar("HuggingFaceModelType", bound = transformers.PreTrainedModel)


def _to_array(x, dtype: jnp.dtype) -> jax.Array:
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = jnp.asarray(x.numpy(), dtype=dtype)
        return DArray(value = x, pspec = None)
    except Exception:
        return DArray(value = jnp.asarray(x, dtype=dtype))


def _parse_path_tokens(key: str) -> list[str | int]:
    tokens: list[str | int] = []
    for tok in key.split("."):
        if tok.isdigit():
            tokens.append(int(tok))
        else:
            tokens.append(tok)
    return tokens




def _get_by_tokens(obj, tokens: list[str | int]):
    cur = obj
    for t in tokens:
        if isinstance(t, int):
            cur = cur[t]
        else:
            cur = getattr(cur, t)
    return cur


def _make_getter(tokens: list[str | int]):
    def getter(tree):
        return _get_by_tokens(tree, tokens)

    return getter


def _try_set(tree, tokens: list[str | int], value: jax.Array):
    try:
        _get_by_tokens(tree, tokens)
    except Exception:
        return tree, False

    getter = _make_getter(tokens)
    try:
        new_tree = eqx.tree_at(getter, tree, value)
        return new_tree, True
    except Exception:
        return tree, False

class HuggingFaceCompatibleModule(tp.Generic[HuggingFaceModelType]):
    hf_model_class: tp.ClassVar[HuggingFaceModelType] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__args__"):
                (cls.hf_model_class,) = base.__args__

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        from_pretrained_kwargs: dict | None = None,
        mesh: Mesh | None = None,
        key: PRNGKeyArray | None = None,
    ):
        """Load HF weights into an eqx module instance.

        - Guesses HF loader if `hf_model_class` is not set: uses AutoModel or AutoModelForMaskedLM
          based on class name suffix "ForMaskedLM".
        - Initializes an eqx module via `init_from_config`.
        - Applies the HF state dict with light key normalization.
        """
        if from_pretrained_kwargs is None:
            from_pretrained_kwargs = {}


        hf_cls = cls.hf_model_class
        if hf_cls is None:
            raise TypeError("cls.hf_model_class is None. Are you sure this model is Hugging Face compatible?")


        th_model = hf_cls.from_pretrained(pretrained_model_name_or_path, **from_pretrained_kwargs)
        th_model.eval()
        hf_config = th_model.config
        state_dict = th_model.state_dict()

        abstract_model = eqx.filter_eval_shape(cls, hf_config, dtype=dtype, params_dtype=params_dtype, key=key)

        key_resolver = getattr(cls, "normalize_hf_key_for_eqx", None)
        model, report = from_state_dict_to_pytree(
            abstract_model,
            state_dict,
            key_resolver,
            params_dtype=params_dtype,
            mesh=mesh
        )
        LOGGER.info(f"State dict application report: {report}")
        return model




def from_state_dict_to_pytree(
    model: _M, 
    state_dict: dict[str, tp.Any], 
    key_resolver: _KeyResolverFn | None = None,
    *, 
    mesh: Mesh| None = None, 
    params_dtype,
):
    if key_resolver is None:
        key_resolver = lambda x: x

    @eqx.filter_jit
    def _apply_state_dict(
        model, state_dict: dict[str, tp.Any], *, params_dtype: jnp.dtype
    ) -> tuple[tp.Any, dict[str, int]]:
        report= {"skipped": 0, "set": 0, "failed": 0}

        updated = model
        for k, v in state_dict.items():
            mapped = key_resolver(k)
            if mapped is None:
                report["skipped"] += 1
                continue
            tokens = _parse_path_tokens(mapped)
            arr = _to_array(v, params_dtype)
            updated, ok = _try_set(updated, tokens, arr)
            if ok:
                report["set"] += 1
            else:
                report["failed"] += 1

        return updated, report

    with mesh if mesh else nullcontext():
        model, report =  _apply_state_dict(model, state_dict, params_dtype=params_dtype)

    return model, report
