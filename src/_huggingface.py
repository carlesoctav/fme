from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
import transformers


TrySetReport = dict[str, int]
HuggingFaceModelType = tp.TypeVar("HuggingFaceModelType", bound = transformers.PreTrainedModel)


def _to_array(x, dtype: jnp.dtype) -> jax.Array:
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        x = jnp.asarray(x.numpy(), dtype=dtype)
        return x
    except Exception:
        # Last-resort conversion
        return jnp.asarray(x, dtype=dtype)


def _parse_path_tokens(key: str) -> list[str | int]:
    """Converts a dot-separated key into tokens. Numeric tokens become indices.

    Example: "encoder.layer.0.attention.self.query.weight" ->
             ["encoder", "layer", 0, "attention", "self", "query", "weight"]
    """
    tokens: list[str | int] = []
    for tok in key.split("."):
        if tok.isdigit():
            tokens.append(int(tok))
        else:
            tokens.append(tok)
    return tokens


def _normalize_hf_key_for_eqx(key: str) -> str | None:
    """Maps common HF keys to eqx attribute paths.

    - Keeps BERT submodule structure mostly intact.
    - Flattens MLM head: "cls.predictions.transform.*" -> "transform.*"
      and "cls.predictions.bias" -> "bias".
    - Skips decoder tied weight ("cls.predictions.decoder.weight").
    """
    # Skip tied decoder weight; our implementation ties to embedding weight
    if key.startswith("cls.predictions.decoder.weight"):
        return None

    if key.startswith("cls.predictions.transform."):
        return key.replace("cls.predictions.transform.", "transform.", 1)
    if key == "cls.predictions.bias":
        return "bias"

    # Default: leave as-is
    return key


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
        current = _get_by_tokens(tree, tokens)
    except Exception:
        return tree, False

    # Attempt simple shape fix for 2D Linear kernels if needed
    if hasattr(current, "shape") and hasattr(value, "shape"):
        if current.shape != value.shape:
            if value.ndim == 2 and (value.T.shape == current.shape):
                value = value.T
            # else: leave as-is; let eqx/tree_at error if incompatible

    getter = _make_getter(tokens)
    try:
        new_tree = eqx.tree_at(getter, tree, value)
        return new_tree, True
    except Exception:
        return tree, False


def _apply_state_dict(
    model, state_dict: dict[str, tp.Any], *, params_dtype: jnp.dtype
) -> tuple[tp.Any, TrySetReport]:
    report: TrySetReport = {"set": 0, "skipped": 0, "failed": 0}
    updated = model
    for k, v in state_dict.items():
        mapped = _normalize_hf_key_for_eqx(k)
        if mapped is None:
            report["skipped"] += 1
            continue
        tokens = _parse_path_tokens(mapped)
        arr = _to_array(v, params_dtype)
        updated, ok = _try_set(updated, tokens, arr)
        if ok:
            report["set"] += 1
        else:
            # Try without leading "bert." if present (useful for bare encoders)
            if mapped.startswith("bert."):
                alt = mapped[len("bert.") :]
                tokens = _parse_path_tokens(alt)
                updated, ok = _try_set(updated, tokens, arr)
            if ok:
                report["set"] += 1
            else:
                report["failed"] += 1
    return updated, report


class HuggingFaceCompatibleModule(tp.Generic[HuggingFaceModelType]):
    hf_model_class: tp.ClassVar[HuggingFaceModelType] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in getattr(cls, "__orig_bases__", []):
            if hasattr(base, "__args__"):
                (cls.hf_class_type,) = base.__args__

    @classmethod
    def init_from_config(
        cls,
        config,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: jax.Array,
        **kwargs,
    ):
        return cls(config, dtype=dtype, params_dtype=params_dtype, key=key, **kwargs)

    @classmethod
    def normalize_hf_key_for_eqx(cls, key: str) -> str | None:
        # Allow subclasses to override mapping rules
        return _normalize_hf_key_for_eqx(key)

    @classmethod
    def apply_state_dict(
        cls,
        model,
        state_dict: dict[str, tp.Any],
        *,
        params_dtype: jnp.dtype = jnp.float32,
    ) -> tuple[tp.Any, TrySetReport]:
        # Allow subclasses to override. By default use generic mapping.
        # Swap in class-specific normalization if provided.
        # We rebind the module-level function to use class hook.
        def class_normalize(key: str):
            return cls.normalize_hf_key_for_eqx(key)

        nonlocal_normalize = globals()["_normalize_hf_key_for_eqx"]
        globals()["_normalize_hf_key_for_eqx"] = class_normalize
        try:
            return _apply_state_dict(model, state_dict, params_dtype=params_dtype)
        finally:
            globals()["_normalize_hf_key_for_eqx"] = nonlocal_normalize

    @classmethod
    def from_huggingface(
        cls,
        pretrained_model_name_or_path: str,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: jax.Array,
        hf_model_kwargs: dict | None = None,
        init_kwargs: dict | None = None,
    ):
        """Load HF weights into an eqx module instance.

        - Guesses HF loader if `hf_model_class` is not set: uses AutoModel or AutoModelForMaskedLM
          based on class name suffix "ForMaskedLM".
        - Initializes an eqx module via `init_from_config`.
        - Applies the HF state dict with light key normalization.
        """
        if hf_model_kwargs is None:
            hf_model_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        try:
            from transformers import (
                AutoModel,
                AutoModelForMaskedLM,
            )
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError("Hugging Face transformers is required.") from e

        # Resolve HF model loader
        hf_cls = cls.hf_model_class
        if hf_cls is None:
            # Heuristic: if this eqx class name ends with ForMaskedLM, use MLM
            if cls.__name__.endswith("ForMaskedLM"):
                loader = AutoModelForMaskedLM
            else:
                loader = AutoModel
        else:
            loader = hf_cls

        th_model = loader.from_pretrained(pretrained_model_name_or_path, **hf_model_kwargs)
        th_model.eval()
        hf_config = th_model.config
        state_dict = th_model.state_dict()

        model = cls.init_from_config(hf_config, dtype=dtype, params_dtype=params_dtype, key=key, **init_kwargs)

        model, report = cls.apply_state_dict(model, state_dict, params_dtype=params_dtype)
        return model, report

