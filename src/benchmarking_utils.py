import logging
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs
from tqdm.auto import tqdm

from .logger import Logger


LOGGER = logging.getLogger("distributed_logger")


M = tp.TypeVar("_M", bound=eqx.Module)
O = tp.TypeVar("_O", bound="Optimizer")

GradTx = GradientTransformation | GradientTransformationExtraArgs
AxisSpec = bool | tp.Callable[[tp.Any], bool]
Wrt = PyTree[AxisSpec]
Aux = dict[str, tp.Any]
Loss = float
Batch = tp.Any

LossFn = tp.Callable[[M, O, Batch, PRNGKeyArray], tuple[Loss, Aux]]
ParallelismPlans = (
    dict[str, tp.Callable[[M], M]] | tp.Sequence[dict[str, tp.Callable[[M], M]]]
)


ModuleInput = tp.TypeVar("_ModuleInput", M, tp.Sequence[M])
OptimizerInput = tp.TypeVar("_OptimizerInput", O, tp.Sequence[O])


def benchmark_loop(
    module: ModuleInput,
    optimizer: OptimizerInput,
    train_step_fn: TrainStepCallable[ModuleInput, OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    logger: Logger | None = None,
    num_steps: int = 100,
    theoretical_flops_per_step: float | None = None,
    trace_steps: tuple[int, int] | None = None,
    trace_dir: str = "./benchmark_traces",
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[ModuleInput, OptimizerInput, dict[str, tp.Any]]:
    import time

    # if logger is None:
    #     raise ValueError("logger is required")
    step_idx = -1
    train_step_times: list[float] = []
    next_batch_times: list[float] = []
    compile_step_time: float | None = None
    try:
        train_iterator = iter(train_loader)
    except TypeError as e:
        raise RuntimeError("train_loader is not iterable") from e
    progress_bar = tqdm(
        total=num_steps + 1,
        desc="Benchmarking",
        disable=jax.process_index() != 0,
        leave=True,
    )
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(train_iterator)
            except StopIteration:
                LOGGER.info("Train data loader exhausted, ending benchmark loop.")
                break
            batch_end = time.perf_counter()
            step_idx += 1
            if (
                trace_steps is not None
                and step_idx == trace_steps[0]
                and jax.process_index() == 0
            ):
                from pathlib import Path

                trace_path = Path(trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Starting JAX profiler trace at step {step_idx}")
                jax.profiler.start_trace(str(trace_path))
            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)

            if step_idx == 0 and jax.process_index() == 0:
                LOGGER.info("Running step 0 (compilation step)...")

            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
                module, optimizer, aux = train_step_fn(
                    module, optimizer, batch, key=step_key
                )
            jtu.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, "block_until_ready")
                else x,
                module,
            )
            jtu.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, "block_until_ready")
                else x,
                optimizer,
            )
            step_end = time.monotonic()

            if (
                trace_steps is not None
                and step_idx == trace_steps[1]
                and jax.process_index() == 0
            ):
                LOGGER.info(f"Stopping JAX profiler trace at step {step_idx}")
                jax.profiler.stop_trace()

            if step_idx == 0:
                compile_step_time = step_end - step_start
                if jax.process_index() == 0:
                    LOGGER.info(f"Step 0 (compilation) took {compile_step_time:.4f}s")
            else:
                train_step_times.append(step_end - step_start)
                next_batch_times.append(batch_end - batch_start)

            progress_bar.update()
        progress_bar.close()
        train_step_times = np.array(train_step_times)
        next_batch_times = np.array(next_batch_times)
        batches_per_sec = 1.0 / train_step_times
        if jax.process_index() == 0:
            LOGGER.info("=" * 30)
            LOGGER.info(" Benchmark Results ".center(30, "="))
            LOGGER.info("=" * 30)
            LOGGER.info(f"Train Step Time (avg): {train_step_times.mean():.4f}s")
            LOGGER.info(f"Train Step Time (median): {np.median(train_step_times):.4f}s")
            LOGGER.info(f"Train Step Time (std): {train_step_times.std():.4f}s")
            LOGGER.info(f"Next Batch Time (avg): {next_batch_times.mean():.4f}s")
            LOGGER.info(f"Batches/sec (avg): {1.0 / train_step_times.mean():.2f}")
            if compile_step_time is not None:
                LOGGER.info(f"Compile Time: {compile_step_time:.4f}s")
            if theoretical_flops_per_step is not None:
                avg_flops = theoretical_flops_per_step / train_step_times.mean()
                LOGGER.info(f"FLOPs/sec (avg): {avg_flops:.2e}")
                LOGGER.info(f"MFU (avg): {avg_flops / theoretical_flops_per_step:.4f}")
            LOGGER.info("=" * 30)
        stats = {
            "train_step_time_mean": float(train_step_times.mean()),
            "train_step_time_median": float(np.median(train_step_times)),
            "train_step_time_std": float(train_step_times.std()),
            "next_batch_time_mean": float(next_batch_times.mean()),
            "batches_per_sec": float(1.0 / train_step_times.mean()),
            "train_step_times": train_step_times.tolist(),
            "next_batch_times": next_batch_times.tolist(),
        }
        if compile_step_time is not None:
            stats["compile_time"] = float(compile_step_time)
        if theoretical_flops_per_step is not None:
            avg_flops = theoretical_flops_per_step / train_step_times.mean()
            stats["flops_per_sec"] = float(avg_flops)
            stats["mfu"] = float(avg_flops / theoretical_flops_per_step)
    except Exception:
        LOGGER.error("Exception during benchmark loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Benchmark loop ended")
    return module, optimizer, stats
