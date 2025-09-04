from src.distributed import get_dp_partition_spec
from src import nn
import jax
import equinox as eqx


key = jax.random.key(0)
linear = nn.Linear(10, 10, key = key)
gg = jax.tree_util.tree_leaves(linear)
print(f"DEBUGPRINT[233]: test_dp.py:9: gg={gg}")
params, static = eqx.partition(linear, eqx.is_array)
gg = jax.tree_util.tree_leaves(params)
print(f"DEBUGPRINT[232]: test_dp.py:10: gg={gg}")

print(f"DEBUGPRINT[231]: test_dp.py:8: static={static}")
print(f"DEBUGPRINT[230]: test_dp.py:8: params={params}")

a = get_dp_partition_spec(linear)
print(f"DEBUGPRINT[229]: test_dp.py:6: a={a}")
