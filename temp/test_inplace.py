import jax
import equinox  as eqx
from equinox import nn

class SuperLinear(eqx.Module):
    linear1: nn.Linear 
    linear2: nn.Linear


    def __call__(
        self,
        x
    ):
        return self.linear1(self.linear2(x))

key = jax.random.key(10)
linear1 = nn.Linear(1, 9, key = key)
linear2 = nn.Linear(9, 1, key = key)
superlinear = SuperLinear(linear1, linear2)
array = jax.random.normal(key, (8))

superlinear = eqx.tree_at(lambda m: m.linear2, superlinear, replace = nn.Linear(8,1, key = key))
output = superlinear(array)

print(f"DEBUGPRINT[177]: test_inplace.py:23: output={output}")

print(f"DEBUGPRINT[176]: test_inplace.py:22: superlinear={superlinear}")

