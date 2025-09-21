import grain
import numpy as np



ds = (
    grain.MapDataset.source(np.arange(1000))
    .seed(seed=45)
    .shuffle()
    .repeat(10)
    .to_iter_dataset()
)

num_steps = 4
ds_iter = iter(ds)

# Read some elements.


for i, x in enumerate( ds ): 
    print(i, x)

# for i in range(num_steps):
#   x = next(ds_iter)
#   print(i, x["label"])
