import numpy as np # linear algebra
import jax
import jax.numpy as jnp
import time


def nansum(x):
    mask = ~jnp.isnan(x)
    x_without_nans = x[mask]
    return x_without_nans.sum()

@jax.jit
def nansum_2(x):
    mask = ~jnp.isnan(x)
    return jnp.where(mask, x, 0).sum()

start_time = time.time()
x = jnp.array([1, 2, jnp.nan, 3, 4])
print(nansum(x))
#print(jax.fit(nansum)(x)) # it will caused an error because the size of `x_without_nans` is dependent on the values within `x`, which is another way of saying its size is dynamic.
print(" Processing time : {:0.4f} sec".format(time.time() - start_time))

start_time = time.time()
print(nansum_2(x))
print(" Processing time : {:0.4f} sec".format(time.time() - start_time))
