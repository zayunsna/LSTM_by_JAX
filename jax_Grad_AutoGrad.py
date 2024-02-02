import numpy as np # linear algebra
import jax
import jax.numpy as jnp
from jax import grad
import time

def fcn(x):
    if x < 3:
        return 3. * x ** 2
    else:
        return -4 * x
    
start_time = time.time()
print(grad(fcn)(2.))
print(grad(fcn)(4.))
print(" Processing time : {:0.4f} sec".format(time.time() - start_time))


@jax.jit
def fcn2(x):
    for i in range(3):
        x = 2 * x
    return x

start_time = time.time()
print(fcn2(3))
print(" Processing time : {:0.4f} sec".format(time.time() - start_time))

@jax.jit
def fcn3(x):
    y = 0.
    for i in range(x.shape[0]):
        y = y + x[i]
    return y

start_time = time.time()
print(fcn3(jnp.array([1.,2.,3.])))
print(" Processing time : {:0.4f} sec".format(time.time() - start_time))
