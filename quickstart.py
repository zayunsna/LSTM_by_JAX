import timeit
import jax.numpy as jnp
from jax import (
    grad,
    jit,
    vmap,
    random
)

## Multiplying Matrices
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

### multiply two big matrices
size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
dot_x = jnp.dot(x, x.T).block_until_ready()
print(dot_x)
