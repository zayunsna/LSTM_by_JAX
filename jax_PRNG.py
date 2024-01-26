import numpy as np
import jax.numpy as jnp
from jax import random
from datetime import datetime

key_id = 1234
key = random.PRNGKey(key_id)

def key_gen(key):
    print("old key {}".format(key))
    new_key, sub_key = random.split(key)
    normal_sample = random.normal(sub_key)
    print(r"    \---SPLIT --> new key   ", new_key)
    print(r"             \--> new subkey", sub_key, "--> normal", normal_sample)
    return new_key

for i in range(5):
    key = key_gen(key)

key = random.PRNGKey(1111)
print("Original key : {}".format(key))

subkeys = random.split(key, num=5)

for i, subkey in enumerate(subkeys):
    print("Subkey no: {} | subkey: {}".format(i+1, subkey))

np.random.seed(key)
comp_np = np.random.randint(0, 10, 5)
print(comp_np)
comp_jax = random.randint(key=key, minval=0, maxval=10, shape=[5])
print(comp_jax)