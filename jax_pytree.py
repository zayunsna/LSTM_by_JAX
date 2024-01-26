import time
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

# Let's see how many leaves they have:
for pytree in example_trees:
  leaves = jax.tree_util.tree_leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")
  
list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

print(jax.tree_map(lambda x: x*2, list_of_lists))

another_list_of_lists = list_of_lists
print(jax.tree_map(lambda x, y: x+y, list_of_lists, another_list_of_lists))

import numpy as np


def init_mlp_params(layer_widths, key):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        # weights are initialized using a normal distribution(gaussian distribution), scaled by 'np.sqrt(2/n_in)`, 
        # which is a common initialzation scheme for neural networks (He initialization).
        _weights = random.normal(key=key, shape=(n_in, n_out))* np.sqrt(2/n_in) 
        _baises = np.ones(shape=(n_out,))
        params.append( dict(weights=_weights,  biases=_baises))
    return params, key

def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights']+layer['biases']) ## a meaning of '@' is matrix multiy [matmul].
    return x @ last['weights']+last['biases']

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x)- y) ** 2) # computes the mean squared error loss.

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    return jax.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)

LEARNING_RATE = 0.0001
key = random.PRNGKey(1234)

params, _key = init_mlp_params([1, 128, 128, 1], key)

jax.tree_map(lambda x: x.shape, params)

params_save = params

key, subkey = random.split(_key)
xs = random.normal(subkey, shape=(128,1))
ys = xs ** 2
params_save = update(params_save, xs, ys)
# xs = np.random.normal(size=(128,1))
# ys = xs ** 2

epoch = 100000
checkpoints = [100, 1000, 10000]
label = ['Epoch 100', 'Epoch 1000', 'Epoch 10000']
params_checkpoints = {}
count = 0

start_time = time.time()
for i in trange(epoch):
    params = update(params, xs, ys)
    if i in checkpoints:
        params_checkpoints[count] = params
        count+=1

print("Total Process time : {:.5f} sec.".format(time.time() - start_time))

size=10
plt.scatter(xs, ys, label="Original data")
plt.scatter(xs, forward(params_save, xs), s=size, label= 'Epoch 1')
for i in range(len(checkpoints)):
    plt.scatter(xs, forward(params_checkpoints[i], xs), s=size, label=label[i])
plt.scatter(xs, forward(params, xs), s=size, label='Result Model Prediction')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
