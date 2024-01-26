import numpy as np # linear algebra
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import time

# x = jnp.arange(10)
# x_np = np.arange(10)
# print(x)
# print(x_np)
# print(type(x))
# print(type(x_np))
# print(x)
# print(x_np)


test_arr = random.uniform(random.PRNGKey(0), (1000, 1000)) # 1000 x 1000 의 random matric 생성

dot1 = jnp.dot(test_arr, test_arr)
dot1.block_until_ready()
# Before comparison between with & without 'block_until_ready()', we need to warm-up step of hardware accelerators, such as GPUs.
# This is a common phenomenon in many computational frameworks that utilize such accelerators.
# In summary, above additional production code is due to the initial overhead for warm-up of hardware accelerators.
# In my case, I write the additional part with 'block_until_ready()' to make a clear ready-state (without any queue job or working process)

start_time = time.time()
dot1 = jnp.dot(test_arr, test_arr)
print(dot1)
print("dot1 Execution time : {:0.5f}".format(time.time() - start_time))
dot1.block_until_ready()


start_time = time.time()
dot2 = jnp.dot(test_arr, test_arr).block_until_ready()
print(dot2)
print("dot2 Execution time : {:0.5f}".format(time.time() - start_time))
# with 'block_until_ready()' is slightly slow than without 'block_until_ready()'


def sum_of_squares(x):
    return jnp.sum(x**2)

sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))
print(sum_of_squares_dx(x))

def sum_squared_error(x, y):
  return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])
print(sum_squared_error_dx(x, y))


#------------------------------------------------

x_sample = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
fcn = x_sample*3 -1+noise

def model(theta, x):
   w, b = theta
   return w*x+b

def loss_fn(theta, x, y):
   pred = model(theta, x)
   return jnp.mean((pred-y)**2)

@jax.jit
def update(theta, x, y, lr=0.1):
   return theta - lr * jax.grad(loss_fn)(theta, x, y)

theta = jnp.array([1.,1.])
start_time = time.time()
for _ in range(1000):
   theta = update(theta, x_sample, fcn)

print("LR Execution time : {:0.5f}".format(time.time() - start_time))

plt.scatter(x_sample, fcn, color='Black', label='Data')
plt.plot(x_sample, model(theta, x_sample), color='r', label='Linear Regression')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

w, b = theta
print(f"w: {w:<.2f}, b: {b:<.2f}")