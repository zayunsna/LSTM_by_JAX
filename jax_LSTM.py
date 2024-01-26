import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, jit
from flax.training import train_state
import optax
import sys
from gen_toyData import DataFrameGenerator_realistic
import matplotlib.pyplot as plt

generator = DataFrameGenerator_realistic(start_date='2023-01-01',
                                         end_date='2024-01-01',
                                         date_unit='D',
                                         correlation_strength=0.05,
                                         variation_scale=5,
                                         trend_direction=0.)
df = generator.generate_time_series_data()

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['main_feature'], label='Main Feature', color='blue')
plt.plot(df['datetime'], df['sub_feature_1'], label='Sub Feature', color='green')
plt.title('Time Series of Main Feature and Sub Feature')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
# plt.show()

class LSTMModel(nn.Module):
    def __init__(self, num_hidden, num_outputs):
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

    @nn.compact
    def __call__(self, x):
        lstm_cell = nn.LSTMCell(name='lstm_cell', features=self.num_hidden)

        batch_size = x.shape[0]
        hidden_state = lstm_cell.initialize_carry(random.PRNGKey(0), (batch_size,))

        out, hidden_state = lstm_cell(hidden_state, x)
        
        out = nn.Dense(self.num_outputs)(out)
        return out

batch_size = 32
sequence_length = 20
input_dim = 1
num_epochs = 100

input_shape = (batch_size, sequence_length, input_dim)  # Define your input shape
model = LSTMModel(num_hidden=128, num_outputs=10)  # Example values

params = model.init(random.PRNGKey(0), jnp.ones(input_shape))

def loss_fn(params, model, x, y):
    predictions = model.apply(params, x)
    loss = jnp.mean((predictions - y) ** 2)  # Mean squared error for example
    return loss

@jax.jit
def train_step(state, x, y):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params, model, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss

def create_sequences(data, sequence_length):
    sequences = []
    output = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        output.append(data[i + sequence_length])  # or data[i + sequence_length + future_step] for forecasting
    return jnp.array(sequences), jnp.array(output)

def data_loader(X, y, key, batch_size):
    num_samples = X.shape[0]
    indices = jnp.arange(num_samples)
    random.permutation(key, indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


sequence_length = 30  # Example sequence length
X, y = create_sequences(df['main_feature'], sequence_length)

optimizer = optax.adam(0.001)

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
key_id = 1234
key = random.PRNGKey(key_id)

for epoch in range(num_epochs):
    key, subkey = random.split(key)
    for x_batch, y_batch in data_loader(X, y, key, batch_size):
        state, loss = train_step(state, x_batch, y_batch)
    print(f'Epoch {epoch}, Loss: {loss}')
