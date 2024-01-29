import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, jit
from flax.training import train_state
import optax
import sys
from gen_toyData import DataFrameGenerator_realistic
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm, trange

generator = DataFrameGenerator_realistic(start_date='2023-01-01',
                                         end_date='2024-01-01',
                                         date_unit='D',
                                         correlation_strength=0.05,
                                         variation_scale=5,
                                         trend_direction=0.)
df = generator.generate_time_series_data()

# plt.figure(figsize=(12, 6))
# plt.plot(df['datetime'], df['main_feature'], label='Main Feature', color='blue')
# plt.plot(df['datetime'], df['sub_feature_1'], label='Sub Feature', color='green')
# plt.title('Time Series of Main Feature and Sub Feature')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()

class LSTMModel(nn.Module):
    # def __init__(self, num_hidden, num_outputs):
    #     self.num_hidden = num_hidden
    #     self.num_outputs = num_outputs
    num_hidden: int 
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        lstm_cell = nn.LSTMCell(name='lstm_cell', features=self.num_hidden)
        batch_size = x.shape[0]
        # hidden_state = lstm_cell.initialize_carry(random.PRNGKey(0), (batch_size,))
        state = lstm_cell.initialize_carry(random.PRNGKey(0), (batch_size, self.num_hidden))

        outputs = []
        for t in range(x.shape[1]):  # Loop over time steps
            # (hidden_state, cell_state), out = lstm_cell((hidden_state, cell_state), x[:, t, :])
            state, out = lstm_cell(state, x[:, t, :])
            outputs.append(out)
        all_outputs = jnp.stack(outputs, axis=1)
        last_output = all_outputs[:, -1, :]

        # Output layer
        decoded = nn.Dense(self.num_outputs)(last_output)
        return decoded
    

def loss_fn(params, model, x, y):
    predictions = model.apply(params, x)
    loss = jnp.mean((predictions - y) ** 2) 
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
        # Create a sequence and reshape it to have a shape of [sequence_length, 1]
        seq = data[i:i + sequence_length]
        sequences.append(seq.values.reshape((sequence_length, 1)))
        output.append([data[i + sequence_length]])
    
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

# print(" X is ")
# print(X)
# print("#"*50)
# print(" Y is")
# print(y)

# sys.exit()
batch_size = 32
input_dim = 1
num_epochs = 10000
lr= 0.001

input_shape = (batch_size, sequence_length, input_dim)  # Define your input shape
model = LSTMModel(num_hidden=128, num_outputs=10)  # Example values

params = model.init(random.PRNGKey(0), jnp.ones(input_shape))

optimizer = optax.adam(lr)

state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
key_id = 1234
key = random.PRNGKey(key_id)

loss_values = []

start_time = time.time()
for epoch in trange(num_epochs):
    key, subkey = random.split(key)
    for x_batch, y_batch in data_loader(X, y, subkey, batch_size):
        state, loss = train_step(state, x_batch, y_batch)
    loss_values.append(loss)
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
        ##TODO : need to develop the function for Best fit params auto-saving

print(" Training is done. ")
print(" Total execution time is {:0.4f} sec.".format(time.time()-start_time))
print(" Minimum Loss value : {:0.5f}".format(min(loss_values)))

with open('lstm_model_params.pkl', 'wb') as model:
    pickle.dump(state.params, model)

plt.plot(loss_values,color='0.5', label='training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,10)
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

with open('lstm_model_params.pkl', 'rb') as model:
    loaded_params = pickle.load(model)

def predict(model, params, input_sequence):
    predictions = model.apply(params, input_sequence)
    return predictions

test_data = df['main_feature']
test_X, test_Y = create_sequences(test_data, sequence_length)
model = LSTMModel(num_hidden=128, num_outputs=10)  # Example values
forecasts = predict(model, loaded_params, test_X)

prediction_times = df['datetime'].iloc[sequence_length:]

flattened_predictions = forecasts.flatten()[:len(prediction_times)]

plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['main_feature'], label='Original Data')
plt.plot(prediction_times, flattened_predictions, label='Predictions', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Forecasting')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()