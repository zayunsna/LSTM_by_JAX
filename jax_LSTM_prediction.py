import jax.numpy as jnp
from flax import linen as nn
from jax import random
from gen_toyData import DataFrameGenerator_realistic
import matplotlib.pyplot as plt
import pickle

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

def create_sequences(data, sequence_length):
    sequences = []
    output = []
    for i in range(len(data) - sequence_length):
        # Create a sequence and reshape it to have a shape of [sequence_length, 1]
        seq = data[i:i + sequence_length]
        sequences.append(seq.values.reshape((sequence_length, 1)))
        output.append([data[i + sequence_length]])
    
    return jnp.array(sequences), jnp.array(output)

with open('lstm_model_params.pkl', 'rb') as model:
    loaded_params = pickle.load(model)

def predict(model, params, input_sequence):
    predictions = model.apply(params, input_sequence)
    return predictions

sequence_length = 30  # Example sequence length

generator = DataFrameGenerator_realistic(start_date='2023-01-01',
                                         end_date='2024-01-01',
                                         date_unit='D',
                                         correlation_strength=0.05,
                                         variation_scale=5,
                                         trend_direction=0.1)
df = generator.generate_time_series_data()

test_data = df['sub_feature_1']
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
plt.show()