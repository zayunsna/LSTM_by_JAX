import pandas as pd
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random, grad, jit
from flax.training import train_state
import optax
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


df = pd.read_csv("IMDB Dataset.csv")
print(df.head())


class LSTMModel(nn.Module):
    def setup(self):
        self.embedding = nn.Embed(max_features, max_len)
        lstm_layer = nn.scan(nn.OptimizedLSTMCell,
                               variable_broadcast="params",
                               split_rngs={"params": False},
                               in_axes=1, 
                               out_axes=1,
                               length=max_len,
                               reverse=False)
        self.lstm1 = lstm_layer()
        self.dense1 = nn.Dense(256)
        self.lstm2 = lstm_layer()
        self.dense2 = nn.Dense(128)
        self.lstm3 = lstm_layer()
        self.dense3 = nn.Dense(64)
        self.dense4 = nn.Dense(2)
        
    @nn.remat    
    def __call__(self, x_batch):
        x = self.embedding(x_batch)
        
        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x_batch),), size=128)
        (carry, hidden), x = self.lstm1((carry, hidden), x)
        
        x = self.dense1(x)
        x = nn.relu(x)
        
        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x_batch),), size=64)
        (carry, hidden), x = self.lstm2((carry, hidden), x)
        
        x = self.dense2(x)
        x = nn.relu(x)
        
        carry, hidden = nn.OptimizedLSTMCell.initialize_carry(jax.random.PRNGKey(0), batch_dims=(len(x_batch),), size=32)
        (carry, hidden), x = self.lstm3((carry, hidden), x)
        
       
        x = self.dense3(x)
        x = nn.relu(x)
        x = self.dense4(x[:, -1])
        return nn.log_softmax(x)
    

max_features = 10000  # Maximum vocab size.
batch_size = 128
max_len = 50 # Sequence length to pad the outputs to.

df = pd.DataFrame()