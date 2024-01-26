#### This part is making the dataframe sample for the function test.
#### Author : Hyoungku Jeon

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs
import math


class DataFrameGenerator_periodical:
    def __init__(self, num_rows:int, frequency:float, frequency_std:float, ratio_accidental:float, n_features:int):
        self.num_rows = num_rows
        self.frequency = frequency
        self.frequency_std = frequency_std
        self.ratio_accidental = ratio_accidental
        self.n_features = n_features
        self.df = self._initialize_df()

    # @staticmethod
    # def elastic_naming(form:str, n:int):
    #     return [f"{form}_{i}" for i in range(n)]
    
    # def _initialize_df(self, start_date:str="2023-01-01"):
    #     date_range = pd.date_range(start=start_date, periods=self.num_rows, freq='S')
    #     columns = self.elastic_naming('Feature', self.n_features)
    #     df = pd.DataFrame(index=date_range, columns=columns)
    #     return df
    @staticmethod
    def elastic_naming(main_feature_name:str, sub_feature_form:str, n_features:int):
        return [main_feature_name] + [f"{sub_feature_form}_{i}" for i in range(1, n_features)]
    
    def _initialize_df(self, start_date:str="2023-01-01"):
        date_range = pd.date_range(start=start_date, periods=self.num_rows, freq='S')
        columns = self.elastic_naming('main_feature', 'sub_feature', self.n_features)
        df = pd.DataFrame(index=date_range, columns=columns)
        return df   


    def calRandomValues(self, seq:int):
        pedestal = 30 + 10 * math.sin(math.pi * self.frequency * seq)
        mean_val = pedestal + 10 * math.sin(2 * math.pi * self.frequency * seq)
        std_val = 2 + math.cos(2 * math.pi * self.frequency_std * seq)
        random_values = np.random.normal(loc=mean_val, scale=std_val, size=(1, self.n_features))
        return random_values


    def gen_df(self, start_date:str="2023-01-01"):
        for i in range(self.num_rows):
            random_values = self.calRandomValues(i)
            self.df.iloc[i] = random_values

        self.df = self.add_accidental_high_values(self.df)
        return self.df.clip(0, 100)

    def add_accidental_high_values(self, df:pd.DataFrame):
        num_high_values = int(self.num_rows * self.ratio_accidental)
        high_value_indices = np.random.choice(self.num_rows, num_high_values, replace=False)

        for col in df.columns:
            high_values = np.random.randint(65, 99, size=num_high_values)
            df.loc[df.index[high_value_indices], col] = high_values
        return df

    def gen_testblobs_df(self, n_samples:int, n_features:int):
        X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
        column_names = self.elastic_naming('Feature', n_features)
        df = pd.DataFrame(X, columns=column_names)
        scaler = StandardScaler()
        return scaler.fit_transform(df)
    
class DataFrameGenerator_realistic:
    def __init__(self, start_date:str, end_date:str, date_unit:str, correlation_strength:float, variation_scale:float=1.0, trend_direction:float=0):
        self.start_date = start_date
        self.end_date = end_date
        self.date_unit = date_unit
        self.correlation_strength = correlation_strength
        self.variation_scale = variation_scale
        self.trend_direction = trend_direction

    def generate_time_series_data(self):

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq=self.date_unit)

        main_feature = np.random.randn(len(date_range)) * self.variation_scale
        main_feature = main_feature.cumsum()

        trend = np.linspace(0, self.trend_direction * len(main_feature), len(main_feature))
        main_feature += trend

        noise_sub = np.random.randn(len(date_range)) * self.variation_scale
        sub_feature = main_feature * self.correlation_strength + (1 - self.correlation_strength) * noise_sub.cumsum()

        random_limit = np.random.randint(5,30, size=4)
        min_main, max_main = min(main_feature)-random_limit[0], max(main_feature)+random_limit[1]
        min_sub, max_sub = min(sub_feature)-random_limit[2], max(sub_feature)+random_limit[3]
        main_feature = 100 * (main_feature - min_main) / (max_main - min_main)
        sub_feature = 100 * (sub_feature - min_sub) / (max_sub - min_sub)

        data = pd.DataFrame({
            'datetime': date_range,
            'main_feature': main_feature,
            'sub_feature_1': sub_feature
        })

        return data

# generator = DataFrameGenerator_realistic(start_date='2023-01-01',
#                                          end_date='2024-01-01',
#                                          date_unit='H',
#                                          correlation_strength=0.05,
#                                          variation_scale=5,
#                                          trend_direction=0.)

# df = generator.generate_time_series_data()

# plt.figure(figsize=(12, 6))
# plt.plot(df['datetime'], df['main_feature'], label='Main Feature', color='blue')
# plt.plot(df['datetime'], df['sub_feature1'], label='Sub Feature', color='green')
# plt.title('Time Series of Main Feature and Sub Feature')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()
    
# num_rows = 600 #5000
# frequency = 1/1300
# frequency_std = 1/1000
# ratio_accidental = 0.001
# n_features = 4

# generator = DataFrameGenerator_periodical(num_rows = num_rows,
#                                           frequency = frequency,
#                                           frequency_std = frequency_std,
#                                           ratio_accidental = ratio_accidental,
#                                           n_features = n_features)
# df = generator.gen_df()

# print(df.head())
# plt.figure(figsize=(12,8))
# plt.plot(df[['main_feature']], label='main_feature')
# for i in range(1, n_features + 1): 
#     feature_name = f'sub_feature_{i}'
#     if feature_name in df.columns:
#         plt.plot(df[feature_name], label=feature_name)
# plt.grid()
# plt.legend()
# plt.show()