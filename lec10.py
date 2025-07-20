import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM


# Read data
pedestrians = pd.read_csv('lec10_pedestrians.csv')

# Define model, small, not necessary reasonable model for quick run
models = [LSTM(input_size=168*3,
               h=168,                    
               max_steps=300,               
               scaler_type='standard',       
               encoder_hidden_size=32,       
               decoder_hidden_size=32)]
# Run fit and forecast
nf = NeuralForecast(models=models, freq='h')
nf.fit(df=pedestrians)
fcst_df = nf.predict(h=168)