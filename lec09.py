import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlforecast import MLForecast
from mlforecast.lag_transforms import Combine, RollingMean
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor


# Read data
pedestrians = pd.read_csv('le09_pedestrians.csv', parse_dates=['ds']).rename(columns={'series_name': 'unique_id', 'pedestrians': 'y'})
pedestrians

# Speciy model
models = [RandomForestRegressor(n_estimators=500, max_features='sqrt', min_samples_leaf=5)]
fcst = MLForecast(
    models=models,
    freq='h',
    lags=[1, 2, 24, 168, 168*2, 168*3],
    date_features=['dayofweek', 'hour', 'month', ],
    lag_transforms={
        1: [RollingMean(window_size=168),
            Combine(
                RollingMean(window_size=168),
                RollingMean(window_size=168*4),
                operator.truediv)
            ],
        168: [RollingMean(window_size=168)]
    }
)

# Permutation Importance
features_data = fcst.preprocess(pedestrians)
eval_size = 500
train_data = pedestrians.iloc[:-eval_size].copy()
eval = features_data.iloc[-eval_size:].copy()
X = eval.drop(['y', 'ds', 'unique_id'], axis=1)
y = eval['y']

fcst.fit(train_data, fitted=True)
model_rf = fcst.models_['RandomForestRegressor']
perm_importance = permutation_importance(
    model_rf,
    X,
    y,
    scoring='neg_mean_squared_error',
    n_repeats=10,
    random_state=42
)
print(perm_importance)

# Create result and plot
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
features = np.array(X.columns)[sorted_idx]
importances = perm_importance.importances_mean[sorted_idx]
std = perm_importance.importances_std[sorted_idx]
plt.figure(figsize=(10, 6))
plt.barh(features, importances, xerr=std, align='center', color='steelblue')
plt.xlabel('Mean Importance')
plt.title('Permutation Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()