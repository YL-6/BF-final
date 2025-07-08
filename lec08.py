
import operator

import optuna
import pandas as pd

from mlforecast import MLForecast
from mlforecast.auto import AutoModel, AutoMLForecast
from mlforecast.lag_transforms import Combine, ExpandingMean, RollingMean
from mlforecast.target_transforms import Differences, LocalBoxCox
from statsforecast import StatsForecast
from sklearn.ensemble import RandomForestRegressor
from utilsforecast.evaluation import evaluate
import utilsforecast.losses as ufl


# Read data
data = pd.read_csv('lec08_pedestrians.csv', parse_dates=['ds']).rename(columns={'series_name': 'unique_id', 'pedestrians': 'y'})

# Specify random forest
# These specifications are too low / supoptimal, just an example and meant to imporve speed of this script
models = [RandomForestRegressor(n_estimators=300, max_features='sqrt', min_samples_leaf=10)]
fcst = MLForecast(
    models=models,
    freq='h',
    lags=[1, 2, 24, 168],
    date_features=['dayofweek', 'hour'],
    target_transforms=[Differences([1])],
    lag_transforms={
        1: [RollingMean(window_size=168)]
    }
)

# Run the forecast
print(fcst.preprocess(data))  # Just to take a look at the data, not necessary
fcst.fit(data)
fc_df = fcst.predict(h=168)
StatsForecast.plot(df=data, forecasts_df=fc_df, max_insample_length=24*7*10)

# Run a cross validation
# These specifications are too low / supoptimal, just an example and meant to imporve speed of this script
cv_df = fcst.cross_validation(data, n_windows=2, h=168)
metrics = [ufl.mse, ufl.mape]
metrics_df = evaluate(df=cv_df.drop(columns={'cutoff'}), metrics=metrics)
metrics_df
print('Cross-Validation finished. Running second half.')

# Specifying hyperparameter tuning
# These specifications are too low / supoptimal, just an example and meant to imporve speed of this script
def rf_config(trial: optuna.Trial):
    return {
        'n_estimators':  trial.suggest_int('n_estimators', 500, 800, step=100),
        'max_features': trial.suggest_int('max_features', 3, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 8, 12)
    }
def rf_init_config(trial: optuna.Trial):
    return {
        'lags': [1, 2, 24, 168, 168*2, 168*3],
        'date_features': ['dayofweek', 'hour', 'month'],
        'target_transforms': [],  
        'lag_transforms': {
            1: [RollingMean(window_size=168),
                Combine(
                    RollingMean(window_size=168),
                    RollingMean(window_size=168*4),
                    operator.truediv)
                ],
            168: [RollingMean(window_size=168)]
        }
    }
def mse_trial(df, train_df):
   return ufl.mse(df, models=["model"])["model"].mean()
# Performing hyparameter tuning
# These specifications are too low / supoptimal, just an example and meant to imporve speed of this script
rf_tuned = AutoModel(
    model=RandomForestRegressor(),
    config=rf_config,
)
tuned_random_forest = AutoMLForecast(
    models={'rf_tuned': rf_tuned},
    freq='h',
    init_config=rf_init_config
).fit(
    data,
    n_windows=5,
    h=168,
    num_samples=2,  # number of combinations to try, should be considerably higher
    loss=mse_trial
)

# Investigating hyperparameter tuning
study = tuned_random_forest.results_['rf_tuned']
print(study.trials_dataframe().sort_values('value'))
print(study.best_params)