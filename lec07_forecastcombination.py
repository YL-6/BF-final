import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, RandomWalkWithDrift, SeasonalNaive, SeasonalWindowAverage, WindowAverage
import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate

# Load data and plot
seasonal_ts = pd.read_csv('lec05_seasonal_ts.csv', parse_dates=['ds'])
StatsForecast.plot(seasonal_ts)

# Create cross-validation and forecast into the future
models = [Naive(), WindowAverage(window_size=12), RandomWalkWithDrift(), AutoARIMA(season_length=12), AutoETS(season_length=12), SeasonalNaive(season_length=12), SeasonalWindowAverage(season_length=12, window_size=3)]
sf = StatsForecast(models=models, freq='MS')
cv_df = sf.cross_validation(df=seasonal_ts, step_size=6, n_windows=7, h=12)
fc_df = sf.forecast(df=seasonal_ts, h=12)

# Calculate error metrics for cross-validation
metrics = evaluate(cv_df.drop(columns={'cutoff'}), metrics=[ufl.mse, ufl.mape])
 
# Choose MSE for accuracy evaluation and prepare metrics data 
metric_for_selection = 'mse'
metrics = metrics[metrics.metric == metric_for_selection].drop('metric', axis=1)
metrics = metrics.melt(id_vars='unique_id', var_name='method', value_name='metric')
metrics = metrics.sort_values(['unique_id', 'metric'])

# Function for calculating different ensemble types
def compute_ensemble_weights(df):
    df = df.copy()
    n_models = len(df)

    df['Best-Fit'] = 0.0
    df.iloc[0, df.columns.get_loc('Best-Fit')] = 1.0

    df['Equal-All'] = 1.0 / n_models

    df['Equal-Best3'] = 0.0
    top_n = min(3, n_models)
    df.iloc[:top_n, df.columns.get_loc('Equal-Best3')] = 1.0 / top_n

    inverse = 1.0 / df['metric']
    df['InverseMSE'] = inverse / inverse.sum()

    df['InverseMSE-Best3'] = 0.0
    top_n = min(3, n_models)
    inverse = 1.0 / df['metric'][:top_n]
    df.iloc[:top_n, df.columns.get_loc('InverseMSE-Best3')] = inverse / inverse.sum()

    long_weights = df.melt(
        id_vars=['method', 'unique_id'],
        value_vars=['Best-Fit', 'Equal-All', 'Equal-Best3', 'InverseMSE', 'InverseMSE-Best3'],
        var_name='ensemble_type',
        value_name='weight'
    )
    long_weights = long_weights[long_weights['weight'] > 0].reset_index(drop=True)

    return long_weights

# Create weights and multiply them with forecasts
weights_df = metrics.groupby('unique_id', group_keys=False).apply(compute_ensemble_weights)
forecasts_long = cv_df.melt(
    id_vars=['unique_id', 'ds', 'cutoff'],
    var_name='method',
    value_name='fc'
)
ensemble_fc = weights_df.merge(forecasts_long, how='left')
ensemble_fc['weighted_fc'] = ensemble_fc['fc'] * ensemble_fc['weight'] 
ensemble_fc = ensemble_fc.groupby(['unique_id', 'ensemble_type', 'ds', 'cutoff']).agg({'weighted_fc': 'sum'}).reset_index()
ensemble_fc = ensemble_fc.pivot(index=['unique_id', 'ds', 'cutoff'], values='weighted_fc', columns='ensemble_type').reset_index()

# Evaluate ensmeble and old models
full_fc = cv_df.merge(ensemble_fc)
metrics = evaluate(full_fc.drop(columns={'cutoff'}), metrics=[ufl.mse, ufl.mape])


# Create ensmeble forecasts into the future using established weights, finally plot it
forecasts_long = fc_df.melt(
    id_vars=['unique_id', 'ds'],
    var_name='method',
    value_name='fc'
)
ensemble_fc = weights_df.merge(forecasts_long, how='left')
ensemble_fc['weighted_fc'] = ensemble_fc['fc'] * ensemble_fc['weight'] 
ensemble_fc = ensemble_fc.groupby(['unique_id', 'ensemble_type', 'ds']).agg({'weighted_fc': 'sum'}).reset_index()
ensemble_fc = ensemble_fc.pivot(index=['unique_id', 'ds'], values='weighted_fc', columns='ensemble_type').reset_index()

# Example plot
StatsForecast.plot(df=seasonal_ts, forecasts_df=ensemble_fc.merge(ensemble_fc.merge(fc_df)), unique_ids=['Newspaper and book retailing'],
                   models=['AutoARIMA', 'AutoETS', 'SeasonalNaive', 'InverseMSE-Best3'], max_insample_length=72)



