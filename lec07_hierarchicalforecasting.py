import pandas as pd

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MiddleOut, TopDown
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


# Read adat 
input = pd.read_csv('lec07_tourism.csv', parse_dates=['ds'])
input['Country'] = 'Australia'

# Specify simple hierarchy and aggregate
spec = [
    ['Country'],
    ['Country' ,'State'],
    ['Country', 'State', 'Region']
]
train_data, S_df, tags = aggregate(df=input, spec=spec)
print(tags)
print(train_data.head(5))

# Forecast base forecasts
sf = StatsForecast(models=[AutoARIMA(season_length=4)], freq='QS')
fc_df = sf.forecast(df=train_data, h=8)

# Reconcile
reconcilers = [
    BottomUp(),
    TopDown(method='proportion_averages'),
    MiddleOut(middle_level="Country/State", top_down_method="proportion_averages"),
]
hrec = HierarchicalReconciliation(reconcilers=reconcilers)
fc_df_reconciled = hrec.reconcile(Y_hat_df=fc_df, Y_df=train_data, S=S_df, tags=tags)

# This renaming is not necessary, but makes plotting etc. much cleaner
fc_df_reconciled = fc_df_reconciled.rename(columns=({
    'AutoARIMA/BottomUp': 'BottomUp',
    'AutoARIMA/TopDown_method-proportion_averages': 'TopDown',
    'AutoARIMA/MiddleOut_middle_level-Country/State_top_down_method-proportion_averages': 'MiddleOut'
}))
fc_df_reconciled





