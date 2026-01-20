import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

def zscore_anomalies(series: pd.Series, threshold: float = 2.5):
    vals = series.fillna(0).values.reshape(-1, 1)
    scaler = StandardScaler()
    z = scaler.fit_transform(vals).flatten()
    anomalies_idx = np.where(np.abs(z) >= threshold)[0]
    return anomalies_idx, z

def moving_average(series: pd.Series, window: int = 7):
    return series.rolling(window=window, min_periods=1).mean()

def arima_forecast(ts_df: pd.DataFrame, value_col: str, steps: int = 7):
    ts_df = ts_df.dropna(subset=[value_col]).copy().sort_values('date')
    series = ts_df[value_col].astype(float)
    try:
        model = ARIMA(series, order=(1,1,1))
        fit = model.fit()
        forecast = fit.forecast(steps=steps)
        future_dates = pd.date_range(ts_df['date'].max(), periods=steps+1, freq='D')[1:]
        return pd.DataFrame({'date': future_dates, 'forecast': forecast})
    except Exception:
        last_val = series.iloc[-1] if len(series) else 0
        future_dates = pd.date_range(ts_df['date'].max(), periods=steps+1, freq='D')[1:]
        return pd.DataFrame({'date': future_dates, 'forecast': [last_val]*steps})
