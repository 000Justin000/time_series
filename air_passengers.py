import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

def plot(series):
    for s in series:
        h = plt.plot(s)
    plt.show(h)

weighted = False

rcParams["figure.figsize"] = (15,6)
data = pd.read_csv("AirPassengers.csv", index_col="Month", parse_dates=["Month"], date_parser=lambda date : pd.datetime.strptime(date, "%Y-%m"))
ts = data["#Passengers"]

if (not weighted):
    rolling_mean = ts.rolling(12).mean()
else:
    rolling_mean = ts.ewm(halflife=12).mean()

oscillation = (ts-rolling_mean).dropna()
decomposition = seasonal_decompose(ts,model="multiplicative")
decomposition.plot(); plt.show()

model = ARIMA(ts,order=(2,1,2),dates=ts.index).fit(disp=-1)
ts_predict = pd.Series(ts.iloc[0], index=ts.index).add(model.fittedvalues.cumsum(),fill_value=0)
plot([ts,ts_predict])

stepwise_model = auto_arima(ts, trace=True, error_action="ignore", suppress_warnings=True, m=12, start_P=2, start_Q=2, max_order=20)
tr = ts["1949-01-01":"1959-12-01"]
te = ts["1960-01-01":]
stepwise_model.fit(tr)
future_forecast = pd.Series(stepwise_model.predict(n_periods=12), index=te.index)
plot([te,future_forecast])
