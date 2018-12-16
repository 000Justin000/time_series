import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

weighted = True

rcParams["figure.figsize"] = (15,6)
data = pd.read_csv("AirPassengers.csv", index_col="Month", parse_dates=["Month"], date_parser=lambda date : pd.datetime.strptime(date, "%Y-%m"))
ts = np.log(data["#Passengers"])

if (not weighted):
    rolling_mean = ts.rolling(12).mean()
else:
    rolling_mean = ts.ewm(halflife=12).mean()

oscillation = (ts-rolling_mean).dropna()

decomposition = seasonal_decompose(ts)
