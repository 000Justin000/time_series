import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams["figure.figsize"] = (15,6)
data = pd.read_csv("AirPassengers.csv", index_col="Month", parse_dates=["Month"], date_parser=lambda date : pd.datetime.strptime(date, "%Y-%m"))
ts = data["#Passengers"]
rolling_mean = ts.rolling(3).mean()
rolling_std = ts.rolling(3).std()
