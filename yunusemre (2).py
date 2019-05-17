import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array, copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def select_date(series, date):
    ser = []
    dummy = series["Day"]
    k = -1
    for i in range(len(series)):
        if dummy[i] == date:
            k = k+1
            ser.append([])
            for j in series.columns:
                ser[k].append(series[j][i])
        else:
            continue
    del dummy, k
    return ser


def select_time(series, time1, time2):
    ser = []
    dummy = series["Time"]
    k = -1
    for i in range(len(series)):
        if (dummy[i][3] == time1[1] and dummy[i][4] == time1[2]) or (
                dummy[i][3] == time2[1] and dummy[i][4] == time2[2]):
            k = k+1
            ser.append([])
            for j in series.columns:
                ser[k].append(series[j][i])
        else:
            continue
    del dummy, k
    return ser


def estimate_holt(df, seriesname, alpha=0.2, slope=0.1, trend="add", estimationlength=2):
    numbers = np.asarray(df[seriesname], dtype='float')
    model = Holt(numbers)
    fit = model.fit(alpha, slope, trend)
    estimate = fit.forecast(estimationlength)
    return estimate


def discard(df, seriesname="Volume"):
    ser = []
    dummy = df[seriesname]
    k = -1
    for i in range(len(df)):
        if dummy[i] != 0 and dummy[i] != "NaN":
            k = k+1
            ser.append([])
            for j in df.columns:
                ser[k].append(df[j][i])
        else:
            continue
    del dummy, k
    return ser


# Q1.a
# Checking the stationarity for the entire period.
# Load data
df = pd.read_csv("HW03_USD_TRY_Trading.txt", sep='\t')
seriesname = 'Close'

# All series
print("\nStationarity for the entire period: ")
series = df[seriesname]
test_stationarity(series)
plt.savefig('fig_1aforall.png')
plt.close()

# Checking the stationarity for May 6th.
print("\nStationarity for May 6th: ")
df6 = select_date(df, '6.05.2019')
df6 = pd.DataFrame(df6)
# While using pandas again for filtered data, headers are gone, so 5th header corresponds to 'close' values.
series = df6[5]
test_stationarity(series)
plt.savefig('fig_1a6thMay.png')
plt.close()


# Q1.b
# Holt forecast using entire data:
# Last two data will be forecasted for test:
fore = df[0:11037]
true = df[11037:]
true = np.asarray(true[seriesname])
result = estimate_holt(fore, seriesname, alpha=0.2, slope=0.1, trend="add")
print("\nHolt estimation for 15:57 and 15:58 (May 6th): ", round(result[0],5) , " ", round(result[1],5) )
print("True values: ", true[0], " ", true[1])
print("RMS value for Holt method is:", round(sqrt(mean_squared_error(true, result, sample_weight=None, multioutput="uniform_average")),5))

# For 16:00, all data is used.
fore = df
result = estimate_holt(fore, seriesname, alpha=0.2, slope=0.1, trend="add")
print("\nHolt estimation for 16:00 (May 6th): ", round(result[-1],5))
# For next day,
result = estimate_holt(fore, seriesname, alpha=0.2, slope=0.1, trend="add", estimationlength=1440)
print("\nHolt estimation for 16:00 (May 7th): ", round(result[-1],5))


# Q1.c
# Discarding data with no volume,
dfnovol = pd.DataFrame(discard(df.fillna(0)))
# Checking stationarity,
print("\nQ1.c) Stationarity for data with no volume: ")
# While using pandas again for filtered data, headers are gone, so 5th header corresponds to 'close' values.
series = dfnovol[5]
test_stationarity(series)
plt.savefig('fig_1c.png')
plt.close()

# Trying to estimate 15:57 and 15:58
fore = dfnovol[0:8118]
true = dfnovol[8118:]
true = np.asarray(true[5])
result = estimate_holt(fore, 5, alpha=0.2, slope=0.1, trend="add")
print("\nQ1.c) Holt estimation for 15:57 and 15:58 (May 6th): ", round(result[0],5) , " ", round(result[1],5) )
print("True values: ", true[0], " ", true[1])
print("RMS value for Holt method is:", round(sqrt(mean_squared_error(true, result, sample_weight=None, multioutput="uniform_average")),5))

# For 16:00, all data is used.
fore = dfnovol
result = estimate_holt(fore, 5, alpha=0.2, slope=0.1, trend="add")
print("\nQ1.c) Holt estimation for 16:00 (May 6th): ", round(result[-1],5))
# For next day,
result = estimate_holt(fore, 5, alpha=0.2, slope=0.1, trend="add", estimationlength=1440)
print("\nQ1.c) Holt estimation for 16:00 (May 7th): ", round(result[-1],5))


# Q2.a
# Checking the stationarity for :57 and :58
print("\nQ2) Stationarity for :57 and :58")
df5758 = select_time(df, ':57', ':58')
df5758_2 = pd.DataFrame(df5758)
# While using pandas again for filtered data, headers are gone, so 5th header corresponds to 'close' values.
series = df5758_2[5]
test_stationarity(series)
plt.savefig('fig_2a_5758.png')
plt.close()


# Q2.b
# Holt forecast using :57 and :58 data:
# Last two data will be forecasted for test:
fore = df5758_2[0:366]
true = df5758_2[366:]
true = np.asarray(true[5])
result = estimate_holt(fore, 5, alpha=0.2, slope=0.1, trend="add")
print("\nHolt estimation for 15:57 and 15:58 (May 6th) (Q2.b): ", round(result[0],5), " ", round(result[1],5) )
print("True values: ", true[0], " ", true[1])
print("RMS value for Holt method is:", round(sqrt(mean_squared_error(true, result, sample_weight=None, multioutput="uniform_average")),5))


# Q2.c
# Discarding data with no volume,
dfnovol = pd.DataFrame(discard(df5758_2.fillna(0),seriesname=6))
# Checking stationarity,
print("\nQ2.c) Stationarity for data with no volume: ")
# While using pandas again for filtered data, headers are gone, so 5th header corresponds to 'close' values.
series = dfnovol[5]
test_stationarity(series)
plt.savefig('fig_2c.png')
plt.close()

# Trying to estimate 15:57 and 15:58
fore = dfnovol[0:270]
true = dfnovol[270:]
true = np.asarray(true[5])
result = estimate_holt(fore, 5, alpha=0.2, slope=0.1, trend="add")
print("\nQ2.c) Holt estimation for 15:57 and 15:58 (May 6th): ", round(result[0],5) , " ", round(result[1],5) )
print("True values: ", true[0], " ", true[1])
print("RMS value for Holt method is:", round(sqrt(mean_squared_error(true, result, sample_weight=None, multioutput="uniform_average")),5))


