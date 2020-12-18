# AR example
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from random import random
import matplotlib.pyplot as plt
import numpy as np


dt = 0.01
data = [x + np.random.normal(1, 1) for x in range(50)]
t = [x*dt for x in range(len(data))]

for i in range(8, len(data)):
    # model = AutoReg(data[:i], lags=3, old_names=False)
    # model = ARIMA(data[:i], order=(2, 0, 1))
    model = ARIMA(data[:i], order=(2, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.predict(1, len(data))
    shift = 1
    ty = [j*dt for j in range(shift, len(yhat)+shift)]
    print(yhat)
    
    plt.plot(t, data)
    plt.plot(ty, yhat)
    plt.show()

