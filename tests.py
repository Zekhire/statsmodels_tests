# AR example
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from random import random
import matplotlib.pyplot as plt
import numpy as np


dt = 0.01
data = [x + np.random.normal(1, 1) for x in range(50)]          # <- stworz wektor probek procesu
t = [x*dt for x in range(len(data))]                            # <- stworz wektor czasu

# w kazdej iteracji doloz kolejna probke procesu, zeby wyznaczyc model
for i in range(8, len(data)):
    # do wyboru:
    # model = AutoReg(data[:i], lags=3, old_names=False)        #
    # model = ARIMA(data[:i], order=(2, 0, 1))                  # <- wyznacz model bazujac na i pierwszych probkach procesu
    model = ARIMA(data[:i], order=(2, 1, 1))                    #
    model_fit = model.fit()
    yhat = model_fit.predict(1, len(data))
    shift = 1
    ty = [j*dt for j in range(shift, len(yhat)+shift)]          # <- wyznacz wektor czasu dla przewidywanych probek
    print(yhat)
    
    plt.plot(t, data)
    plt.plot(ty, yhat)
    plt.show()

