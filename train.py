#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from pandas.plotting import lag_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import mse, rmse, meanabs

import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.tools import diff


# In[2]:


df = pd.read_csv("countries-aggregated.csv",parse_dates=True,index_col="Date")
df = df[df["Country"]=="Germany"]


# ## Inspection

# In[3]:


df["Confirmed"].plot()
plt.title("Confirmed Covid Cases in Germany")
plt.show()


# In[4]:


result = seasonal_decompose(df['Confirmed'], model='additive')  # model='add' also works
fig = result.plot();
fig.set_size_inches((12, 9))


# In[5]:


lag_plot(df["Confirmed"])
plt.title("Lag Plot")
plt.show()


# Strong indication of Autocorrelation

# In[6]:


plot_acf(df["Confirmed"],title="Autocorrelation of Confirmed Covid Cases",lags=50);


# In[7]:


plot_pacf(df["Confirmed"],title="Partial Autocorrelation of Confirmed Covid Cases",lags=10);


# ## Dickey-Fuller Test

# In[8]:


from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data

    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val

    print(out.to_string())          # .to_string() removes the line "dtype: float64"

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


# In[9]:


adf_test(df["Confirmed"],title="Confirmed Covid Cases in Germany")


    # As seen from the plot is the begining, the data is non-stationary and is seeing an exponential trend

    # ## Choosing ARIMA Parameters

    # In[10]:


from pmdarima import auto_arima


# In[11]:


stepwise_fit = auto_arima(df["Confirmed"], start_p=0, start_q=0,
                          max_p=6, max_q=3, m=3,
                          seasonal=True,
                          d=None, trace=True,
                          error_action='ignore',   #  if an order does not work
                          suppress_warnings=True,  
                          stepwise=True)  


# In[12]:


stepwise_fit.summary()


# In[13]:


df["diff_2"] = diff(df["Confirmed"],k_diff=2)


# In[14]:


adf_test(df["diff_2"])


# In[15]:


plot_acf(df["diff_2"].dropna(),lags=10);


# In[16]:


plot_pacf(df["diff_2"].dropna(),lags=10);


# ## ARIMA

# In[17]:


from statsmodels.tsa.arima.model import ARIMA


# In[18]:


train = df["Confirmed"].iloc[: len(df)-30]
test = df["Confirmed"].iloc[len(df)-30:]


# In[19]:


model = ARIMA(train,order=(0,2,2))
arima_results = model.fit()
arima_results.summary()


# In[20]:


start = len(train)
end = len(train)+len(test)-1
predictions = arima_results.predict(start=start,end=end,dynamic=False,typ="levels").rename("ARIMA (0,2,2) predictions")


# In[21]:


predictions.plot()
plt.plot(test,label="actuals")
plt.legend()
plt.savefig("ARIMA_Preds.png",dpi=120)
plt.close()

# In[22]:


df["predictions"] = predictions


# In[23]:


df = df.convert_dtypes()


# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


# In[27]:


mae = mean_absolute_error(test,predictions)
rmse = math.sqrt(mean_squared_error(test,predictions))
print("MAE is:",mae)
print("RMSE is:",rmse)

with open("arima_metrics.txt","w") as op:
    op.write("MAE of ARIMA is:  %2.1f%%\n"%mae)
    op.write("RMSE ARIMA is:  %2.1f%%\n"%rmse)



# ## SARIMA

# In[28]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[29]:


model = SARIMAX(train,order=(0,2,2),seasonal_order=(2,0,2,3))
sarima_results = model.fit()
sarima_results.summary()


# In[30]:


predictions = sarima_results.predict(start=start,end=end,dynamic=False,typ="levels").rename("SARIMAX (0,2,2)(2,0,2,3) predictions")


# In[32]:


predictions.plot()
plt.plot(test,label="actuals")
plt.legend()
plt.savefig("sarima_pred.png",dpi=120)
plt.close()


# In[34]:


mae = mean_absolute_error(test,predictions)
rmse = math.sqrt(mean_squared_error(test,predictions))
print("MAE is:",mae)
print("RMSE is:",rmse)

with open("sarima_metrics.txt","w") as op:
    op.write("MAE of SARIMA is:  %2.1f%%\n"%mae)
    op.write("RMSE SARIMA is:  %2.1f%%\n"%rmse)


# In[ ]:




