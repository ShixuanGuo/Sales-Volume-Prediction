#!/usr/bin/env python
# coding: utf-8

# # 1. Time Series Model

# In[128]:


from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA


# In[129]:


# Read in data and parse dates
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m')
df_data = pd.read_csv('data.csv', parse_dates=['YM'],date_parser=dateparse)
df_H=df_data[df_data['SalesRegion']=='H']
df_J=df_data[df_data['SalesRegion']=='J']


# In[131]:


# test series' stationarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    '''
    Plot the rolling statistics and check time-series' stationary using ADF score
    
    Parameters:
    ----------
    timeseries: dataframe
    '''
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['VolumeHL'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[ ]:


# Hyperparameter tuning
def best_hyperparameter(ts_log_train,ts_log_test):
    '''
    Find the best hyperparameters using BIC score
    
    Parameters:
    ----------
    ts_log_train: series
        adjusted training time series
    ts_log_test: series
        adjusted testing time series
    Return:
    ---------
    p: int
        the best number of previous/lagged Y values are accounted for for each time point 
    q: int
        the best number of previous/lagged error values are accounted for for each time point
    '''
    pmax = int(len(ts_log_train)/10) 
    qmax = int(len(ts_log_train)/10) 
    
    #Build BIC score matrix
    bic_matrix = []
    for p in range(pmax+1):
        tmp = []
        for q in range(qmax+1):
            try: 
                tmp.append(ARIMA(ts_log_train, (p,1,q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    
    #Choose hyperparameter combination which has the lowest BIC score
    p,q = bic_matrix.stack().astype(float).idxmin()
    return p,q


# ## 2.1 Monthly total volume prediction

# In[ ]:


# Monthly total volume by region
df_H_Sum=pd.DataFrame(df_H.groupby('YM')['VolumeHL'].sum())
df_J_Sum=pd.DataFrame(df_J.groupby('YM')['VolumeHL'].sum())


# In[ ]:


# Time series adjustment: remove trend and seasonality
ts_log = np.log(df_H_Sum)
ts_log_diff = ts_log - ts_log.shift(periods=1)
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# split train and test dataset
n=int(len(ts_log)*0.7)
ts_log_train=ts_log[:n]
ts_log_test=ts_log[n:]
# find the best hyperparameter
p,q=best_hyperparameter(ts_log_train,ts_log_test)
print(u'BIC minimum p and q is：%s、%s' %(p,q))


# In[15]:


# Fit ARIMA model and make prediction
model = ARIMA(ts_log, (p,1,q)).fit() 
ts_log_hat=model.forecast(1)[1]


# ## 2.2 Monthly volume prediction by SKU

# In[ ]:


SKU_H=df_H['SKU Code'].unique()


# In[82]:


y_test=[]
y_hat=[]
SKU_H=df_H['SKU Code'].unique()
for code in SKU_H:
    df_H1=df_H.loc[df_data['SKU Code']==code,['YM','VolumeHL']].sort_values('YM')
    df_H1.set_index('YM',inplace=True)
    
    ts_log = np.log(df_H1)
    ts_log_diff = ts_log - ts_log.shift(periods=1)
    ts_log_diff.dropna(inplace=True)
    test_stationarity(ts_log_diff)
    
    n=int(len(ts_log)*0.7)
    ts_log_train=ts_log[:n]
    ts_log_test=ts_log[n:]
    
    p,q=best_hyperparameter(ts_log_train,ts_log_test)
    print(u'BIC minimum p and q is：%s、%s' %(p,q))

    model = ARIMA(ts_log, (p,1,q)).fit() 
    ts_log_hat=model.forecast(len(ts_log_test))[1]
    ts_hat_inverse=[]
    for i in range(len(ts_log_hat)):
        ts_hat_inverse+=list(np.exp(ts_log_hat[i]+ts_log.values[n+i-1]))
    y_test.append(list(df_H1['VolumeHL'][:n]))
    y_hat.append(ts_hat_inverse)


# In[83]:


# Calculate WAPE
error=0
total=0
for i in range(len(y_hat)):
    total+=sum(y_test[i])
    for j in range(len(y_hat[i])):
        error+=abs(y_hat[i][j]-y_test[i][j])
print(error, total)
mse=error/total
print(mse)

