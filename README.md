```python
# import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as pdr
%matplotlib inline
import yfinance as yf
```


```python
# Define date range
startdate = '2023-01-01'
enddate = '2024-12-31'
# Fetch Apple, Tesla and Google's stock data from Yahoo Finance
sample = yf.download (['AAPL', 'TSLA', 'GOOGL'], start = startdate, end = enddate)
print (sample.head())
```

    YF.download() has changed argument auto_adjust default to True


    [*********************100%***********************]  3 of 3 completed

    Price            Close                               High             \
    Ticker            AAPL      GOOGL        TSLA        AAPL      GOOGL   
    Date                                                                   
    2023-01-03  123.632530  88.695946  108.099998  129.395518  90.616763   
    2023-01-04  124.907700  87.660904  113.639999  127.181268  90.218675   
    2023-01-05  123.583107  85.789841  110.339996  126.301500  87.153325   
    2023-01-06  128.130234  86.924423  113.059998  128.792531  87.272764   
    2023-01-09  128.654129  87.601181  119.769997  131.876670  89.621528   
    
    Price                          Low                               Open  \
    Ticker            TSLA        AAPL      GOOGL        TSLA        AAPL   
    Date                                                                    
    2023-01-03  118.800003  122.742873  88.098795  104.639999  128.782649   
    2023-01-04  114.589996  123.642412  86.854753  107.519997  125.431607   
    2023-01-05  111.750000  123.326101  85.491273  107.160004  125.668857   
    2023-01-06  114.389999  123.454601  84.456228  101.809998  124.561732   
    2023-01-09  123.519997  128.397123  87.441946  117.110001  128.970458   
    
    Price                                 Volume                       
    Ticker          GOOGL        TSLA       AAPL     GOOGL       TSLA  
    Date                                                               
    2023-01-03  89.163703  118.470001  112117500  28131200  231402800  
    2023-01-04  89.920100  109.110001   89113600  34854800  180389000  
    2023-01-05  87.053802  110.510002   80962700  27194400  157986300  
    2023-01-06  86.377045  103.000000   87754700  41381500  220911100  
    2023-01-09  87.939567  118.959999   70790800  29003900  190284000  


    



```python
# Displays the stocks data
print(sample)
# Uses pandas to change the display setting
```

    Price            Close                                High              \
    Ticker            AAPL       GOOGL        TSLA        AAPL       GOOGL   
    Date                                                                     
    2023-01-03  123.632530   88.695946  108.099998  129.395518   90.616763   
    2023-01-04  124.907700   87.660904  113.639999  127.181268   90.218675   
    2023-01-05  123.583107   85.789841  110.339996  126.301500   87.153325   
    2023-01-06  128.130234   86.924423  113.059998  128.792531   87.272764   
    2023-01-09  128.654129   87.601181  119.769997  131.876670   89.621528   
    ...                ...         ...         ...         ...         ...   
    2024-12-23  254.989655  194.406113  430.600006  255.369227  194.875573   
    2024-12-24  257.916443  195.884399  462.279999  257.926411  195.884399   
    2024-12-26  258.735504  195.375000  454.130005  259.814335  196.523671   
    2024-12-27  255.309296  192.538254  431.660004  258.415896  195.095322   
    2024-12-30  251.923019  191.020004  417.410004  253.221595  192.328495   
    
    Price                          Low                                Open  \
    Ticker            TSLA        AAPL       GOOGL        TSLA        AAPL   
    Date                                                                     
    2023-01-03  118.800003  122.742873   88.098795  104.639999  128.782649   
    2023-01-04  114.589996  123.642412   86.854753  107.519997  125.431607   
    2023-01-05  111.750000  123.326101   85.491273  107.160004  125.668857   
    2023-01-06  114.389999  123.454601   84.456228  101.809998  124.561732   
    2023-01-09  123.519997  128.397123   87.441946  117.110001  128.970458   
    ...                ...         ...         ...         ...         ...   
    2024-12-23  434.510010  253.171646  189.931255  415.410004  254.490204   
    2024-12-24  462.779999  255.009620  193.557078  435.140015  255.209412   
    2024-12-26  465.329987  257.347047  194.156402  451.019989  257.906429   
    2024-12-27  450.000000  252.782075  190.430680  426.500000  257.546826   
    2024-12-30  427.000000  250.474615  188.902433  415.750000  251.952985   
    
    Price                                  Volume                       
    Ticker           GOOGL        TSLA       AAPL     GOOGL       TSLA  
    Date                                                                
    2023-01-03   89.163703  118.470001  112117500  28131200  231402800  
    2023-01-04   89.920100  109.110001   89113600  34854800  180389000  
    2023-01-05   87.053802  110.510002   80962700  27194400  157986300  
    2023-01-06   86.377045  103.000000   87754700  41381500  220911100  
    2023-01-09   87.939567  118.959999   70790800  29003900  190284000  
    ...                ...         ...        ...       ...        ...  
    2024-12-23  192.398415  431.000000   40858800  25675000   72698100  
    2024-12-24  194.615856  435.899994   23234700  10403300   59551800  
    2024-12-26  194.925505  465.160004   27237100  12046600   76366400  
    2024-12-27  194.725737  449.519989   42355300  18891400   82666800  
    2024-12-30  189.581658  419.399994   35557500  14264700   64941000  
    
    [501 rows x 15 columns]



```python
# Using iloc index to specify the columns needed 
sample = sample.iloc [:, 0:3]
# Renaming the columns 
sample.columns = ('APPLE', 'GOOGLE', 'TESLA')
```


```python
print(sample)
```

                     APPLE      GOOGLE       TESLA
    Date                                          
    2023-01-03  123.632530   88.695946  108.099998
    2023-01-04  124.907700   87.660904  113.639999
    2023-01-05  123.583107   85.789841  110.339996
    2023-01-06  128.130234   86.924423  113.059998
    2023-01-09  128.654129   87.601181  119.769997
    ...                ...         ...         ...
    2024-12-23  254.989655  194.406113  430.600006
    2024-12-24  257.916443  195.884399  462.279999
    2024-12-26  258.735504  195.375000  454.130005
    2024-12-27  255.309296  192.538254  431.660004
    2024-12-30  251.923019  191.020004  417.410004
    
    [501 rows x 3 columns]



```python
print(BarPlot_sample)
```

    APPLE     188.766780
    GOOGLE    140.720748
    TESLA     223.712455
    dtype: float64



```python
BarPlot_sample= sample.mean()
```


```python
# Reassigning the dataframe
BarPlot_sample = pd.DataFrame (data = BarPlot_sample, columns = ['values'])
```


```python
# Create scatter plot
plt.figure(figsize=(10, 6))
sns.barplot (x = BarPlot_sample.index, y = 'values', data = BarPlot_sample)
# Label Axes
plt.xlabel ('STOCKS')
plt.ylabel ('VALUES')
plt.title ('MEAN CLOSING STOCK PRICE')
plt.yticks(ticks=range(0, 350, 50));

```


    
![png](output_8_0.png)
    



```python
print (Apple_sample.head())
```

                     APPLE     GOOGLE       TESLA
    Date                                         
    2023-01-03  123.632530  88.695953  108.099998
    2023-01-04  124.907707  87.660896  113.639999
    2023-01-05  123.583107  85.789841  110.339996
    2023-01-06  128.130219  86.924408  113.059998
    2023-01-09  128.654129  87.601173  119.769997



```python
Apple_sample = sample
```


```python
# Create Apple Stock line chart
years = Apple_sample.index
plt.figure(figsize=(10, 6))
Values = Apple_sample ['APPLE']
plt.plot (years, Values)
plt.xticks (rotation = 45)
# Label Axes
plt.xlabel ('YEARS')
plt.ylabel ('CLOSE')
plt.title ('TREND IN APPLE');
```


    
![png](output_11_0.png)
    



```python
print (sample.head())
```

                     APPLE     GOOGLE       TESLA
    Date                                         
    2023-01-03  123.632530  88.695953  108.099998
    2023-01-04  124.907707  87.660896  113.639999
    2023-01-05  123.583107  85.789841  110.339996
    2023-01-06  128.130219  86.924408  113.059998
    2023-01-09  128.654129  87.601173  119.769997



```python
# Create Google stock lime chart
Google_sample = sample
years = Google_sample.index
plt.figure(figsize=(10, 6))  
Values = Google_sample ['GOOGLE']
plt.plot (years, Values)
plt.xticks (rotation = 45)
# Label Axes
plt.xlabel ('YEARS')
plt.ylabel ('CLOSE')
plt.title ('TREND IN GOOGLE');
```


    
![png](output_13_0.png)
    



```python
# Create Tesla stock line chart 
Tesla_sample = sample
years = Tesla_sample.index
plt.figure(figsize=(10, 6))  #
Values = Tesla_sample ['TESLA']
plt.plot (years, Values)
plt.xticks (rotation = 45)
# Label Axes
plt.xlabel ('YEAR')
plt.ylabel ('CLOSE')
plt.title ('TREND IN TESLA');
```


    
![png](output_14_0.png)
    



```python
print(sample)
```

                     APPLE      GOOGLE       TESLA
    Date                                          
    2023-01-03  123.632530   88.695946  108.099998
    2023-01-04  124.907700   87.660904  113.639999
    2023-01-05  123.583107   85.789841  110.339996
    2023-01-06  128.130234   86.924423  113.059998
    2023-01-09  128.654129   87.601181  119.769997
    ...                ...         ...         ...
    2024-12-23  254.989655  194.406113  430.600006
    2024-12-24  257.916443  195.884399  462.279999
    2024-12-26  258.735504  195.375000  454.130005
    2024-12-27  255.309296  192.538254  431.660004
    2024-12-30  251.923019  191.020004  417.410004
    
    [501 rows x 3 columns]



```python
DailyReturns_sample = sample
```


```python
DailyReturns_sample = DailyReturns_sample [['APPLE', 'GOOGLE', 'TESLA']].pct_change()
```


```python

```


```python
plt.figure(figsize=(10, 6))
sns.kdeplot(DailyReturns_sample['APPLE'], label='APPLE', shade=True)
sns.kdeplot(DailyReturns_sample['GOOGLE'], label='GOOGLE', shade=True)
sns.kdeplot(DailyReturns_sample['TESLA'], label = 'TESLA', shade = True)

plt.title('KDE Plot of Daily Returns: AAPL vs MSFT')
plt.xlabel('Daily Return')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

#
```

    /data/user/0/ru.iiec.pydroid3/cache/ipykernel_8573/3436900718.py:2: FutureWarning: 
    
    `shade` is now deprecated in favor of `fill`; setting `fill=True`.
    This will become an error in seaborn v0.14.0; please update your code.
    
      sns.kdeplot(DailyReturns_sample['APPLE'], label='APPLE', shade=True)
    /data/user/0/ru.iiec.pydroid3/cache/ipykernel_8573/3436900718.py:3: FutureWarning: 
    
    `shade` is now deprecated in favor of `fill`; setting `fill=True`.
    This will become an error in seaborn v0.14.0; please update your code.
    
      sns.kdeplot(DailyReturns_sample['GOOGLE'], label='GOOGLE', shade=True)
    /data/user/0/ru.iiec.pydroid3/cache/ipykernel_8573/3436900718.py:4: FutureWarning: 
    
    `shade` is now deprecated in favor of `fill`; setting `fill=True`.
    This will become an error in seaborn v0.14.0; please update your code.
    
      sns.kdeplot(DailyReturns_sample['TESLA'], label = 'TESLA', shade = True)



    
![png](output_19_1.png)
    



```python
Volatility_sample = sample
```


```python

Volatility_sample = Volatility_sample[['APPLE', 'GOOGLE', 'TESLA']].std()
```


```python
Volatility_sample = Volatility_sample * np.sqrt(252)
```


```python
print(Volatility_sample)
```

    APPLE     446.154263
    GOOGLE    435.378817
    TESLA     923.376457
    dtype: float64



```python

```