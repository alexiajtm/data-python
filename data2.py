import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta 


data = yf.download('MSFT', start='2009-01-01', end='2024-01-01')


print(data)
print(data.head())
print(data.tail())
print(data.info())
print(data.describe())


plt.figure(figsize=(12,6))
plt.plot(data.index, data['High'], label='Prix High')

plt.xlabel('Date')
plt.ylabel('Prix')
plt.show()