import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv("shapna.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.strftime("%Y-%m-%d")
df = df.set_index(pd.DatetimeIndex(df['Date'].values))

print(df)

#visualize data
plt.figure(figsize=(12.2, 4.5))
plt.title('Close Price')
plt.plot(df['Close Price'])
plt.show()
