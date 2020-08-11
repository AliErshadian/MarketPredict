import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = pd.read_csv("shapna.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.strftime("%Y-%m-%d")
df = df.set_index(pd.DatetimeIndex(df['Date'].values))



#visualize data
def visualize():
    plt.figure(figsize=(12.2, 4.5))
    plt.title('Buy & Sell Plot', fontsize=18)

    #plot 3 moving average
    plt.plot(df['Close Price'], label='Close Price', color='blue', alpha = 0.35)
    plt.plot(ShortEMA, label='Short/Fast EMA', color='red', alpha = 0.35)
    plt.plot(MiddleEMA, label='Middle/Medium EMA', color='orange', alpha = 0.35)
    plt.plot(LongEMA, label='Long/Slow EMA', color='green', alpha = 0.35)
    plt.scatter(df.index, df['Buy'], color = 'green', marker='^', alpha = 1)
    plt.scatter(df.index, df['Sell'], color = 'red', marker='v', alpha = 1)

    plt.xlabel('Date', fontsize =18)
    plt.ylabel('Close Price', fontsize=18)

    plt.show()
    print(df)


#3 moving avg
ShortEMA = df['Close Price'].ewm(span=5, adjust= False).mean()
MiddleEMA = df['Close Price'].ewm(span=21, adjust= False).mean()
LongEMA = df['Close Price'].ewm(span=63, adjust= False).mean()

#add 3 moving avg to dataframe
df['Short'] = ShortEMA
df['Middle'] =MiddleEMA
df['Long'] = LongEMA


#function to buy and sell the stock
def buy_sell_function(data):
    buy_list = []
    sell_list = []
    flag_long = False
    flag_short = False

    for i in range(0, len(data)):
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Close Price'][i])
            sell_list.append(np.nan)
            flag_short = True
        elif flag_short == True and data['Short'][i] > data['Middle'][i]:
            sell_list.append(data['Close Price'][i])
            buy_list.append(np.nan)
            flag_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_long == False and flag_short == False:
            buy_list.append(data['Close Price'][i])
            sell_list.append(np.nan)
            flag_long = True
        elif flag_long == True and data['Short'][i] < data['Middle'][i]:
            sell_list.append(data['Close Price'][i])
            buy_list.append(np.nan)
            flag_long = False
        else:
            buy_list.append(np.nan)
            sell_list.append(np.nan)

    return (buy_list, sell_list)

#add buy and sell signal to data frame
df['Buy'] = buy_sell_function(df)[0]
df['Sell'] = buy_sell_function(df)[1]


visualize()