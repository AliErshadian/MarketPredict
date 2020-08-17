import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#read csv file into data frame
df = pd.read_csv("shapna.csv")
#set date to right foramt and into the index of data farme
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.strftime("%Y-%m-%d")
df = df.set_index(pd.DatetimeIndex(df['Date'].values))


#Exponential Moving Average function
def EMAfunc(_span):
    EMA = df['Close Price'].ewm(span=_span, adjust=False).mean()
    return EMA

#add 3 moving avg to data frame
df['Short'] = EMAfunc(5)
df['Middle'] = EMAfunc(21)
df['Long'] = EMAfunc(63)


#Macd and signal line indicators
MACD = EMAfunc(12) - EMAfunc(26)
MACDsignal = MACD.ewm(span=9, adjust=False).mean()

#add macd and signal line to data frame
df['MACD'] = MACD
df['MACD Signal'] = MACDsignal

#function to buy and sell the stock using EMA
def buy_sell_function(data):
    buyEMA_list = []
    sellEMA_list = []
    flag_EMA_long = False
    flag_EMA_short = False

    buyMACD_list = []
    sellMACD_list = []
    flag_MACD = False


    for i in range(0, len(data)):
        #EMA buy and sell calculating
        if data['Middle'][i] < data['Long'][i] and data['Short'][i] < data['Middle'][i] and flag_EMA_long == False and flag_EMA_short == False:
            buyEMA_list.append(data['Close Price'][i])
            sellEMA_list.append(np.nan)
            flag_EMA_short = True
        elif flag_EMA_short == True and data['Short'][i] > data['Middle'][i]:
            sellEMA_list.append(data['Close Price'][i])
            buyEMA_list.append(np.nan)
            flag_EMA_short = False
        elif data['Middle'][i] > data['Long'][i] and data['Short'][i] > data['Middle'][i] and flag_EMA_long == False and flag_EMA_short == False:
            buyEMA_list.append(data['Close Price'][i])
            sellEMA_list.append(np.nan)
            flag_EMA_long = True
        elif flag_EMA_long == True and data['Short'][i] < data['Middle'][i]:
            sellEMA_list.append(data['Close Price'][i])
            buyEMA_list.append(np.nan)
            flag_EMA_long = False
        else:
            buyEMA_list.append(np.nan)
            sellEMA_list.append(np.nan)


        #MACD buy and sell calculating
        if data['MACD'][i] > data['MACD Signal'][i]:
            sellMACD_list.append(np.nan)
            if flag_MACD == False:
                buyMACD_list.append(data['Close Price'][i])
                flag_MACD = True
            else:
                buyMACD_list.append(np.nan)
        elif data['MACD'][i] < data['MACD Signal'][i]:
            buyMACD_list.append(np.nan)
            if flag_MACD == True:
                sellMACD_list.append(data['Close Price'][i])
                flag_MACD = False
            else:
                sellMACD_list.append(np.nan)
        else:
            buyMACD_list.append(np.nan)
            sellMACD_list.append(np.nan)

    return (buyEMA_list, sellEMA_list, buyMACD_list, sellMACD_list)



#add buy and sell signals to data frame
buy_sell = buy_sell_function(df)
df['BuyEMA'] = buy_sell[0]
df['SellEMA'] = buy_sell[1]
df['BuyMACD'] = buy_sell[2]
df['SellMACD'] = buy_sell[3]



#machine learning part
def PredictMarketPrice(data):
    _df = data['Close Price']

    #Create a variable to predict 'x' ays out into the future
    future_days = 25

    #Create a new column (target) shifted 'x' units/days up
    _df['Prediction'] = _df[['Close Price']].shift(-future_days)

    #Create the feature data set (X) and convert it to a numpy array and remove the last 'x' rows/days
    X = np.array(_df.drop(['Prediction'], 1))[:-future_days]

    #Create the target data set (Y) and convert it to a numpy array and get all the target values except the last 'x' rows/days
    Y = np.array(_df['Prediction'])[:-future_days]

    #Split the data into 75% training and 25% testing
    x_train, x_test, y_tarin, y_test = train_test_split(X, Y, test_size= 0.25)

    #Create the models
    #Create the decision tree regressor model
    tree = DecisionTreeRegressor().fit(x_train, y_tarin)

    #Create the linear regression model
    lr = LinearRegression().fit(x_train, y_tarin)

    #Get the last 'x' rows of the future data set
    x_future = _df.drop(['Prediction'], 1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    #Show the model tree prediction
    tree_prediction = tree.predict(x_future)
    print(tree_prediction)

    print()

    #Show the model linear regression prediction
    lr_prediction = lr.predict(x_future)
    print(lr_prediction)




#visualize data
def EMAvisualize():
    plt.figure(figsize=(12.2, 4.5))
    plt.title('Buy & Sell Plot', fontsize=18)
    #plot 3 moving average
    plt.plot(df['Close Price'], label='Close Price', color='blue', alpha = 0.35)
    plt.plot(EMAfunc(5), label='Short/Fast EMA', color='red', alpha = 0.35)
    plt.plot(EMAfunc(21), label='Middle/Medium EMA', color='orange', alpha = 0.35)
    plt.plot(EMAfunc(63), label='Long/Slow EMA', color='green', alpha = 0.35)
    plt.scatter(df.index, df['BuyEMA'], color= 'green', marker='^', alpha = 1)
    plt.scatter(df.index, df['SellEMA'], color= 'red', marker='v', alpha = 1)
    plt.xticks(rotation=45)
    plt.xlabel('Date', fontsize =18)
    plt.ylabel('Close Price', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()
    print(df)

def MACDvisualize():
    plt.figure(figsize=(12.2, 4.5))
    plt.plot(df['Close Price'], label='Close Price', color='black', alpha = 0.35)
    plt.plot(df.index, MACD, label='MACD Line', color='red')
    plt.plot(df.index, MACDsignal, label='Signal Line', color='blue')
    plt.scatter(df.index, df['BuyMACD'], color='green', marker='^', alpha=1)
    plt.scatter(df.index, df['SellMACD'], color='red', marker='v', alpha=1)
    plt.legend(loc='upper left')
    plt.show()
    print(df)

def visualize():
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close Price'], label='Close Price')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.legend(loc='upper left')
    plt.show()




#EMAvisualize()
#MACDvisualize()
#visualize()