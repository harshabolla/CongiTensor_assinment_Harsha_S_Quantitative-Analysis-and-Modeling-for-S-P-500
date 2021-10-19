#importing libiries 
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

#from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates

from datetime import datetime
from sklearn import preprocessing;
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import linear_model;


import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
#get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("cs-1.csv")
print(df.head())



print(df.info())
df['date'] = df['date'].map(lambda t: datetime.strptime(str(t), '%Y-%m-%d'))

#df = df.set_index('date')
#checking for na values
df.isnull().sum()
#droping navalues 
df = df.dropna()
print(df.describe())

#For the sake of visualization, Let's create extract year from the date column
df['year'] = pd.DatetimeIndex(df["date"]).year
df['month'] = pd.DatetimeIndex(df["date"]).month
df['date'] = pd.DatetimeIndex(df["date"]).date

#Since the year 2017 is the most recent year with dataset of over 4 months, let's explore that
# Creating a ColumnDataSource instance to act as a reusable data source for ploting

print(df["Name"].unique())
#Give an exploratory analysis on any one stock describing it’s key statisticaltendencies.

#Give an exploratory analysis on any one stock describing it’s key statisticaltendencies.

#We'll focus on one
walmart = df.loc[df['Name'] == 'WMT']
print(walmart.head())

print(walmart.info())

#Create a copy to avoid the SettingWarning .loc issue 
walmart_df = walmart.copy()

# Change to datetime datatype.
walmart_df.loc[:,'date'] = pd.to_datetime(walmart.loc[:,'date'], format="%Y/%m/%d")

print(walmart_df.info())

#Let’s calculate returns 
walmart_df['Returns'] = (walmart_df['open'] - walmart_df['close'])/walmart_df['open']

walmart_df['Returns'].hist()

plt.title('walmart Stock Price Returns Distribution')

plt.show()



# Let us plot Walmart Stock Price
# First Subplot
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
ax1.plot(walmart_df["date"], walmart_df["close"])
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title("walmart Close Price History")
ax2.plot(walmart_df["date"], walmart_df["high"], color="green")
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Stock Price")
ax2.set_title("walmart High Price History")


# Second Subplot
f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
ax1.plot(walmart_df["date"], walmart_df["low"], color="red")
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title("Walmart Low Price History")
ax2.plot(walmart_df["date"], walmart_df["volume"], color="orange")
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Stock Price")
ax2.set_title("Walmart's Volume History")

#do some feture enggenering 
walmart_df['First Difference'] = walmart_df['close'] - walmart_df['close'].shift()
walmart_df['First Difference'].plot(figsize=(16, 12))

#Adding the new features to understand the price variations of stock within a day and from previous day, these additional features will help in predicting the day close stock price with atmost accuracy as well as these features helps the model to classsify the volatility of a stock

walmart_df['changeduringday'] = ((walmart_df['high'] - walmart_df['low'] )/ walmart_df['low'])*100

walmart_df['changefrompreviousday'] = (abs(walmart_df['close'].shift() - walmart_df['close'] )/ walmart_df['close'])*100

print("**The new features 'changeduring day & change from previous day are added to the dataset. Note: The first row for change from previous day for each stock is NA or blank always")
print(walmart_df.head())

walmart_df.hist(bins=50, figsize=(20,15))
plt.show()


corr_matrix = walmart_df.corr()

corr_matrix["close"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["high", "low", "open", "changefrompreviousday", "changeduringday", "volume"]

scatter_matrix(walmart_df[attributes], figsize=(20, 15))

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().run_line_magic('matplotlib', 'inline')
corr = walmart_df[["high", "low", "open", "changefrompreviousday", "changeduringday", "volume"]].corr()

# generate a mask for the lower triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 12))

# generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, 
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax);


df_close = pd.DataFrame(walmart_df['close'])
df_close.index = pd.to_datetime(walmart_df['date'])




#Plotting again the closing price graph for walmart
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

plt.figure(figsize=(8, 6))
plt.plot(df_close, color='g')
plt.title('Walmart Closing Price', weight='bold', fontsize=16)
plt.xlabel('Date', weight='bold', fontsize=14)
plt.ylabel('Stock Price', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth = 0.5)





#Autocorrelation plot
from statsmodels.tsa import stattools

acf_djia, confint_djia, qstat_djia, pvalues_djia = stattools.acf(df_close,
                                                             unbiased=True,
                                                             nlags=50,
                                                             qstat=True,
                                                             fft=True,
                                                             alpha = 0.05)

plt.figure(figsize=(7, 5))
plt.plot(pd.Series(acf_djia), color='r', linewidth=2)
plt.title('Autocorrelation of Walmart Closing Price', weight='bold', fontsize=16)
plt.xlabel('Lag', weight='bold', fontsize=14)
plt.ylabel('Value', weight='bold', fontsize=14)
plt.xticks(weight='bold', fontsize=12, rotation=45)
plt.yticks(weight='bold', fontsize=12)
plt.grid(color = 'y', linewidth = 0.5)


# prepare data for the prediction
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) #cross validation 

    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;


forecast_col = 'close'#choosing which column to forecast
forecast_out = 50 #how far to forecast 
test_size = 0.2; #the size of my test set

X_train, X_test, Y_train, Y_test , X_lately =prepare_data(walmart_df,forecast_col,forecast_out,test_size); #calling the method were the cross validation and data preperation is in

learner = linear_model.LinearRegression(); #initializing linear regression model
learner.fit(X_train,Y_train); #training the linear regression model

score=learner.score(X_test,Y_test);#testing the linear regression model
forecast= learner.predict(X_lately); #set that will contain the forecasted data

response={};#creting json object
response['test_score']=score; 
response['forecast_set']=forecast;

print(response);




f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.plot(X_lately, forecast, color="yellow")
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("Stock Price")
ax1.set_title("Walmart Close Price Predicted")

ax2.scatter(X_test, Y_test, color="green")
ax2.set_xlabel("day", fontsize=12)
ax2.set_ylabel("Stock Price")
ax2.set_title("Walmart actual Close Price")
print(score)


# "1. Volatility Index - Out of all the 500 stocks in the dataset, establish a weekly"
# "volatilityindex which ranks stocks on the basis of intraday price movements.(Weekly volatility Index 
# "implies that it is to be calculated on a weekly time frame and bothintraday 
# "as well as weekly change in price needs to be used in calculating volatility)"

count_df = pd.DataFrame(df.Name.value_counts(),columns=["Name","Count"]).reset_index()
lis_valid_share = list(count_df["index"])
final_df = df[df.Name.isin(lis_valid_share)]
data_by_year = final_df.groupby("year")
data_by_asset = final_df.groupby("Name")
print(df.year.value_counts())
#for year 2013
year = 2013
data2 = data_by_year.get_group(year)
final_pivot = data2.pivot(index = "date",columns = "Name",values = "close")
daily_volatility = final_pivot.pct_change().apply(lambda x:np.log(1+x)).std()
weekly_volatility  = daily_volatility.apply(lambda x: x*np.sqrt(5))
WV = pd.DataFrame(weekly_volatility).reset_index()
WV.columns = ["Name","Volatility"]

#ToP Volatile stocks
weekly_top10_volatility = WV.sort_values(by = "Volatility",ascending = False)
weekly_low10_volatility = WV.sort_values(by = "Volatility",ascending = True)
print("weekly_hig10_volatility.")
print(weekly_top10_volatility.head(10))



#Least Volatile stocks
weekly_low10_volatility = WV.sort_values(by = "Volatility",ascending = True)
weekly_low10_volatility.head(10)
print("weekly_low10_volatility.")

