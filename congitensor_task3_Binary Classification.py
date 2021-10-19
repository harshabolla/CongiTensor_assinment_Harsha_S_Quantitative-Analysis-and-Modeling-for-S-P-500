
#Binary Classification - Given a stock and it’s data, you have to predict whether it willclose lower than it opened (red) or higher than it opened (green)

#Import the libraries
import pandas as pd
import datetime as dt
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#cs-1 file to be in same folder
df=pd.read_csv("cs-1.csv")
#date format 
df['date'] = df['date'].map(lambda t: datetime.strptime(str(t), '%Y-%m-%d'))
#checking for null
df.isnull().sum()
#dropping null values
df = df.dropna()
#checking shape of dataset
print(df.shape)
#Binary Classification - Given a stock and it’s data, you have to predict whether it willclose lower than it opened (red) or higher than it opened (green) [Continued on the nextpage]""""""

#Manipulate the data set 
#Create the target column
df['Price_Up_down'] = np.where(df['close'] > df['open'], 1, 0)
#Remove the date column
remove_list = ['date','Name'] 
df = df.drop(columns=remove_list)
#Show the data
print(df)
#finding correlationbetween features 
correlation =df.corr()
cols= ['#00876c','#85b96f','#f7e382','#f19452','#d43d51']
plt.figure(figsize=(15,9))
sns.heatmap(correlation,cmap=cols ,annot=True, linewidths=0.5)
plt.show()

#pair plot
sns.pairplot(df, hue='Price_Up_down',corner=True)

#distrubution of data
def disbution_of_data(feture):
    plt.figure(figsize=(15,9))
    sns.distplot(feture,color='green',bins=100)
    
disbution_of_data(df['open'])
disbution_of_data(df['high'])
disbution_of_data(df['low'])
disbution_of_data(df['low'])
disbution_of_data(df['volume'])
disbution_of_data(df['Price_Up_down'])

f,ax=plt.subplots(1,2,figsize=(14,6))
df['Price_Up_down'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=False)
ax[0].set_title('Price_Up_down')
ax[0].set_ylabel('Price_Up_down')
sns.countplot('Price_Up_down',data=df,ax=ax[1])
ax[1].set_title('Price_Up_down')
plt.show()

cols_to_use = ['open', 'high', 'low', 'close', 'volume', 'Price_Up_down']
fig = plt.figure(figsize=(8, 20))
plot_count = 0
for col in cols_to_use:
    plot_count += 1
    plt.subplot(7, 1, plot_count)
    plt.scatter(range(df.shape[0]), df[col].values)
    plt.title("Distribution of "+col)
plt.show()

#histogram of all fetures
p =df.hist(figsize = (20,20))


#fininding the feture importance using XGboost
import xgboost as xgb

train_y = df['Price_Up_down']
train_X = df.drop(['Price_Up_down'], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 10,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

import warnings
warnings.filterwarnings('ignore')
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
remain_num = 99

fig, ax = plt.subplots(figsize=(10,8))
xgb.plot_importance(model, max_num_features=remain_num, height=0.8, ax=ax)
plt.show()
#Model bulding
#Split the data set into a feature or independent data set (X) and a target or dependent data set (Y)
x = df.iloc[:, 0:df.shape[1] -1].values #Get all the rows and columns except for the target column
y = df.iloc[:, df.shape[1]-1].values  #Get all the rows from the target column

#Split the data again but this time into 80% training and 20% testing data sets
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.20, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
ran_clf = RandomForestClassifier(max_depth=2, random_state=0)

rfmodel = ran_clf.fit(x_train,y_train)
y_pred2 = rfmodel.predict(x_test)
accuracy_score(y_test,y_pred2)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cn = confusion_matrix(y_test,y_pred2)
sns.heatmap(cn,annot=True)
print(confusion_matrix(y_test,y_pred2))
print(accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

from sklearn import metrics
#IMPORTANT: first argument is true values, second argument is predicted probabilities
# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred2)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Attrition classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()


#Create and train the decision tree Classifier model
tree = DecisionTreeClassifier().fit(x_train, y_train)

#Check how well the model did on the training data set
print("traning score" ,tree.score(x_train, y_train))


#Check how well the model did on the test data set
print( "testscore", tree.score(x_test, y_test))
#Show the actual values from the test data set
print(y_test)
