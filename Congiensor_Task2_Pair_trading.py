#import libiries 
import pandas as pd
import datetime as dt
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

#load data set 

df=pd.read_csv("cs-1.csv")
#change date variable 
df['date'] = pd.to_datetime(df['date'])
df['year'] =df['date'].dt.year
#checking nann
print(df.isnull().sum())
#removing null values 
df = df.dropna()
#To compare two stocks, both should follow the same date range of data. I found that some stocks had a smaller date range:
df.Name.value_counts()
#Let’s remove these sets of stocks to continue with further processing:
count_df=pd.DataFrame(df.Name.value_counts()[:470], columns=["Name", "Count"]).reset_index()
list_valid_shares=list(count_df["index"])
final_df=df[df.Name.isin(list_valid_shares)]
print(final_df.head())
#We have the data from 2013-2019. Our goal is to find the most similar stock for any specific year. We will take the data for the year 2018:
data_by_year=final_df.groupby("year")

data_2013=data_by_year.get_group(2013)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2013.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2013=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2013")
print(sc_2013)



data_2014=data_by_year.get_group(2014)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2014.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2014=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2014")
print(sc_2014)



data_2015=data_by_year.get_group(2015)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2015.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2015=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2015")
print(sc_2015)




data_2016=data_by_year.get_group(2016)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2016.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2016=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2016")
print(sc_2016)




data_2017=data_by_year.get_group(2016)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2017.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2017=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2017")
print(sc_2017)

data_2018=data_by_year.get_group(2018)
#Let’s make the date column the index and make our data pivot for comparing different stocks:
pivot_df=data_2018.pivot(index="date",columns="Name", values="close")
#Finding Similarities
#To find the similarities, we will use pandas’s corr method:
corr_mat=pivot_df.corr(method ='pearson').apply(lambda x : x.abs())
#select the top ten pairs to give us the top five relationships between stocks for pair trading:
sorted_corr = corr_mat.unstack().sort_values(kind="quicksort", ascending=False)
sc_2018=pd.DataFrame(sorted_corr, columns=["Value"])[470:475]
print("5 strongest pairs for every year 2018")
print(sc_2018)
