
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
from pandas import *
import numpy as np 
import os
import pandas.plotting as pdp
import matplotlib.pyplot as plt
import statsmodels
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
os.chdir('C:\\Users\\ShaRo\\Downloads')


# # Data Loading

# In[2]:


movieTitleDf = pd.read_csv('movie_titles.csv', engine='python', names = ['movieID','RYear','movieTitle', '1','2','3'], usecols=['movieID','RYear','movieTitle'])


# In[3]:


ccdf = pd.DataFrame(columns = ['cusID', 'Rating', 'Datetime','movieID'])
for i in range(4):
    cdf = pd.read_csv('combined_data_'+str(i+1)+'.txt', engine='python', nrows=500000, names = ['cusID', 'Rating', 'Datetime','movieID'])
    ccdf = pd.concat([ccdf, cdf])
ccdf = ccdf.reset_index()


# We'll be using only 500k rows to decrease the running time.

# ## MovieID Handling

# In[4]:


for i, row in ccdf.iterrows():
    if row.isna()['Rating'] == True:
        mID = row['cusID'][0:-1]
    else:
        ccdf.iat[i,4] = mID
    
ccdf = ccdf.dropna()
ccdf = ccdf.drop(['index'],axis=1)


# ## Check for na leftovers

# In[9]:


ccdf.isnull().any()


# In[10]:


movieTitleDf.isnull().any()


# In[11]:


movieTitleDf[movieTitleDf.isnull()['RYear'] == True]


# In[12]:


list(movieTitleDf[movieTitleDf.isnull()['RYear'] == True]['movieID'])


# In[13]:


len(movieTitleDf[movieTitleDf.isnull()['RYear'] == True])


# #### Although there are only 7 instances and we'd wish to delete them, since there are many instances that correspond to those movies, we'd like to impute the average value.

# In[14]:


int(movieTitleDf['RYear'].mean())


# ## Col types handling

# In[5]:


movieTitleDf['RYear'] = movieTitleDf['RYear'].fillna(movieTitleDf['RYear'].mean())


# In[6]:


movieTitleDf['RYear'] = movieTitleDf['RYear'].astype('int64')


# In[7]:


movieTitleDf.dtypes


# In[7]:


ccdf['cusID'] = pd.to_numeric(ccdf['cusID'], downcast='signed')
ccdf['Rating'] = pd.to_numeric(ccdf['Rating'], downcast='signed')
ccdf['Datetime'] = pd.to_datetime(ccdf['Datetime'])
ccdf['movieID'] = pd.to_numeric(ccdf['movieID'], downcast='signed')
ccdf.dtypes


# In[8]:


ccdf = pd.merge(ccdf, movieTitleDf, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)


# In[20]:


ccdf.describe()


# In[21]:


ccdf.columns


# ## Ratings Histogram

# In[22]:


plt.figure(figsize=(5, 5))
ax = ccdf["Rating"].groupby(ccdf["Rating"]).count().plot(kind="bar")

plt.xticks(fontsize=10, rotation='vertical')
plt.xlabel("Rating", fontsize=10)  
plt.ylabel("Count", fontsize=10)  


# In[29]:


ccdf["Rating"].agg([np.mean, np.median, np.var])


# # Date Time Analysis

# In[33]:


plt.figure(figsize=(15, 5))
ax = plt.subplot(111)
ax.spines["top"].set_visible(True)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.xticks(fontsize=10, rotation='vertical')
plt.xlabel("DateTime", fontsize=10)  
plt.ylabel("Count", fontsize=10)
plt.title("Number of observations across time")
plt.hist(list(ccdf['Datetime']), color="green", bins = 400);

print("Next, we'll want to distinguish a trend in corresponds to the distribution of observations across time.")


# In[51]:


print("Movie's Year Values: \n"+str(ccdf.RYear.unique()))
print("\nnumber of unique values: "+str(len(ccdf.RYear.unique())))


# In[74]:


table = pivot_table(ccdf, values=["Rating"], index=["Datetime"], aggfunc=[np.mean, np.var, np.median, lambda x:len(x)])
table.plot(subplots=True, figsize=(15,5));


# We can't learn anything from this analysis, since the variance is lowering with growing number of observations and the means stays the same.
# the jumps in the median happen because of the "holes" mentioned earlier.

# In[35]:


#DateTime Year histogram
ccdf["Datetime"].groupby(ccdf["Datetime"].dt.year).count().plot(kind="bar")


# In[9]:


ccdf['Datetime-Year'] = ccdf['Datetime'].dt.year
table = pivot_table(ccdf, values=["Rating"],index=['Datetime-Year'], aggfunc=np.average)
table.plot(kind="bar")


# In[106]:


ccdf.boxplot(column = ['Rating'], by = ['Datetime-Year'], figsize= (15,2));


# It can be inferred the people tended to right higher in more advanced years. 2005's Raters rate highest. <br>Box-Plot tells us, it was not a coincidence. the rating falls between the same limits.

# In[36]:


#DateTime Month histogram
ccdf["Datetime"].groupby(ccdf["Datetime"].dt.month).count().plot(kind="bar")


# In[34]:


#DateTime Day histogram
ccdf["Datetime"].groupby(ccdf["Datetime"].dt.day).count().plot(kind="bar")


# We don't notice anything too odd for to explore further out of those analysis. we would've if we saw big difference in the data on those, views.

# In[79]:


#Released Year histogram
plt.figure(figsize=(15, 5))
ccdf["RYear"].groupby(ccdf["RYear"]).count().plot(kind="bar");
print("as could be expected, most of the ratings are for newer movies, but there is too many 'incongruencies' in the graph, we'd like too look further.and bear that in mind for further analysis.\ncan be due to good years of cinema - biased raters-movies, altered database, some popular movies or other reasons.")


# ## Rolling Statistics of Rating

# In[100]:


plt.figure(figsize=(15, 5))
#Against Rating's Submitted Timestamp
table1 = pivot_table(ccdf, values=["Rating"],index=["Datetime"], aggfunc=np.average).rolling(window=12, center=False).mean()

#Against Movie's Released Year
table2 = pivot_table(ccdf, values=["Rating"],index=["RYear"], aggfunc=np.average).rolling(window=12, center=False).mean()


plt.xlabel("DateTime", fontsize=10)  
plt.ylabel("Cummulative Average Rating", fontsize=10)
plt.title("Cummulative Average Rating across Rating's Datetime")
plt.plot(table1, color="blue");


# In[99]:


plt.figure(figsize=(15, 5))
plt.xlabel("RYear", fontsize=10)  
plt.ylabel("Cummulative Average Rating", fontsize=10)
plt.title("Cummulative Average Rating across Movie's year")
plt.plot(table2, color="yellow");


# Against Rating's Timestamp we can see a clear upward trend: newer submitted rating tends to be higer. regardless of movie's year of release, which behaves differently, there are 2 peaks in ratings: 1970's and 2000's Movies. <br><br>Next, We'll want to recall the Rating's average for movie's year in order to understand the bigger picture.

# In[101]:


table = pivot_table(ccdf, values=["Rating"],index=['RYear'], aggfunc=np.average);
plt.figure(figsize=(15, 5))
plt.xlabel("RYear", fontsize=10)  
plt.ylabel("Average Rating", fontsize=10)
plt.title("Average Rating across Movie's year")
plt.plot(table);


# We can't be sure of anything special because on the small number of observations in some year's (the gaps in between the graph)

# In[202]:


plt.matshow(ccdf.corr())
ccdf.corr()


# As seen before, Datetime-year slightly correlates with the rating. and the ratings with the movie's themselves (which won't help us to predict new movie's ratings.<br>
# another conclusion we can arrive at is there is no correlation between the customers and their ratings. 

# In[120]:


table = pd.crosstab(columns=ccdf['Datetime-Year'],index=ccdf["RYear"], values=ccdf['Rating'], aggfunc=np.average).round(2)
print(table.sample(15))


# <br><br>Counts-cross table:

# In[121]:


table = pd.crosstab(columns=ccdf['Datetime-Year'],index=ccdf["RYear"], values=ccdf['Rating'], aggfunc=lambda x:len(x))
print(table.sample(15))


# There are too many Nan values due to 0 observations for the cross of ratings' year and movie's year. (not rated movie's year)

# # UserID  & MovieID Analysis

# In[100]:


ccdf['cusID'].groupby(ccdf['Rating']).nunique()


# In[147]:


table = pivot_table(ccdf, values=["Rating"],index=["cusID"], aggfunc=[lambda x: len(x), np.average, np.var], margins=True)
print(table.sample(15))
print("\naverage number of ratings per user: %.2f" % np.mean(table['<lambda>','Rating'].round(2)))
print("with min and max of: "+ str((min(table['<lambda>','Rating']),max(table['<lambda>','Rating']))))
print("average rating between users : %.2f" % np.mean(table['average','Rating'].round(2)))
print("average variance between users' rating: %.2f" % np.mean(table['var','Rating'].round(2)))


# In[149]:


sns.distplot(pivot_table(ccdf, values=["Rating"],index=["cusID"], aggfunc=np.average))
plt.xlabel("Average Rating", fontsize=10)  
plt.ylabel("Probability", fontsize=10)  
plt.suptitle('Distribution plot for average rating of user', fontsize=16)


# In[163]:


print("Percent of users rated only once in the dataset: %.2f" % (len(table[table['<lambda>','Rating'] == 1])/len(table['<lambda>','Rating'])))
table2 = table[table['<lambda>','Rating'] == 1]['average','Rating']
print("that explain the odd looking plot. thus we'll try to omit those rows.")
sns.distplot(table2)
plt.xlabel("Average Rating", fontsize=10)  
plt.ylabel("Probability", fontsize=10)  
plt.suptitle('Distribution plot for average rating of user', fontsize=16)


# This time the plot looks standard for this kind of population.

# ### We'll use the same kind of analyzation for Movies in order to find oddities:

# In[175]:


sns.distplot(pivot_table(ccdf, values=["Rating"],index=["movieID"], aggfunc=np.average))
plt.xlabel("Average Rating", fontsize=10)  
plt.ylabel("Probability", fontsize=10)  
plt.suptitle('Distribution plot for average rating of movie', fontsize=16)


# In[10]:


table = pivot_table(ccdf, values=["Rating"],index=["movieID"], aggfunc=lambda x:len(x))
sns.distplot(table)
plt.xlabel("Number of observations", fontsize=10)  
plt.ylabel("Probability", fontsize=10)
plt.suptitle('Distribution plot for number of observations for movie', fontsize=16)


# trying to fit a distribution to the above distplot

# In[13]:


ccdf.columns


# In[31]:


#Movie Quantiles
table = pd.DataFrame(pivot_table(ccdf, values=["Rating"],index=["movieID"], aggfunc=np.average).to_records())['Rating']
st = stats.probplot(table, plot=plt)
print("Rating's across Movies: \nslope, intercept, r^2 accordingly: "+str(st[1]))
print("from this analysis it can be guaranteed that Ratings across movies are distributed normally and we need'nt look further")


# In[33]:


#Customer Quantiles
table = pd.DataFrame(pivot_table(ccdf, values=["Rating"],index=["cusID"], aggfunc=np.average).to_records())['Rating']
st = stats.probplot(table, plot=plt)
print("Rating's across Customers: \nslope, intercept, r^2 accordingly: "+str(st[1]))
print("from this analysis it can be guaranteed that Ratings across cutomers are distributed normally and we need'nt look further")
print("from the plot it can be seen that, like we earlier, there are some extreme rating customers")


# ## Rating / Year DF-Frequency Table 

# In[35]:


table = pivot_table(ccdf, values=["Rating"],index=["cusID"], aggfunc=[np.average,lambda x:len(x),np.std])
table.columns = ['Avg','Customers Count','Std']
table2 = pivot_table(ccdf, values=["Rating"],index=["cusID"], aggfunc=lambda x:stats.mode(x)[0][0])
table2.columns = ['Mode']

table3 = pd.merge(pd.DataFrame(table.to_records()), pd.DataFrame(table2.to_records()), how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

table3.sample(25)


# #### Customers tend to rate differently, different bias (users who tend to right higher or lower than average), deviation (in their ratings),  moreover there are too many who users don't have enough instances to even be considered and we would like to remove from the dataset.

# In[287]:


table = pivot_table(ccdf, values=["Rating"],index=["movieID"], aggfunc=[np.average,lambda x:len(x),np.std])
table.columns = ['Avg','Customers Count','Std']
table2 = pivot_table(ccdf, values=["Rating"],index=["movieID"], aggfunc=lambda x:stats.mode(x)[0][0])
table2.columns = ['Mode']

table3 = pd.merge(pd.DataFrame(table.to_records()), pd.DataFrame(table2.to_records()), how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

table3.sample(25)


# #### Movies tend to rated differently, they have higher instances and the standard deviation is more or less the same for all movies.

# ## Number of unique customers to rate movie analysis

# In[43]:


pivot_table(ccdf, values=["cusID"],index=["movieID"], aggfunc= lambda x: len(x.unique())).hist(bins=50)
plt.xlabel("Num of customers", fontsize=10)  
plt.ylabel("Count of Movies", fontsize=10)  
plt.suptitle('Histogram plot: count cutomers per movie', fontsize=16)


# ## Number of unique movies to be rated by customer

# In[40]:


bins= range(20)
pivot_table(ccdf, values=["movieID"],index=["cusID"], aggfunc= lambda x: len(x.unique())).hist(bins=bins, edgecolor="k")
plt.xlabel("Num of Unique rated Movies", fontsize=10)  
plt.ylabel("Count Customers", fontsize=10)
plt.xticks(bins)
plt.suptitle('Histogram plot: count movies per customers', fontsize=16)


# This side by side analysis shows us, on the one hand an evenly distributed number of users to rate a movie, that we'd expected to see, and on the other hand, complicated histogram of customers or popular movies tend to be watched and rated more or other reason.

# In[288]:


# Rating's Count
table = pivot_table(ccdf, values=["cusID"],index=["Rating"], aggfunc= lambda x: len(x))
table.loc['Total'] = table.sum()
table


# # In Conclusion:

# with the data in hand we could deduce that each cutomer acts differently for each movie but with a specific "rating bias" to begin with.
# In order to deduce more information and analyze the data further we'll need to transform the data and fit different ditributions to each 'group' of measurements.
# other than that, on the first look, that data distributes normally and evenly.
