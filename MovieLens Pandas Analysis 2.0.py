
# coding: utf-8

# In[488]:


import pandas as pd 
import numpy as np 
import os
import matplotlib
import pandas.plotting as pdp
from pandas import *
import matplotlib.pyplot as plt
import statsmodels
from scipy.stats import sem


#Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded


# In[2]:


os.chdir('C:\\Users\\ShaRo\\Downloads\\ml-1m')


# In[3]:


#Data Loading

moviesDf = pd.read_csv('movies.dat', sep='::', engine='python', names = ['MovieID', 'Title', 'Genres'])
moviesDf['Genres'] = moviesDf['Genres'].apply(lambda x: x.split('|'))
moviesDf['Year'] = moviesDf['Title'].apply(lambda x: x[-5:-1])
moviesDf['Title'] = moviesDf['Title'].apply(lambda x: x[0:-7])
moviesDf=moviesDf.reindex(columns=['MovieID','Title','Year', 'Genres'])
moviesDf['Year'] = moviesDf['Year'].apply(pd.to_numeric)
moviesDf['Title'] = moviesDf['Title'].astype('category')
moviesDf['#Genres'] = moviesDf['Genres'].apply(lambda x: len(x))

usersDf = pd.read_csv('users.dat', sep='::', engine='python', names = ['UserID','Gender','Age','Occupation','Zip-code'])

ratingsDf = pd.read_csv('ratings.dat', sep='::', engine='python', names = ['UserID','MovieID','Rating','Timestamp'])
ratingsDf['Datetime'] = pd.to_datetime(ratingsDf['Timestamp'], unit='s')
ratingsDf = ratingsDf.drop(['Timestamp'], axis=1)
ratingsDf['Rating'] = ratingsDf['Rating'].apply(pd.to_numeric)


# In[4]:


Occupation = {0:  "other"
,1:"academic/educator"
,2:"artist"
,3:"clerical/admin"
,4:"college/grad student"
,5:"customer service"
,6:"doctor/health care"
,7:"executive/managerial"
,8:"farmer"
,9:"homemaker"
,10:"K-12 student"
,11:"lawyer"
,12:"programmer"
,13:"retired"
,14:"sales/marketing"
,15:"scientist"
,16:"self-employed"
,17:"technician/engineer"
,18:"tradesman/craftsman"
,19:"unemployed"
,20:"writer"}

usersDf['Occupation'] = usersDf['Occupation'].apply(lambda x:Occupation[x])
usersDf['Occupation'] = usersDf['Occupation'].apply(lambda x: str(x))


# In[5]:


moviesDf.head()


# In[6]:


usersDf.head()


# In[91]:


ratingsDf.head()


# In[19]:


moviesDf.describe()


# In[87]:


usersDf.describe()


# In[92]:


ratingsDf.describe()


# In[7]:


#Missing Values:
moviesDf.isnull().values.any()


# In[31]:


usersDf.isnull().values.any()


# In[32]:


ratingsDf.isnull().values.any()


# In[7]:


#Data Merging

ruDf = pd.merge(ratingsDf, usersDf, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

rmDf = pd.merge(ratingsDf, moviesDf, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

rmuDf = pd.merge(rmDf, usersDf, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)

rmuDf = rmuDf.drop(['MovieID'], axis=1)


# In[8]:


print('shape check:')
print(usersDf.shape)
print(moviesDf.shape)
print(ratingsDf.shape)
print(ruDf.shape)
print(rmDf.shape)
print(rmuDf.shape)


# In[9]:


#Data Analysis

rmuDf.corr()


# In[23]:


rmuDf.cov()


# In[22]:


rmuDf.describe()


# We can notice that there is'nt a significant correlation / covariance between the numeric variables.
# specificly, we'd look for rating variable and other explaining variables.

# In[10]:


print("Number of Users Rated the Movie:")
table = pivot_table(rmuDf, values=["Rating"],index=["Title"], aggfunc=lambda x:len(x))

table.rename(index=str, columns={"Rating" :"#Rating"} ,inplace=True)
flattened = pd.DataFrame(table.to_records())
rm11Df = pd.merge(rmuDf, flattened, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True, copy=True, indicator=False, validate=None)
rm11Df.head()

rm11Df.corr()


# We can see the #Rating column does correlates, but not much, with the rating column.

# # Time Analysis

# In[489]:


#Time Histogram

plt.figure(figsize=(12, 9))  
ax = plt.subplot(111)  
ax.spines["top"].set_visible(True)  
ax.spines["right"].set_visible(False)  
ax.get_xaxis().tick_bottom()  
ax.get_yaxis().tick_left()  
plt.xticks(fontsize=10, rotation='vertical')
plt.xlabel("Date", fontsize=10)  
plt.ylabel("Count", fontsize=10)  
plt.hist(list(ratingsDf['Datetime']), color="#3F5D7D", bins=150);


# this analysis shows us the distribution of instances across the timeline.
# most of the ratings occured between May 2000 and Januray 2001. with 2 of 80k and 170k at August 2000 and December 2000.
# this alone can't determine any seasonality, and we can't disqualify any dataset collection effects.

# In[20]:


#Avg Rating over Years
# under the assumption that age distributes normally.
table = pivot_table(rmuDf, values=["Rating"],index=["Year"], aggfunc=[np.average,np.std,lambda x:len(x)/10000])
print(table.head())

table.plot(figsize = (10,5))
plt.axhline(0, color='k')


# As we from the plots above, most of the ratings are between 5.2000 - 1.2001. we'd like to look for difference or effect in movies between those years' rating and others.
# 
# The second plot, which shows us statistics of average and standard deviation between rows in this date, together with a normalized count, helps us deduce there isn't any seasonality or trends we'd like to look further.

# The next analysis focused on the distribution of genres across time.
# We'd like to find any clues of influence to the ratings in Genres' choise in every year.

# In[25]:


def explode(df, columns):
    idx = np.repeat(df.index, df[columns[0]].str.len())
    a = df.T.reindex_axis(columns).values
    concat = np.concatenate([np.concatenate(a[i]) for i in range(a.shape[0])])
    p = pd.DataFrame(concat.reshape(a.shape[0], -1).T, idx, columns)
    return pd.concat([df.drop(columns, axis=1), p], axis=1).reset_index(drop=True)

table = rmuDf.ix[:,["Year","Genres","Rating"]]
result = explode(table,['Genres'])
print("\n\nThere can be various amount of genres permutations in the Dataset:\n")
print(table.head())
print("\n\nIndividual - Genres' Pivot-Table:\n")
print(result.head())

mostFrequent = result.groupby('Year')['Genres'].agg(lambda x: x.value_counts().index[0])
print("\n\nMost frequently appeared genres choise in that year\n")
print (mostFrequent.tail())


# In[215]:


result3 = result.groupby('Year')['Genres'].nunique()
result3.sort_index().plot(figsize = (10,5), title = "Number of UNIQUE Genres Per Year")
plt.axhline(0, color='k')
print("\nRemembering the distribution of the ratings across the timeline,there is'nt sufficient evidence to suggest an apparent effect in the number of unique genres choosen that year. Although we can notice the growing number of choosen unique genres, a growing diversity might mean bigger variation in rating, that we'd like to look further")


# In[36]:


result.head()


# In[54]:


result4 = pivot_table(result, values=["Rating"],index=["Year", "Genres"], aggfunc=[np.average,np.std])
print("Pivot: statistics of Ratings across year and genres")
print("\n\n")
print(result4.head())
print(result4.tail(20))


# In[60]:


result.head()


# In[164]:


result5 = result.groupby(['Year','Genres']).size().groupby('Year').agg(np.std)
result6 = result5.reset_index()
result6.columns = ['Year','std Rating']
print(result6.tail(15))


# In[264]:


result7 = result.groupby(['Year','Genres']).agg(['count']).reset_index()
result8 = result.groupby(['Year','Genres']).agg([np.average]).reset_index()
#result7['Year-Genres'] = result7['Year'].map(str) +'-'+ result7['Genres']

result7 = result7.drop(['Genres'], axis=1)
result8 = result7.drop(['Genres'], axis=1)
result7.columns = result8.columns.get_level_values(0)
result8.columns = result8.columns.get_level_values(0)
result8 = result8.groupby(['Year']).agg(['std'])
result8.columns = result8.columns.get_level_values(0)

result7 = result7.groupby(['Year']).agg(['sum'])

result7.columns = result7.columns.get_level_values(0)

result7 = result7.reset_index()
result8 = result8.reset_index()
print('Counts of Ratings Per Year:\n')
print(result7.tail())
result7.columns = ['Year', 'ratingsCount']
result8.columns = ['Year', 'std-avgRating among genres']
print('\nstd of Ratings Per Year:\n')
print(result8.tail())

result9 = pd.merge(result7 ,result8, on='Year')
print(result9.tail())
ax = result9.plot(figsize = (10,5), title = "Rating Counts Per Genre - std among Genres Per Year")
plt.axhline(0, color='k')
ax.set_xlabel("Index")

print ("\nCorrelation calc: \n")
print(result9.corr())

print("\nPutting side by side the graph of standard deviation of average rating per genre in a year and the number of ratings and calculating it's correlation can explain the variation in average rating across genres, and by this we eliminate the effect of genres in the dataset.\nBy this we'll choose to deduce there isn't any significant effect in genres-years variables")


# In[9]:


print ("Most Rated Genres Frequencies by Year")
mostFrequent.value_counts()


# In[10]:


table = pivot_table(moviesDf, values=["MovieID"],index=["Year"], aggfunc=lambda x:len(x))
table.sort_index().plot(figsize = (10,5), title = "Number of Movies in DB Per Year")
plt.axhline(0, color='k')


# In[59]:


#Genre Histogram
table = moviesDf.ix[:,["MovieID","Genres"]]
table = explode(table,['Genres'])
g = table["Genres"].hist(bins = table["Genres"].nunique())
for tick in g.get_xticklabels():
        tick.set_rotation(90)


# # Age Analysis

# In[284]:


agePivot = usersDf.groupby('Age', as_index=True).count()
print('count of Users:')
print(agePivot['UserID'])
print("The average Age of user is: "+str('{0:.3g}'.format(usersDf['Age'].mean()))+" and variance: "+str('{0:.5g}'.format(usersDf['Age'].var())))
print("We'd like to look further for Age analysis: average and var of ratings in each age category")

agePivot['UserID'].plot.bar()
plt.axhline(0, color='k')


# In[322]:


table = pivot_table(rmuDf, values=["Rating"],index=["Age"], aggfunc=[np.median,np.average,np.var,lambda x:len(x)])
print(table.head())
table.columns = table.columns.get_level_values(0)
#print(table.columns)
table.drop(['<lambda>'], axis=1).plot.bar()

table = pivot_table(rmuDf, values=["Age"],index=["Rating"], aggfunc=[np.median, np.average,np.var,lambda x:len(x)])
print('\n')
print(table.head())
table.columns = table.columns.get_level_values(0)
table.drop(['<lambda>'], axis=1).plot.bar()

#scipy.stats.ttest_rel(rmuDf['average'], cat2['values'])
print("\nSince the median is stable across all age group we can't identify a age-rating abnormalziations")


# # Occupation Analysis

# In[114]:


#Occupation Histogram:

g = usersDf['Occupation'].hist(bins=20)
for tick in g.get_xticklabels():
        tick.set_rotation(90)


# In[323]:


# Checking for Difference of Rating among Occupation:
# under the assumption that age distributes normally.
table = pd.crosstab(rmuDf['Occupation'],rmuDf['Gender'], values=rmuDf['Rating'], aggfunc=[np.median, np.average, np.std])
table.sort_values(list(table.columns.values), ascending=False)

print(table)

table.plot.bar(figsize = (10,5))
plt.axhline(0, color='k')


# # Gender Analysis

# In[350]:


print('Differentiating Gender:\n')
genderPivot = usersDf.groupby('Gender', as_index=True).count()['UserID']
print(genderPivot)
print('\nthere are as much as 2.5 times male indentified users than female indentified users.\n')
genderPivot2 = rmuDf.groupby('Gender', as_index=True).count()['UserID']
print(genderPivot2)
print('\nthere are as much as 3 times male records than female records.\n')

print('\nWe will try to see whether we should treat them differently and whether a further analysis is needed:\n      if male and female rate differently.\n')
table = pivot_table(rmuDf, values=["Rating"],index=['Title'], columns=["Gender"], aggfunc=np.average)
table.sample(35).plot.bar()
plt.axhline(0, color='k')

table2 = table.var(axis=1).sum()
print("The total variance between female and male (summed for all movies) is "+str('{0:.5g}'.format(table2)))
print("\n which in the meaning of ratings, is not much and we further deduce to keep them intact.")


# In order to understand the effect of rating over Gender & Age we'll use cross-tabulation:

# In[352]:


pd.crosstab(rmuDf['Age'],rmuDf['Gender'], values=rmuDf['Rating'], aggfunc=[np.average,np.var,np.median])


# It can be inferred that:<br>
# 1)female users tend to rate higher for all age groups (with little significane).<br>
# 2)the first age category's variance is higher for both male and female.<br>
# 3) 50 - 56 and 56+ age categories average - rating is higher.

# In[374]:


from scipy.stats import ttest_ind

cat1 = rmuDf[rmuDf['Age']==1]
cat2 = rmuDf[rmuDf['Age']!=1]

print("We'll use T-test to determine if two sets of data are significantly different from each other:\n1-18 Age olds' rating and the others:      \nA small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.\n")

print(ttest_ind(cat1['Rating'], cat2['Rating']))


# In[376]:


print("Lastly, checking the minimum and maximum values of the distribution of ratings:")
plt = rmuDf.boxplot(column='Rating', by = ['Gender'])
ttl = plt.title
ttl.set_position([.5, 1.15])


# In[375]:


plt = rmuDf.boxplot(column='Rating', by = ['Age'])
ttl = plt.title
ttl.set_position([.5, 1.15])


# As seen before, although the median doesn't change, people of age tend to rate generously.

# # Movies Analysis

# In[470]:


# number of users to rate a movie
table = pivot_table(rmuDf, values=["Rating"],index=["Title"], aggfunc=[np.average,lambda x:len(x), np.var, np.median]).dropna()
table.columns = table.columns.get_level_values(0)
#top 90% percentile movies - dropping 10% of the movies with least number of ratings
table = table[table['<lambda>']> table['<lambda>'].quantile(0.1)]
print(table.sort_values(by='<lambda>').head)
plot1 = table.sort_values(by='<lambda>').drop(['<lambda>'], axis=1)['average'].plot(figsize=(17,4))
print("from the plot we recognize 2 sets of movies, by average rating, \n low rating and high rating movies, so we'll try to differentiate them and look further. \n the premise behind is there are some extraordinary movies with higher than average rating and they behave differently.")
plot1.axes.get_xaxis().set_ticks([])
plot1.axes.set_xlabel('sampled movies sorted by num of raters')
plot1.axes.set_ylabel('movie average rating')


# In[469]:


plot2 = table.sort_values(by='<lambda>').drop(['<lambda>'], axis=1)['var'].plot(figsize=(17,4))
plot2.axes.get_xaxis().set_ticks([])
plot2.axes.set_xlabel('sampled movies sorted by num of raters')
plot2.axes.set_ylabel('movie var rating')


# In[471]:


plot3 = table.sort_values(by='<lambda>').drop(['<lambda>'], axis=1)['median'].plot(figsize=(17,4))
plot3.axes.get_xaxis().set_ticks([])
plot3.axes.set_xlabel('sampled movies sorted by num of raters')
plot3.axes.set_ylabel('movie median rating')


# In[480]:


table['c'] = pd.qcut(table['average'], 7, labels=list(range(7)))
#print(tableH.to_frame())
#tableC = pd.merge(table,tableH.to_frame(), how='inner')
print(table.groupby('c').agg(np.average))
#print(table[table['c']=='low'])


# Analyzing the 3 plots and table, we can see a clear trend: as number of raters rises, higher the rating.

# In[485]:


print("distance of rating's timestamp-datetime from movie's year")
#table = pd.concat([pd.to_datetime(rmuDf['Datetime'] - (pd.to_datetime(rmuDf['Year'], format='%Y')+pd.Timedelta(182.5, unit='d'))), rmuDf['Rating']], axis=1)
table = rmuDf
table['dateDist'] = pd.to_datetime(rmuDf['Datetime'] - (pd.to_datetime(rmuDf['Year'], format='%Y')+pd.Timedelta(182.5, unit='d')))
#table.columns = ['dateDist','Rating']
table['dateDist'] = table['dateDist'].values.astype(np.int64)
table.corr()


# As we can see there is some correlation between rating the distance between rating's timestamp and movie's date.
