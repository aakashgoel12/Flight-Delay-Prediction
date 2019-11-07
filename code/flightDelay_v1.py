
# coding: utf-8

# ****
# __author__ = Aakash Goel
# 
# __python_version__ = 2.7
# 
# __program_version__ = v1.0
# 
# __date__ = 04-November-19
# 
# __email__ = aakashgoel12@gmail.com
# ****

# In[117]:

from __future__ import division
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.options.display.max_columns = 150
import numpy as np
import os,gc,holidays,datetime
from math import radians, sin, cos, acos,ceil

#sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error,mean_absolute_error

## Modelling imports
from sklearn.ensemble import GradientBoostingRegressor
from catboost import Pool,CatBoostClassifier,CatBoostRegressor
# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')


# ## STEP I: Setting up Problem

# Lets set some **background** first:
# 
# Customer browsing through some flight booking website and want to book flight for some specific date, time, source and destination.
# 
# **IDEA:** If during flight booking, we can show to customer whether the flight he/she considering for booking is likely to arrive on time or not. Additionaly, if flight is expected to delay, also show delayed time.
# 
# If customer know that the flight is likely to be late, he/she might choose to book another flight.
# 
# From **Modelling Propective**, need to set two goals:
#     
#     GOAL I: Predict whether flight is going to delay or not.
#     GOAL II: If flight delays, predict amount of time by which it delays.

# ## STEP II:Getting Data

# Data expo 2009 (http://stat-computing.org/dataexpo/2009/the-data.html) provides flight data from year 1987 to year 2008.
# 
# For this notebook, used data for year 2004. 
# 
# Link for downloading data: http://stat-computing.org/dataexpo/2009/2004.csv.bz2
# 
# Also, using **external data** like airport, carrier, plane details. (http://stat-computing.org/dataexpo/2009/supplemental-data.html)
# 

# In[2]:

display(os.listdir('../input/'))


# In[3]:

get_ipython().system(u' bzip2 -dk ../input/2004.csv.bz2')


# In[4]:

## check if unzipped file also exists with name '2004.csv'
display(os.listdir('../input/'))


# ## STEP III: Clean and Prepare data

# All Ideas regarding feature engineering:
# 
# ****
# **Scheduled departure and arrival time** (Feature - 1)
#     > convert scheduled departure and arrival time into hour
# **Source Airport (Origin)** (Feature - 2)     
#     > Get Total number of flights pre and post one hour from source airport for specific day    
#     > Get historical average delay for each source airport
# **Flying Date**  (Feature - 3)
#     > Using holiday calendar of USA, create binary variable if flying day is holiday or not
#     > Using feature day of week, create feature weekend i.e get if it is saturday/sunday or not
#     > create extended weekend variable with buffer of 1 day i.e. if day of week is Friday or Monday then its extended weekend
#     > create week number from day of month
# **Source and destination airport ** (Feature - 4)
#     > From source and destination airport code, Get state, city,latitude and longitude
#         > Using lat,long for source and destination airport, create distance as feature between airports
#         > Using State, create feature 'same_state' i.e. if both source and destination lies in same state
#         > Using city, create city level features (like population, forecasted weather details like rainy, cloudy)
#         > Get count of airports in each state i.e. state level feature
# ** Unique Carrier (company)** (Feature - 5)       
#     > create historical average delay as feature for each carrier (company)
#     > create percentages of flights for each carrier (company)
# **Tail Number** (Feature - 6)    
#     > Using TailNumber, get feature like manufacturing year (Age of aircraft), engine type. 
# 

# In[5]:

data = pd.read_csv('../input/2004.csv')


# In[6]:

display(data.shape)


# In[162]:

display(data.sample(frac=0.1,random_state=42).head(3))


# **Observation:** Column 'ArrDelay' contains continuous values and it contains negative value when flight arrive before scheduled arrival time and positive value when flight get delay. 

# #### Functions used in below feature engineering

# In[8]:

# display number of null values for each column if exist 
def check_null_values(df):
    if df.isnull().values.any():
        display(df.isnull().sum())
    else:
        display("Null values don't exist")

# convert year, month, day to date
def get_date(row):
    return datetime.datetime(year=row['Year'], month=row['Month'], day=row['DayofMonth'])

# Getting historical average delay; parameters column_name, data, buffer (for historical)
def hist_avg_delay(row, col_name,df, buffer_days):
    df = df.ix[0:row.name-1]
    df = df[df[col_name]==row[col_name]]
    df = df[(row['date']-df['date'])<=pd.to_timedelta(buffer_days, unit='d')]
    return round(df['ArrDelay'].mean(),2)

# find if holiday exist for specific day
def get_holiday(x,holiday_list):
    if x in holiday_list:
        return 1
    else:
        return 0

# find if day is saturday/sunday or not
def get_sat_sun(x):
    if x==6 or x==7:
        return 1
    else:
        return 0

# find if day is extended weekend or not
def extended_weekend(x):
    if x==1 or x==5:
        return 1
    else:
        return 0

# Get actual distance between source and destination airport
def get_actual_distance(row):
    slat = radians(row['lat_x'])
    slon = radians(row['long_x'])
    elat = radians(row['lat_y'])
    elon = radians(row['long_y'])
    dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
    return dist

def check_same_state(row):
    if row['state_x']==row['state_y']:
        return 1
    else:
        return 0

# number of unique values
def get_unique_values(df,column):
    display("No of unique values for column {}: {}".format(column,df[column].nunique()))


# #### check CRSDepTime and CRSArrTime column (Feature-1)

# In[9]:

get_unique_values(data,'CRSDepTime')
get_unique_values(data,'CRSArrTime')


# Number of unique values are quite high in both columns and can have negative impact on modelling exercise. Format of column values are **hhmm** i.e. 4 digit number.
# 
# We can reduce the number of unique values by binning i.e. dividing every number in both column by 100 and idea behind it is that it doesn't matter whether flight departs at **10:10 A.M. or 10:20 A.M.** as both led to **10**. 

# In[10]:

data['CRSDepTime'] = data['CRSDepTime']//100
data['CRSArrTime'] = data['CRSArrTime']//100

get_unique_values(data,'CRSDepTime')
get_unique_values(data,'CRSArrTime')


# Ideally, should have 24 unique values (0 to 23) only but CRSArrTime have 25 unique values after binning. Lets see reason for this and if we can reduce 25 to 24.

# In[11]:

display(data['CRSDepTime'].unique())


# 24 is one of unique value in column CRSArrTime. lets map it to 0 as **hhmm 2400 and 0000** refer to same.

# In[12]:

data['CRSArrTime'] = data['CRSArrTime'].replace({24:0})

get_unique_values(data,'CRSDepTime')
get_unique_values(data,'CRSArrTime')


# In[13]:

# (x.value_counts(normalize=True)*100).reset_index().sort_values(by='index',ascending=False).set_index('index')


# #### Get no of flights from: pre&post 1 hour (Feature - 2)

# **Idea** is to get the total number of flights pre and post one hour for source airport for a specific day,time.
# 
# Example - Let say, want to find total number of flights pre and post one hour for 
#     > source airport 'ORD'
#     > date (12-Jan-2004)
#     > time 6 A.M.

# In[14]:

## convert year,month,day in to date
data['date'] = data.apply(get_date,axis=1)
data['date'] = pd.to_datetime(data['date'])
gpby_colums = ['date','Origin','CRSDepTime']
data_gpby = data.groupby(gpby_colums).agg({'FlightNum':'count'}).reset_index().rename(columns={'FlightNum':'flight_count'})
data_gpby['pre'] = data_gpby['CRSDepTime']+1
data_gpby['post'] = data_gpby['CRSDepTime']-1


# In[15]:

## getting no of flights before 1 hour
data = pd.merge(data,data_gpby[['date','Origin','pre','flight_count']],how='left',               left_on = gpby_colums,right_on = ['date','Origin','pre'])
## getting no of flights after 1 hour
data = pd.merge(data,data_gpby[['date','Origin','post','flight_count']],how='left',               left_on = gpby_colums,right_on = ['date','Origin','post'])
data = data.fillna({'flight_count_x':0,'flight_count_y':0})
## summation of both pre and post 1 hour
data['flight_pre_post_count'] = data['flight_count_y']+data['flight_count_x']
data.drop(['pre','post','flight_count_y','flight_count_x'],inplace=True,axis=1)

# validation of above logic
t=6;yr=2004;month=1;day=12;origin='ORD'
assert data[(data['Year']==yr)&(data['Month']==month)&(data['DayofMonth']==day)     &(data['Origin']==origin)&(data['CRSDepTime']==t)]['flight_pre_post_count'].unique()[0]==    data[(data['Year']==yr)&(data['Month']==month)&(data['DayofMonth']==day)&(data['Origin']==origin)                &(((data['CRSDepTime']==t+1))|(data['CRSDepTime']==t-1))].shape[0]


# In[16]:

display(data.head(3))


# #### Historical average Delay per airport 

# Idea: Get historical delay for each source airport. Taken buffer of 90 days i.e. to get historical data, use past 90 days data only. 
# 
# **Caution**: Data leakage should be avoided.

# In[17]:

gpby_col = ['Origin','date']
## get average delay for each airport at date level
date_avg_delay = data.groupby(by=gpby_col).agg({'ArrDelay':'mean'}).reset_index()
date_avg_delay = date_avg_delay.sort_values(by='date')
date_avg_delay.reset_index(inplace=True)
## get historical average delay for each airport
date_avg_delay['hist_avg_delay_origin'] = date_avg_delay.apply(hist_avg_delay, df=date_avg_delay,                                                               col_name='Origin',buffer_days=90,axis=1)
## Merge historical average delay at day, airport level with complete data 
data = pd.merge(data,date_avg_delay[['Origin','date','hist_avg_delay_origin']],how='left',left_on = gpby_col,right_on=gpby_col)


# In[18]:

pd.isnull(data['hist_avg_delay_origin']).sum()


# In[19]:

data.dropna(subset=['hist_avg_delay_origin'],inplace=True)


# #### Feature on flying date  (Feature - 3)

# In[21]:

# Unique Years and holidays
unique_years = data['Year'].unique()
holiday_list = holidays.UnitedStates(years=unique_years).keys()

# Get if day is holiday or not
data['holiday'] = data['date'].apply(get_holiday,holiday_list=holiday_list)
# Get if day is saturday/sunday or not
data['weekend'] = data['DayOfWeek'].apply(get_sat_sun)
# Get if extended weekend or not
data['extended_weekend'] = data['DayOfWeek'].apply(extended_weekend)
# Get week number from day of month
data['week_number'] = (data['DayofMonth']/7).apply(ceil)


# #### check Airport code (Origin and Dest) - Get External data (Feature - 4)

# In[22]:

airport = pd.read_csv('../input/airports.csv')
display(airport.head(2))

# count no of airports at state level
airport = pd.merge(airport,airport.groupby('state')['iata'].count().reset_index().rename(columns={'iata':'no_airports'}),how='inner',on='state')
display(airport.head(3))


# In[23]:

# Count of airports at state level
plt.figure(figsize=(20,6))
airport.groupby('state')['iata'].count().plot.bar(title='Count of Airports at state level')


# In[24]:

# verify if all airport codes present in external data
display(len(set(data['Origin']).difference(set(airport['iata']))))
display(len(set(data['Dest']).difference(set(airport['iata']))))


# In[25]:

## Getting details for source airport
data = pd.merge(data,airport[['iata','city','state','country','lat','long','no_airports']],how='left',left_on='Origin',right_on='iata')
display(data.head(2))
## Getting details for destination airport
data = pd.merge(data,airport[['iata','city','state','country','lat','long']],how='left',left_on='Dest',right_on='iata')
display(data.head(2))


# In[26]:

pd.isnull(data[['lat_x','lat_y']]).sum()


# In[27]:

# if we drop here, not a significant number of loss in rows
data.dropna(subset=['lat_x','lat_y'],inplace=True)
display(data.shape)


# Distance is given in data but it may or may not be actual distance. let say if flight get diverted, distance will not be actual one.  

# In[28]:

# calculate actual distance b/w origin airport and destination airport
data['actual_distance'] = data[['lat_x','lat_y','long_x','long_y']].apply(get_actual_distance,axis=1)


# In[29]:

# check if any other country than USA
display(data['country_x'].unique())
display(data['country_y'].unique())


# In[30]:

#check if source and destination state same
display((data['state_x']==data['state_y']).value_counts(normalize=True))

data['same_state'] = data.apply(check_same_state,axis=1)
display(data['same_state'].value_counts(normalize=True))


# In[31]:

import gc
gc.collect()


# Lets see count of Airports at State level (Origin and Destination wise). Idea is if count for some states is very very less, group those states together to reduce number of categorical variables.

# In[160]:

# data.groupby(['state_x'])['Origin'].describe().reset_index()


# In[33]:

plt.figure(figsize=(18,6))
data.groupby('state_x')['Origin'].count().plot.bar(title='Count of Origin Airports at Origin state level')


# From above graph, can see count of some state_x is very very less, can group together (KS,ND,SD,VI,VT,WY,WY). Rule: count<10k

# In[161]:

# data.groupby(['state_y'])['Dest'].describe().reset_index()


# In[35]:

plt.figure(figsize=(18,6))
data.groupby('state_y')['Dest'].count().plot.bar(title='Count of Destination Airports at Destination state level')


# From above graph, can see count of some state_y is very very less, can group together (ND,SD,VI,VT,WY,WY). Rule: count<10k

# #### Carrier Level Feature (feature - 5)

# ##### Get percentage of flight for each carrier

# In[36]:

(data['UniqueCarrier'].value_counts(normalize=True)*100).plot.bar(title='Percentages of data per carrier')


# In[37]:

carrier_per = pd.DataFrame(data['UniqueCarrier'].value_counts(normalize=True)*100).                reset_index().rename(columns={'index':'UniqueCarrier','UniqueCarrier':'carrier_per'})

data = pd.merge(data,carrier_per,how='left',on='UniqueCarrier')
display(data.head())


# ##### Historical delay per carrier

# In[38]:

gpby_col = ['UniqueCarrier','date']
## get average delay for each carrier at date level
date_avg_delay = data.groupby(by=gpby_col).agg({'ArrDelay':'mean'}).reset_index()
date_avg_delay = date_avg_delay.sort_values(by='date')
date_avg_delay.reset_index(inplace=True)
## get historical average delay for each carrier
date_avg_delay['hist_avg_delay_carrier'] = date_avg_delay.apply(hist_avg_delay, df=date_avg_delay                                                                ,col_name='UniqueCarrier',buffer_days=120,axis=1)
## Merge historical average delay at day, carrier level with complete data 
data = pd.merge(data,date_avg_delay[['UniqueCarrier','date','hist_avg_delay_carrier']],how='left',left_on = gpby_col,right_on=gpby_col)


# #### Get external data for TailNum (Feature - 6)

# In[39]:

# number of unique tail number
data['TailNum'].nunique()


# In[40]:

taildata = pd.read_csv('../input/plane-data.csv')
display(taildata.shape)
display(taildata.tail(3))


# In[41]:

# verify if all tailnum present in external data
display(len(set(data['TailNum']).difference(set(taildata['tailnum']))))


# In[42]:

taildata.rename(columns={'year':'manufacture_year','tailnum':'TailNum'},inplace=True)
data = pd.merge(data,taildata[['TailNum','type','aircraft_type','engine_type','manufacture_year','status']],how='left',on='TailNum')

display(data.head(3))


# In[43]:

display(pd.isnull(data['manufacture_year']).sum()/data.shape[0])
display(pd.isnull(data['aircraft_type']).sum()/data.shape[0])
display(pd.isnull(data['engine_type']).sum()/data.shape[0])


# As we can see that 36% of data is NaN for manufacturer year after merging with tail data. So, better to drop taildata and TailNum column as well. Original Idea was to **calculate aircraft age** from using manufacture year.

# In[44]:

data.drop(['TailNum','type','aircraft_type','engine_type','manufacture_year','status'],axis=1,inplace=True)


# #### Check FlightNum

# In[45]:

data['FlightNum'].nunique()


# Don't derive any features for flight as if we get new flight. should drop FlightNum as large number of unique values exist and will not be helpful for prediction

# #### Variation in Arrival Delay at Month Level

# In[46]:

col_cat_analyse = 'Month'
col_val_analyse = 'ArrDelay'
# x = data[(data[col_cat_analyse]=='OO') | (data[col_cat_analyse]=='NW') | (data[col_cat_analyse]=='WN')]
df = data[[col_cat_analyse,col_val_analyse]].sample(frac=0.0001)
# sns.set(style="whitegrid")
f = plt.figure(figsize=(15,8))
ax = sns.boxplot(x=col_cat_analyse, y=col_val_analyse, data=df,palette="Set3",hue=col_cat_analyse)
f.suptitle('Average Delay per carrier (sample rows)', fontsize=10)
plt.xlabel('Carrier')
plt.ylabel('Arrival delay')
plt.legend(loc='upper right')


# #### Variation in Arrival Delay at Day of week Level

# In[47]:

col_cat_analyse = 'DayOfWeek'
col_val_analyse = 'ArrDelay'
# x = data[(data[col_cat_analyse]=='OO') | (data[col_cat_analyse]=='NW') | (data[col_cat_analyse]=='WN')]
df = data[[col_cat_analyse,col_val_analyse]].sample(frac=0.0001)
# sns.set(style="whitegrid")
f = plt.figure(figsize=(19,8))
ax = sns.boxplot(x=col_cat_analyse, y=col_val_analyse, data=df,palette="Set3",hue=col_cat_analyse)
f.suptitle('Average Delay per carrier (sample rows)', fontsize=10)
plt.xlabel('Carrier')
plt.ylabel('Arrival delay')
plt.legend(loc='upper right')


# #### Average Delay for sample carrier for sample rows

# In[48]:

col_cat_analyse = 'UniqueCarrier'
col_val_analyse = 'ArrDelay'
x = data[(data[col_cat_analyse]=='OO') | (data[col_cat_analyse]=='NW') | (data[col_cat_analyse]=='WN')]
df = x[[col_cat_analyse,col_val_analyse]].sample(frac=0.0001)
# sns.set(style="whitegrid")
f = plt.figure(figsize=(19,8))
ax = sns.boxplot(x=col_cat_analyse, y=col_val_analyse, data=df,palette="Set3",hue=col_cat_analyse)
f.suptitle('Average Delay per carrier (sample rows)', fontsize=10)
plt.xlabel('Carrier')
plt.ylabel('Arrival delay')
plt.legend(loc='upper right')


# #### Checking null values

# In[49]:

check_null_values(data)


# For column 'ArrDelay', missing values are 141541. Let's investigate the reason.
# 
# Get subset of data where flight is either cancelled or diverted. In both cases, flight will delay for sure and should have some value for 'ArrDelay'. 

# In[50]:

# subset of data where flight is either cancelled or diverted
x = data[(data['Cancelled']==1) | (data['Diverted']==1)]


# In[51]:

# check null values for that subset of data
check_null_values(x)


# As we can see above, missing values for **ArrDelay** is still same i.e. 141541. It means that ArrDelay have NaN when flight is either cancelled or diverted. And we already know that ArrDelay contains continuous value where positive value indicate that flight get delayed. So, while **imputation** for ArrDelay, it is safe to impute its values by any positive number say 1 (for Goal -I) i.e. flights didn't arrive on time or simply delayed. 
# 
# **Caution**: Imputation actually depends on Goal as well i.e. Is it regression or classification? It will **not be correct** way if we impute ArrDelay by 1 and use it for regression. Better to discard NaN rows for regression as we already have sufficient number data points. 

# In[52]:

# data['ArrDelay'].fillna(1,inplace=True)
data.dropna(subset=['ArrDelay'],inplace=True)


# In[53]:

check_null_values(data)


# ##### Feature Selection

# Now, need to **eliminate** columns that aren't relevant to a predictive model and then again check status of missing values.

# Two types of feature can be removed:
# 
#     1) Type I feature: Features which are not useful for model. Example - TailNum, Year,
#     2) Type II feature: Features which can not be captured at the time of flight booking. 
#     Example - ArrTime, DepTime, ActualElapsedTime, CRSElapsedTime, AirTime,DepDelay,Distance, TaxiIn, TaxiOut, Cancelled, CancellationCode, Diverted, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay, 'ArrDelay'
# 
# 
# However, feature 'ArrDelay' can not be removed from data as it will serve as output variable.

# In[54]:

display(data.columns)


# In[55]:

col_to_remove = ['Year','DepTime','ArrTime','FlightNum','ActualElapsedTime', 'CRSElapsedTime', 'AirTime','DepDelay','Origin', 'Dest','Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay','WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay','date','iata_x',                 'city_x','country_x', 'lat_x', 'long_x','iata_y', 'city_y','country_y',                'lat_y', 'long_y','DayofMonth']

remaining_cols = list(set(data.columns).difference(col_to_remove))

display(remaining_cols)


# In[56]:

# get data with remaining columns only
data = data[remaining_cols]
display(data.shape)


# As you can see, we reduced number of columns from 29 (present in original data) to 13. 

# In[57]:

# check null values again in filtered data
check_null_values(data)


# In[58]:

# simply remove NaN as we already have large number of data points and facing resource issues. 
data.dropna(inplace=True)
check_null_values(data)


# Cool !!. Null Values don't exist at all in data. Till Now, missing values have been replaced and the list of columns has been narrowed down to those most relevant to the model.

# #### Create Data for both Goals

# Goal I is classification problem, need to discretize ArrDelay. ArrDelay will have two values 0 and 1. 0 signifies flight arrived on time and 1 signifies flight delayed. To discretize, use **simple rule** i.e. replace ArrDelay by 1 where values greater than 15 else 0.
# 
# Goal II is reression problem, let ArrDelay as it is i.e. continuous. 

# In[59]:

# create output variable for goal-I i.e. 0 or 1 using 15 minute rule.
data['ArrDelay_classification'] = (data['ArrDelay']>15).astype(int)
# rename ArrDelay for regression
data.rename(columns={'ArrDelay':'ArrDelay_regression'},inplace=True)


# In[60]:

# def get_clean_data(input_file_path,dump_file_path):
#     gc.collect()
#     # filtering columns
#     remaining_cols = ['Month', 'Dest', 'ArrDelay', 'DayOfWeek',\
#                       'CRSArrTime', 'DayofMonth', 'Origin','CRSDepTime','FlightNum','UniqueCarrier']
#     # loading data
#     try:
#         data = pd.read_csv(input_file_path,usecols = remaining_cols)
#         gc.collect()
#     except Exception as e:
#         display("Loading Error: {}".format(e))
#         return
#     # dropping NaN rows
#     data.dropna(inplace=True)
#     # binning column - CRSDepTime, CRSArrTime
#     data['CRSDepTime'] = data['CRSDepTime']//100
#     data['CRSArrTime'] = data['CRSArrTime']//100
#     # handling special case of 2400 and 0000 hhmm
#     data['CRSArrTime'] = data['CRSArrTime'].replace({24:0})
#     # binning validation
#     assert data['CRSDepTime'].nunique()==24
#     assert data['CRSArrTime'].nunique()==24
#     # dummify categorical data
#     gc.collect()
#     #data = pd.get_dummies(data, columns=['Origin', 'Dest'])
#     #gc.collect()
#     # validation - drop of columns 'Origin','Dest'
#     #assert 'Dest' not in data.columns
#     #assert 'Origin' not in data.columns
#     # create output variable for goal-I i.e. 0 or 1 using 15 minute rule.
#     data['ArrDelay_classification'] = (data['ArrDelay']>15).astype(int)
#     # rename ArrDelay for regression
#     data.rename(columns={'ArrDelay':'ArrDelay_regression'},inplace=True)
#     # save data
#     try:
#         #data.to_csv(dump_file_path,index=False)
#         display("Data preparation and dump completes sucessfully")
#         return data
#     except Exception as e:
#         display("Dumping Error: {}".format(e))
#         return


# In[61]:

# input_file_path = '../input/2004.csv'
# dump_file_path = '../input/2004_clean_data.csv'
# data = get_clean_data(input_file_path,dump_file_path)


# In[62]:

display(data.shape)


# #### check distribution of Flight Delay

# In[63]:

data['ArrDelay_classification'].value_counts(normalize=True)


# ## Step IV: Building  Model 

# For this Notebook, to solve both goals can use any of below approach. We will go through both approach and compare the result.
# 
# **Approach 1:** 
# 
# Make Independent models for both goals. Difference will be on side of output variable only i.e. for Goal-I, output variable will be discrete (0 or 1) and for Goal-II, output variable will be continuous.
# 
# **Approach 2:**
# 
# Make only regression model and on top of its output apply 15 minute rule to simply classify in 0 or 1.
# 
# To **compare** the output from approach 1 and approach 2, both approach should have same dataset. So, we have dropped rows where ArrDelay is NaN.

# #### Get training and test Data for both Goal-I and Goal-II

# In[64]:

g1_y = data['ArrDelay_classification']
g2_y = data['ArrDelay_regression']
data.drop(['ArrDelay_classification','ArrDelay_regression'],axis=1,inplace=True)
# get goal-I data
g1_train_x, g1_test_x, g1_train_y, g1_test_y = train_test_split(data, g1_y, test_size=0.07,                                                                stratify=g1_y,random_state=42)

g1_train_x, g1_val_x, g1_train_y, g1_val_y = train_test_split(g1_train_x, g1_train_y, test_size=0.07,                                                                stratify=g1_train_y,random_state=42)

# get goal-II data
g2_train_x, g2_test_x, g2_train_y, g2_test_y = train_test_split(data, g2_y, test_size=0.07,                                                                random_state=42)

g2_train_x, g2_val_x, g2_train_y, g2_val_y = train_test_split(g2_train_x, g2_train_y, test_size=0.07,                                                                random_state=42)

display("Train data: {}".format(g1_train_x.shape))
display("Test data: {}".format(g1_test_x.shape))


# #### Approach - 1

# ###### Goal-I: classification Model

# In[65]:

display(data.dtypes)


# In[66]:

col_to_object = ['CRSArrTime','DayOfWeek','extended_weekend','weekend',                 'holiday','CRSDepTime','UniqueCarrier','state_y','state_x','Month','same_state','week_number']

for col in col_to_object:
    g1_train_x[col] = g1_train_x[col].astype(str)
    g1_test_x[col] = g1_test_x[col].astype(str)
    g1_val_x[col] = g1_val_x[col].astype(str)

categorical_features_indices = np.where((g1_train_x.dtypes == 'object'))[0]
display(categorical_features_indices)


# In[67]:

## See No of unique values in each categorical variable
display(data[col_to_object].nunique())


# In[68]:

clf=CatBoostClassifier(iterations = 400, depth = 5, learning_rate=0.045,loss_function='Logloss',use_best_model=True,
                     logging_level='Verbose',eval_metric = 'AUC')


# In[69]:

history = clf.fit(g1_train_x, g1_train_y,cat_features=categorical_features_indices,eval_set=(g1_val_x,g1_val_y),plot=True)


# In[70]:

g1_train_pred_prob = clf.predict_proba(g1_train_x)
g1_val_pred_prob = clf.predict_proba(g1_val_x)
g1_test_pred_prob = clf.predict_proba(g1_test_x)


# In[71]:

display("Train Auc Score {}".format(roc_auc_score(g1_train_y, g1_train_pred_prob[:, 1])))
display("Val Auc Score {}".format(roc_auc_score(g1_val_y, g1_val_pred_prob[:, 1])))
display("Test Auc Score {}".format(roc_auc_score(g1_test_y, g1_test_pred_prob[:, 1])))


# Train, validation and test AUC score are quite similar i.e. very minute difference between there score. Good sign that model is **not overfitting**. 

# #### Saving Model

# In[101]:

clf.save_model('models/catboost_classification_v1')


# #### Result Visualization

# In[75]:

fpr, tpr, _ = roc_curve(g1_test_y, g1_test_pred_prob[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# In[90]:

g1_train_x.dtypes.index


# In[91]:

## code taken from KAGGLE (https://www.kaggle.com/mistrzuniu1/tutorial-eda-feature-selection-regression) ##
feature_score = pd.DataFrame(list(zip(g1_train_x.dtypes.index, clf.get_feature_importance(Pool(g1_train_x, label=g1_train_y,                cat_features=categorical_features_indices)))),columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

## visualize

plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()


# ###### Goal-II: Regression Model

# In[127]:

from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Copy data so that if any changes made to this data, not affect original data. 
tx2 = deepcopy(g2_train_x)
ty2 = deepcopy(g2_train_y)

vx2 = deepcopy(g2_val_x)
vy2 = deepcopy(g2_val_y)

tex2 = deepcopy(g2_test_x)
tey2 = deepcopy(g2_test_y)


# In[129]:

# Assigning number to category
for c in col_to_object:
    lbl = LabelEncoder() 
    lbl.fit(list(tx2[c].values)) 
    tx2[c] = lbl.transform(list(tx2[c].values))
    vx2[c] = lbl.transform(list(vx2[c].values))
    tex2[c] = lbl.transform(list(tex2[c].values))


# ##### LinearRegression

# In[139]:

lr = LinearRegression()
lr.fit(tx2,ty2)


# In[142]:

display("Train MAPE Score {}".format(mean_absolute_error(ty2, lr.predict(tx2))))
display("Val MAPE Score {}".format(mean_absolute_error(vy2,lr.predict(vx2))))
display("Test MAPE Score {}".format(mean_absolute_error(tey2, lr.predict(tex2))))


# In[154]:

display("Train R2 Score {}".format(r2_score(ty2, lr.predict(tx2))))
display("Val R2 Score {}".format(r2_score(vy2,lr.predict(vx2))))
display("Test R2 Score {}".format(r2_score(tey2, lr.predict(tex2))))


# #### Lasso, Elastic, Ridge

# In[155]:

model_elastic = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005))
model_ridge = Ridge(alpha = 5)
model_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005))


# In[156]:

model_lasso.fit(tx2,ty2)

model_elastic.fit(tx2,ty2)
model_ridge.fit(tx2,ty2)


# #### LASSO Evaluation

# In[158]:

display("Train MAPE Score {}".format(mean_absolute_error(ty2, model_lasso.predict(tx2))))
display("Val MAPE Score {}".format(mean_absolute_error(vy2,model_lasso.predict(vx2))))
display("Test MAPE Score {}".format(mean_absolute_error(tey2, model_lasso.predict(tex2))))


# ##### Ridge Evaluation

# In[152]:

display("Train MAPE Score {}".format(mean_absolute_error(ty2, model_ridge.predict(tx2))))
display("Val MAPE Score {}".format(mean_absolute_error(vy2,model_ridge.predict(vx2))))
display("Test MAPE Score {}".format(mean_absolute_error(tey2, model_ridge.predict(tex2))))


# #### CatboostRegressor

# In[76]:

bestparams = { 'depth': 4,
 'iterations': 400,
 'l2_leaf_reg': 3,
 'learning_rate': 0.2,
 'thread_count': 4,
 'loss_function': 'RMSE'}


# In[110]:

# bestparams = { 'depth': 4,
#               'l2_leaf_reg':4,
#  'iterations': 400,
#  'learning_rate': 0.15,
#  'loss_function': 'RMSE','border_count':15}


# In[111]:

model=CatBoostRegressor(**bestparams)
history = model.fit(g2_train_x,g2_train_y,cat_features=categorical_features_indices,eval_set=(g2_val_x,g2_val_y),plot=True)


# In[112]:

g2_train_pred_value = model.predict(g2_train_x)
g2_val_pred_value = model.predict(g2_val_x)
g2_test_pred_value = model.predict(g2_test_x)


# ##### CatboostEvaluation

# In[113]:

display("Train RMSE Score {}".format(np.sqrt(mean_squared_error(g2_train_y, g2_train_pred_value))))
display("Val RMSE Score {}".format(np.sqrt(mean_squared_error(g2_val_y, g2_val_pred_value))))
display("Test RMSE Score {}".format(np.sqrt(mean_squared_error(g2_test_y, g2_test_pred_value))))


# In[116]:

display("Train MAPE Score {}".format(mean_absolute_error(g2_train_y, g2_train_pred_value)))
display("Val MAPE Score {}".format(mean_absolute_error(g2_val_y, g2_val_pred_value)))
display("Test MAPE Score {}".format(mean_absolute_error(g2_test_y, g2_test_pred_value)))


# #### Features Visualization

# In[114]:

## code taken from KAGGLE (https://www.kaggle.com/mistrzuniu1/tutorial-eda-feature-selection-regression) ##
feature_score = pd.DataFrame(list(zip(g2_train_x.dtypes.index, clf.get_feature_importance(Pool(g2_train_x, label=g2_train_y,                cat_features=categorical_features_indices)))),columns=['Feature','Score'])
feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort', na_position='last')

## visualize

plt.rcParams["figure.figsize"] = (12,7)
ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
ax.set_title("Catboost Feature Importance Ranking", fontsize = 14)
ax.set_xlabel('')
rects = ax.patches
labels = feature_score['Score'].round(2)
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 0.35, label, ha='center', va='bottom')

plt.show()


# #### Approach - 2

# As Regression model not working fine, drop idea of using this approach.

# ## Step V : Future Scope

# **External Dataset**
# - **Aircraft** dataset can scrapped using TailNum feature [1]
# - **Weather** dataset can also be one of important feature
# - **Airport level** dataset in terms of number of runways, capacity of airport can be helpful
# - Data for **city** of source airport like population, city status (Business city) can also be helpful
# - **Festival calendar** data at region level can also be helpful 
# ****
# **Feature Engineering**
# - Some **magic features** like ratio of airport capacity and city's population can also be helpful
# - Output variable in Goal-I i.e delay or not can be more categorized into small delay, large delay
# - High level feature on departure time like 'early morning','morning','afternoon','evening','night' can be created
# - High level feature on states **region** wise Like East, west
# - **Mean encoding** of Categorical variables with smoothing can also be important where categorical values are large in 
# number like state (source and destination airport) 
# - Binning of less occurence of values for some categorical variable can be done like for some states in column source 
# state and destination state
# - Window/Buffer days can be adjusted while calculating historical level feature
# - Flight direct or connecting
# - Machine learning model should know that a feature is cyclical like Dayofweek, departure tome in hour format [2] 
# - Removing Bias from dataset like some delays are unpredictable like delay due to accidient
# - More features can be derived on  flights delay per airline and per airport level
# ****
# **Modelling**
# - Code can be more formalised (more of functions/class than free form)
# - Experiment on approach for creation of dataset (training, validation and test set) for modelling can be performed like on time basis (out of time validation)
# - Cross validation technique while model taining and prediction can also be used
# - Hyperparameter Tuning can also be performed but as of now hardware resource (less computation power) is constraint.
# - Ensemble, Stacking techniques can also be experimented to improve results.
# 
# 1) https://www.flightradar24.com/data/aircraft/ + {each tail no}
# 
# 
# 2) https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes 
