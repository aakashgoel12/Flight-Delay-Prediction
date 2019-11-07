# Flight-Delay-Prediction


## STEP I: Setting up Problem
Lets set some **background** first:

Customer browsing through some flight booking website and want to book flight for some specific date, time, source and destination.

**IDEA:** If during flight booking, we can show to customer whether the flight he/she considering for booking is likely to arrive on time or not. Additionaly, if flight is expected to delay, also show delayed time.

If customer know that the flight is likely to be late, he/she might choose to book another flight.

From **Modelling Propective**, need to set two goals:
    
    GOAL I: Predict whether flight is going to delay or not.
    GOAL II: If flight delays, predict amount of time by which it delays.


## STEP II:Getting Data
Data expo 2009 (http://stat-computing.org/dataexpo/2009/the-data.html) provides flight data from year 1987 to year 2008.

For this notebook, used data for year 2004. 

Link for downloading data: http://stat-computing.org/dataexpo/2009/2004.csv.bz2

Also, using **external data** like airport, carrier, plane details. (http://stat-computing.org/dataexpo/2009/supplemental-data.html)


## STEP III: Clean and Prepare data
All Ideas regarding feature engineering:
****

**Scheduled departure and arrival time** (Feature - 1)
- convert scheduled departure and arrival time into hour
**Source Airport (Origin)** (Feature - 2)     
- Get Total number of flights pre and post one hour from source airport for specific day    
- Get historical average delay for each source airport
**Flying Date**  (Feature - 3)
- Using holiday calendar of USA, create binary variable if flying day is holiday or not
- Using feature day of week, create feature weekend i.e get if it is saturday/sunday or not
- create extended weekend variable with buffer of 1 day i.e. if day of week is Friday or Monday then its  extended weekend
- create week number from day of month
**Source and destination airport ** (Feature - 4)
- From source and destination airport code, Get state, city,latitude and longitude
    - Using lat,long for source and destination airport, create distance as feature between airports
    - Using State, create feature 'same_state' i.e. if both source and destination lies in same state
    - Using city, create city level features (like population, forecasted weather details like rainy,cloudy)
    - Get count of airports in each state i.e. state level feature
** Unique Carrier (company)** (Feature - 5)       
- create historical average delay as feature for each carrier (company)
- create percentages of flights for each carrier (company)
**Tail Number** (Feature - 6)    
- Using TailNumber, get feature like manufacturing year (Age of aircraft), engine type. 

## Step IV: Building  Model 

For this Notebook, to solve both goals can use any of below approach. We will go through both approach and compare the result.

**Approach 1:** 

Make Independent models for both goals. Difference will be on side of output variable only i.e. for Goal-I, output variable will be discrete (0 or 1) and for Goal-II, output variable will be continuous.

**Approach 2:**

Make only regression model and on top of its output apply 15 minute rule to simply classify in 0 or 1.

To **compare** the output from approach 1 and approach 2, both approach should have same dataset. So, we have dropped rows where ArrDelay is NaN.

## Step V : Future Scope
**External Dataset**
- **Aircraft** dataset can scrapped using TailNum feature [1]
- **Weather** dataset can also be one of important feature
- **Airport level** dataset in terms of number of runways, capacity of airport can be helpful
- Data for **city** of source airport like population, city status (Business city) can also be helpful
- **Festival calendar** data at region level can also be helpful 
****
**Feature Engineering**
- Some **magic features** like ratio of airport capacity and city's population can also be helpful
- Output variable in Goal-I i.e delay or not can be more categorized into small delay, large delay
- High level feature on departure time like 'early morning','morning','afternoon','evening','night' can be created
- High level feature on states **region** wise Like East, west
- **Mean encoding** of Categorical variables with smoothing can also be important where categorical values are large in 
number like state (source and destination airport) 
- Binning of less occurence of values for some categorical variable can be done like for some states in column source 
state and destination state
- Window/Buffer days can be adjusted while calculating historical level feature
- Flight direct or connecting
- Machine learning model should know that a feature is cyclical like Dayofweek, departure tome in hour format [2] 
- Removing Bias from dataset like some delays are unpredictable like delay due to accidient
- More features can be derived on  flights delay per airline and per airport level
****
**Modelling**
- Code can be more formalised (more of functions/class than free form)
- Experiment on approach for creation of dataset (training, validation and test set) for modelling can be performed like on time basis (out of time validation)
- Cross validation technique while model taining and prediction can also be used
- Hyperparameter Tuning can also be performed but as of now hardware resource (less computation power) is constraint.
- Ensemble, Stacking techniques can also be experimented to improve results.
****
1) https://www.flightradar24.com/data/aircraft/ + {each tail no}

2) https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes 
