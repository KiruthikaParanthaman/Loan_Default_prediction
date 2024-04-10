<h1 style="text-align: center;">Loan Default Prediction </h1>

### Load Libraries
Import necessary Libraries like Numpy,Pandas for Data Exploration, Matplotlib,Seaborn for Data Visualization and other necessary Pre-processing,modelling,imputing libraries from sklearn


```python
#import necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler,PowerTransformer,OneHotEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
```

### Extract data
Csv file format is converted into pandas Dataframe for efficient analysis and modelling using read_csv method


```python
# Load loan default dataset
df = pd.read_csv("loan_default_prediction_project.csv")
```

### Explore data
Dataset has 1000 Rows and 12 columns with columns like Age,Gender,Income,Employment Status,Credit Score etc.Int64,Float64,Object are the three major datatypes present.Gender column has 208 null values and Employment Status has 94 null values.Both columns are categorical in nature.


```python
#Find number of Rows and columns in dataset
df.shape
print("No of Rows    :", df.shape[0])
print("No of Columns :", df.shape[1])
```

    No of Rows    : 1000
    No of Columns : 12
    


```python
df.columns
```




    Index(['Age', 'Gender', 'Income', 'Employment_Status', 'Location',
           'Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance',
           'Loan_Status', 'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months'],
          dtype='object')




```python
#Overview of dataset
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Income</th>
      <th>Employment_Status</th>
      <th>Location</th>
      <th>Credit_Score</th>
      <th>Debt_to_Income_Ratio</th>
      <th>Existing_Loan_Balance</th>
      <th>Loan_Status</th>
      <th>Loan_Amount</th>
      <th>Interest_Rate</th>
      <th>Loan_Duration_Months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56</td>
      <td>Male</td>
      <td>71266.105175</td>
      <td>Employed</td>
      <td>Suburban</td>
      <td>639</td>
      <td>0.007142</td>
      <td>27060.188289</td>
      <td>Non-Default</td>
      <td>13068.330587</td>
      <td>18.185533</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46</td>
      <td>NaN</td>
      <td>37283.054601</td>
      <td>Employed</td>
      <td>Suburban</td>
      <td>474</td>
      <td>0.456731</td>
      <td>40192.994312</td>
      <td>Default</td>
      <td>15159.338369</td>
      <td>11.727471</td>
      <td>69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>Male</td>
      <td>69567.036392</td>
      <td>Employed</td>
      <td>Urban</td>
      <td>750</td>
      <td>0.329231</td>
      <td>25444.262759</td>
      <td>Default</td>
      <td>6131.287659</td>
      <td>17.030462</td>
      <td>69</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>Female</td>
      <td>72016.087392</td>
      <td>Employed</td>
      <td>Rural</td>
      <td>435</td>
      <td>0.052482</td>
      <td>3122.213749</td>
      <td>Non-Default</td>
      <td>37531.880251</td>
      <td>16.868949</td>
      <td>26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25</td>
      <td>Female</td>
      <td>32161.988250</td>
      <td>Unemployed</td>
      <td>Suburban</td>
      <td>315</td>
      <td>0.450236</td>
      <td>19197.350445</td>
      <td>Non-Default</td>
      <td>41466.397989</td>
      <td>18.891582</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 12 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Age                    1000 non-null   int64  
     1   Gender                 792 non-null    object 
     2   Income                 1000 non-null   float64
     3   Employment_Status      906 non-null    object 
     4   Location               1000 non-null   object 
     5   Credit_Score           1000 non-null   int64  
     6   Debt_to_Income_Ratio   1000 non-null   float64
     7   Existing_Loan_Balance  1000 non-null   float64
     8   Loan_Status            1000 non-null   object 
     9   Loan_Amount            1000 non-null   float64
     10  Interest_Rate          1000 non-null   float64
     11  Loan_Duration_Months   1000 non-null   int64  
    dtypes: float64(5), int64(3), object(4)
    memory usage: 93.9+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income</th>
      <th>Credit_Score</th>
      <th>Debt_to_Income_Ratio</th>
      <th>Existing_Loan_Balance</th>
      <th>Loan_Amount</th>
      <th>Interest_Rate</th>
      <th>Loan_Duration_Months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>40.986000</td>
      <td>60705.291386</td>
      <td>571.094000</td>
      <td>0.485502</td>
      <td>25239.656186</td>
      <td>27636.369345</td>
      <td>11.538267</td>
      <td>42.221000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.497852</td>
      <td>24594.030383</td>
      <td>163.395983</td>
      <td>0.296466</td>
      <td>14202.689890</td>
      <td>12925.200961</td>
      <td>4.883342</td>
      <td>17.116867</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>20010.775440</td>
      <td>250.000000</td>
      <td>0.000628</td>
      <td>80.059377</td>
      <td>5060.998602</td>
      <td>3.003148</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>29.000000</td>
      <td>40820.367273</td>
      <td>429.000000</td>
      <td>0.220606</td>
      <td>13597.494593</td>
      <td>16756.405848</td>
      <td>7.483547</td>
      <td>27.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>42.000000</td>
      <td>60547.119929</td>
      <td>567.000000</td>
      <td>0.483633</td>
      <td>25439.429898</td>
      <td>27938.066858</td>
      <td>11.537942</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>52.000000</td>
      <td>80338.552791</td>
      <td>720.250000</td>
      <td>0.735476</td>
      <td>37305.466739</td>
      <td>39151.564233</td>
      <td>15.499129</td>
      <td>57.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>64.000000</td>
      <td>150000.000000</td>
      <td>849.000000</td>
      <td>0.999849</td>
      <td>49987.578171</td>
      <td>49986.843702</td>
      <td>19.991438</td>
      <td>71.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Find number of Null values
#Gender and Employment Status has null values which is to be handled
df.isnull().sum()
```




    Age                        0
    Gender                   208
    Income                     0
    Employment_Status         94
    Location                   0
    Credit_Score               0
    Debt_to_Income_Ratio       0
    Existing_Loan_Balance      0
    Loan_Status                0
    Loan_Amount                0
    Interest_Rate              0
    Loan_Duration_Months       0
    dtype: int64




```python
#Explore Gender and Employment Status for handling null values
#Equal distribution of Female, Male Employed
dfge = df.groupby(by=["Gender", "Employment_Status"], as_index=False).size()
dfge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Employment_Status</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>Employed</td>
      <td>236</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>Unemployed</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Employed</td>
      <td>233</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Unemployed</td>
      <td>127</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Gender and Employment Loan Status distribution
dfg = df.groupby(by=["Gender", "Loan_Status"], as_index=False).size()
dfe = df.groupby(by=["Employment_Status", "Loan_Status"], as_index=False).size()
df2 = pd.concat([dfg,dfe],axis=1)
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Loan_Status</th>
      <th>size</th>
      <th>Employment_Status</th>
      <th>Loan_Status</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Female</td>
      <td>Default</td>
      <td>88</td>
      <td>Employed</td>
      <td>Default</td>
      <td>119</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>Non-Default</td>
      <td>311</td>
      <td>Employed</td>
      <td>Non-Default</td>
      <td>477</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Male</td>
      <td>Default</td>
      <td>73</td>
      <td>Unemployed</td>
      <td>Default</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Non-Default</td>
      <td>320</td>
      <td>Unemployed</td>
      <td>Non-Default</td>
      <td>250</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Exploring relationship between male female income category
#On Analysis no solid relationship could be found
dfm = df.loc[df['Gender']=='Male','Income'].describe()
dff = df.loc[df['Gender']=='Female','Income'].describe()
dfc = pd.concat([dfm,dff],axis = 1)
print("'Male Vs Female Income Distribution'")
dfc
```

    'Male Vs Female Income Distribution'
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>393.000000</td>
      <td>399.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60075.210977</td>
      <td>61099.839227</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24034.305409</td>
      <td>25375.480561</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20010.775440</td>
      <td>20125.208415</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41407.414138</td>
      <td>40765.924242</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60592.955097</td>
      <td>59825.000643</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78333.879004</td>
      <td>82288.361303</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150000.000000</td>
      <td>150000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Exploring Relationship between Employed,Unemployed Vs Income Category
dfem = df.loc[df['Employment_Status']=='Employed','Income'].describe()
dfum = df.loc[df['Employment_Status']=='Unemployed','Income'].describe()
dfemm = pd.concat([dfem,dfum],axis = 1)
print("'Employed Vs Unemployed Income Distribution'")
dfemm
```

    'Employed Vs Unemployed Income Distribution'
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>596.000000</td>
      <td>310.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60527.676396</td>
      <td>61145.794617</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24680.034657</td>
      <td>24802.896800</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20010.775440</td>
      <td>20257.461088</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41420.598679</td>
      <td>41003.269591</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60458.913323</td>
      <td>60945.569590</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80880.201407</td>
      <td>80529.362620</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150000.000000</td>
      <td>150000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_encode = df.copy()
```

### Data Cleaning
Gender and Employment Status has null values of 208, 94 each. Both columns are categorical in nature and contain null values. As these are categorical values fillna with median/mode is not helpful much.KnnImputer looks like a better idea. Both columns are as objects and has to be converted to numerical values for knnimputation to work. One-hot Encoding doenn't work because it considers nan values as seperate class and Label Encoder cannot handle null values. Hence, Label Encoding is done manually with with map function and after handling null values imputation has been done with knnimputer.


```python
#Label Encoding Categorical values with map to preserve nan for imputation
df_encode['Gender']=df_encode['Gender'].map({'Male':0,'Female':1})
df_encode['Employment_Status'] = df_encode['Employment_Status'].map({'Unemployed' : 0,'Employed' : 1})
df_encode['Location'] = df_encode['Location'].map({'Rural': 0,'Suburban' : 1,'Urban' : 2})
df_encode['Loan_Status'] = df_encode['Loan_Status'].map({'Default' : 0,'Non-Default' : 1})
```


```python
#Impute Nan Values in Gender and Employment Status with KNNimputer
imputer = KNNImputer(n_neighbors=1)
df_nan = imputer.fit_transform(df_encode)
```


```python
#Converting numpy array to dataframe
df_encoded = pd.DataFrame(df_nan, columns=['Age', 'Gender', 'Income', 'Employment_Status', 'Location',
       'Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance',
       'Loan_Status', 'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months'])
```


```python
#Check null values after imputation
df_encoded.isnull().sum()
```




    Age                      0
    Gender                   0
    Income                   0
    Employment_Status        0
    Location                 0
    Credit_Score             0
    Debt_to_Income_Ratio     0
    Existing_Loan_Balance    0
    Loan_Status              0
    Loan_Amount              0
    Interest_Rate            0
    Loan_Duration_Months     0
    dtype: int64




```python
#Gender,Employment Vs Loan Status distribution After Encoding
dfg = df.groupby(by=["Gender", "Loan_Status"], as_index=False).size()
dfag = df_encoded.groupby(by=["Gender", "Loan_Status"], as_index=False).size()
dfe = df.groupby(by=["Employment_Status", "Loan_Status"], as_index=False).size()
dfae = df_encoded.groupby(by=["Employment_Status", "Loan_Status"], as_index=False).size()
df2 = pd.concat([dfg,dfe],axis=1)
df3 = pd.concat([dfag,dfae],axis=1)
print("Before Encoding :")
print(df2)
print("After Encoding:")
print(df3)
```

    Before Encoding :
       Gender  Loan_Status  size Employment_Status  Loan_Status  size
    0  Female      Default    88          Employed      Default   119
    1  Female  Non-Default   311          Employed  Non-Default   477
    2    Male      Default    73        Unemployed      Default    60
    3    Male  Non-Default   320        Unemployed  Non-Default   250
    After Encoding:
       Gender  Loan_Status  size  Employment_Status  Loan_Status  size
    0     0.0          0.0    88                0.0          0.0    62
    1     0.0          1.0   393                0.0          1.0   270
    2     1.0          0.0   107                1.0          0.0   133
    3     1.0          1.0   412                1.0          1.0   535
    

### Univariate Analysis


```python
df_encoded.hist(figsize=(10,10))
plt.show()
```


    
![png](output_24_0.png)
    



```python
sns.pairplot(df_encoded,hue = 'Loan_Status',diag_kind ='kde')
plt.show()

```


    
![png](output_25_0.png)
    



    
![png](output_25_1.png)
    



```python
#Box-plot Visualization
numerical_columns = ['Age','Income','Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance',
       'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months']
fig = plt.figure()
fig = plt.figure(figsize=(10, 10))
for i in range(len(numerical_columns)):
    var = numerical_columns[i]
    sub = fig.add_subplot(3,3, i + 1)
    df_encoded[var].plot(kind = 'box')

```


    <Figure size 640x480 with 0 Axes>



    
![png](output_26_1.png)
    


### Bi-variate Analysis
On Bi-variate Analysis using Box-plot visualization with independant variables in X-column and Target Variable in Y-column, we can observe that Income column has some outliers.But such values looks like a valid outlier.Hence frequent outlier handling methods like outlier removal/modification is not undertaken. However,if left unhandled outliers impacts the prediction. Hence, inorder to reduce the impact of outlier, normalization of data is done after train/test split to reduce the impact.


```python
numerical_columns = ['Age','Income','Credit_Score', 'Debt_to_Income_Ratio', 'Existing_Loan_Balance',
                    'Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months']
fig = plt.figure()
fig = plt.figure(figsize=(10, 10))
for i in range(len(numerical_columns)):
    var = numerical_columns[i]
    sub = fig.add_subplot(3,3, i + 1)
    df_encoded.boxplot(column = var, by = "Loan_Status", ax = sub,figsize = (3,3),layout = (3,3),fontsize = 0.1)
```


    <Figure size 640x480 with 0 Axes>



    
![png](output_28_1.png)
    


### Feature Creation
EMI/NMI Ratio is a key factor in Loan Sanction in Banking sector.It gives a glimpse of what portion of monthly income is spent on repaying Equated Monthly Installments. Generally, banks service loan to individuals with emi/nmi ratio less than 0.70.


```python
#Create new feature EMI/NMI ratio
#p-principal,r-interest rate per month n-no of installments
def emi_calculator(p, R, n):
    r = R/(12*100)
    emi = p * r * ((1+r)**n)/((1+r)**n - 1)
    return emi
df_encoded['Current_Emi_Nmi_Ratio']= emi_calculator(df_encoded['Existing_Loan_Balance'],df_encoded['Interest_Rate'],df_encoded['Loan_Duration_Months'])/(df['Income']/12)
```


```python
#Debt to Income Ratio based on current Loan Balance
df_encoded['Current_dbi'] = df['Existing_Loan_Balance']/df['Income']
```


```python
#Find number of Emi's missed months.If Existing Loan Balance is greater than Loan amount it symbolizes that either Emi or Interest portion 
# has not been serviced.In indian counterpart if emi is unserviced for more than 3 months it is considered default. Here missed Emi is 
#found by differencing loan amount and existing loan balance and dividing it by emi. 
df_encoded['Emi_ratio'] = (df_encoded['Existing_Loan_Balance']-df_encoded['Loan_Amount'])/emi_calculator(df_encoded['Existing_Loan_Balance'],df_encoded['Interest_Rate'],df_encoded['Loan_Duration_Months'])

```


```python
df_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Income</th>
      <th>Employment_Status</th>
      <th>Location</th>
      <th>Credit_Score</th>
      <th>Debt_to_Income_Ratio</th>
      <th>Existing_Loan_Balance</th>
      <th>Loan_Status</th>
      <th>Loan_Amount</th>
      <th>Interest_Rate</th>
      <th>Loan_Duration_Months</th>
      <th>Current_Emi_Nmi_Ratio</th>
      <th>Current_dbi</th>
      <th>Emi_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56.0</td>
      <td>0.0</td>
      <td>71266.105175</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>639.0</td>
      <td>0.007142</td>
      <td>27060.188289</td>
      <td>1.0</td>
      <td>13068.330587</td>
      <td>18.185533</td>
      <td>59.0</td>
      <td>0.117379</td>
      <td>0.379706</td>
      <td>20.071727</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46.0</td>
      <td>1.0</td>
      <td>37283.054601</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>474.0</td>
      <td>0.456731</td>
      <td>40192.994312</td>
      <td>0.0</td>
      <td>15159.338369</td>
      <td>11.727471</td>
      <td>69.0</td>
      <td>0.258633</td>
      <td>1.078050</td>
      <td>31.153719</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.0</td>
      <td>0.0</td>
      <td>69567.036392</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>750.0</td>
      <td>0.329231</td>
      <td>25444.262759</td>
      <td>0.0</td>
      <td>6131.287659</td>
      <td>17.030462</td>
      <td>69.0</td>
      <td>0.100173</td>
      <td>0.365752</td>
      <td>33.256326</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60.0</td>
      <td>1.0</td>
      <td>72016.087392</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>435.0</td>
      <td>0.052482</td>
      <td>3122.213749</td>
      <td>1.0</td>
      <td>37531.880251</td>
      <td>16.868949</td>
      <td>26.0</td>
      <td>0.024027</td>
      <td>0.043354</td>
      <td>-238.629541</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25.0</td>
      <td>1.0</td>
      <td>32161.988250</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>315.0</td>
      <td>0.450236</td>
      <td>19197.350445</td>
      <td>1.0</td>
      <td>41466.397989</td>
      <td>18.891582</td>
      <td>66.0</td>
      <td>0.175280</td>
      <td>0.596896</td>
      <td>-47.403103</td>
    </tr>
  </tbody>
</table>
</div>



### Pearson Correlation
As expected Current Dbi and Current Emi/Nmi shows strong co-relation with Existing Loan Balance.Current dbi and Current Emi/Nmi ratio shows strong relation.However, they are all below the threshold of 0.80.Hence,not dropped.


```python
#Using Pearson Correlation to find the relation
plt.figure(figsize=(12,10))
cor = df_encoded.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
```


    
![png](output_35_0.png)
    


### Train-Test data split
Data is converted into Train,Validation and Test in the ratio 80:10:10. Random State is assigned to enable the uniformity in prediction accuracy. Startify was specified to make the distribution of target variable in test,valid split mimics the distribution in Target variable


```python
# Splitting dataset into Train,Validate and Test
indep_var = df_encoded.loc[:,['Age', 'Gender', 'Income', 'Employment_Status', 'Location','Credit_Score', 'Debt_to_Income_Ratio',
        'Existing_Loan_Balance','Loan_Amount', 'Interest_Rate', 'Loan_Duration_Months','Current_dbi','Current_Emi_Nmi_Ratio','Emi_ratio']]
target_var = df_encoded.loc[:,['Loan_Status']]

X = indep_var
y = target_var
#Splitting dataset into Train and Validation
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.20,stratify = y, random_state=42)

#Splitting Validation to validation and test
X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5)
```


```python
X_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>Income</th>
      <th>Employment_Status</th>
      <th>Location</th>
      <th>Credit_Score</th>
      <th>Debt_to_Income_Ratio</th>
      <th>Existing_Loan_Balance</th>
      <th>Loan_Amount</th>
      <th>Interest_Rate</th>
      <th>Loan_Duration_Months</th>
      <th>Current_dbi</th>
      <th>Current_Emi_Nmi_Ratio</th>
      <th>Emi_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>192</th>
      <td>61.0</td>
      <td>0.0</td>
      <td>48430.765495</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>508.0</td>
      <td>0.232404</td>
      <td>22454.169634</td>
      <td>7508.731931</td>
      <td>5.056996</td>
      <td>13.0</td>
      <td>0.463634</td>
      <td>0.440701</td>
      <td>8.402806</td>
    </tr>
    <tr>
      <th>909</th>
      <td>37.0</td>
      <td>1.0</td>
      <td>28151.391045</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>676.0</td>
      <td>0.256830</td>
      <td>12738.379316</td>
      <td>49290.832256</td>
      <td>10.188553</td>
      <td>57.0</td>
      <td>0.452496</td>
      <td>0.120562</td>
      <td>-129.237424</td>
    </tr>
    <tr>
      <th>365</th>
      <td>19.0</td>
      <td>0.0</td>
      <td>61550.868760</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>392.0</td>
      <td>0.994775</td>
      <td>24876.465801</td>
      <td>27337.951475</td>
      <td>5.758004</td>
      <td>30.0</td>
      <td>0.404161</td>
      <td>0.173966</td>
      <td>-2.758541</td>
    </tr>
    <tr>
      <th>450</th>
      <td>46.0</td>
      <td>0.0</td>
      <td>31357.759181</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>668.0</td>
      <td>0.236849</td>
      <td>23705.103792</td>
      <td>24110.555559</td>
      <td>10.778670</td>
      <td>12.0</td>
      <td>0.755957</td>
      <td>0.800816</td>
      <td>-0.193750</td>
    </tr>
    <tr>
      <th>431</th>
      <td>43.0</td>
      <td>1.0</td>
      <td>78519.719227</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>595.0</td>
      <td>0.223086</td>
      <td>24909.935978</td>
      <td>27773.739308</td>
      <td>6.704376</td>
      <td>71.0</td>
      <td>0.317244</td>
      <td>0.065102</td>
      <td>-6.722784</td>
    </tr>
  </tbody>
</table>
</div>



### Handling imbalanced data using SMOTENC
The Default sample dataset is way less than Non-dafault. Because of the imbalanced data, model will learn only the non-default parameters well as the data is comparitively higher than default dataset. So it is necessary to balance the model. Smote or Synthetic Minority Oversampling Technique handles only numerical values. Hence SMOTE_NC technique was used to handle the categorical and numerical values oversampling 


```python
# Using SMOTENC to handle categorical and Numerical data
#SMOTE handles only numerical data well
smote_nc = SMOTENC(categorical_features=[1,3,4], random_state=0)
X_train_res, y_train_res = smote_nc.fit_resample(X_train, y_train.to_numpy().ravel())
```


```python
print("Before SMOTE :")
print(y_train.value_counts())
print("After SMOTE :")
print("No of defauts :",(y_train_res==0).sum() )
print("No of non-defaults :",(y_train_res==1).sum())
print(y_train_res.shape)
```

    Before SMOTE :
    Loan_Status
    1.0            644
    0.0            156
    dtype: int64
    After SMOTE :
    No of defauts : 644
    No of non-defaults : 644
    (1288,)
    


```python
# Re-labelling Train dataset for one-hot Encoding
X_train_res['Gender']=X_train_res['Gender'].map({0 : 'Male', 1 : 'Female'})
X_train_res['Employment_Status'] = X_train_res['Employment_Status'].map({0 : 'Unemployed',1 : 'Employed' })
X_train_res['Location'] = X_train_res['Location'].map({0 : 'Rural' , 1 : 'Suburban', 2 : 'Urban'})

```


```python
X_train_res.isnull().sum()

```




    Age                      0
    Gender                   0
    Income                   0
    Employment_Status        0
    Location                 0
    Credit_Score             0
    Debt_to_Income_Ratio     0
    Existing_Loan_Balance    0
    Loan_Amount              0
    Interest_Rate            0
    Loan_Duration_Months     0
    Current_dbi              0
    Current_Emi_Nmi_Ratio    0
    Emi_ratio                0
    dtype: int64



### One-Hot Encoding
The categorical variables in dataset are nominal i.e., no inherent order is present in the dataset. Hence categorical variables are being One-hot-Encoded


```python
# One Hot Encoding Train dataset
#Set_output gives output as dataframe
of_enc = OneHotEncoder(sparse_output=False,handle_unknown='ignore').set_output(transform="pandas")
Xtrain_encoded1 = of_enc.fit_transform(X_train_res[["Gender","Employment_Status","Location"]])

#concating the encodings
Xtrain_encoded = pd.concat([X_train_res,Xtrain_encoded1],axis=1).drop(columns=["Gender",'Employment_Status','Location'])
```


```python
Xtrain_encoded.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Income</th>
      <th>Credit_Score</th>
      <th>Debt_to_Income_Ratio</th>
      <th>Existing_Loan_Balance</th>
      <th>Loan_Amount</th>
      <th>Interest_Rate</th>
      <th>Loan_Duration_Months</th>
      <th>Current_dbi</th>
      <th>Current_Emi_Nmi_Ratio</th>
      <th>Emi_ratio</th>
      <th>Gender_Female</th>
      <th>Gender_Male</th>
      <th>Employment_Status_Employed</th>
      <th>Employment_Status_Unemployed</th>
      <th>Location_Rural</th>
      <th>Location_Suburban</th>
      <th>Location_Urban</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>61.0</td>
      <td>48430.765495</td>
      <td>508.0</td>
      <td>0.232404</td>
      <td>22454.169634</td>
      <td>7508.731931</td>
      <td>5.056996</td>
      <td>13.0</td>
      <td>0.463634</td>
      <td>0.440701</td>
      <td>8.402806</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37.0</td>
      <td>28151.391045</td>
      <td>676.0</td>
      <td>0.256830</td>
      <td>12738.379316</td>
      <td>49290.832256</td>
      <td>10.188553</td>
      <td>57.0</td>
      <td>0.452496</td>
      <td>0.120562</td>
      <td>-129.237424</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.0</td>
      <td>61550.868760</td>
      <td>392.0</td>
      <td>0.994775</td>
      <td>24876.465801</td>
      <td>27337.951475</td>
      <td>5.758004</td>
      <td>30.0</td>
      <td>0.404161</td>
      <td>0.173966</td>
      <td>-2.758541</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46.0</td>
      <td>31357.759181</td>
      <td>668.0</td>
      <td>0.236849</td>
      <td>23705.103792</td>
      <td>24110.555559</td>
      <td>10.778670</td>
      <td>12.0</td>
      <td>0.755957</td>
      <td>0.800816</td>
      <td>-0.193750</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.0</td>
      <td>78519.719227</td>
      <td>595.0</td>
      <td>0.223086</td>
      <td>24909.935978</td>
      <td>27773.739308</td>
      <td>6.704376</td>
      <td>71.0</td>
      <td>0.317244</td>
      <td>0.065102</td>
      <td>-6.722784</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Saving One-hot Encoding
with open("one_hot_encoder.pkl",'wb') as f:
    pickle.dump(of_enc,f)
```

### Standardizing and Normalizing the dataset


```python
# standardization 
sc = StandardScaler()
scaled_data = sc.fit_transform(Xtrain_encoded)

#PowerTransformer
pt = PowerTransformer()
normalized_data = pt.fit_transform(Xtrain_encoded)
```


```python
#Saving standard scaler
with open("standard_scalar.pkl",'wb') as f:
    pickle.dump(sc,f)
```


```python
#Saving power Transformer
with open("normal_data.pkl",'wb') as f:
    pickle.dump(pt,f)
    1
```

### Logistic Regression


```python
#Logistic Regression
log = LogisticRegression()
log.fit(scaled_data,y_train_res.ravel())
```




<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" checked><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
#Re-labelling Validation data
X_val['Gender']=X_val['Gender'].map({0 : 'Male', 1 : 'Female'})
X_val['Employment_Status'] = X_val['Employment_Status'].map({0 : 'Unemployed',1 : 'Employed' })
X_val['Location'] = X_val['Location'].map({0 : 'Rural' , 1 : 'Suburban', 2 : 'Urban'})

```


```python
#One hot Encoding validation data
log_onehot = pd.read_pickle("\one_hot_encoder.pkl")
X_val_encoded1 = log_onehot.transform(X_val[["Gender","Employment_Status","Location"]])

#Joining encoded data with validation data
X_val_encoded = pd.concat([X_val,X_val_encoded1],axis=1).drop(columns=["Gender",'Employment_Status','Location'])

#Scaling Encoded data
sc_pickle = pd.read_pickle("standard_scalar.pkl")
scaled_val = sc_pickle.fit_transform(X_val_encoded)

#Normalizing validation data
normal_pickle = pd.read_pickle("normal_data.pkl")
normal_val = normal_pickle.fit_transform(X_val_encoded)
```


```python
#predicting validation data
Xval_array = scaled_val
yval_array = y_val.to_numpy()
pred_log_val = log.predict(Xval_array)
print(classification_report(yval_array, pred_log_val))
```

                  precision    recall  f1-score   support
    
             0.0       0.23      0.60      0.33        20
             1.0       0.83      0.49      0.61        80
    
        accuracy                           0.51       100
       macro avg       0.53      0.54      0.47       100
    weighted avg       0.71      0.51      0.56       100
    
    


```python
#Confusion matrix
cm = confusion_matrix(yval_array, pred_log_val)
display = ConfusionMatrixDisplay(cm)
display.plot()
plt.show()
```


    
![png](output_57_0.png)
    


### Decision Tree


```python
tree = DecisionTreeClassifier()
tree.fit(scaled_data,y_train_res.ravel())
```




<style>#sk-container-id-5 {color: black;}#sk-container-id-5 pre{padding: 0;}#sk-container-id-5 div.sk-toggleable {background-color: white;}#sk-container-id-5 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-5 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-5 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-5 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-5 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-5 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-5 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-5 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-5 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-5 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-5 div.sk-item {position: relative;z-index: 1;}#sk-container-id-5 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-5 div.sk-item::before, #sk-container-id-5 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-5 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-5 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-5 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-5 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-5 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-5 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-5 div.sk-label-container {text-align: center;}#sk-container-id-5 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-5 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" checked><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier()</pre></div></div></div></div></div>




```python
#predicting validation data
Xval_array = scaled_val
pred_tree_val = tree.predict(Xval_array)
print(classification_report(yval_array, pred_tree_val))
```

                  precision    recall  f1-score   support
    
             0.0       0.26      0.45      0.33        20
             1.0       0.83      0.69      0.75        80
    
        accuracy                           0.64       100
       macro avg       0.55      0.57      0.54       100
    weighted avg       0.72      0.64      0.67       100
    
    


```python
#Grid Search
param_grid ={"max_depth": [50,100,150,200],'min_samples_split': [1,3,9,12,15],
    'min_samples_leaf':[1,3,5,9,12]}
                 
                        
search = GridSearchCV(tree, param_grid).fit(normalized_data,y_train_res.ravel())

print("The best hyperparameters are ",search.best_params_)
```


```python
tree1 = DecisionTreeClassifier(max_depth=100)
tree1.fit(normalized_data,y_train_res.ravel())
pred_tree_val = tree1.predict(normal_val)
print(classification_report(yval_array, pred_tree_val))
```

                  precision    recall  f1-score   support
    
             0.0       0.29      0.55      0.38        20
             1.0       0.85      0.66      0.75        80
    
        accuracy                           0.64       100
       macro avg       0.57      0.61      0.56       100
    weighted avg       0.74      0.64      0.67       100
    
    


```python
tree1.feature_importances_
```




    array([0.08838454, 0.12532593, 0.12765624, 0.12641376, 0.05910984,
           0.1363285 , 0.09972281, 0.08025238, 0.03249688, 0.08639198,
           0.0017762 , 0.        , 0.00329885, 0.01140886, 0.01524571,
           0.00441133, 0.0017762 ])



### XGBClassifier


```python
#XGBClassifier
xgb = XGBClassifier()
xgb.fit(normalized_data,y_train_res.ravel())
pred_xgb = xgb.predict(normal_val)
print(classification_report(yval_array, pred_xgb))
```

                  precision    recall  f1-score   support
    
             0.0       0.25      0.20      0.22        20
             1.0       0.81      0.85      0.83        80
    
        accuracy                           0.72       100
       macro avg       0.53      0.53      0.53       100
    weighted avg       0.70      0.72      0.71       100
    
    

### Balanced Bagging Classifier


```python
#base_estimator = DecisionTreeClassifier()
bbc = BalancedBaggingClassifier(estimator=tree1, sampling_strategy='auto', replacement=False, random_state=42)
```


```python
bbc.fit(normalized_data,y_train_res.ravel())
pred_bb = bbc.predict(normal_val)
print(classification_report(yval_array, pred_bb))
```

                  precision    recall  f1-score   support
    
             0.0       0.29      0.60      0.39        20
             1.0       0.86      0.64      0.73        80
    
        accuracy                           0.63       100
       macro avg       0.58      0.62      0.56       100
    weighted avg       0.75      0.63      0.67       100
    
    

### Knn Classifier


```python
knn = KNeighborsClassifier()
knn.fit(normalized_data,y_train_res.ravel())

```




<style>#sk-container-id-7 {color: black;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>




```python
#Predicting Validation data
pred_knn = knn.predict(normal_val)
print(classification_report(yval_array, pred_knn))
```

                  precision    recall  f1-score   support
    
             0.0       0.19      0.55      0.29        20
             1.0       0.79      0.42      0.55        80
    
        accuracy                           0.45       100
       macro avg       0.49      0.49      0.42       100
    weighted avg       0.67      0.45      0.50       100
    
    

### Hyper-parameter Tuning


```python
#Grid Search
param_grid ={"learning_rate" : (0.15,0.20),
                        "n_estimators": [50,100,200,500]}
                 
                        
search = GridSearchCV(adab, param_grid, cv=5).fit(scaled_data,y_train_res.ravel())

print("The best hyperparameters are ",search.best_params_)
```

    The best hyperparameters are  {'learning_rate': 0.15, 'n_estimators': 50}
    

### AdaBoost


```python
adab = AdaBoostClassifier(estimator = tree1,learning_rate = 0.15, n_estimators= 50)
adab.fit(normalized_data,y_train_res.ravel())
pred_adab = adab.predict(normal_val)
print(classification_report(yval_array, pred_adab))
```

                  precision    recall  f1-score   support
    
             0.0       0.29      0.55      0.38        20
             1.0       0.85      0.66      0.75        80
    
        accuracy                           0.64       100
       macro avg       0.57      0.61      0.56       100
    weighted avg       0.74      0.64      0.67       100
    
    

### Random Forest Classifer


```python
randf = RandomForestClassifier(n_estimators= 50)
randf.fit(normalized_data,y_train_res.ravel())
pred_rand = randf.predict(normal_val)
print(classification_report(yval_array, pred_rand))
```

                  precision    recall  f1-score   support
    
             0.0       0.28      0.55      0.37        20
             1.0       0.85      0.64      0.73        80
    
        accuracy                           0.62       100
       macro avg       0.56      0.59      0.55       100
    weighted avg       0.73      0.62      0.66       100
    
    

### Gradient Booster


```python
#Gradient Booster
grad = GradientBoostingClassifier(n_estimators= 100)
grad.fit(scaled_data,y_train_res.ravel())
pred_grad = grad.predict(scaled_val)
print(classification_report(yval_array, pred_grad))
```

                  precision    recall  f1-score   support
    
             0.0       0.21      0.45      0.29        20
             1.0       0.81      0.57      0.67        80
    
        accuracy                           0.55       100
       macro avg       0.51      0.51      0.48       100
    weighted avg       0.69      0.55      0.59       100
    
    

### Testing 


```python
#Re-labelling Testing data
X_test['Gender']=X_val['Gender'].map({0 : 'Male', 1 : 'Female'})
X_test['Employment_Status'] = X_test['Employment_Status'].map({0 : 'Unemployed',1 : 'Employed' })
X_test['Location'] = X_test['Location'].map({0 : 'Rural' , 1 : 'Suburban', 2 : 'Urban'})

```


```python
#One hot Encoding Testing data
log_onehot = pd.read_pickle("E:\Winnie Documents\Guvi\project\Capstone\Loan Default prediction\one_hot_encoder.pkl")
X_test_encoded1 = log_onehot.transform(X_test[["Gender","Employment_Status","Location"]])

#Joining encoded data with testing data
X_test_encoded = pd.concat([X_test,X_test_encoded1],axis=1).drop(columns=["Gender",'Employment_Status','Location'])

#Scaling testing data
sc_pickle = pd.read_pickle("E:\Winnie Documents\Guvi\project\Capstone\Loan Default prediction\standard_scalar.pkl")
scaled_test = sc_pickle.fit_transform(X_test_encoded)

#Normalizing testing data
normal_pickle = pd.read_pickle("E://Winnie Documents//Guvi//project//Capstone//Loan Default prediction//normal_data.pkl")
normal_test = normal_pickle.fit_transform(X_test_encoded)
```


```python
#Ada booster
pred_adab_test = adab.predict(normal_test)
print(classification_report(y_test, pred_adab_test))
```

                  precision    recall  f1-score   support
    
             0.0       0.16      0.32      0.21        19
             1.0       0.79      0.62      0.69        81
    
        accuracy                           0.56       100
       macro avg       0.48      0.47      0.45       100
    weighted avg       0.67      0.56      0.60       100
    
    


```python
#Balanced Bagging
bbc_test = bbc.predict(normal_test)
print(classification_report(y_test, bbc_test))
```

                  precision    recall  f1-score   support
    
             0.0       0.13      0.32      0.18        19
             1.0       0.76      0.51      0.61        81
    
        accuracy                           0.47       100
       macro avg       0.44      0.41      0.40       100
    weighted avg       0.64      0.47      0.53       100
    
    


```python
#XGB Classifier
xgb_test = xgb.predict(normal_test)
print(classification_report(y_test, xgb_test))
```

                  precision    recall  f1-score   support
    
             0.0       0.19      0.26      0.22        19
             1.0       0.81      0.74      0.77        81
    
        accuracy                           0.65       100
       macro avg       0.50      0.50      0.50       100
    weighted avg       0.69      0.65      0.67       100
    
    


```python
#Decision Tree
tree_test = tree1.predict(normal_test)
print(classification_report(y_test, tree_test))
```

                  precision    recall  f1-score   support
    
             0.0       0.21      0.42      0.28        19
             1.0       0.82      0.63      0.71        81
    
        accuracy                           0.59       100
       macro avg       0.52      0.53      0.50       100
    weighted avg       0.71      0.59      0.63       100
    
    
