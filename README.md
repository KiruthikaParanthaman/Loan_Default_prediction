# Loan Default Prediction

**Objective**:

Aim of the project is to predict whether bank customer will default his/her loan or not. This Dataset contains 1000 Rows and 12 columns.Age,Gender,Income, Employment Status, Location, Credit Score,Debt-to-Income Ratio,Existing Loan Balance,Loan Amount, Interest Rate,Loan Duration Months are the columns features.

**Approach**:
* Dataset was converted from Csv file into pandas dataframe
* Exploratory Data Analysis was carried out. There were 208 null values in Gender and 94 null values in Employment Status.Male,Female are the subcategories and Rural,Urban,Suburban are the subcategories of Gender      and Loction respectively.
* To fill the null values of categorical variables, Label Encoding was undertaken and Knnimputer was used to fill the missing categorical values
* Univariate and Bi-variate Analysis undertaken.Pairplot showed
