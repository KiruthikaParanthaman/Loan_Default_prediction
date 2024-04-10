# Loan Default Prediction

**Objective**:

Aim of the project is to predict whether bank customer will default his/her loan or not. This Dataset contains 1000 Rows and 12 columns.Age,Gender,Income, Employment Status, Location, Credit Score,Debt-to-Income Ratio,Existing Loan Balance,Loan Amount, Interest Rate,Loan Duration Months are the columns features.

**Approach**:
* Dataset was converted from Csv file into pandas dataframe
* **Exploratory Data Analysis** was carried out. There were 208 null values in Gender and 94 null values in Employment Status.Male,Female are the subcategories and Rural,Urban,Suburban are the subcategories of Gender      and Loction respectively.
* To fill the null values of categorical variables, Label Encoding was undertaken and **Knnimputer** was used to fill the missing categorical values
* Univariate and Bi-variate Analysis undertaken.Pairplot showed no significant relation.
* **Feature Creation** -  New Features like Current Emi/Nmi Ratio,Current dbi,Emi Ratio was created
* **Pearson Correaltion** visualization with sns heatmap unsertaken to find the correlation between variables.
* Data was split into Train,Validation and Test in 80:10:10 ratio
* Imbalance in Target variable(Loan Default : 156 Loan Non-default : 644) was handled by creating synthetic minority samples using **SMOTE-NC**
* Categorical variables were **one-hot encoded** as they were nominal and not ordinal as no inherent order in categorical features found
* Standadization using **standardscaler** and normalization using **powertransformer** was undertaken.Either of them was used in machine learning algorithm based on the resultant accuracy
* Dataset was trained using Logistic Regression, Decision Tree classifier, AdaBoost, XGBoost, Gradient Classifier,knncluster, Random Forest and Balanced Bagging Classifier. XGBClassifier gave the highest accuracy 
  of 0.71 with validation data and with test accuracy of 0.65
* To view the full detailed approach [click here](https://github.com/KiruthikaParanthaman/Loan_Default_prediction/blob/main/Capstone%20Loan%20Default%20Prediction.ipynb)

**Summary**:

 * The Dataset didnt represent the real world scenario. For Example: In many cases loan balance outstanding was 20 times more than loan amount. In real world scenario if customer defaults more than 3 Emi they     
   will be declared defaulter.Loans provided to customer with score as less as 300 and customer with no employment status but with income level of 150000.
 * No independant variables had positive correlation of more than 0.2 and hence no significant relations could be indentified by advanced machine learning models like adaboost, gradientboost even after some     
   feature creation
 * And Finally data is the king. We need more features and dataset reflecting real world scenario to further improve the accuracy of our model.

