Importing Packages


```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
```

Loading Data and Viewing Descriptive Stats


```python
df = pd.read_csv("default of credit card clients.csv")
```


```python
print(df.shape)
df.describe().transpose()
```

    (30000, 25)
    




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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ID</td>
      <td>30000.0</td>
      <td>15000.500000</td>
      <td>8660.398374</td>
      <td>1.0</td>
      <td>7500.75</td>
      <td>15000.5</td>
      <td>22500.25</td>
      <td>30000.0</td>
    </tr>
    <tr>
      <td>LIMIT_BAL</td>
      <td>30000.0</td>
      <td>167484.322667</td>
      <td>129747.661567</td>
      <td>10000.0</td>
      <td>50000.00</td>
      <td>140000.0</td>
      <td>240000.00</td>
      <td>1000000.0</td>
    </tr>
    <tr>
      <td>SEX</td>
      <td>30000.0</td>
      <td>1.603733</td>
      <td>0.489129</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>EDUCATION</td>
      <td>30000.0</td>
      <td>1.853133</td>
      <td>0.790349</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>MARRIAGE</td>
      <td>30000.0</td>
      <td>1.551867</td>
      <td>0.521970</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>AGE</td>
      <td>30000.0</td>
      <td>35.485500</td>
      <td>9.217904</td>
      <td>21.0</td>
      <td>28.00</td>
      <td>34.0</td>
      <td>41.00</td>
      <td>79.0</td>
    </tr>
    <tr>
      <td>PAY_0</td>
      <td>30000.0</td>
      <td>-0.016700</td>
      <td>1.123802</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_2</td>
      <td>30000.0</td>
      <td>-0.133767</td>
      <td>1.197186</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_3</td>
      <td>30000.0</td>
      <td>-0.166200</td>
      <td>1.196868</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_4</td>
      <td>30000.0</td>
      <td>-0.220667</td>
      <td>1.169139</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_5</td>
      <td>30000.0</td>
      <td>-0.266200</td>
      <td>1.133187</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_6</td>
      <td>30000.0</td>
      <td>-0.291100</td>
      <td>1.149988</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>BILL_AMT1</td>
      <td>30000.0</td>
      <td>51223.330900</td>
      <td>73635.860576</td>
      <td>-165580.0</td>
      <td>3558.75</td>
      <td>22381.5</td>
      <td>67091.00</td>
      <td>964511.0</td>
    </tr>
    <tr>
      <td>BILL_AMT2</td>
      <td>30000.0</td>
      <td>49179.075167</td>
      <td>71173.768783</td>
      <td>-69777.0</td>
      <td>2984.75</td>
      <td>21200.0</td>
      <td>64006.25</td>
      <td>983931.0</td>
    </tr>
    <tr>
      <td>BILL_AMT3</td>
      <td>30000.0</td>
      <td>47013.154800</td>
      <td>69349.387427</td>
      <td>-157264.0</td>
      <td>2666.25</td>
      <td>20088.5</td>
      <td>60164.75</td>
      <td>1664089.0</td>
    </tr>
    <tr>
      <td>BILL_AMT4</td>
      <td>30000.0</td>
      <td>43262.948967</td>
      <td>64332.856134</td>
      <td>-170000.0</td>
      <td>2326.75</td>
      <td>19052.0</td>
      <td>54506.00</td>
      <td>891586.0</td>
    </tr>
    <tr>
      <td>BILL_AMT5</td>
      <td>30000.0</td>
      <td>40311.400967</td>
      <td>60797.155770</td>
      <td>-81334.0</td>
      <td>1763.00</td>
      <td>18104.5</td>
      <td>50190.50</td>
      <td>927171.0</td>
    </tr>
    <tr>
      <td>BILL_AMT6</td>
      <td>30000.0</td>
      <td>38871.760400</td>
      <td>59554.107537</td>
      <td>-339603.0</td>
      <td>1256.00</td>
      <td>17071.0</td>
      <td>49198.25</td>
      <td>961664.0</td>
    </tr>
    <tr>
      <td>PAY_AMT1</td>
      <td>30000.0</td>
      <td>5663.580500</td>
      <td>16563.280354</td>
      <td>0.0</td>
      <td>1000.00</td>
      <td>2100.0</td>
      <td>5006.00</td>
      <td>873552.0</td>
    </tr>
    <tr>
      <td>PAY_AMT2</td>
      <td>30000.0</td>
      <td>5921.163500</td>
      <td>23040.870402</td>
      <td>0.0</td>
      <td>833.00</td>
      <td>2009.0</td>
      <td>5000.00</td>
      <td>1684259.0</td>
    </tr>
    <tr>
      <td>PAY_AMT3</td>
      <td>30000.0</td>
      <td>5225.681500</td>
      <td>17606.961470</td>
      <td>0.0</td>
      <td>390.00</td>
      <td>1800.0</td>
      <td>4505.00</td>
      <td>896040.0</td>
    </tr>
    <tr>
      <td>PAY_AMT4</td>
      <td>30000.0</td>
      <td>4826.076867</td>
      <td>15666.159744</td>
      <td>0.0</td>
      <td>296.00</td>
      <td>1500.0</td>
      <td>4013.25</td>
      <td>621000.0</td>
    </tr>
    <tr>
      <td>PAY_AMT5</td>
      <td>30000.0</td>
      <td>4799.387633</td>
      <td>15278.305679</td>
      <td>0.0</td>
      <td>252.50</td>
      <td>1500.0</td>
      <td>4031.50</td>
      <td>426529.0</td>
    </tr>
    <tr>
      <td>PAY_AMT6</td>
      <td>30000.0</td>
      <td>5215.502567</td>
      <td>17777.465775</td>
      <td>0.0</td>
      <td>117.75</td>
      <td>1500.0</td>
      <td>4000.00</td>
      <td>528666.0</td>
    </tr>
    <tr>
      <td>default payment next month</td>
      <td>30000.0</td>
      <td>0.221200</td>
      <td>0.415062</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Setting and Normalizing Target and Predictor Variables


```python
target_column = ['default payment next month'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ID</td>
      <td>30000.0</td>
      <td>0.500017</td>
      <td>0.288680</td>
      <td>0.000033</td>
      <td>0.250025</td>
      <td>0.500017</td>
      <td>0.750008</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>LIMIT_BAL</td>
      <td>30000.0</td>
      <td>0.167484</td>
      <td>0.129748</td>
      <td>0.010000</td>
      <td>0.050000</td>
      <td>0.140000</td>
      <td>0.240000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>SEX</td>
      <td>30000.0</td>
      <td>0.801867</td>
      <td>0.244565</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>EDUCATION</td>
      <td>30000.0</td>
      <td>0.308856</td>
      <td>0.131725</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.333333</td>
      <td>0.333333</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>MARRIAGE</td>
      <td>30000.0</td>
      <td>0.517289</td>
      <td>0.173990</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.666667</td>
      <td>0.666667</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>AGE</td>
      <td>30000.0</td>
      <td>0.449184</td>
      <td>0.116682</td>
      <td>0.265823</td>
      <td>0.354430</td>
      <td>0.430380</td>
      <td>0.518987</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_0</td>
      <td>30000.0</td>
      <td>-0.002087</td>
      <td>0.140475</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_2</td>
      <td>30000.0</td>
      <td>-0.016721</td>
      <td>0.149648</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_3</td>
      <td>30000.0</td>
      <td>-0.020775</td>
      <td>0.149608</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_4</td>
      <td>30000.0</td>
      <td>-0.027583</td>
      <td>0.146142</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_5</td>
      <td>30000.0</td>
      <td>-0.033275</td>
      <td>0.141648</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_6</td>
      <td>30000.0</td>
      <td>-0.036388</td>
      <td>0.143748</td>
      <td>-0.250000</td>
      <td>-0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT1</td>
      <td>30000.0</td>
      <td>0.053108</td>
      <td>0.076345</td>
      <td>-0.171672</td>
      <td>0.003690</td>
      <td>0.023205</td>
      <td>0.069560</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT2</td>
      <td>30000.0</td>
      <td>0.049982</td>
      <td>0.072336</td>
      <td>-0.070917</td>
      <td>0.003033</td>
      <td>0.021546</td>
      <td>0.065052</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT3</td>
      <td>30000.0</td>
      <td>0.028252</td>
      <td>0.041674</td>
      <td>-0.094505</td>
      <td>0.001602</td>
      <td>0.012072</td>
      <td>0.036155</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT4</td>
      <td>30000.0</td>
      <td>0.048524</td>
      <td>0.072156</td>
      <td>-0.190671</td>
      <td>0.002610</td>
      <td>0.021369</td>
      <td>0.061134</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT5</td>
      <td>30000.0</td>
      <td>0.043478</td>
      <td>0.065573</td>
      <td>-0.087723</td>
      <td>0.001901</td>
      <td>0.019527</td>
      <td>0.054133</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>BILL_AMT6</td>
      <td>30000.0</td>
      <td>0.040421</td>
      <td>0.061928</td>
      <td>-0.353141</td>
      <td>0.001306</td>
      <td>0.017752</td>
      <td>0.051160</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT1</td>
      <td>30000.0</td>
      <td>0.006483</td>
      <td>0.018961</td>
      <td>0.000000</td>
      <td>0.001145</td>
      <td>0.002404</td>
      <td>0.005731</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT2</td>
      <td>30000.0</td>
      <td>0.003516</td>
      <td>0.013680</td>
      <td>0.000000</td>
      <td>0.000495</td>
      <td>0.001193</td>
      <td>0.002969</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT3</td>
      <td>30000.0</td>
      <td>0.005832</td>
      <td>0.019650</td>
      <td>0.000000</td>
      <td>0.000435</td>
      <td>0.002009</td>
      <td>0.005028</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT4</td>
      <td>30000.0</td>
      <td>0.007771</td>
      <td>0.025227</td>
      <td>0.000000</td>
      <td>0.000477</td>
      <td>0.002415</td>
      <td>0.006463</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT5</td>
      <td>30000.0</td>
      <td>0.011252</td>
      <td>0.035820</td>
      <td>0.000000</td>
      <td>0.000592</td>
      <td>0.003517</td>
      <td>0.009452</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>PAY_AMT6</td>
      <td>30000.0</td>
      <td>0.009865</td>
      <td>0.033627</td>
      <td>0.000000</td>
      <td>0.000223</td>
      <td>0.002837</td>
      <td>0.007566</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>default payment next month</td>
      <td>30000.0</td>
      <td>0.221200</td>
      <td>0.415062</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Creating Train and Test Sets


```python
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)
```

    (21000, 24)
    (9000, 24)
    

Setting Neural Net Parameters


```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train.ravel())

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
```

Checking Accuracy of Train Dataset


```python
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
```

    [[15613   732]
     [ 3039  1616]]
                  precision    recall  f1-score   support
    
               0       0.84      0.96      0.89     16345
               1       0.69      0.35      0.46      4655
    
        accuracy                           0.82     21000
       macro avg       0.76      0.65      0.68     21000
    weighted avg       0.80      0.82      0.80     21000
    
    

Checking Accuracy of Test Dataset


```python
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
```

    [[6682  337]
     [1249  732]]
                  precision    recall  f1-score   support
    
               0       0.84      0.95      0.89      7019
               1       0.68      0.37      0.48      1981
    
        accuracy                           0.82      9000
       macro avg       0.76      0.66      0.69      9000
    weighted avg       0.81      0.82      0.80      9000
    
    
