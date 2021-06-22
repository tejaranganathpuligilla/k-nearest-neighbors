# KNN - K Nearest Neighbours

KNN is a supervised ML algorithm mainly used for classification predictive problems 

Understanding the KNN with a project 

## Import some libraries required 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Pandas is used for data analysis. The library allows various data manipulation operations such as merging, reshaping, selecting, as well as data cleaning, and data wrangling features

Numpy is used in the industry for array computing

Seaborn is a Python library used for enhanced data visualization

#Importing the data set

get the dataset and then check its data so that there are no null values or any objects

```python
df = pd.read_csv('KNN_Project_Data')
dh.head()
```
![image](https://user-images.githubusercontent.com/82372055/122861035-628f7780-d33c-11eb-8bed-d3d4264850a9.png)

## Now make a Pairplot with the HUE=Target 

```python
sns.pairplot(df, hue = 'TARGET CLASS')
```

![image](https://user-images.githubusercontent.com/82372055/122861142-9074bc00-d33c-11eb-9e72-aa6327087a75.png)


from this pairplot we can see the relation between the features 

## Standerdizing the dataset 

Standardization is done so that the algorithm will not consider the big valued features as the useful and small valued features as noise 

so that the intensity of data points will not change 

standardizing makes all values in the range of [-1,1] if negative values are there in the dataset or else just in the range of [0,1]

## importing the library 

```python
from sklearn.preprocessing import StandardScaler

```

## Create a StandardScaler() object called scaler then fit and transform the data 

```python
myscaler = StandardScaler()
myscaler.fit(X = df.drop('TARGET CLASS', axis = 1))
X = myscaler.transform(X = df.drop('TARGET CLASS', axis = 1))
```

## TRAIN TEST SPLIT 

Splitting the data for the training and testing process

### import the train_test_split from sklearn.model_selection

```python 
from sklearn.model_selection import train_test_split

```
### Continuing with the TTS process

```python
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
```

## Using KNN

Import KNeighborsClassifier from scikit learn

```python
myKNN = KNeighborsClassifier(n_neighbors = 1)
```
 Fitting this KNN model to the training data
```python
myKNN.fit(X_train, y_train)

```
Here all the process done in KNN algorithm is done 

for each point in the data set 
                    -Calculate the distance between test data and each row of training data 
                      with the help of any of the method namely: Euclidean, Manhattan, or Hamming distance. 
                      The most commonly used method to calculate distance is Euclidea

                    - sort them in ascending order
 
                    - top K rows from the sorted array
 
                    - top K rows from the sorted array

## Predictions and Evaluations


Let's evaluate our KNN model! 

Using the predict method with our KNN model and x_test 

```python
y_predict = myKNN.predict(X_test)

```


checking confusion matrics and classification report 

```python
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))

```
![image](https://user-images.githubusercontent.com/82372055/122861275-cade5900-d33c-11eb-8798-678b63292e7f.png)
![image](https://user-images.githubusercontent.com/82372055/122861366-f5c8ad00-d33c-11eb-98f0-db2f5783b900.png)

So as a human we can not know which k value should be used so to get the best results output

We use the elbow method to pick a good k value 

 Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list.

```python
err_rates = []
for idx in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = idx)
    knn.fit(X_train, y_train)
    pred_idx = knn.predict(X_test)
    err_rates.append(np.mean(y_test != pred_idx))
```
now creating the plot so that to check where we are getting less error for the corresponding value 

```python
plt.style.use('ggplot')
plt.subplots(figsize = (10,6))
plt.plot(range(1,40), err_rates, linestyle = 'dashed', color = 'blue', marker = 'o', markerfacecolor = 'red')
plt.xlabel('K-value')
plt.ylabel('Error Rate')
plt.title('Error Rate vs K-value')
```
![image](https://user-images.githubusercontent.com/82372055/122861417-0c6f0400-d33d-11eb-852c-e221573f8eed.png)

## Now retraining the model with the new K value 

```python
myKNN = KNeighborsClassifier(n_neighbors = 31)
myKNN.fit(X_train,y_train)
y_predict = myKNN.predict(X_test)
```
![image](https://user-images.githubusercontent.com/82372055/122861445-1c86e380-d33d-11eb-9318-4cd70a801912.png)

