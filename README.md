# Ensemble Methods Code Examples

Using SciKit-Learn to practice with ensemble methods


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```

## Austin Pet Dataset

Trying to predict whether a given pet will be adopted

### Data Preparation


```python
austin_df = pd.read_csv("austin.csv")
```


```python
from shelter_preprocess import preprocess_df
```


```python
# normally we would do preprocessing after train-test split
# in this case the preprocessing is all just "hard-coded"
df = preprocess_df(austin_df)
```


```python
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
      <th>is_dog</th>
      <th>age_in_days</th>
      <th>is_female</th>
      <th>adoption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>46</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>136</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>575</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>748</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (908, 4)




```python
from sklearn.model_selection import train_test_split
```


```python
y = df["adoption"]
X = df.drop("adoption", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
```

### Modeling

Stacking classifier composed of:

1. Random forest classifier (ensemble of trees with 2 kinds of randomness)
2. kNN classifier
3. Logistic regression

Then it uses the default final estimator—a logistic regression—to aggregate the answers from the other models


```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
```


```python
rfc = RandomForestClassifier(random_state=2020)
```


```python
knn = KNeighborsClassifier()
```


```python
lr = LogisticRegression(penalty='none', random_state=2020)
```


```python
# similar to a pipeline, the stacking classifier wants you to give
# each model a human-readable name string in addition to just the
# model variable (initially I didn't notice this in the docs)
estimators = [
    ("random_forest", rfc),
    ("knn", knn),
    ("logistic_regression", lr)
]
```


```python
stack = StackingClassifier(estimators=estimators)
```


```python
%time
stack.fit(X_train, y_train)
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 7.15 µs





    StackingClassifier(cv=None,
                       estimators=[('random_forest',
                                    RandomForestClassifier(bootstrap=True,
                                                           ccp_alpha=0.0,
                                                           class_weight=None,
                                                           criterion='gini',
                                                           max_depth=None,
                                                           max_features='auto',
                                                           max_leaf_nodes=None,
                                                           max_samples=None,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None,
                                                           min_samples_leaf=1,
                                                           min_samples_split=2,
                                                           min_weight_fraction_leaf=0.0,
                                                           n_estimators=100,...
                                                         p=2, weights='uniform')),
                                   ('linear_regression',
                                    LogisticRegression(C=1.0, class_weight=None,
                                                       dual=False,
                                                       fit_intercept=True,
                                                       intercept_scaling=1,
                                                       l1_ratio=None, max_iter=100,
                                                       multi_class='auto',
                                                       n_jobs=None, penalty='none',
                                                       random_state=2020,
                                                       solver='lbfgs', tol=0.0001,
                                                       verbose=0,
                                                       warm_start=False))],
                       final_estimator=None, n_jobs=None, passthrough=False,
                       stack_method='auto', verbose=0)



### Model Evaluation


```python
stack.score(X_test, y_test)
```




    0.7224669603524229




```python
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
```




    0.7004405286343612




```python
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```




    0.7224669603524229




```python
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```




    0.5198237885462555



### Summary for Austin Dataset

The Austin dataset has 908 rows and 4 columns.  The task is predicting whether an animal will be adopted based on:

1. Whether the animal is a dog
2. Whether the animal is female
3. The age of the animal in days

In this case, **the kNN model has the best performance, and there is no improvement when it is stacked with a random forest and logistic regression model**. This may be related to the fact that there are so few features.

Based on this simple analysis (no hyperparameter tuning, no feature engineering, no additional models attempted) we would most likely choose the kNN model as our final, best model

## SciKit-Learn Breast Cancer Dataset

Trying to predict whether a patient has breast cancer, based on 30 features

### Data Preparation


```python
from sklearn.datasets import load_breast_cancer
```


```python
data = load_breast_cancer()
```


```python
print(data.target_names)
```

    ['malignant' 'benign']



```python
print(data.feature_names)
```

    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']



```python
data.data.shape
```




    (569, 30)




```python
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=2020)
```


```python
# I didn't do this for time reasons in the original demo, but in order to
# make sure the logistic regression can converge with all these features,
# we want to scale the data
from sklearn.preprocessing import StandardScaler
```


```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Modeling

We'll just go ahead and use the exact same stack from the previous example

(In reality you would want to do more EDA, make a more intentional decision)


```python
%time
stack.fit(X_train, y_train)
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 5.25 µs





    StackingClassifier(cv=None,
                       estimators=[('random_forest',
                                    RandomForestClassifier(bootstrap=True,
                                                           ccp_alpha=0.0,
                                                           class_weight=None,
                                                           criterion='gini',
                                                           max_depth=None,
                                                           max_features='auto',
                                                           max_leaf_nodes=None,
                                                           max_samples=None,
                                                           min_impurity_decrease=0.0,
                                                           min_impurity_split=None,
                                                           min_samples_leaf=1,
                                                           min_samples_split=2,
                                                           min_weight_fraction_leaf=0.0,
                                                           n_estimators=100,...
                                                         p=2, weights='uniform')),
                                   ('linear_regression',
                                    LogisticRegression(C=1.0, class_weight=None,
                                                       dual=False,
                                                       fit_intercept=True,
                                                       intercept_scaling=1,
                                                       l1_ratio=None, max_iter=100,
                                                       multi_class='auto',
                                                       n_jobs=None, penalty='none',
                                                       random_state=2020,
                                                       solver='lbfgs', tol=0.0001,
                                                       verbose=0,
                                                       warm_start=False))],
                       final_estimator=None, n_jobs=None, passthrough=False,
                       stack_method='auto', verbose=0)



### Model Evaluation


```python
stack.score(X_test, y_test)
```




    0.9790209790209791




```python
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
```




    0.958041958041958




```python
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
```




    0.951048951048951




```python
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
```




    0.965034965034965



#### Looking at the Confusion Matrices


```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(y_test, stack.predict(X_test))
```




    array([[59,  2],
           [ 1, 81]])




```python
confusion_matrix(y_test, rfc.predict(X_test))
```




    array([[58,  3],
           [ 3, 79]])




```python
confusion_matrix(y_test, knn.predict(X_test))
```




    array([[57,  4],
           [ 3, 79]])




```python
confusion_matrix(y_test, lr.predict(X_test))
```




    array([[59,  2],
           [ 3, 79]])



### Summary for Breast Cancer Dataset

We are trying to predict whether a given patient has breast cancer based on 30 different features, with a dataset of 569 records.

In this case, the **stacking classifier was more than the sum of its parts**.  Each of the estimators inside the stacking classifier got around 95-96% accuracy, and the stacking classifier got closer to 98% accuracy.

As also shown by the confusion matrices, the different models are getting things wrong in different ways.  For example, the kNN model has the most false positives, and the logistic regression has the least false positives.  This makes this combination of models + dataset a good candidate for stacking.

Therefore based on this simple analysis (no hyperparameter tuning, no feature engineering, no additional models attempted) we would most likely choose the stacking model as our final, best model

## Conclusion

Stacking classifiers can leverage multiple different models, which can sometimes result in better metrics than any individual model

In this case, we were able to run the analysis fairly quickly, since there was not a lot of data.  If you are using significantly larger datasets, you may need to be more selective with which models you use, and a stacking classifier may end up being too slow


```python

```
