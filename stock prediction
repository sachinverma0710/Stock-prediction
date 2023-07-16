#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

#importing dataset
df = pd.read_csv('Iris.csv')

df.head()

#Displaying the basic statistics about data
df.describe()

#Displaying the information regarding the data type
df.info()

#Displaying the number of sample for each class
df['Species'].value_counts()

#Checking for the null values
df.isnull().sum()

#Displaying the number of rows and columns
df.shape

#Visualizing the data
sns.set_style('whitegrid')
sns.FacetGrid(df, hue = 'Species').map(plt.scatter, 'SepalLengthCm','SepalWidthCm').add_legend()
plt.show()

#Scatterplot for the different features of flower
colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
plt.subplots(figsize=(15,10))
plt.subplot(2,2,1)
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()

plt.subplot(2,2,2)
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()

plt.subplot(2,2,3)
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()

plt.subplot(2,2,4)
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()

sns.pairplot(df.drop(['Id'],axis=1), hue='Species')
plt.show()

df['Sepal_diff'] = df['SepalLengthCm'] - df['SepalWidthCm']
df['Petal_diff'] = df['PetalLengthCm'] - df['PetalWidthCm']

df['Sepal_Petal_Len_diff'] = df['SepalLengthCm'] - df['PetalLengthCm']
df['Sepal_Petal_Wid_diff'] = df['SepalWidthCm'] - df['PetalWidthCm']

df['Sepal_Petal_Len_Wid_diff'] = df['SepalLengthCm'] - df['PetalWidthCm']
df['Sepal_Petal_Wid_Len_diff'] = df['SepalWidthCm'] - df['PetalLengthCm'] 

sns.pairplot(df[['Species', 'Sepal_diff', 'Petal_diff', 'Sepal_Petal_Len_diff','Sepal_Petal_Wid_diff', 'Sepal_Petal_Len_Wid_diff', 'Sepal_Petal_Wid_Len_diff']], hue='Species')
plt.show()

df.head(10)

df.drop(['Id'], axis =1, inplace = True)

df.head(10)

for i in df.columns:
    if i == 'Species':
        continue
    sns.set_style('whitegrid')
    sns.FacetGrid(df,hue='Species')\
    .map(sns.distplot,i)\
    .add_legend()
    plt.show()

#Building the decision tree classifier algorithm
!pip install graphviz
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score

x = df[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

#Spliting the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#Splitting the dataset into validation training and testing
xt, xv, yt, yv = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

Iris_clf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_clf.fit(xt, yt)

tree.plot_tree(Iris_clf)

print('Accuracy score is:',cross_val_score(Iris_clf, xt, yt, cv=3, scoring='accuracy').mean())

#Check the validation data
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

Y_hat = Iris_clf.predict(xv)


print('Accuracy score for validation test data is:',accuracy_score(yv, Y_hat))
multilabel_confusion_matrix(yv, Y_hat)

YT_hat = Iris_clf.predict(X_test)
YT_hat

print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Y_test, YT_hat)*100,'%')
multilabel_confusion_matrix(Y_test , YT_hat)
