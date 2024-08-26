# import libraries

import streamlit as st
import pandas as pd
import numpy as np
import statistics
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# titles and info
st.title('RN Tenure Prediction Application')

st.info('This application will allow you to predict whether an RN will be a high (10+ years), medium (6+ years), or low (< 6 years) tenure employee.')

# load dataset
url = 'https://raw.githubusercontent.com/jennifergurley/capstone/main/rndatafinal.csv'
data = pd.read_csv(url)

#display data
with st.expander('View Dataset'):
     st.write('RN Data Final')
     data

# calculate and view statistics
with st.expander('Review Data Statistics'):
     st.write('Basic Statistics')
     df = pd.DataFrame(data)
     print(df.describe())
     st.write('File Info')
     data.info()
     st.write('Check for Null Values')
     data.isnull().sum()
     st.write('Calculate Median Tenure')
     statistics.median(data.tenure)

# add field to show user defined tenure categories

with st.expander('Add Tenure Categories'):
     data['tenurecategory'] = ['high tenure' if x > 10 else 'medium tenure' if x > 6 else 'low tenure' for x in data.tenure]

with st.expander('Data Visualization'):
     st.bar_chart(data=data, x="gradrn", y="tenure")
     st.scatter_chart(data=data, x="active", y="tenure")
     st.scatter_chart(data=data, x="degree", y="tenure")
     st.scatter_chart(data=data, x="volcerts", y="tenure")
     st.scatter_chart(data=data, x="rnyears", y="tenure")
     st.scatter_chart(data=data, x="lasthire", y="tenure")
     st.scatter_chart(data=data, x="avghours", y="tenure")
     st.scatter_chart(data=data, x="avgoncall", y="tenure")
     st.scatter_chart(data=data, x="lateorcallin", y="tenure")
     st.scatter_chart(data=data, x="avgshiftwork", y="tenure")
     st.scatter_chart(data=data, x="retentionrisk", y="tenure")
     st.scatter_chart(data=data, x="hiresource", y="tenure")
     st.scatter_chart(data=data, x="performance", y="tenure")

# split into training and testing
X = pd.DataFrame(data[['gradrn', 'degree', 'volcerts', 'rnyears', 'lasthire', 'avghours', 'avgoncall', 'lateorcallin', 'avgshiftwork', 'retentionrisk', 'hiresource', 'performance']])
y = pd.DataFrame(data[['tenurecategory']])
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20)

# one-hot encoding
X_train_hot = pd.get_dummies(X_train)
X_test_hot = pd.get_dummies(X_test)

X_train_hot

y_train

print(y_train.shape)

X_test_hot

y_test

knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')

knn.fit(X_train_hot,y_train.values.ravel())

pred = knn.predict(X_test_hot)

error_rate = []

for i in range(1,160):
     knn = KNeighborsClassifier(n_neighbors=i)
     knn.fit(X_train_hot,y_train.values.ravel())
     pred_i = knn.predict(X_test_hot)
     error_rate.append(np.mean(pred_i != y_test.values.ravel()))


plt.figure(figsize=(10,6))
plt.plot(range(1,160),error_rate, color='blue',linestyle='dashed')

error_rate

print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))









