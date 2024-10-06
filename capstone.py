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


# create application title and overview
st.title('Registered Nurse Tenure Prediction Application')

st.info('This application will allow you to predict whether an RN will be a high (7+ years), medium (3+ years), or low (< 3 years) tenure employee.')

# load dataset
url = 'https://raw.githubusercontent.com/jennifergurley/capstone/main/rndatafinal.csv'
data = pd.read_csv(url)


# streamlit expanders are used to keep the application from being too cluttered
# each expander will have a descriptive title for ease of selection

st.header('Data Review and Exploration')

# display dataset
with st.expander('View the Registered Nurse Dataset'):
     st.write('** Registered Nurse Dataset **')
     data


# calculate and view statistics
with st.expander('Review Data Statistics'):
     st.write('** Basic Statistics **')
     df = pd.DataFrame(data)
     st.write(df.describe())
     st.write('** File Info **')
     st.write(data.info())
     st.write('** Check for Null Values **')
     st.write(data.isnull().sum())
     st.write('** Median Tenure **')
     st.write(statistics.median(data.tenure))


# add field to show user defined tenure categories
with st.expander('Add Tenure Categories'):
     st.write('High tenure is 7+ years; medium tenure is between 3 and 7 years; low tenure is less than 3 years')
     data['tenurecategory'] = ['high tenure' if x > 7 else 'medium tenure' if x > 3 else 'low tenure' for x in data.tenure]
     data


# create scatter plots
with st.expander('Data Visualization for Selected Attributes'):
     st.write('** Tenure vs Years as RN **')
     st.scatter_chart(data=data, x="tenure", y="rnyears")
     st.write('** Tenure vs Average Hours Worked **')
     st.scatter_chart(data=data, x="tenure", y="avghours")
     st.write('** Tenure vs Grad RN Orientation Attendance **')
     st.scatter_chart(data=data, x="tenure", y="gradrn")
     st.write('** Tenure vs Voluntary Certifications **')
     st.scatter_chart(data=data, x="tenure", y="volcerts")



# split into training and testing
X = pd.DataFrame(data[['gradrn', 'degree', 'volcerts', 'rnyears', 'lasthire', 'avghours', 'avgoncall', 'lateorcallin', 'avgshiftwork', 'retentionrisk', 'hiresource', 'performance']])
y = pd.DataFrame(data[['tenurecategory']])
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.20)

# one-hot encoding
X_train_hot = pd.get_dummies(X_train)
X_test_hot = pd.get_dummies(X_test)

# display training and test sets
with st.expander('Training and Test Sets'):
     st.write('Data has been one-hot encoded')
     st.write(' ')
     st.write('X_train')
     X_train_hot
     st.write('y_train')
     y_train
     st.write('X_test')
     X_test_hot
     st.write('y_test')
     y_test


# create sidebar to get user inputs
with st.sidebar:
     
# user input to set KNN parameters
     st.header('Select KNN parameters')
     st.write('Select the KNN parameters you would like to use')
     udneighbors = st.slider ('Select KNN neighbors: ', 2, 30, 10)
     udmetric = st.selectbox ('Select KNN metric: ', ('minkowski', 'euclidean'))


with st.sidebar:
     st.header('User Manual')
     with st.expander('Overview'):
          st.write('The Registered Nurse Tenure Prediction Application consists of three sections:')
          st.subheader('Data Review and Exploration')
          st.write('Allows you to view the dataset, statistics, visualizations, and training / test datasets.')
          st.write('Build and Fit the KNN Model allows you to select the parameters you wish to use to run a KNN model against the data and view the results. You can also fine-tune the model by selecting different parameters.')
          st.write('RN Retention Actions allows you to input parameters about an RN to get a personalized list of suggestions to help with employee engagement.') 
     with st.expander('Getting Started'):
          st.write('The application is available online at https://capstone-jennifergurley.streamlit.app/. No login is required. If you are unable to access the application, please contact jennifergurley@gmail.com for assistance.')
     with st.expander('DATA REVIEW AND EXPLORATION section'):
          st.subheader('Viewing the Dataset')
          st.write('The dataset will automatically load when you access the application. To view the data, select the ‘View the Registered Nurse Dataset’ link in the main section to expand the screen. You may scroll down or to the right to view all 200 records and 17 columns.')
          st.subheader('Reviewing Statistics')
          st.write('To view basic statistics for the dataset, select the ‘Review Data Statistics’ link in the main section.')
          st.subheader('Add Tenure Categories')
          st.write('Tenure is the predictive variable for this model and has been divided into three categories: high tenure (7+ years), medium tenure (3-7 years), and low tenure (less than 3 years). You may view these categories by expanding the ‘Add Tenure Categories’ link.')
          st.subheader('Reviewing Visualizations')
          st.write('Visualizations for selected attributes can be viewed by selecting the ‘Data Visualization for Selected Attributes’ link.')
          st.subheader('Viewing the Train and Test Datasets')
          st.write('The dataset has been divided into 80% training and 20% test, which can be viewed by selecting the ‘Training and Test Sets’ link.')
     with st.expander('BUILD AND FIT THE KNN MODEL section'):
          st.subheader('Setting the KNN Model Parameters')
          st.write('In the sidebar, you may choose to model data for 2 – 30 nearest neighbors by using the slider or select the Minkowski or Euclidean metric from the drop-down.')
          st.write('You can also fine-tune your selections by expanding the Error Rates, Confusion Matrix, Classification Report, and Plotted Error Rates sections and updating the parameters.')
     with st.expander('RN RETENTION ACTIONS section'):
          st.subheader('Modeling a Test Case')
          st.write('Enter basic information about an RN to get a personalized retention actions report based on their degree level and experience as an RN.')


st.header('Build and Fit the KNN Model')
st.write('From the sidebar, select the KNN parameters you would like to use to build the model, then use the expanders below to view the results. You may reset the parameters to fine-tune the model.')


# build and fit model

knn = KNeighborsClassifier(n_neighbors=udneighbors, metric=udmetric)

knn.fit(X_train_hot,y_train.values.ravel())

pred = knn.predict(X_test_hot)

with st.expander('Error Rates'):
     error_rate = []
     for i in range(1,160):
          knn = KNeighborsClassifier(n_neighbors=i)
          knn.fit(X_train_hot,y_train.values.ravel())
          pred_i = knn.predict(X_test_hot)
          error_rate.append(np.mean(pred_i != y_test.values.ravel()))
     error_rate

with st.expander('Confusion Matrix'):
     st.write(confusion_matrix(y_test,pred))

with st.expander('Classification Report'):
     st.write(classification_report(y_test,pred))


with st.expander('Plot Error Rates'):
     st.line_chart(error_rate)
     #plt.figure(figsize=(10,6))
     #plt.plot(range(1,160),error_rate, color='blue',linestyle='dashed')
     



# user input to generate report to optimize retention based on RN attributes

st.header('RN Retention Actions')
st.write('Enter information about a new hire RN below to see actions you can take to help retain them longer.')

rnname = st.text_input ('What is this RNs name? ')
degree = st.selectbox ('Highest RN Degree?:', ('Associates', 'Bachelors', 'Masters'))
rnyears = st.slider ('How long has employee had RN license?', 0, 45, 10)
st.write(f'Expand the RN Retention Actions section below to see suggestions for {rnname}.')


# create input feature dataset
inputdata = {'degree': degree,
             'rnyears': rnyears}
             
inputdf = pd.DataFrame(inputdata, index=[0])



#display user input data
with st.expander('See Your RN Retention Actions Report'):
     st.write(f'{rnname} has a {degree} degree and {rnyears} years experience.')
     st.write(' ')
     st.write(f'Here are some actions you can take to increase retention for {rnname}:')
     st.write(' ')
     if degree == "Associates":
          st.write(f'Consider offering tuition reimbursement to encourage {rnname} to earn a BSN.')
     if rnyears < 5:
          st.write(f'Since {rnname} has less than 5 years experience, attending the Grad RN Orientation Program may be beneficial.')
     if rnyears > 15:
          st.write(f'{rnname} has more than 15 years experience and may value the chance to be a preceptor or attend leadership training.')
     st.write(f'Work with {rnname} to set short and long term goals. Employees with goals are more engaged.')


