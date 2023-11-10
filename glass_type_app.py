import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


feature_col=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
@st.cache()
def prediction(_model,feature_col):
  glass_type=_model.predict([feature_col])
  if glass_type[0]==1:
    return('building windows float processed')
  elif glass_type[0]==2:
    return('building windows non float processed')
  elif glass_type[0]==3:
    return('vehicle windows float processed')  
  elif glass_type[0]==4:
    return('vehicle windows non float processed')
  elif glass_type[0]==5:
    return('containers')
  elif glass_type[0]==6:
    return('tableware')
  else:
    return('headlamp') 

st.sidebar.title('Exploratory Data Analysis')
st.title('Glass Type Predictor')

if st.sidebar.checkbox('Show raw data'):
  st.subheader('Glass Type Dataset')
  st.dataframe(glass_df)

st.sidebar.subheader('Scatter Plot')
# Choosing x-axis values for the scatter plot.
# Add a multiselect in the sidebar with the 'Select the x-axis values:' label
# and pass all the 9 features as a tuple i.e. ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe') as options.
# Store the current value of this widget in the 'features_list' variable.
feature_list=st.sidebar.multiselect('Select the x-axis values:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)
for i in feature_list:
  st.subheader(f'scatter plot between {i} and glasstype')
  plt.figure(figsize=(10,5))
  sns.scatterplot(x=glass_df[i],y=glass_df['GlassType'])
  st.pyplot()



st.sidebar.subheader('Boxplot')
feature_list2=st.sidebar.multiselect('Select the features for boxplot:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
for i in feature_list2:
  st.subheader(f'Boxplot for {i}')
  plt.figure(figsize=(10,5))
  sns.boxplot(x=glass_df[i])
  st.pyplot()

st.sidebar.subheader('Visualisation Selector')
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
plot_types=st.sidebar.multiselect('Select the Charts/Plots:', ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if 'Histogram' in plot_types:
  feature_list=st.sidebar.selectbox('Select the features for Histogram:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  st.subheader(f'Histogram for {feature_list}')
  plt.figure(figsize=(10,5))
  plt.hist(x=glass_df[feature_list],bins='sturges',edgecolor='red')
  st.pyplot()

if 'Box Plot' in plot_types:
  feature_list2=st.sidebar.selectbox('Select the features for boxplot:', ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

  st.subheader(f'Boxplot for {feature_list2}')
  plt.figure(figsize=(10,5))
  sns.boxplot(x=glass_df[feature_list2])
  st.pyplot()

if 'Count Plot' in plot_types:
  st.subheader('Count Plot')
  plt.figure(figsize=(10,5))
  sns.countplot(x=glass_df['GlassType'])
  st.pyplot()

if 'Pie Chart' in plot_types:
  st.subheader('Pie Chart')
  plt.figure(figsize=(10,5))
  plt.pie(glass_df['GlassType'].value_counts())
  st.pyplot()

if 'Correlation Heatmap' in plot_types:
  st.subheader('Corrrelation Heatmap')
  plt.figure(figsize=(10,5))
  sns.heatmap(glass_df.corr(), annot=True)
  st.pyplot()

if 'Pair Plot' in plot_types:
  st.subheader('Pair Plot')
  plt.figure(figsize=(10,5))
  sns.pairplot(glass_df)
  st.pyplot()

st.sidebar.subheader('Select your values')
ri=st.sidebar.slider('Ri',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na=st.sidebar.slider('Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg=st.sidebar.slider('Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al=st.sidebar.slider('Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
si=st.sidebar.slider('Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k=st.sidebar.slider('K',float(glass_df['K'].min()),float(glass_df['K'].max()))
ca=st.sidebar.slider('Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
ba=st.sidebar.slider('Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
fe=st.sidebar.slider('Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))

st.sidebar.subheader('Choose Classifier')
# Add a selectbox in the sidebar with label 'Classifier'.
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyperparameters:')
  c=st.sidebar.number_input('C value',1,100,step=1)
  gamma=st.sidebar.number_input('gamma',1,100,step=1)
  kernel=st.sidebar.radio('kernel',('linear', 'rbf', 'poly'))
    # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
    # This 'if' statement must be inside the above 'if' statement.
  if st.sidebar.button('Classify'):
    svc_model=SVC(C = c, kernel = kernel, gamma = gamma).fit(X_train,y_train)
    y_pred=svc_model.predict(X_test)
    accuracy=svc_model.score(X_train,y_train)
    glass_type=prediction(svc_model, [ri,na,mg,al,si,k,ca,ba,fe])
    st.write(f'the predicted glass type is {glass_type}')
    st.write(f' the accuracy of the model is {accuracy}')
    st.write(confusion_matrix(y_test,y_pred))

if classifier == 'Random Forest Classifier':
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
    max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)

    if st.sidebar.button('Classify'):
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
        rf_clf.fit(X_train,y_train)
        accuracy = rf_clf.score(X_test, y_test)
        glass_type = prediction(rf_clf, [ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        y_pred=rf_clf.predict(X_test)
        st.write(confusion_matrix(y_test,y_pred))
        

if classifier == 'Logistic Regression':
    st.sidebar.subheader("Model Hyperparameters")
    c = st.sidebar.number_input("c", 1, 100, step = 1)
    max_iter=st.sidebar.slider('Maximum Iteration', 10, 100, step=10)
    if st.sidebar.button('Classify'):
        st.subheader("Logistic Regression")
        lg=LogisticRegression(C = c, max_iter = max_iter)
        lg.fit(X_train,y_train)
        accuracy = lg.score(X_test, y_test)
        glass_type = prediction(lg, [ri, na, mg, al, si, k, ca, ba, fe])
        st.write("The Type of glass predicted is:", glass_type)
        st.write("Accuracy", accuracy.round(2))
        y_pred=lg.predict(X_test)
        st.write(confusion_matrix(y_test,y_pred))
