import pickle as pickle
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def clean_data():
 data = pd.read_csv("data/diabetes.csv")
 # make numerical feature categorical
#  data['Pregnancies'] = data['Pregnancies'].astype('category')
#  data['Outcome'] = data['Outcome'].astype('category')
#  # create a boolean mask for categorical columns
#  categorical_mask = (data.dtypes == 'category')
#  # get list of categorical column names
#  categorical_columns = data.columns[categorical_mask].tolist()
#  categorical_columns = ['Pregnancies']
#  dummies = pd.get_dummies(data[categorical_columns],dummy_na = True,drop_first=True)# one-hot-encoding
#  data = pd.concat([data,dummies], axis=1)# make the dummies and concat with original data
#  data.drop(categorical_columns,inplace=True, axis=1)# drop the original columns
#  data['Outcome'] = data['Outcome'].astype('int')
 print(data.head())
 print(data.dtypes)
 return data

def add_sidebar():
 st.sidebar.header('Input Parameters')
 data = clean_data()
 slider_labels = [
    ("Pregnancies", 'Pregnancies'),
    ("Glucose",'Glucose'),
    ("BloodPressure",'BloodPressure'),
    ("SkinThickness",'SkinThickness'),
    ("Insulin",'Insulin'),
    ("BMI",'BMI'),
    ("DiabetesPedigreeFunction",'DiabetesPedigreeFunction'),
    ("Age",'Age')
  ]
 
 input_dict = {}

 for label, key in slider_labels:
  input_dict[key] = st.sidebar.slider(
   label,
   min_value=float(0),
   max_value=float(data[key].max()),
   value=float(data[key].mean())
  )
 
 return input_dict

def get_scaled_value(input_dict):
 data = clean_data()
 x = data.drop(['Outcome'], axis=1)

 scaled_dict = {}

 for key, value in input_dict.items():
  max_val = x[key].max()
  min_val = x[key].min()
  scaled_value = (value - min_val)/ (max_val - min_val)
  scaled_dict[key] = scaled_value

 return scaled_dict

def get_radar_chart(input_data):
 
 input_data = get_scaled_value(input_data)
 categories = ['Glucose','BloodPressure','SkinThickness','Insulin'
    ,'BMI','DiabetesPedigreeFunction','Age']

 fig = go.Figure()

 fig.add_trace(go.Scatterpolar(
      r=[
       input_data['Glucose'], input_data['BloodPressure']
       ,input_data['Insulin'], input_data['SkinThickness'], input_data['Age']
       ,input_data['BMI'],input_data['DiabetesPedigreeFunction']
      ],
      theta=categories,
      fill='toself',
      name='Value'
 ))


 fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=False
 )

 return fig

def add_predictions(input_data):
 model = pickle.load(open('model/model.pkl', 'rb'))
 scaler = pickle.load(open('model/scaler.pkl', 'rb'))
 input_array = np.array(list(input_data.values())).reshape(1,-1)

 input_array_scaled = scaler.transform(input_array)
 
 prediction = model.predict(input_array_scaled)
 st.subheader('Cell cluster prediction')
 st.write('The cell cluster is:')
 #st.write(prediction)
 if prediction[0] == 0:
  st.write("<span class= 'diagnosis Not_diabetic'>Not Diabetic</span>", unsafe_allow_html=True)
 else:
  st.write("<span class= 'diagnosis diabetic'>Diabetic</span>", unsafe_allow_html=True)
  

 st.write('Probability of Not being diabetic:', model.predict_proba(input_array_scaled)[0][0])
 st.write('Probability of being diabetic:', model.predict_proba(input_array_scaled)[0][1])
 st.write('This app can assist medical professionals in making initial diagnosis, and should not be used as a substitute for a professional diagnosis.')

def main():
 st.set_page_config(
  page_title="Diabetes Predictor",
  page_icon=":Doctor",
  layout='wide',
  initial_sidebar_state='expanded'
 )
 
 with open('assets/style.css') as f:
  st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


 input_data = add_sidebar()
 #st.write(input_data)

 with st.container():
  st.title('Diabetes Predictor')
  st.write('Please connect this app to your lab results to help diagnose diabetes. ' \
  'This app predicts using machine learning model whether a patient is diabetic or not. You can play around with the sidebar for various measurements (Input parameters) and see the results by right hand side. ')

 col1,col2 = st.columns([4,1])
 with col1:
  radar_chart = get_radar_chart(input_data)
  st.plotly_chart(radar_chart)
 with col2:
  add_predictions(input_data)

if __name__=='__main__':
 main()