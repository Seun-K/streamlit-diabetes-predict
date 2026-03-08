# 🩺 Diabetes Risk Prediction Using Machine Learning
## 📌 Project Overview
Early detection of diabetes is critical for improving patient outcomes and enabling timely medical intervention. This project develops a machine learning classification model to predict the likelihood of diabetes based on patient clinical features.
Beyond predictive modelling, this project emphasizes model interpretability and accessibility. SHAP (SHapley Additive exPlanations) was used to explain model predictions, and the final model was operationalized through an interactive Streamlit web application, enabling users to generate real-time predictions.

## 🎯 Objectives
*	Build a machine learning model to predict diabetes risk using patient clinical data
*	Compare multiple classification algorithms to determine the best-performing model
*	Improve transparency using SHAP explainability techniques
*	Deploy an interactive web application to allow live predictions

## 📂 Dataset
The dataset contains patient clinical measurements commonly associated with diabetes risk.
Example Features
*	Glucose level
*	Blood pressure
*	Body Mass Index (BMI)
*	Insulin level
*	Skin thickness
*	Age
*	Diabetes pedigree function
The target variable indicates whether a patient is likely to have diabetes.

## 🧠 Machine Learning Approach
The modelling pipeline consisted of several key stages.

### 1️⃣ Data Preprocessing
•	Handling missing or zero values in clinical measurements
•	Exploratory data analysis to understand feature distributions
•	Feature scaling where required
•	Splitting data into training and testing sets

### 2️⃣ Model Training
Several classification algorithms were trained and evaluated:
*	Logistic Regression
*	Decision Tree
*	Random Forest
*	Support Vector Machine (SVM)
*	XGBoost (Extreme Gradient Boosting)
XGBoost is a powerful gradient boosting algorithm widely used in predictive modelling due to its strong performance, ability to capture nonlinear relationships, and built-in regularization to prevent overfitting.

### 3️⃣ Model Evaluation
Models were evaluated using standard classification metrics:
*	Accuracy
*	Precision
*	Recall
*	F1 Score
The best-performing model was selected for deployment in the web application.

## 🔍 Model Explainability with SHAP
To improve transparency and trust in the model, SHAP (SHapley Additive exPlanations) was used to interpret predictions.
SHAP provides:

### Global Interpretability
Identifies which clinical features contribute most to predicting diabetes risk across the dataset.

### Local Interpretability
Explains why the model produced a specific prediction for an individual patient, highlighting how each feature influences the final outcome.
This approach makes the model more understandable and supports responsible use of machine learning in healthcare analytics.

## 🌐 Web Application
To operationalize the model, an interactive web application was developed using Streamlit.
Key Features
*	Simple input interface for patient health metrics
*	Real-time diabetes risk prediction
*	Interactive model explanations
*	Accessible via a public web link
The app enables users to input clinical features and instantly receive a prediction from the trained machine learning model.

## 🚀 Deployment
The Streamlit application was deployed using Streamlit Community Cloud, allowing the model to be accessed directly through a web browser without requiring local setup.
This demonstrates the full lifecycle of a machine learning project:
Model Development → Model Interpretation → Application Deployment

## 🛠 Tools & Technologies
*	Python
*	Scikit-learn
*	XGBoost
*	Pandas
*	NumPy
*	SHAP
*	Streamlit
*	Matplotlib / Seaborn
*	Visual Studio Code

## 📊 Key Skills Demonstrated
*	Machine learning classification modelling
*	Gradient boosting with XGBoost
*	Data preprocessing and feature engineering
*	Model evaluation and comparison
*	Explainable AI using SHAP values
*	Web application development with Streamlit
*	Deployment of machine learning applications to the cloud

## 💡 Project Impact
This project demonstrates how machine learning can support early disease detection while maintaining transparency through explainable AI techniques
By deploying the model in an interactive web application, predictive analytics is transformed into a practical and accessible tool for users to explore diabetes risk predictions in real time.

## 👤 Seun Komolafe
Developed as part of a machine learning project demonstrating predictive modelling, explainable AI, and deployment of an end-to-end data science application.

