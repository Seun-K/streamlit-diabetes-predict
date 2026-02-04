import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle as pickle
import shap as shap

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

def create_model(data):
 x = data.drop(['Outcome'], axis=1)
 y = data['Outcome']
 # scale the data
 scaler = StandardScaler()
 x = scaler.fit_transform(x)
 # split the data
 x_train,x_test,y_train,y_test = train_test_split (
  x, y, test_size=0.2,stratify=y, random_state=42
 )


 ###################################################################################################################
#  model = RandomForestClassifier(random_state=42, min_samples_leaf=5, n_jobs=-1, max_depth=10,
#  n_estimators=200, oob_score=True)
#  sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
#  model.fit(x_train,y_train,sample_weight=sample_weights)
#  model.oob_score_

####################################################################################### Random forest hyperparameter
#  model = RandomForestClassifier(random_state=42, n_jobs=-1)
#  params = {
#     'max_depth': [2,3,5,10,20],
#     'min_samples_leaf': [5,10,20,50,100,200],
#     'n_estimators': [10,25,30,50,100,200]
#  }

 

#  # Instantiate the grid search model
#  grid_search = GridSearchCV(estimator=model,
#  param_grid=params,
#  cv = 4,
#  verbose=1, scoring="accuracy")

#  # time
#  grid_search.fit(x_train, y_train)
#  rf_best = grid_search.best_estimator_
#  print(rf_best)

# ###################################################################################### xgb hyperparameter
#  model = xgboost.XGBClassifier(objective='binary:logistic', seed=42)
#  param_grid = {
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'n_estimators': [100, 200, 300],
#     'subsample': [0.5, 0.7]
#  }
#  grid_search = GridSearchCV(
#     estimator= model,
#     param_grid=param_grid,
#     scoring='roc_auc',  # Example scoring metric
#     cv=3,
#     verbose=1,
#     n_jobs=-1
#  )
#  grid_search.fit(x_train, y_train)
#  best = grid_search.best_estimator_
#  print(best)
 ##############################################################################################
 model = xgboost.XGBClassifier(objective='binary:logistic', seed=42,
 max_depth = 3, learning_rate = 0.01, subsample= 0.7, n_estimators=200,)
 sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
 model.fit(x_train,y_train,sample_weight=sample_weights)
 ##############################################################################################
# #  ratio = float(sum(y_train == 0)) / sum(y_train == 1)
# #  # Initialize the classifier with the adjusted weight
# #  model = XGBClassifier(scale_pos_weight=ratio, use_label_encoder=False, eval_metric='logloss')
# #  model.fit(x_train, y_train)


#  sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
#  model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#  model.fit(x_train, y_train, sample_weight=sample_weights)


# #  # train
# #  model = LogisticRegression()
# #  model.fit(x_train,y_train)

 # test
 y_pred = model.predict(x_test)
 y_pred_train = model.predict(x_train)
 print('AUC of our model:', roc_auc_score(y_test,y_pred))
 print('AUC of our model:', roc_auc_score(y_train,y_pred_train))
 print("Classification report:\n", classification_report(y_test,y_pred))


#  # train model
#  dtrain = xgboost.DMatrix(data = x_train, label = y_train)
#  dtest = xgboost.DMatrix(data = x_test, label = y_test)
#  evallist = [(dtest,'test'), (dtrain, 'train')]

#  # default parameters
#  #param = {'max_depth':10,'nthread':8,'silent':1,'learning_rate':0.15,'subsample':0.7,'colsample_bytree':0.8,'colsample_bylevel':0.8,'objective': 'binary:logistic','eval_metric': ['auc', 'aucpr']}
#  param = {'subsample':0.1,'colsample_bytree':0.8,'max_depth':8,'nthread':8,'learning_rate':0.15,'objective': 'binary:logistic','eval_metric': ['auc', 'aucpr']}
#  num_round = 20
#  model = xgboost.train(param, dtrain, num_round, evallist, verbose_eval = 1)

#  # test
#  y_pred = model.predict(dtest)
#  y_pred_train = model.predict(dtrain)
#  predictions = [round(value) for value in y_pred]
#  train_predictions = [round(value) for value in y_pred_train]
#  print('Accuracy of our model:', accuracy_score(y_test,predictions))
#  print('Accuracy of our model:', accuracy_score(y_train,train_predictions))
#  print("Classification report:\n", classification_report(y_test,predictions))
 return model, scaler

#def test_model(model): 

def main():

 data = clean_data()
 # Create the model
 model, scaler = create_model(data)
 with open('model/model.pkl', 'wb') as f:
  pickle.dump(model,f)

 with open('model/scaler.pkl', 'wb') as f:
  pickle.dump(scaler,f)

if __name__=='__main__':
 main()
    