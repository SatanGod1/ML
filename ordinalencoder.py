import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

X = pd.read_csv('data.csv',index = 'Id')
X.dropna(subset = ['SalePrice'],axis = 0,inplace = True)
y = X.SalePrice
X.drop(['SalePrice'],axis = 1,inplace = True)

cols_with_missing = [col for col in X.columns
                    if X[col].isnull().any()]
X.drop(cols_with_missing,axis = 1)
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state = 21,train_size = 0.8,test_size = 0.2)

def score_dataset(X_train,X_valid,y_train,y_valid) :
  model = RandomForestRegressor(n_estimators = 200,random_state = 21)
  model.fit(X_train,y_train)
  preds = model.predict(X_valid)
  mae = meaan_absolute_error(y_valid,preds)
  return mae

object_cols = [col for col in X_train.columns
              if X_train[col].dtype == 'object']
good_label_cols = [col for col in object_cols
                  if set(X_valid[col]).issubset(set(X_train[col]))]
bad_label_cols = list(set(object_cols) - set(good_label_cols))

X_train.drop(bad_label_cols)
X_valid.drop(bad_label_cols)

label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print(score_dataset(label_X_train,label_X_valid,y_train,y_valid))
