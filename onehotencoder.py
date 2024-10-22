import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = pd.read_csv('data.csv' , index = 'Id')
X.dropna(subset = ['SalePrice'],axis = 0, inplace = True)
y = X.SalePrice
X.drop(['SalePrice'], axis = 1 , inplace = True )

cols_with_missing = [col for col in X.columns
                    if X[col].isnull().any()]
X.drop(cols_with_missing,axis = 1 , inplace = True)

X_train,X_valid,y_train,y_valid = train_test_split(X , y , random_state = 21 , train_size = 0.8 , test_size = 0.2)

def score_dataset(X_train,X_valid,y_train,y_valid) :
  model = RandomForestRegressor(n_estimators = 200 , random_state = 21)
  model.fit(X_train,y_train)
  preds = model.predict(X_valid)
  mae = mean_absolute_error(y_valid,preds)
  return mae

object_cols = [col for col in X_train.columns
              if X_train[col].dtype == 'object']

low_cardinality_cols = [cname for cname in object_cols
                       if X_train[cname].nunique() < 10 and X_train[cname].dtype == 'object']
OH_encoder = OneHotEncoder(handle_unknown = 'ignore' , sparse_output = False)
OH_train_cols = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_valid_cols = pd.DataFrame(OH-encoder.transform(X_valid[object_cols]))

OH_train_cols.index = X_train.index
OH_valid_cols.index = X_valid.index

# num_train_cols = X_train.select_dtype(exclude = 'object' , axis = 1)
# num_valid_cols = X_valid.select_dtype(exclude = 'object' , axis = 1)

num_train_cols = X_train.drop(object_cols , axis = 1)
num_valid_cols = X_valid.drop(object_cols , axis = 1)

OH_X_train = pd.concat([num_train_cols,OH_train_cols] , axis = 1)
OH_X_valid = pd.concat([num_valid_cols,OH_valid_cols] , axis = 1)

OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print(score_dataset(OH_X_train,OH_X_valid,y_train,y_valid))
