# from os import PathLike
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from joblib import dump
# import pandas as pd
# import pathlib

# df = pd.read_csv(pathlib.Path('data/laptop-price.csv'))
# y = df.pop('Price')
# columns_to_drop = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os']
# X = df
# X = df.drop(columns_to_drop, axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# print ('Training model.. ')
# clf = RandomForestClassifier(n_estimators = 10,
#                             max_depth=2,
#                             random_state=0)
# clf.fit(X_train, y_train)
# print ('Saving model..')

# dump(clf, pathlib.Path('model/laptop-v1.joblib'))



# from os import PathLike
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from joblib import dump
# import pandas as pd
# import pathlib
# import numpy as np

# df = pd.read_csv(pathlib.Path('data/laptop-price.csv'))
# y = df.pop('Price')
# columns_to_drop = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os']
# X = df.drop(columns_to_drop, axis=1)
# print (X.head())
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# X=imp.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=44, shuffle =True)

# print ('Training model.. ')
# clf = RandomForestRegressor(n_estimators=100,
#                             max_depth=7, 
#                             random_state=33)
# clf.fit(X_train, y_train)
# print ('Saving model..')

# print('Random Forest Regressor Train Score is : ' , clf.score(X_train, y_train))

# dump(clf, pathlib.Path('model/laptop-v1.joblib'))



from os import PathLike
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
import pathlib
import numpy as np

df = pd.read_csv(pathlib.Path('data/laptop-price.csv'))
y = df.pop('Price')
columns_to_drop = ['Company', 'TypeName', 'Cpu_brand', 'Gpu_brand', 'Os']
X = df.drop(columns_to_drop, axis=1)
print (X.head())
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imp.fit_transform(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

seed = np.random.seed(0)

print ('Training model.. ')
rf_model = RandomForestRegressor(max_depth=12 ,min_samples_leaf=10, random_state=seed)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
print('MAE for RandomForestRegressor:', mean_absolute_error(val_y, rf_val_predictions))
rf_model.fit(train_X, train_y)
print ('Saving model..')

dump(rf_model, pathlib.Path('model/laptop-v1.joblib'))