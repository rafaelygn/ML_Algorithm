'''
--------------------------------------------------------------------------------------------------
        SVM REGRESSOR
--------------------------------------------------------------------------------------------------
'''

# Import Lib's
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando dataset
dataset = pd.read_csv('train.csv', sep=',')

#https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

def data_info(data):        
    info = pd.DataFrame()
    info['var'] = data.columns
    info['# missing'] = list(data.isnull().sum())
    info['% missing'] = info['# missing'] / data.shape[0]
    info['types'] = list(data.dtypes)
    info['unique values'] = list(len(data[var].unique()) for var in data.columns)
    return info


'''
--------------------------------------------------------------------------------------------------
        EXPLORING YOUR DATASET
--------------------------------------------------------------------------------------------------
'''

dataset.head()
dataset.info()
dataset.describe()
data_info(dataset)

# Features most importants
corr_matrix	=	dataset.corr()
corr_matrix["SalePrice"].sort_values(ascending=False)
# Correlation 
from pandas.tools.plotting import scatter_matrix
attributes = ["SalePrice","OverallQual","GrLivArea","GarageCars"]
scatter_matrix(dataset[attributes],	figsize=(12,	8))

# Count_values
dataset.OverallQual.value_counts()

# PLOT 2D
dataset.plot(kind="scatter", x="OverallQual", y="SalePrice", alpha=0.5)
# PLOT 4D - Size (s) it's not necessary
dataset.plot(kind="scatter", x="OverallQual", y="GrLivArea", alpha=0.5, s=dataset["GarageCars"]*100, c=dataset["SalePrice"], label="SalePrice" ,cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

''' 
--------------------------------------------------------------------------------------------------
        DATA PREP 
--------------------------------------------------------------------------------------------------
'''
# SELECTING X and y
X = dataset.loc[:,["OverallQual","GrLivArea","GarageCars"]]
y = dataset.iloc[:,-1]
# X = dataset.iloc[:,[1,3]].values

# HANDLING MISSING DATA

# Create list of numerical features and categorical features
num_col = []
cat_col = []
for col in X.columns:
    if X[col].dtype=='object':
        cat_col.append(col)
    else:
        num_col.append(col)

X.info()
data_info(X)
# fillnan with mean/median/most_frequent
# Numerical features
from sklearn.preprocessing import Imputer
for col in num_col:
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X[col].values.reshape(len(X),1))
    X[col] = imputer.transform(X[col].values.reshape(len(X),1))

# Categorical features
X.fillna(value='NULL', inplace=True)

X.info()
# Drop Nan
X.dropna(0, axis=1, inplace = True)
y.dropa(0, axis=1, inplace = True)

# ENCODING CATEGORICAL VALUES
cat_col = ["Sex", "Pclass", 'SibSp']
for col in cat_col:
    X = pd.get_dummies(X, columns=[col], prefix = [col], drop_first=True)
# ENCODING 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X(X.iloc[:,1])
X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])

# FEATURE SCALING
# Saving the original values 
Xo = X.copy()
yo = y.copy()
'''
    # Mean normalization
X = ((X-X.mean())/(X.max()-X.min()))
    # Inverse Transform 
X = (Xo.max()-Xo.min())*X+Xo.mean()
    # Min-Max scaling
X = ((X-X.min())/(X.max()-X.min()))
    # Inverse Tranform
X = (Xo.max()-Xo.min())*X+Xo.min()
'''
    # Standardisation
    #X = ((X-X.mean())/X.std())
num_col = ["Age"]
from sklearn.preprocessing import StandardScaler
for col in num_col:
    sc_X = StandardScaler()
    X[col] = sc_X.fit_transform(X[col].reshape((len(X),1)))

    # y Standardisation
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape((len(y),1)))

    # MinMax
num_col = ["Age"]
from sklearn.preprocessing import MinMaxScaler
for col in num_col:
    minmaxscaler = MinMaxScaler()
    X[col] = minmaxscaler.fit_transform(X[col].reshape((len(X),1)))

    # Robust Scaler
num_col = ["Age"]
from sklearn.preprocessing import RobustScaler
for col in num_col:
    robscaler = RobustScaler()
    X[col] = robscaler.fit_transform(X[col].reshape((len(X),1)))

robscaler_y = RobustScaler()
y = robscaler_y.fit_transform(y.reshape((len(y),1)))


''' 
--------------------------------------------------------------------------------------------------
        SPLIT THE DATA SET INTO TRAINING AND TEST SET
--------------------------------------------------------------------------------------------------
'''
# HOLD OUT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # (Ordered)
'''
Pos=8
X_train, y_train  = X[:Pos,:] , y[:Pos]
X_test, y_test = X[Pos:,:], y[Pos:]
#OBS.: iloc para df e sr[:,:] para series
'''

''' 
--------------------------------------------------------------------------------------------------
        TRAINING THE MODEL
--------------------------------------------------------------------------------------------------
'''
# if is a array like (n,) you have to reshape it with .reshape(n,1)
# sometimes you have to put df.values.reshape
X_train = X_train.reshape(np.size(X_train,0),np.size(X_train,1))
X_test = X_test.reshape(np.size(X_test,0),np.size(X_test,1))
y_train = y_train.reshape(len(y_train), 1)
y_test = y_test.reshape(len(y_test),1)

# Fit your method to your training set
from sklearn.svm import SVR
regressor = SVR(kernel='rbf', degree=3,
                C=1.0,
                shrinking =True,
                )
regressor.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=100, weights='uniform', p=2)
regressor.fit(X, y)


from sklearn import linear_model
regressor = linear_model.ElasticNet(alpha = 0.01, l1_ratio=0.5, random_state=42)
regressor.fit(X, y)



'''
--------------------------------------------------------------------------------------------------
        PREDICT 
--------------------------------------------------------------------------------------------------
'''
# y_pred
y_pred = regressor.predict(X_test)

# Predict a specific Xi value
X_new = np.array([9,1500]).reshape((1,2))
y_new = sc_y.inverse_transform(regressor.predict( sc_X.transform(X_new) ))
print(y_new)

'''
--------------------------------------------------------------------------------------------------
        EVALUATING MODEL
--------------------------------------------------------------------------------------------------
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error , r2_score
print("Mean test: ", y_test.mean())
print("Mean pred: ", y_pred.mean())
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("RMSE: ",mean_squared_error(y_test, y_pred)**(1/2))
print("R2: ",r2_score(y_test, y_pred))

y_pred = robscaler_y.inverse_transform(y_pred)
y_test = robscaler_y.inverse_transform(y_test)
robscaler_y.inverse_transform(0.21)
'''
--------------------------------------------------------------------------------------------------
        CROSS VALIDATION
--------------------------------------------------------------------------------------------------
'''
# K-FOLD or LOOCV

# K-Folds:
k=10
# LOOCV
k = len(X_train)

results = pd.DataFrame()

from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
cv_kfold = KFold(10, shuffle = False, random_state=42)


from sklearn.model_selection import cross_validate
metrics_lt = ['mean_absolute_error', 'mean_squared_error', 'r2']
for m in metrics_lt:
    metrics_cv = cross_validate(regressor, X=X, y=np.ravel(y), 
                               scoring=m, cv=cv_kfold, return_train_score=True)
    dict_res = {m + '_Treino': metrics_cv['train_score'], 
                m + '_Teste': metrics_cv['test_score']}
    res_aux = pd.DataFrame(dict_res)
    results = pd.concat([results, res_aux], axis = 1)
results = pd.concat([results, np.transpose(pd.DataFrame(results.mean(), columns=['mean']))], axis = 0)
results = pd.concat([results, np.transpose(pd.DataFrame(results.std(), columns=['std']))], axis = 0)

''' 
--------------------------------------------------------------------------------------------------
        GRID SEARCH
--------------------------------------------------------------------------------------------------
'''
parameters = {'kernel':('rbf','poly', 'sigmoid'), 
              'degree':[2, 3, 5, 10],
              'C':[0.1, 0.5, 1.0, 3.0],
              'epsilon': [0.01, 0.1, 1.0]
             }

from sklearn.model_selection import GridSearchCV
gscv = GridSearchCV(regressor, parameters, scoring ='mean_absolute_error', cv=5)
gscv.fit(X_train,y_train)
gscv.best_estimator_
gscv.best_score_
y_pred = gscv.best_estimator_.predict(X_test)
print("Mean test: ", y_test.mean())
print("Mean pred: ", y_pred.mean())
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE: ",mean_squared_error(y_test, y_pred))
print("RMSE: ",mean_squared_error(y_test, y_pred)**(1/2))
print("R2: ",r2_score(y_test, y_pred))

''' 
--------------------------------------------------------------------------------------------------
        VISUALISING THE RESULTS
--------------------------------------------------------------------------------------------------
'''
# X scaled into X originalm or you can use Xo
X_plot_train = robscaler.inverse_transform(X_train)
X_plot_test = robscaler.inverse_transform(X_test)

# PLOT 2D
    # TRANING
plt.scatter(X_plot_train[:,1], y_train, color='purple', alpha=0.5)
plt.title('Train set')
plt.scatter(X_plot_train[:,1], regressor.predict(X_train), color='blue')
plt.show()

    # TEST
plt.scatter(X_plot_test[:,0], y_test, color='orange' , alpha=0.5)
plt.scatter(X_plot_test[:,0], y_pred, color='blue',alpha=0.3)
plt.title('Test set')
plt.show()

#  PLOT 3D
    # y_pred
d_test = {'x1': X_plot_test[:,0], 'x2':X_plot_test[:,1] }
df_test = pd.DataFrame(d_test)
df_test.plot(kind="scatter", x="x1", y="x2", alpha=0.5, c=y_pred, label="SalePrice" ,cmap=plt.get_cmap("jet"), colorbar=True, title="y_pred")
plt.legend()

    # y_test
y_test= y_test.reshape((292,1))
df_test.plot(kind="scatter", x="x1", y="x2", alpha=0.5, c=y_test.reshape((292,)), label="SalePrice" ,cmap=plt.get_cmap("jet"), colorbar=True, title="y_test")
plt.legend()


