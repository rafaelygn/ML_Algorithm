'''
--------------------------------------------------------------------------------------------------
        RANDOM FOREST CLASSIFIER
--------------------------------------------------------------------------------------------------
'''

# Import Lib's
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('train_titanic.csv')

#https://www.kaggle.com/c/titanic/data

def data_info(data):        
    info = pd.DataFrame()
    info['var'] = data.columns
    info['# missing'] = list(data.isnull().sum())
    info['% missing'] = info['# missing'] / data.shape[0]
    info['types'] = list(data.dtypes)
    info['unique values'] = list(len(data[var].unique()) for var in data.columns)
    return info

def	plot_roc_curve(fpr,	tpr,	label=None):
	plt.plot(fpr,tpr,linewidth=2,label=label)
	plt.plot([0,1],[0,1],'k--')
	plt.axis([0,	1,	0,	1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
    
def ks_score(y_pred_proba, y_real):
    from scipy import stats
    df_ks = pd.concat([pd.DataFrame(y_pred_proba), pd.DataFrame(np.ravel(y_real))], axis=1)
    df_ks.columns = ['proba', 'real']
    df_true1 = df_ks[df_ks['real']==1]
    df_true0 = df_ks[df_ks['real']==0]
    print(stats.ks_2samp(df_true0.proba, df_true1.proba).statistic)

def ks_plot(y_pred_proba, y_real):
    from scipy import stats
    df_ks = pd.concat([pd.DataFrame(y_pred_proba), pd.DataFrame(np.ravel(y_real))], axis=1)
    df_ks.columns = ['proba', 'real']
    df_true1 = df_ks[df_ks['real']==1]
    df_true0 = df_ks[df_ks['real']==0]
    df_true1.sort_values('proba', inplace = True)
    df_true0.sort_values('proba', inplace = True)
    suma =1/len(df_true1)
    somatoria = []
    for row in range(len(df_true1)):
        somatoria.append((row+1)*suma)
    suma = 1/len(df_true0)
    somatoria0 =[]
    for row in range(len(df_true0)):
        somatoria0.append((row+1)*suma)
    df_true1['SS'] = somatoria
    df_true0['SS'] = somatoria0
    plt.plot(df_true0.proba, df_true0.SS)
    plt.plot(df_true1.proba, df_true1.SS)
    plt.legend(['curve 0', 'curve 1'])
    plt.show()

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
corr_matrix["Survived"].sort_values(ascending=False)
# Correlation 
from pandas.tools.plotting import scatter_matrix
attributes = ["Survived","Fare","Pclass","Age"]
scatter_matrix(dataset[attributes],	figsize=(12,8))

# Count_values
dataset.Pclass.value_counts()

# PLOT 2D
dataset.plot(kind="scatter", x="OverallQual", y="SalePrice", alpha=0.5)
# PLOT 4D - Size (s) it's not necessary
dataset.plot(kind="scatter", x="OverallQual", y="GrLivArea", alpha=0.5, s=dataset["GarageCars"]*100, c=dataset["SalePrice"], label="SalePrice" ,cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()

# BOXPLOT
import seaborn as sns
sns.boxplot(x="Pclass", y="Age", hue="Survived",data=dataset, palette="Set3")

''' 
--------------------------------------------------------------------------------------------------
        DATA PREP 
--------------------------------------------------------------------------------------------------
'''
# SELECTING X and y
X = dataset.loc[:,["Fare", "Sex", "Pclass","Age"]]
y = dataset.loc[:,"Survived"]


# HANDLING MISSING DATA
X.info()
# Create list of numerical features and categorical features
num_col = []
cat_col = []
for col in X.columns:
    if X[col].dtype=='object':
        cat_col.append(col)
    else:
        num_col.append(col)

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
cat_col = ["Sex", "Pclass"]
for col in cat_col:
    X = pd.get_dummies(X, columns=[col], prefix = [col], drop_first=True)
# ENCODING 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X(X.iloc[:,1])
X.iloc[:,3] = labelencoder_X.fit_transform(X.iloc[:,3])
print(LabelEncoder.get_params.mro)

labelencoder_y = LabelEncoder()
labelencoder_y.fit(y)
y = labelencoder_y.fit_transform(y)


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


''' 
--------------------------------------------------------------------------------------------------
        SPLIT THE DATA SET INTO TRAINING AND TEST SET
--------------------------------------------------------------------------------------------------
'''
# HOLD OUT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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
from sklearn.ensemble import  RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',
                                    n_estimators=30,
                                    max_depth=5,
                                    oob_score=True)
classifier.fit(X_train, y_train)

from sklearn.naive_bayes  import  GaussianNB
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X,y)

from sklearn.naive_bayes  import   BernoulliNB
classifier =  GaussianNB()
classifier.fit(X,y)

from sklearn.svm import  SVC
classifier = SVC(C=1.0,
                 kernel = 'linear',
                 shrinking =False
                 )
classifier.fit(X, y)



'''
--------------------------------------------------------------------------------------------------
        PREDICT 
--------------------------------------------------------------------------------------------------
'''
# y_pred and y_proba
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)

# Predict a specific Xi value
X_new = np.array([1,0,0,1,1,1]).reshape((1,6))
y_new = classifier.predict( sc_X.transform(X_new) )
y_new_proba = classifier.predict_proba(sc_X.transform(X_new))
print(y_new)
print(y_new_proba)



'''
--------------------------------------------------------------------------------------------------
        EVALUATING MODEL
--------------------------------------------------------------------------------------------------
'''
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, confusion_matrix

print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ",recall_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("F1: ",f1_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test,y_pred_proba[:,1]))

conf_mtx = confusion_matrix(y_test, y_pred)

classifier.oob_score_
classifier.feature_importances_

# ROC Curve
from sklearn.metrics import	roc_curve
fpr,tpr,thresholds = roc_curve(y_test,	y_pred_proba[:,1])
plot_roc_curve(fpr,	tpr)
plt.show()

# Kolmogov Smirnov - Score
ks_score(y_pred_proba[:,1], y_test)
ks_plot(y_pred_proba[:,1], y_test)


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
from sklearn.model_selection import KFold
cv_kfold = KFold(10, shuffle = False, random_state=42)
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut

from sklearn.model_selection import cross_validate
metrics_lt = ['precision', 'recall', 'accuracy','f1','roc_auc']
for m in metrics_lt:
    metrics_cv = cross_validate(classifier, X=X, y=np.ravel(y), 
                               scoring=m, cv=cv_kfold, return_train_score=True)
    dict_res = {m + '_Treino': metrics_cv['test_score'], 
                m + '_Teste': metrics_cv['train_score']}
    res_aux = pd.DataFrame(dict_res)
    results = pd.concat([results, res_aux], axis = 1)
results = pd.concat([results, np.transpose(pd.DataFrame(results.mean(), columns=['mean']))], axis = 0)
results = pd.concat([results, np.transpose(pd.DataFrame(results.std(), columns=['std']))], axis = 0)

''' 
--------------------------------------------------------------------------------------------------
        GRID SEARCH
--------------------------------------------------------------------------------------------------
'''
parameters = {'n_estimators':[10,30, 50, 100, 300], 
              'criterion':('gini', 'entropy'),
              'max_depth':[3, 5, 10, 30],
             }

from sklearn.model_selection import GridSearchCV
gscv = GridSearchCV(classifier, parameters, scoring ='roc_auc', cv=5)
gscv.fit(X_train,y_train)
gscv.best_estimator_
gscv.best_score_
y_pred = gscv.best_estimator_.predict(X_test)
y_pred_proba = gscv.best_estimator_.predict_proba(X_test) 

print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ",recall_score(y_test, y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("F1: ",f1_score(y_test, y_pred))
print("AUC: ", roc_auc_score(y_test,y_pred_proba[:,1]))
conf_mtx = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", conf_mtx)


''' 
--------------------------------------------------------------------------------------------------
        VISUALISING THE RESULTS
--------------------------------------------------------------------------------------------------
'''
# X scaled into X originalm or you can use Xo
X_plot_train = np.   sc_X.inverse_transform(X_train[:,-1])
X_plot_test = sc_X.inverse_transform(X_test)


X_train[:,[0,1]]
from matplotlib.colors import ListedColormap
X_set, y_set = X_train ,  y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - .5, stop=X_set[:, 0].max() + .5, step=0.1),
                     np.arange(start=X_set[:, 1].min() - .5, stop=X_set[:, 1].max() + .5, step=0.1))
plt.plcontourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'blue'))(i),  alpha=0.5,label=j)
plt.title('K-NN')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.legend()
plt.show()

# Visualizing the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.1))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('K-NN (Test set)')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.legend()
plt.show()




