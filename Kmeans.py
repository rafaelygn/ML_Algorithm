'''
--------------------------------------------------------------------------------------------------
        K-MEANS
--------------------------------------------------------------------------------------------------
'''
import random
random.seed(42)
# %reset -f
# Import Lib's
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando dataset
dataset = pd.read_csv('Q2.csv', sep=';')
dataset = pd.read_csv('agrupamento.csv', sep=',')
centroid = pd.read_csv('centroides_iniciais.csv', sep=',')
#https://www.kaggle.com/c/titanic/data


'''
--------------------------------------------------------------------------------------------------
        EXPLORING YOUR DATASET
--------------------------------------------------------------------------------------------------
'''

dataset.head()
dataset.info()
dataset.describe()


# Features most importants
corr_matrix	=	dataset.corr()
corr_matrix["Survived"].sort_values(ascending=False)
# Correlation 
from pandas.tools.plotting import scatter_matrix
attributes = ["Survived","Fare","Pclass","Age"]
scatter_matrix(dataset[attributes],	figsize=(12,	8))

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
X = dataset.iloc[:,:]
cen = centroid.iloc[:8,:]
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
X.columns
# Drop Nan
X.drop('CustomerID', axis=1, inplace=True)
X.dropna(0, axis=1, inplace = True)
y.dropa(0, axis=1, inplace = True)

# ENCODING CATEGORICAL VALUES
cat_col = ["Genre"]
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
y = labelencoder_X.fit_transform(y)


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
        ELBOW method to find the optimal number of cluster 
--------------------------------------------------------------------------------------------------
'''
# We use Elbow method to fund the optimal number of cluster
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', 
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.scatter(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# inertia is the sum of squared distances of samples
# to their cluster center

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
d_sil = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', 
                    max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)
    silh = silhouette_score(X, y_kmeans, metric='euclidean')
    d_sil.append(silh)
plt.scatter(range(2, 11), d_sil)
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()



''' 
--------------------------------------------------------------------------------------------------
        APPLYING K-MEANS
--------------------------------------------------------------------------------------------------
'''
kini = np.array([[3,0],[5,0]])
kmeans = KMeans(n_clusters=8, init=cen, max_iter=300, random_state=42)
y_kmeans = kmeans.fit_predict(X)
kmeans.cluster_centers_
kmeans.inertia_

'''
--------------------------------------------------------------------------------------------------
        EVALUATING MODEL
--------------------------------------------------------------------------------------------------
'''

# Silhouette Coefficient
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
silhouette_score(X, y_kmeans, metric='euclidean')

# WCSS
kmeans.inertia_

# Outer Validation
label_true = dataset.label
label_pred = y_kmeans
# Adjusted Rand Index
metrics.adjusted_rand_score(label_true, label_pred)
# It does not necessary the same label, you can rename and get the same score

''' 
--------------------------------------------------------------------------------------------------
        VISUALISING THE RESULTS
--------------------------------------------------------------------------------------------------
'''
# Visulizing the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


