'''
--------------------------------------------------------------------------------------------------
        Hierarchical Clustering
--------------------------------------------------------------------------------------------------
'''
# %reset -f
# Import Lib's
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importando dataset
dataset = pd.read_csv('Mall_Customers.csv', sep=',')
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
X = dataset.iloc[:,[3,4]].values
# X = dataset.iloc[:,[1,3]].values

# HANDLING MISSING DATA
# fillnan with mean/median/most_frequent
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X.iloc[:,0].values.reshape((len(X),1)))
X.iloc[:,0] = imputer.transform(X.iloc[:,0].values.reshape((len(X),1)))
X.info()
X.drop(0, axis=1, inplace=True)
# Drop Nan
X.dropna(inplace = True)
y.dropa(inplace = True)

# ENCODING CATEGORICAL FEATURES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# The values 0,1,2,etc into categorical values
labelencoder_X = LabelEncoder()
X.iloc[:,1] = labelencoder_X.fit_transform(X.iloc[:, 1])
# Here we create the dummies
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#Delete a COlumn in a array
X = np.delete(X, 0, axis=1)


# FEATURE SCALING
# Saving the original values 
Xo = X
    # Standardisation
    #X = ((X-X.mean())/X.std())
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

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

''' 
--------------------------------------------------------------------------------------------------
        DENDROGRAM
--------------------------------------------------------------------------------------------------
'''
# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()



''' 
--------------------------------------------------------------------------------------------------
        APPLYING HIERARCHICAL
--------------------------------------------------------------------------------------------------
'''
# Fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
hc.

'''
--------------------------------------------------------------------------------------------------
        EVALUATING MODEL
--------------------------------------------------------------------------------------------------
'''

# Silhouette Coefficient
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
silhouette_score(X, y_hc, metric='euclidean')

For
d_sil = []
for i in range(2, 11):
    hc = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    silh = silhouette_score(X, y_hc, metric='euclidean')
    d_sil.append(silh)
plt.scatter(range(2, 11), d_sil)
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()



# Outer Validation
label_true = dataset.label
label_pred = y_hc
# Adjusted Rand Index
metrics.adjusted_rand_score(label_true, label_pred)
# It does not necessary the same label, you can rename and get the same score

''' 
--------------------------------------------------------------------------------------------------
        VISUALISING THE CLUSTERS
--------------------------------------------------------------------------------------------------
'''
# Visulizing the clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(X[y_hc==5, 0], X[y_hc==4, 1], s=100, c='black', label='Cluster 6')


plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


''' 
--------------------------------------------------------------------------------------------------
        DENDOGRAM with DISTANCE MATRIX
--------------------------------------------------------------------------------------------------
'''

import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt


mat = np.array([[0.0, 2.0, 6.0, 10.0, 9.0],
                [2.0, 0.0, 5.0, 9.0,  8.0], 
                [6.0, 5.0, 0.0, 4.0,  5.0],
                [10.0, 9.0, 4.0, 0.0, 3.0],
                [9.0, 8.0, 5.0, 3.0,  0.0]])
dists = squareform(mat)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, labels=["0", "1", "2","3", "4"])
plt.title("test")
plt.show()

# How to calculate distance_matrix
from scipy.spatial import distance_matrix
p = dataset.iloc[:5,[2,4]].values

distance_matrix(p,p)

d = np.dot(p,p.T)
norm = (p**2).sum(0, keepdims=True)
d / norm
d / norm / norm.T
1 - d / norm / norm.T

1-pairwise_distances(p, metric='cosine')