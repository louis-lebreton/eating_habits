"""
MAIN

12/2023
Econometrics & Health Project

Kaggle dataset :
https://www.kaggle.com/datasets/mariaren/covid19-healthy-diet-dataset

Packages
"""
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans # K-Means
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error # Machine Learning metrics
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

import geopandas as gpd # to create maps

import statsmodels.api as sm # to get a summary of the regression
from adjustText import adjust_text  # PCA : to adjust the position of the text


# repository
os.chdir("C:/Users/lebre/OneDrive/Bureau/Projet Sante/Scripts")

#################################################################################################
# Preliminary treatments ########################################################################
#################################################################################################

# data import
df= pd.read_csv('../Data/raw/Food_Supply_kcal_Data.csv')
df=df.iloc[:,:-7] # we only keep obesity and diet variables

(df.isna().sum()/len(df))*100 # % of missing data
# only Obesity has missing data (>1.76%)

# countries with missing obesity value
pays_null_o=df[df.Obesity.isnull()]
pays_null_o

len(df)

# data frame description
print("HEAD : ------------------------------------------------")
print(df.head())
print("SHAPE : -----------------------------------------------")
print(df.shape) 
print("DTYPES : ----------------------------------------------")
print(df.dtypes)
print("DESCRIBE : --------------------------------------------")
print(df.describe())
print("INFO : ------------------------------------------------")
print(df.info())
print("ISNULL : ----------------------------------------------")
print(df.isnull) 
print(df.Country.unique()) # countries


# qualitative var columns
qual_colonnes=df.select_dtypes(include=['object','category']).columns 
# remove categorical columns
df_num=df.drop(qual_colonnes,axis=1)


#################################################################################################
# correlation matrix ############################################################################
#################################################################################################

corr_mat=df_num.corr(method='pearson').round(3)
print(corr_mat)

sns.heatmap(corr_mat,annot=False,cmap='coolwarm')
plt.title('matrice de corrélation')
# plt.show()

# correlations : obesity
corr_mat_slice=corr_mat[['Obesity']].sort_values(by='Obesity',ascending=True)
print(corr_mat_slice[:-1]) # sans obesity
plt.figure(figsize=(2,10))
sns.heatmap(corr_mat_slice[:-1],annot=False,cmap='coolwarm')
plt.title("Correlation of Foods with Obesity",fontweight='bold')
plt.show()

#################################################################################################
# K-means #######################################################################################
#################################################################################################

df_kmeans= df_num.iloc[:,:-1] # obesity removal
df_kmeans.dropna(inplace=True)  # NaN removal
df_kmeans

# Elbow method
inertia_list = []

K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_kmeans)
    inertia_list.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, inertia_list, 'bx-')
plt.xlabel('nb clusters')
plt.ylabel('inertia')
plt.title('Elbow method')
plt.show()

# I choose 3 clusters

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_kmeans)

print(kmeans.labels_) # my clusters
# adding clusters to the base df thanks to a merge on the indexes
df_kmeans['cluster']=kmeans.labels_


df_kmeans=pd.merge(df_kmeans,df['Obesity'],left_index=True,right_index=True)
# means of clusters
means=df_kmeans.groupby(['cluster']).mean()
means
# export excel
# means.to_excel(r'means.xlsx',index =False)
# std of clusters
sd=df_kmeans.groupby(['cluster']).std()
sd
# nb countries for each cluster
df_kmeans.groupby(['cluster']).size()

# countries for each cluster
df_join=pd.merge(df,df_kmeans['cluster'],left_index=True,right_index=True)
pays_cluster=df_join[['cluster','Country']].sort_values(by=['cluster'])
pays_cluster


#################################################################################################
# PCA ###########################################################################################
#################################################################################################

df_num.dropna(inplace=True)  # remove NaNs
pca = PCA(n_components=2) # 2 principal components
pca.fit(df_num)

# Ratio of explained variance by the first two principal components
print(pca.explained_variance_ratio_)
# Sum of ratios of explained variance by the first two principal components
sum(pca.explained_variance_ratio_)
# Principal components: coordinates
print(pca.components_)

components=pca.components_

##### Correlation Circle #####
plt.figure(figsize=(10,10))

texts=[] # To adjust the text
threshold=0.1  # Threshold to filter short vectors

for i, (x, y) in enumerate(zip(components[0],components[1])):
    if np.sqrt(x**2+y**2)>=threshold:  # Filter based on threshold
        plt.plot([0,x],[0,y], color='k')
        texts.append(plt.text(x,y,df_num.columns[i],fontsize=12))
    
# Automatically adjust text
adjust_text(texts)

# Adding the circle
circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='purple', linestyle='--')
plt.gca().add_artist(circle)
plt.axhline(y=0, color="grey", linestyle="--")
plt.axvline(x=0, color="grey", linestyle="--")
plt.xlim(-1,1)
plt.ylim(-1,1)

# Axis labels
plt.xlabel('Principal Component 1 (63.55 %)')
plt.ylabel('Principal Component 2 (15.78 %)')

plt.show()

##### Display of countries #####

result = pca.fit_transform(df_num)

# Remember what we said about the sign of eigenvectors that might change?
pc1 = result[:,0]
pc2 = result[:,1]

# plotting in 2D
df1=df.dropna(inplace=False)  # Remove NaNs
countries = df1['Country'].tolist()

def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for i, c in enumerate(countries):
        if i%2==0: # Display every other one
            plt.scatter(pc1[i], pc2[i], label = c, s = 20)
            try:
                ax.annotate(c, (pc1[i],pc2[i]))
            except:
                ax.annotate(c, (pc1[i],pc2[i]))
    
    
    ax.set_xlabel('Principal Component 1 (63.55 %)')
    ax.set_ylabel('Principal Component 2 (15.78 %)')
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
    

    plt.grid()
    plt.axis([-25,30,-15,20])
    plt.show()
    
plot_scatter(pc1, pc2)

#################################################################################################
# Maps ##########################################################################################
#################################################################################################

# Geographical data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
merged = world.merge(df, left_on='name', right_on='Country', how='left')

# Map
plt.figure(figsize=(15,10))
fig, ax = plt.subplots(1, 1)

# Test variables: Vegetal Products, Sugar & Sweeteners, Alcoholic Beverages
merged.plot(column='Alcoholic Beverages', ax=ax, legend=True, legend_kwds={'shrink': 0.5}, cmap="RdPu",
            missing_kwds={'color': 'lightgrey', 'label': 'Missing Data'})

ax.set_frame_on(False) # Remove frame around the map
# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig('../Maps/map.png', dpi=700, bbox_inches='tight') # Save the map


#################################################################################################
# Regression tree & Random Forest ###############################################################
#################################################################################################

print(df_num)
df_num.dropna(inplace=True)  # suppression des NaN

x_train=df_num.drop('Obesity',axis=1) # vars explicatives
y_train=df_num['Obesity'] # var à expliquer


# division : samples train et test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0)
x_train.shape # 125 lignes
x_test.shape # 42 lignes
y_train.shape
y_test.shape

# Regression tree

arbre_reg=tree.DecisionTreeRegressor(max_depth=4)
arbre_reg.fit(x_train,y_train)

plt.figure(figsize=(20,10))
tree.plot_tree(arbre_reg,filled=True)

plt.show()

# Random Forest
random_forest = RandomForestRegressor()
random_forest.fit(x_train, y_train)


