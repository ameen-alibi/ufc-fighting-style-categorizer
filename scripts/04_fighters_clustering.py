#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: 02_data_cleaning.ipynb
Conversion Date: 2025-10-21T20:37:39.172Z
"""

from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

fights_df = pd.read_csv('data/Fights.csv')


fights_df['Avg Fight Time'] = fights_df['Fight_Time'].str.split(':').str[0].astype(int) * 60 \
    + fights_df['Fight_Time'].str.split(':').str[1].astype(int) \
    + (fights_df['Round'] - 1) * 5 * 60

# Metrics to aggregate for each fighter
metrics = ['Fighter_Id', 'KD', 'STR', 'TD', 'SUB', 'Ctrl', 'Sig. Str. %', 'Head_%', 'Body_%', 'Leg_%',
           'Distance_%', 'Clinch_%', 'Ground_%', 'Sub. Att', 'Rev.']

# Per-fight rows for Fighter_1
df1 = fights_df[['Fighter_1', 'Round', 'Avg Fight Time'] + [f'{m}_1' for m in metrics]].rename(
    columns={'Fighter_1': 'Full Name', **{f'{m}_1': m for m in metrics}}
)

# Per-fight rows for Fighter_2
df2 = fights_df[['Fighter_2', 'Round', 'Avg Fight Time'] + [f'{m}_2' for m in metrics]].rename(
    columns={'Fighter_2': 'Full Name', **{f'{m}_2': m for m in metrics}}
)

# Concating the two dataframes and averaging each fighter's statistics
fighter_stats = (
    pd.concat([df1, df2], ignore_index=True)
      .groupby('Fighter_Id')
      .agg({col: 'mean' for col in df1.select_dtypes(include='number').columns})
      .round(2)
      .sort_index()
)
fighter_stats.rename(columns={'Round': 'Avg Rounds'}, inplace=True)

# Merging fighters with their aggregated stats
fighters_df = pd.read_csv('data/Fighters.csv')
# The join should be inner to avoid missing values from both sides (I trued both left and right joins)
fighters_df = fighters_df.join(fighter_stats, on='Fighter_Id', how='inner')

# Get each fighter's weightclass from the Fights dataset

fighters_weight_class = pd.concat(
    [
        fights_df[['Fighter_Id_1', 'Weight_Class']].rename(
            columns={'Fighter_Id_1': 'Fighter_Id'}),
        fights_df[['Fighter_Id_2', 'Weight_Class']].rename(
            columns={'Fighter_Id_2': 'Fighter_Id'})
    ]
)

fighters_weight_class['Fighter_Id'] = fighters_weight_class['Fighter_Id'].astype(
    'str')

# Keep only the first occurence. Fights are already in a chronological order and we need to get last weight class a fighter played in
fighters_weight_class = fighters_weight_class.drop_duplicates(
    subset=['Fighter_Id'], keep='first')

fighters_weight_class.set_index('Fighter_Id', inplace=True)

# Join with fighters_df to add weight class information
fighters_df = fighters_df.merge(
    fighters_weight_class, on='Fighter_Id', how='left')


# Extract gender from Weight_Class
fighters_df['Gender'] = fighters_df['Weight_Class'].str.startswith(
    'Women').map({True: 'Female', False: 'Male'})

# fighters_df.set_index('Full Name',inplace=True)

# The data is here let's get to preprocessing

# 25% of fighters have missing Reach + I do not think Reach is important for clustering fighters into different styles
fighters_df.drop(columns='Reach', inplace=True)
# Dropping fighters with missing Ht. or Wt. (Whether they are retired or for whatever reason these values are missing)
# I think dropping them from the dataframe won't do any harm
fighters_df.dropna(subset=['Wt.', 'Ht.'], inplace=True)

# Imputing Ctrl and Significant Strikes Percentage with aggregated numbers of similar fighters
imputer = KNNImputer()
imputer.fit(fighters_df[['Ctrl', 'Sig. Str. %']])
fighters_df[['Ctrl', 'Sig. Str. %']] = imputer.transform(
    fighters_df[['Ctrl', 'Sig. Str. %']])

fighters_df.set_index('Fighter_Id', inplace=True)

cols_to_drop = ['Full Name', 'Nickname']
X = fighters_df.drop(columns=cols_to_drop)
cat_cols = ['Stance']
X['Stance'] = X['Stance'].astype('category')


# scale numeric columns in-place
num_cols = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# k-means assumes continuous variables. The use on categorical data, even when one-hot encoded,is questionable.
# It sometimes works "okayish" but barely ever workd "good".

# Feature Engineering : Adding a win ratio feature
# X['Win Ratio'] = X['W'] / (X['W'] + X['L'] + X['D'])

# Finding this comment helped me improve my model
X.drop(columns=['Stance', 'Belt', 'Avg Rounds',
       'Weight_Class', 'Gender'], inplace=True)


def train_kmeans(X):
    kms = []
    cluster_centers = []
    silhouette_scores = []
    inertias = []
    calinski_scores = []
    bouldin_scores = []

    for i in range(2, 20):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X)
        labels = km.predict(X)
        kms.append(km)
        cluster_centers.append(km.cluster_centers_)
        silhouette_scores.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)
        calinski_scores.append(calinski_harabasz_score(X, labels))
        bouldin_scores.append(davies_bouldin_score(X, labels))

    return [kms, bouldin_scores, inertias, silhouette_scores, calinski_scores, cluster_centers]


kms, bouldin_scores, inertias, silhouette_scores, calinski_scores, cluster_centers = train_kmeans(
    X)


# - There is no clear `Elbow` point in the Inertia graph
# - Based only on the previous graphs we are so much far from having clear and separated clusters


# Reduce numeric features to 2D
reducer = umap.UMAP(n_components=2)
embedding = pd.DataFrame([])
embedding[['UMAP_1', 'UMAP_2']] = reducer.fit_transform(X)

# PCA dimensionality Reduction

pca = PCA(n_components=2, random_state=42)
pca.fit(X)
X_pca = pd.DataFrame([])
X_pca[['PCA_1', 'PCA_2']] = pca.transform(X)

pca_kms, bouldin_scores, inertias, silhouette_scores, calinski_scores, cluster_centers = train_kmeans(
    X_pca)
pca_km = pca_kms[2]

# This decision is based on the PCA visualizations performed .
# We are going to continue with 4 clusters :) You can find why in the readme file
km = kms[2]
fighters_df['Cluster'] = km.predict(X)


# Print fighters of interest (That I already I know) and their cluster assignments
def explore_clusters(col):
    fighters_list = [
        'Islam Makhachev',
        'Ilia Topuria',
        'Jon Jones',
        'Daniel Cormier',
        'Max Holloway',
        'Demetrious Johnson',
        'Dricus Du Plessis',
        'Alex Pereira',
        'Magomed Ankalaev',
        'Israel Adesanya',
        'Khabib Nurmagomedov',
        'Alexander Volkanovski',
        'Merab Dvalishvili',
        'Umar Nurmagomedov',
        'Conor McGregor',
        'Francis Ngannou'
    ]
    for fighter in fighters_list:
        print(fighter)
        print(fighters_df[fighters_df['Full Name'] == fighter][col])
        print()


# There is no clear separated clusters. However the UMAP visualization seems to be the best, having the minimum of overlapping points

# Testing the model on the reduced dataset
pca_km.fit(X_pca)
fighters_df['PCA Cluster'] = pca_km.predict(X_pca)

# > `PCA_km` model is far more **precise** than `km`
#
# So PCA Cluster is gonna be our Fighting Style column


explore_clusters('PCA Cluster')

# Explore each cluster summary statistics to gain insights about fighting style.


# # Clustering Analysis of UFC Fighter Styles
#
# After completing these two visualizations we can confidently say:
#
# - **Cluster 0** is for strikers: fighters in this group have the highest knockdown average, strikes and significant strikes percentage.
# - **Cluster 1** is for wrestlers: fighters who tend more to the ground game having the biggest control time, submissions, ground strikes percentage, submission attempts and more features distinguishing wrestlers.
# - **Cluster 2** is for hybrid fighters: having a mix of both striking & wrestling.
# - **Cluster 3** has nothing significant: so they will be labeled as 'No Fighting Style'.


style_dummies = pd.get_dummies(fighters_df['PCA Cluster'])
temp_df = pd.concat([fighters_df.select_dtypes(
    exclude=['object', 'category', 'boolean']), style_dummies], axis=1)
corr_matrix = temp_df.corr()
corr_matrix = corr_matrix.loc[corr_matrix.drop(
    columns=[0, 1, 2, 3]).columns, [0, 1, 2, 3]]

# Using correlation to map numerical fighting classes to their meaning
striking_cols = ['KD', 'STR', 'Sig. Str. %',
                 'Head_%', 'Body_%', 'Leg_%', 'Distance_%']
wrestling_cols = ['TD', 'SUB', 'Ctrl', 'Ground_%', 'Sub. Att', 'Rev.',]

striker = corr_matrix.transpose()[striking_cols].mean(axis=1).idxmax()
wrestler = corr_matrix.transpose()[wrestling_cols].mean(axis=1).idxmax()
no_style = corr_matrix.transpose().mean(axis=1).idxmin()
# We use 6 - sum(already_assigned_classes) to get the index of the remaining class which is hybrid
hybrid = 6 - (striker+wrestler+no_style)

# Building the hashmap dynammically
map_dict = {
    striker: 'Striker',
    wrestler: 'Wrestler',
    hybrid: 'Hybrid',
    no_style: 'No Clear Style',
}

fighters_df['Fighting Style'] = fighters_df['PCA Cluster'].map(map_dict)

# Converting the Fighting Style column into categorical
fighters_df['Fighting Style'] = fighters_df['Fighting Style'].astype(
    'category')
# Finished working with the numerical categories.
# It's time to drop them
fighters_df.drop(columns=['Cluster', 'PCA Cluster'], inplace=True)

# Visualizing how each Fighting Style correlates to fighter's statistics
# I need to do this one hot encoding because corr method does not accept any non numerical feature
style_dummies = pd.get_dummies(fighters_df['Fighting Style'])
numeric_with_style = pd.concat([fighters_df.select_dtypes(
    exclude=['object', 'bool', 'category']), style_dummies], axis=1)

# Also removing feature that the model was not trained on
no_train_cols = ['Ht.', 'Wt.', 'W', 'L', 'D', 'Avg Rounds']
numeric_with_style.drop(columns=no_train_cols, inplace=True)


corr_matrix = numeric_with_style.corr()


# ### This is absolutely amazing!
#
# This correlation heatmap reveals a **clear relationship** between fighting styles and their defining characteristics:
#
# - **Strikers**: Strong positive correlation with striking metrics (STR, KD, Sig. Str. %, Distance_%)
# - **Wrestlers**: Highly correlated with ground game metrics (SUB, TD, Ctrl, Sub. Att, Rev.)
# - **Hybrid Fighters**: Negative correlations with specialized metrics, indicating balanced skill distribution
# - **No Clear Style**: Strong negative correlation with technical metrics across the board
#
# These correlations validate our clustering approach perfectly!


fighters_df.head()

distances = pca_km.transform(X_pca)

def distance_to_membership(distances, alpha=1.0):
    exp_distances = np.exp(alpha * -(distances+1e-10))
    memberships = exp_distances / exp_distances.sum(axis=1, keepdims=True)

    return memberships


memberships = distance_to_membership(distances, alpha=2.0)

fighters_df['Striker_Membership'] = memberships[:, striker]
fighters_df['Wrestler_Membership'] = memberships[:, wrestler]
fighters_df['Hybrid_Membership'] = memberships[:, hybrid]
fighters_df['NoStyle_Membership'] = memberships[:, no_style]



fighters_df.to_csv("data/Fighters Stats.csv")
