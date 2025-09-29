# UFC Fighting Style Categorizer 

---

### Project Overview
UFC Fighting Style Categorizer is a project that I built out of pure love for both ML and the MMA sport. It is mainly about building a model that is able to confidently assign each fighter to his fighting style. To achieve this, I first gathered data by scraping it from the UFC stats website, cleaned it, performed an EDA, and finally built a clustering model that determines fighters’ styles based on different fight statistics (Knockdowns, Takedowns, Strikes, ...).

---

### Collecting Data
I used BeautifulSoup to scrape Events, Fights, and Fighters from the UFC Stats website. Most of the data was arranged into paginated HTML tables, so the process in this step was nearly the same for all of the datasets. I grabbed each of the Events’ and Fights’ unique identifiers in order to keep the relational structure of the data for later use cases.  

For the fighters dataset, I generated a unique identifier using the fighter’s full name and nickname. Finally, instead of just scraping the fighters’ stats, I extracted them from the fights dataset using joins and aggregation functions.

You can find both raw and preprocessed data on [Kaggle](https://www.kaggle.com/datasets/aminealibi/ufc-fights-fighters-and-events-dataset)

---

### Data Cleaning
In this process I mainly used these techniques:
- Dropping columns with a large portion of missing data  
- Using KNN imputation to preserve the spherical aspect of data if it exists (for simpler clustering later)  
- Dropping samples with crucial features missing  
- Dropping highly correlated or redundant columns  
- Freeing space and optimizing storage by:  
  - Casting low-cardinality objects to categorical type  
  - Identifying small integers (`int8`) for features that cannot be very large (e.g., W, L, D)  

---

### Data Analysis
I used seaborn visualizations to explore various aspects of the data, correlations, and also answered questions based on different barplots, scatterplots, pie charts, and correlation heatmaps.

---

### Clustering Fighters
To categorize unlabeled data, hard clustering is a great option. Fitting the KMeans model to the whole dataset performed poorly with various `n_clusters` values.  

<img width="1990" height="790" alt="image" src="https://github.com/user-attachments/assets/49882c32-2bde-4bdc-98a3-28181e5383c4" />


The model had the best silhouette score with 2 clusters, but fighters have much more versatile fighting styles than only two. That's why I decided to continue with `n_clusters = 4`.  

This is a visualization of the clusters after using different dimensionality reduction techniques such as PCA, UMAP, and LLE:  

<img width="2534" height="790" alt="image" src="https://github.com/user-attachments/assets/a7574b68-ed4b-47c2-8921-fa2908f1b17e" />

As we can see, there are no clear and separated clusters. There are many overlapping data points.  

So I decided to fit the model to the PCA-reduced dataset, as it conserves the overall structure of the data but also reduces its complexity. It simply performed much better.  

<img width="1990" height="790" alt="image" src="https://github.com/user-attachments/assets/db9d3894-7dc5-440b-af5e-16a67e83b609" />

This is how the clusters turned out:  

<img width="2534" height="790" alt="image" src="https://github.com/user-attachments/assets/9c901aa6-35c5-4e4e-be3e-3965ba010be4" />

---

#### Labelling Clusters
Using a correlation heatmap, I can confidently label clusters like this:  

<img width="1817" height="1589" alt="image" src="https://github.com/user-attachments/assets/55449536-a08b-4f87-9f08-ddc2a388987c" />

- **Strikers**: Strong positive correlation with striking metrics (STR, KD, Sig. Str. %, Distance_%)  
- **Wrestlers**: Highly correlated with ground game metrics (SUB, TD, Ctrl, Sub. Att, Rev.)  
- **Hybrid Fighters**: Negative correlations with specialized metrics, indicating balanced skill distribution  
- **No Clear Style**: Strong negative correlation with technical metrics across the board  
