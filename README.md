This is a Python notebook for a Data Science Capstone Project focused on analyzing credit card customer behavior. 
It aims to make it easier for Banks to accept/reject customers based on their data by segmenting customers into 4 
groups, namely, Denied, For Review 1, For Review 2, and Approved. This serves as out Capstone Project from the 5-day 
Data Science Workshop conducted by the University of the Philippines Diliman Mathematics Foundation Inc. On April 14-15 
and June 25-27, 2025 which aims to provide us practical experience in applying our knowledge on Exploratory Data Analysis, Data Cleaning,
Data Visualization, and Machine Learning models.

**Key Questions Addressed
The project starts with these guiding questions to drive the analysis:**

1. How do different types of usage behavior affect a customer’s ability to make high percent credit card payments?
2. What is the optimal credit limit that ensures payment of credit card debt?
3. Does balance affect the mode of payment?
4. What types of credit card holders are present in the dataset? 


  ** Notebook Contents Outline**
The notebook is divided into logical steps, covering data loading,EDA, preprocessing, unsupervised learning, supervised learning, 
and an interactive prediction tool. Below is a detailed outline, with a strong focus on the machine learning parts.

**1. SETUP AND LOADING**
  Import necessary libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn (for preprocessing, clustering, classification, and metrics).
  Load the Kaggle's dataset: credit card information.
  
**2. Exploratory Data Analysis (EDA) and Data Cleaning**\
   **Data Inspection:** Shape (8950, 18), info, summary statistics, no duplicates, missing values check (132 in CREDIT_LIMIT, 313 in MINIMUM_PAYMENTS)\
   **Distributions:** Histograms and boxplots for all numerical features to visualize spread, central tendency, skewness,\
   and outliers (e.g., many outliers in PURCHASES and BALANCE).\
   **Outlier and Missing Value Handling:**\
     Detect outliers using IQR method and replace with NaN.\
     Impute NaNs (including outliers) with median values.\
   **Relationships Between Variables:**\
      Correlation heatmap (e.g., strong positive correlation between PURCHASES and PAYMENTS ~0.92).\
      Pairplots (focused on key features like BALANCE, PURCHASES, CREDIT_LIMIT, colored by PURCHASES_FREQUENCY\
      
**3. FEATURE ENGINEERING**
     Create AVG_TRX_AMOUNT = PURCHASES / PURCHASES_TRX to capture average purchase size.\
     
**4. Data Preprocessing for Modeling (Focus: Preparing for ML)**\
     Standardize all features using StandardScaler() to ensure equal weighting in distance-based algorithms.\
     Fit PCA on scaled data (18 components).\
     **Scree plot:** Cumulative explained variance shows 11 components capture ~90% variance (chosen to retain 90% while reducing noise).\
     Transform data to 11 principal components (pca_df_scale).\

**5. Unsupervised Learning: Customer Segmentation (Key ML Section)**\
     **Algorithm:**\ 
     **K-Means Clustering on PCA-transformed data.**\
        Elbow Method (using Yellowbrick's KElbowVisualizer): Optimal k=4 clusters (based on distortion score).\
     **Fit K-Means:** \
        KMeans(n_clusters=4, init='k-means++', n_init=12).\
     **Radar Chart (Polar Plot):** \
        Mean feature values per cluster using Plotly\
     **Interpretation (Cluster Profiles):**\
       **Denied (Cluster 0):** Low activity, infrequent usage—likely inactive or low-value customers (target for retention?).\
       **For Review 1 (Cluster 1):** High cash advances, high balance—possible budget-conscious users limiting monthly expenses.\
       **Approved (Cluster 2):** High purchases (mostly one-off), high payments—reliable, high-value spenders (target for premium offers/higher limits).\
       **For Review 2 (Cluster 3):** Frequent installment purchases—installment-focused users.\
**6. Supervised Learning: Cluster Classification (Key ML Section)**\
     **Goal:**\
       Predict a customer's cluster based on features to enable real-time assessment (e.g., for credit decisions or targeting).\
     **Algorithm:** \
       Support Vector Machine (SVM) with RBF kernel (SVC(kernel='rbf'))\
     **Evaluation Metrics:**\
       Accuracy: ~0.93 .\
       Confusion Matrix: Heatmap showing minimal misclassifications above 0.9 precision,recall, and f1-score across all 4 classes.\
       Classification Report: Precision/Recall/F1 >0.90 across clusters.\
   
   






