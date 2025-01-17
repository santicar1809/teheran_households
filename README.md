# Clustering and Prediction of Properties in Teheran

## Project Description

This project focuses on clustering and predicting property prices in Teheran using a dataset of property features. The workflow involves preprocessing, exploratory data analysis (EDA), clustering, and regression modeling. Below are the detailed steps:

### 1. Data Preprocessing

- **Data Cleaning**:
  - Verify the dataset for missing (NaN) values and duplicate entries.
  - Handle missing values by either dropping rows/columns or imputing them appropriately.
  - Remove duplicates to ensure data consistency.
  
- **Type Verification**:
  - Check the data types of all columns to ensure compatibility for modeling.
  - Identify and address outliers that may impact the analysis.

### 2. Exploratory Data Analysis (EDA)

- Conduct in-depth analysis to uncover insights within the dataset.
- Focus on the relationships between property features and their correlation with the target variable (price).
- Visualize distributions and relationships between variables to understand the data better.

### 3. Feature Engineering for Clustering

- Normalize the data to standardize features for clustering algorithms.
- Use visualization techniques (e.g., PCA or t-SNE) to simplify the representation of clusters.

### 4. Clustering

- Apply unsupervised learning algorithms:
  - **K-Means**: Determine optimal cluster count using the elbow method and silhouette score.
  - **DBSCAN**: Identify clusters with varying densities.
  - **Hierarchical Clustering**: Visualize relationships using a dendrogram.

### 5. Feature Engineering for Regression

- Encode categorical data into numeric format to make it compatible with regression algorithms.
- Ensure all features are scaled and transformed appropriately for regression tasks.

### 6. Regression Modeling

- **Data Preparation**:
  - Separate the target variable (price) from the features.
  - Use pipelines to automate scaling and model training.

- **Algorithms Used**:
  - LightGBM
  - CatBoost
  - XGBoost
  - Decision Tree
  - Random Forest
  - Linear Regression
  - Neural Network

- **Evaluation Metrics**:
  - Compare model performance using metrics such as:
    - R² Score
    - Mean Squared Error (MSE)

### 7. Model Testing

- Evaluate models on a test dataset to assess their generalization performance.
- Compare the results across different models to determine the best-performing algorithm.

## Results

The project demonstrates clustering and prediction techniques on property data, highlighting the differences in model performance and providing actionable insights into property valuation in Teheran. Clustering provides a segmentation of properties, while regression models help in accurately predicting property prices.

## Files

- **Data**: Preprocessed datasets and raw files.
- **EDA**: Visualizations and analysis reports.
- **Models**: Trained clustering and regression models.
- **Results**: Performance metrics and comparisons.

## Tools and Libraries

- **Clustering Algorithms**: KMeans, DBSCAN, Hierarchical Clustering
- **Regression Algorithms**: LightGBM, CatBoost, XGBoost, Random Forest, Neural Network
- **Libraries**:
  - Python (NumPy, Pandas, Matplotlib, Seaborn)
  - Scikit-learn
  - LightGBM, CatBoost, XGBoost
  - TensorFlow/Keras for Neural Networks
  - Scipy

## Conclusion

This project showcases the application of clustering and regression models to analyze and predict property prices in Teheran. The results provide valuable insights into the property market and demonstrate the importance of robust preprocessing, feature engineering, and algorithm selection in achieving accurate predictions.

