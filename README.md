# Clustering and prediction of properties in Teheran

## Project description

1. A dataset is given with features of properties in Teheran to cluster every house, but first preprocessing of the data is needed so we verify if the data has nan values or duplicated values. In case data has this values we have to drop them or impute them. After that we verify each type of the columns data for modeling and identify outliers.

2. EDA is realized to find insights within data, focusing on the apartment features and the variables correlation.

3. Features engineer is developed to normalize data an visualize clusteres from the non-supervised algoithms more easily.

4. We used algorithms as KMEANS, DBSCAN AND Hierachical Clustering to group data, using techniques like elbow method, silouette method, dendogram method and so on.

5. Features engineer is needed again to encode data from the preprocessing dataset so the categorical data is transformed into numbers that can be processed by the regression algorithms.

6. Once we have our data ready, we used different regression models to predict the price of the properties based on the features and separating the target (price). We used a pipeline to scale and train data with different algorithms as Lightgbm, Catboost, XGboost, Decision Tree, Random Forest, Linear Regression and a neural network, so we can compare the performance of each model using metrics as R2 and Mean squared error.

7. Lastly we test our data with each dataset to evaluate our models without previusly been trained.