#  Online Retail Data Analysis and Modeling

1) Overview
This project explores the Online Retail Dataset from the UCI Machine Learning Repository using R. The goal is to apply various statistical and machine learning techniques to understand customer purchasing behavior, perform dimensionality reduction, clustering, classification, and multivariate analysis.

2) Objectives
* Perform Exploratory Data Analysis (EDA).
* Clean and preprocess the dataset.
* Apply dimensionality reduction techniques (PCA).
* Explore unsupervised learning methods (K-Means).
* Apply supervised learning techniques (K-Nearest Neighbors).
* Conduct Canonical Correlation Analysis (CCA).

3) Dataset
Source: UCI Machine Learning Repository
Link: Online Retail Dataset
      Contains transactional data from a UK-based online retailer.

4) Data Cleaning
* Removed null values and duplicates.
* Created new features like TotalPrice and HighSpender.
* Converted appropriate columns to categorical (as.factor).

5) Techniques Used
   * EDA
   * Bar plots for top countries by transaction
   * Histograms and boxplots for numeric features
   
   * Dimensionality Reduction - Principal Component Analysis (PCA)
   
   * Clustering (Unsupervised Learning) - K-Means Clustering with Elbow Method and Silhouette Score
     
   * Classification (Supervised Learning) - K-Nearest Neighbors (KNN) with Accuracy, Precision, Recall, F1
   
   * Canonical Correlation Analysis (CCA) - Explored relationships between variable sets

6) Results
     * Found optimal number of clusters using Elbow & Silhouette methods.
     * Achieved high classification accuracy with KNN on predicting High Spenders.
     * PCA helped reduce dimensionality while retaining most variance.

7) Requirements
   * R 4.x
   * Libraries: tidyverse, ggplot2, cluster, factoextra, caret, MASS, CCA, dplyr, psych, class, gridExtra

