# AI in Medical Fields

## Overview
This repository provides an overview of **AI in Medical Fields** course at Cairo University, Faculty of Engineering. Each discussion topic aimed at expanding our knowledge beyond the common techniques and each assignment explores different ML concepts implemented from scratch.

## Acknowledgements

This project was submitted as part of the **SBE3021 AI in Medical Fields course, Spring 2024** under the supervison of **[Dr. Inas Yassine](https://www.linkedin.com/in/inas-yassine-15ab4b4/?originalSubdomain=eg)** and **[ENG. Merna Bibars](https://merna-atef.github.io/)**.

## 💡 Key Discussion Topics from Lectures

### 1. Outlier Detection Methodologies
- **Statistical Methods**:
  - Z-score analysis
  - IQR (Interquartile Range) method
  - Assumption of independence vs. dependency in multivariate data
- **Modern Approaches**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - DBSCAN for density-based detection

### 2. Advanced Stopping Criteria in ML
- **Beyond Traditional Methods**:
  - Early stopping with validation set
  - Cross-validation based stopping
  - Learning rate scheduling
- **Alternative Approaches**:
  - Gradient-based criteria
  - Model complexity metrics
  - Ensemble validation techniques

### 3. Cluster Assignment Metrics
- **Distance Measures**:
  - Cosine similarity
  - Manhattan distance
  - Minkowski distance
  - Correlation-based distances
- **Advanced Techniques**:
  - Kernel-based similarity measures
  - Probabilistic approaches
  - Density-based metrics

### 4. Non-Linear Metrics in Tree-Based Models
- **Alternatives to Information Gain**:
  - Gini impurity
  - Chi-square test
  - Gain ratio
- **Advanced Splitting Criteria**:
  - MDL (Minimum Description Length)
  - Distance-based measures
  - Custom domain-specific metrics

### 5. Additional Topics of Interest
- Trade-offs between model complexity and interpretability
- Impact of feature scaling on different distance metrics
- Handling high-dimensional data in clustering

## 📚 Assignments

### Assignment 1: Introduction to Machine Learning
- **Dataset**: Breast Cancer Wisconsin (Diagnostic) Database
- **Topics Covered**:
  - Data preprocessing and exploration
  - K-Nearest Neighbors (KNN) implementation
  - Model evaluation and confusion matrix analysis
- **Key Results**: 
  - Successfully implemented KNN classifier
  - Achieved model optimization through data normalization

### Assignment 2: Regression Analysis
- **Topics Covered**:
  - Linear Regression implementation
  - Polynomial feature transformation
  - Model evaluation metrics (MSE, R² Score)
- **Key Findings**:
  - Training R² Score: 0.55
  - Test performance analysis revealed overfitting issues
  - Implemented data scaling for improved results

### Assignment 3: Logistic Regression
- **Topics Covered**:
  - Binary classification
  - Feature preprocessing
  - Model regularization
- **Highlights**:
  - Implemented oversampling techniques
  - Achieved accuracy improvement through feature scaling
  - ROC AUC Score analysis

### Assignment 4: Support Vector Machines
- **Topics Covered**:
  - SVM implementation with different kernels
  - Hyperparameter tuning using GridSearchCV
  - Cancer dataset classification
- **Results**:
  - Best parameters: C=10, kernel='rbf'
  - Training accuracy: 98.8%
  - Test accuracy: 99.3%

### Assignment 5 & 6: Decision Trees
- **Topics Covered**:
  - Decision tree implementation
  - ID3 algorithm
  - Feature importance analysis
- **Key Implementation**:
  - Custom ID3 classifier
  - Tree visualization
  - Model evaluation metrics

### Assignment 7: Unsupervised Learning
- **Topics Covered**:
  - K-means clustering
  - Dealing with non-globular cluster shapes
  - Principal Component Analysis (PCA)
  - Dimensionality reduction
- **Implementation Details**:
  - Custom PCA function
  - Clustering visualization
  - Eigenvalue analysis

### Assignment 9: Movie Recommender System
- **Details can be found here**.

## 🛠️ Technologies Used
- Python 3.11
- Key Libraries:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
