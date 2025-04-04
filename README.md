# 🛍️ E-commerce Product Analysis – Data Science Capstone Project

## 📌 Project Overview

This capstone project focuses on analyzing Amazon product data using both **Unsupervised** and **Supervised Learning** techniques. The goal is to uncover patterns in product features, group similar products using clustering, and classify product categories using machine learning models.

---

## 🧠 Objectives

- 📊 Perform **K-Means Clustering** to group similar products based on price, rating, and reviews.
- 🤖 Build and evaluate classification models to predict the product category.
- 🛠️ Use **Grid Search** for hyperparameter tuning to improve model accuracy.

---

## 🗂️ Dataset

The dataset contains details of Amazon products, including:

- `Product Name`
- `Price`
- `Rating`
- `Reviews`
- `Category`

---

## 🔧 Methodology

### 1. Data Preprocessing
- Handled missing values and data type conversions.
- Standardized features using `StandardScaler`.
- Encoded product categories using `LabelEncoder`.

### 2. Unsupervised Learning (Clustering)
- Used **K-Means** to find optimal clusters via the Elbow Method.
- Visualized clusters based on product `Price` and `Rating`.

### 3. Supervised Learning (Classification)
- Trained the following models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - k-Nearest Neighbors (k-NN)
  - Random Forest
- Evaluated with Accuracy and F1 Score.

### 4. Hyperparameter Tuning
- Used `GridSearchCV` for fine-tuning the above models.
- Compared model performance before and after tuning.

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook / VS Code

---

## 📈 Results

- Identified **optimal number of clusters** for product grouping.
- The **Random Forest** model (after tuning) achieved the best classification performance.
- Visualizations were generated to interpret clusters and model accuracies.

---


