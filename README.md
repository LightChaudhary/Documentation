### My AI & Machine Learning Learning Journey

This repository is dedicated to documenting my learning journey in AI and Machine Learning. I aim to understand concepts deeply, practice coding, and improve day by day. Every project, notebook, and exercise here represents a step toward mastering these topics and applying them in real-world scenarios.

## Task 1: Understanding Decision Tree Classifier

**Goal:**
To understand how a Decision Tree Classifier works using labeled examples and features.

## Task 2: Iris Flower Classification  

**Description:**  
Classifies Iris flowers (setosa, versicolor, virginica) using a Random Forest model. Trains on a dataset and predicts on a test set, showing accuracy.  

**Files:**  
- `Iris-Train.csv` – Training data (included)  
- `Iris-Test.csv` – Test data (included)  
- `irisdatasets_train_test.py` – Python script  

## Task 3: Linear Regression House Price

Goal:
To predict (create) a random house price based on area using a Linear Regression model and visualize the relationship.

## Task 4: Linear Regression with Train–Test Split

Goal:
To train a Linear Regression model using house area and price data, evaluate its performance using Mean Squared Error (MSE) and R² score, and visualize both the training and testing results with a regression line.

## Task 5: Multiple Feature House Price Prediction with Scaling

Goal:
To predict house prices using multiple features (area, number of bedrooms, and house age) with a Linear Regression model, applying feature scaling (StandardScaler) to improve model performance before training and prediction.

## Task 6: Multiple Feature Linear Regression with Error Evaluation

Goal:
To train a Linear Regression model using multiple house features and evaluate its performance using Mean Squared Error (MSE) by comparing actual and predicted house prices.

## Task 7 : Polynomial Regression for Non-Linear Data

Goal:
To model and visualize a non-linear relationship between input and output using Polynomial Regression (degree 2) by transforming features and fitting a Linear Regression model on the transformed data.

## Task 8: Logistic Regression with Train–Test Split

**Goal:** To train a Logistic Regression model using study hours and pass/fail data, evaluate its performance using accuracy, confusion matrix, and classification report, and make predictions for new inputs.

**Dataset:** Hours studied vs Result (0 = Fail, 1 = Pass)  
**Libraries Used:** NumPy, scikit-learn  

**Steps:**  
1. Split data into training and testing sets  
2. Train Logistic Regression model  
3. Predict test set results  
4. Evaluate accuracy and generate confusion matrix & classification report  
5. Predict new values and probability of passing  

**Outcome:** Model predicts pass/fail based on study hours and gives probabilities for each class.

## Task 9: Decision Tree Classifier with Train–Test Split

**Goal:** To train a Decision Tree model using study hours and pass/fail data, evaluate its performance using accuracy, and make predictions for new inputs.

**Dataset:** Hours studied vs Result (0 = Fail, 1 = Pass)  
**Libraries Used:** NumPy, scikit-learn  

**Steps:**  
1. Split data into training and testing sets (stratified)  
2. Train Decision Tree model with max depth = 2  
3. Predict test set results  
4. Evaluate accuracy  
5. Predict new values (e.g., 6.5 hours)  

**Outcome:** Model predicts pass/fail based on study hours with a decision tree approach.

# Task 10: Random Forest Classifier with Train–Test Split

**Goal:** To train a Random Forest model using study hours and pass/fail data, evaluate its performance using accuracy, and make predictions for new inputs.

**Dataset:** Hours studied vs Result (0 = Fail, 1 = Pass)  
**Libraries Used:** NumPy, scikit-learn  

**Steps:**  
1. Split data into training and testing sets (stratified)  
2. Train Random Forest model with 70 estimators and max depth = 2  
3. Predict test set results  
4. Evaluate accuracy  
5. Predict new values (e.g., 6.5 hours) and check feature importance  

**Outcome:** Model predicts pass/fail using an ensemble of decision trees and shows feature importance.

# Task 11: Support Vector Machine (SVM) Classification

**Goal:** Predict student results using Linear and RBF SVM models.

**Dataset:** Student performance
**Features:** Study_Hours, Attendance, Previous_Score
**Target:** Result (Pass/Fail)

**Libraries:** Pandas, Scikit-learn

**Steps:**
1. Load dataset and select features
2. Split data into train and test sets
3. Apply StandardScaler
4. Train Linear SVM and RBF SVM
5. Evaluate accuracy
6. Tune C and gamma values

**Outcome:** Compared Linear vs RBF SVM and found optimal hyperparameters for best accuracy.

# Task 12: K-Nearest Neighbors (KNN) Classification

**Goal:** Predict student results using the KNN algorithm.

**Dataset:** Student performance  
**Features:** Study_Hours, Attendance, Previous_Score  
**Target:** Result (Pass/Fail)

**Libraries:** Pandas, Scikit-learn

**Steps:**  
1. Load dataset and select features  
2. Perform stratified train–test split  
3. Apply StandardScaler  
4. Train KNN with k = 11  
5. Evaluate using accuracy and classification report

**Outcome:** KNN model successfully classifies student results using nearest neighbors.

# Task 13: Multiclass Logistic Regression Classification

**Goal:** Perform multiclass classification using Logistic Regression and predict new student data.

**Dataset:** Synthetic dataset (make_classification)  
**Features:** 4 numerical features  
**Target:** 3 Classes (Multiclass)

**Libraries:** NumPy, Scikit-learn

**Steps:**  
1. Generate synthetic multiclass dataset  
2. Perform stratified train–test split  
3. Apply StandardScaler  
4. Train Logistic Regression (multinomial)  
5. Evaluate using accuracy and classification report  
6. Predict class and probability for a new sample

**Outcome:** Logistic Regression successfully classifies multiple classes and predicts probabilities for new inputs.
