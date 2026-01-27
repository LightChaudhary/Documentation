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

### Projects
# Spam Email Classifier (Machine Learning - NLP)

A real-world Spam vs Ham (Not Spam) email classification system built using Python, Natural Language Processing (NLP), and Machine Learning.

This project uses the Enron Email Dataset and applies text cleaning, TF-IDF vectorization, and supervised learning to automatically detect spam emails.

## What This Project Does
Given an email, the system predicts whether it is: 
* SPAM
* HAM (Not Spam)
It learns patterns from thousands of real emails and builds a statistical model that understands which words and phrases indicate spam.

## Machine Learning Pipeline
The project follows workflow: 
Raw Emails
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Train/Test Split
     ↓
Machine Learning Model
     ↓
Evaluation
     ↓
Live Email Prediction

## Dataset
We use the Enron Spam Dataset, which contains real corporate emails.
Dataser structure: 
| Column   | Description         |
| -------- | ------------------- |
| Subject  | Email subject       |
| Message  | Email body          |
| Spam/Ham | Label (spam or ham) |
| Date     | Email date          |

Total emails : 33,716
Balanced classes:
* Spam ≈ 17,171
* Ham ≈ 16,545

### Step 1- Data Preparation

We combine the subject and message into one text field.
We also convert labels into numbers.

### Step 2- Text 
Each email is cleaned using:
* Lowercasing
* Removing numbers
* Removing punctuation
* Removing stopwords (like the, is, and, to)
* Tokenization
This makes the model focus only on meaningful words.

### Step 3- Feature Engineering(TF-TDF)
We convert text into numbers using TF-IDF (Term Frequency – Inverse Document Frequency).
This means:
* Words that appear often in a single email get more weight
* Words that appear in almost every email (like “the”) get less weight
This creates a numeric vector representation of each email.

### Step 4- Model Training & Evaluation
The model is trained on labeled email data and evaluated on unseen emails.
**Results**
Accuracy: 98.23%

Precision, Recall, F1-Score:
Spam and Ham are both predicted with extremely high reliability.

### Step 5- Live Email Prediction
You can type any email, and the model will classify it:

Example: 
Enter an email:
Congratulations! You have won $10,000. Click here to claim.

Output:
SPAM

Enter an email:
Hey, there is a meeting tomorrow at 2pm.

Output:
HAM (Not Spam)

## Technologies Used 
* Python
* Pandas
* Scikit-learn
* Regular Expressions
* NLP (TF-IDF)
* Jupyter Notebook / VS Code

## Current Status
* Data loaded
* Text cleaned
* Features extracted
* Model trained
* Model evaluated
* Real email predictions working

## Next Improvements (Future Work)
* Add multiple models
* Compare model performance
* Tune hyperparameters
* Save & load trained models
* Build a web or API interface
* Use deep learning (BERT)
