# Titanic Survival Prediction

## Project Overview
This project involves participating in the Kaggle competition "Titanic: Machine Learning from Disaster," where the task is to predict survival based on passenger information. The goal is to fit and compare three different learning algorithms, including a linear and a non-linear learner, and select the best model for submission.

## Data
- **Dataset:** Titanic: Machine Learning from Disaster (available on Kaggle)
- **Description:** The dataset contains information about the passengers aboard the Titanic, including features such as age, sex, passenger class, and whether they survived.

## Task
1. **Load Data:**
   - Load the Titanic dataset from Kaggle.

2. **Exploratory Data Analysis and Pre-processing:**
   - Data cleaning.
   - Identification and treatment of missing values and outliers.
   - Feature engineering.
   - Generate at least three plots:
     - Scatter plot to inspect relationships between two variables.
     - Histogram to show distributions (e.g., age).
     - Pie chart to show the relationship between a categorical variable and survival.
   - Print a basic data description.
   - Print descriptive statistics.

3. **Partition Data:**
   - Split the data (excluding Kaggle's test set) into train, validation, and test sets.

4. **Model Training:**
   - Fit three different learning algorithms on the training set.
   - Perform hyper-parameter search if necessary.
   - Select the best model based on validation set performance.

5. **Model Evaluation:**
   - Print the results of all three models on the test set, including accuracy, F1-score, and AUC.

6. **Submission:**
   - Save the predictions of the best model on Kaggle's test set to `submission.csv`.

## Deliverable
- A Jupyter Notebook (`Titanic_Survival_Prediction.ipynb`) containing the complete code that trains all three models and evaluates their performance.
- The notebook should print accuracy, F1-score, and AUC for each model under the "Results" section.

## Example Output
Data description
Number of examples
train: X
valid: Y
test: Z
Number of features: XX
Number of examples per class
class 0: YY
class 1: ZZ

Descriptive statistics
Age
mean: XXX
median: YYY
standard deviation: ZZZ

Results
Model 1 <print the name>
Accuracy: XXXX
F1-score: YYYY
AUC: ZZZ
Model 2 <print the name>
Accuracy: XXXXX
F1-score: YYYYY
AUC: ZZZZ
Model 3 <print the name>
Accuracy: XXXXX
F1-score: YYYYY
AUC: ZZZZ


## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run
1. Ensure you have Python 3 and the necessary libraries installed.
2. Clone this repository: `git clone https://github.com/mredshaw/Titanic_Survival_Prediction.git`
3. Open the Jupyter Notebook: `Titanic_Survival_Prediction.ipynb`
4. Execute the cells in the notebook to train the models and evaluate their performance.

## Key Insights
This project demonstrates the application of machine learning algorithms to predict survival on the Titanic. By comparing multiple models and evaluating their performance, we can identify the best approach for this classification task.
