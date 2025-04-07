# Predicting Cardiovascular Diseases using Machine Learning

This project aims to develop machine learning models to predict the risk of cardiovascular diseases (CVDs) using patient health data.

## Project Overview

Cardiovascular diseases are a leading cause of death globally. Early prediction and intervention can significantly reduce mortality rates. This project utilizes machine learning algorithms to analyze patient data, including medical history, lifestyle factors, and clinical measurements, to predict the likelihood of CVDs.

## Dataset
The dataset used in this project was obtained from Doppala, B. P., & Bhattacharyya, D. (2021) and is available on Mendeley Data. It contains records of patients with cardiovascular disease-related data, with a total of 1000 records. The dataset includes 13 features, such as:

Age

Gender

Blood Pressure

Cholesterol levels

Smoking habits

Family history

The dataset was preprocessed to handle missing values, encode categorical variables, and scale numerical features for better model performance. The features were also analyzed for any correlations, and feature engineering techniques were applied where necessary.
## Methodology

The project follows these steps:

1.  **Data Exploration and Preprocessing:**
    * Loading and cleaning the dataset.
    * Handling missing values.
    * Encoding categorical features.
    * Scaling numerical features.
    * Exploratory data analysis (EDA) to understand feature distributions and relationships.
2.  **Feature Selection/Engineering (Optional):**
    * Identifying important features using techniques like feature importance from tree-based models or statistical tests.
    * Creating new features from existing ones if necessary.
3.  **Model Development:**
    * Training various machine learning models (e.g., Logistic Regression, Random Forest, Support Vector Machines, Gradient Boosting, Neural Networks).
    * Splitting the data into training and testing sets.
    * Hyperparameter tuning using techniques like GridSearchCV or RandomizedSearchCV.
4.  **Model Evaluation:**
    * Evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
    * Visualizing results using confusion matrices and ROC curves.
5.  **Model Deployment (Optional):**

## Files

* `data/`: Contains the dataset.
* `notebooks/`: Contains Jupyter Notebooks with code for data preprocessing, model training, and evaluation.
* `models/`:  Contains saved machine learning models.
* `README.md`: Project documentation.
* `requirements.txt`: List of Python dependencies.

## Requirements

To run this project, you need the following Python libraries:

pandas: For data manipulation and analysis.

numpy: For numerical operations and array handling.

matplotlib: For data visualization (plotting graphs and charts).

seaborn: For advanced data visualization and statistical plotting.

scikit-learn: For machine learning algorithms, model selection, and evaluation.

tensorflow or keras: For building and training neural networks (e.g., LSTM model).

xgboost: For gradient boosting models.

statsmodels: For statistical modeling and hypothesis testing.

scipy: For scientific computing, including optimization and integration
