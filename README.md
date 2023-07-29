# Bharat-Intern-Task--2-Titanic-Classification

## Overview
This repository contains the code and resources for the "Titanic Classification" data science project completed as part of the Data Science internship with "BHARAT INTERN." The project's main objective was to predict the survival of passengers onboard the Titanic based on various features like age, gender, ticket class, and more.

## Project Description
The sinking of the RMS Titanic on its maiden voyage in April 1912 is one of the most infamous shipwrecks in history. This project revolves around the Kaggle's Titanic dataset, which provides information about passengers, including whether they survived or not. The goal was to build a classification model using machine learning techniques to predict the survival outcome of passengers based on certain features.

## Dataset
The dataset used for this project, "titanic.csv," contains the following columns:
- `PassengerId`: Unique identifier for each passenger.
- `Survived`: Target variable (0 = No, 1 = Yes).
- `Pclass`: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- `Name`: Passenger's name.
- `Sex`: Gender of the passenger (male or female).
- `Age`: Age of the passenger in years.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Ticket`: Ticket number.
- `Fare`: Fare paid for the ticket.
- `Cabin`: Cabin number.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Methodology
The project was divided into the following main steps:

1. **Data Preprocessing**: Exploring and cleaning the dataset, handling missing values, and converting categorical variables into numerical format.

2. **Exploratory Data Analysis (EDA)**: Analyzing the data, visualizing relationships between features and the target variable, and gaining insights into the dataset.

3. **Feature Engineering**: Creating new features or modifying existing ones to improve model performance.

4. **Model Selection**: Evaluating different classification algorithms, including logistic regression, decision trees, random forests, and support vector machines, to identify the best performing model.

5. **Model Training and Evaluation**: Splitting the dataset into training and testing sets, training the chosen model on the training data, and evaluating its performance on the testing data using appropriate metrics.

6. **Hyperparameter Tuning**: Fine-tuning the model by optimizing hyperparameters to achieve better results.

7. **Conclusion**: Summarizing the project's findings and the performance of the final model.

## Requirements
- Python (version X.X.X)
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Usage
1. Clone this repository to your local machine using `git clone`.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open the Jupyter Notebook `Titanic_Classification.ipynb`.
4. Follow the step-by-step instructions in the notebook to explore the dataset, preprocess the data, train different models, and make predictions.
5. Experiment with hyperparameter tuning to improve the model's performance.
6. Share your findings and results with the team at "BHARAT INTERN."

