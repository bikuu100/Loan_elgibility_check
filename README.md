
---

# Loan Eligibility Check

## Description
This repository contains the code for an exploratory data analysis (EDA) and a decision tree classifier model for loan eligibility prediction. The model achieves 98% accuracy.

## Table of Contents
- [Description](#description)
- [Data Handling and Preprocessing](#data-handling-and-preprocessing)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Inspection](#2-data-inspection)
  - [3. Data Cleaning](#3-data-cleaning)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Handling Outliers](#5-handling-outliers)
  - [6. Feature Selection](#6-feature-selection)
  - [7. Splitting the Data](#7-splitting-the-data)
  - [8. Data Transformation](#8-data-transformation)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Data Visualization](#data-visualization)
- [Save Preprocessed Data and Model](#save-preprocessed-data-and-model)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Data Handling and Preprocessing

### 1. Data Collection
- **Gather Data**: Ensure all relevant data is collected from various sources.

### 2. Data Inspection
- **Understand Data Structure**: Use functions like `head()`, `info()`, and `describe()` in pandas to understand the structure and summary statistics.
- **Check for Missing Values**: Use `isnull().sum()` to check for missing values.

### 3. Data Cleaning
- **Handle Missing Values**: Depending on the nature of the data and the proportion of missing values, you can:
  - **Remove rows/columns** with missing values using `dropna()`.
  - **Impute missing values** using methods like mean, median, mode, or more sophisticated techniques like K-Nearest Neighbors (KNN).
    ```python
    df['numerical_feature'].fillna(df['numerical_feature'].mean(), inplace=True)
    df['categorical_feature'].fillna(df['categorical_feature'].mode()[0], inplace=True)
    ```
- **Remove Duplicates**: Use `drop_duplicates()` to remove duplicate rows.
- **Correct Data Types**: Ensure numerical features are of numeric type and categorical features are of object type.
  ```python
  df['numerical_feature'] = df['numerical_feature'].astype(float)
  df['categorical_feature'] = df['categorical_feature'].astype('category')
  ```

### 4. Feature Engineering
- **Encoding Categorical Variables**:
  - **Label Encoding** for ordinal categories.
    ```python
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['education_level'] = le.fit_transform(df['education_level'])
    ```
  - **One-Hot Encoding** for nominal categories.
    ```python
    df = pd.get_dummies(df, columns=['gender', 'occupation', 'marital_status'])
    ```

### 5. Handling Outliers
- **Detect Outliers**: Use visualization techniques like box plots or statistical methods like Z-scores.
  ```python
  df['z_score_age'] = (df['age'] - df['age'].mean()) / df['age'].std()
  df['z_score_income'] = (df['income'] - df['income'].mean()) / df['income'].std()
  df['z_score_credit_score'] = (df['credit_score'] - df['credit_score'].mean()) / df['credit_score'].std()
  ```
- **Handle Outliers**: You can remove or transform outliers based on the context of your data.
  ```python
  df = df[(df['z_score_age'].abs() <= 3) & (df['z_score_income'].abs() <= 3) & (df['z_score_credit_score'].abs() <= 3)]
  df.drop(columns=['z_score_age', 'z_score_income', 'z_score_credit_score'], inplace=True)
  ```

### 6. Feature Selection
- **Correlation Analysis**: Use correlation matrices to identify highly correlated features.
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  correlation_matrix = df.corr()
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
  plt.show()
  ```
- **Feature Importance**: Use methods like feature importance from tree-based models or mutual information.

### 7. Splitting the Data
- **Train-Test Split**: Split the data into training and testing sets to evaluate model performance.
  ```python
  from sklearn.model_selection import train_test_split
  X = df.drop('loan_status', axis=1)
  y = df['loan_status']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

### 8. Data Transformation
- **Normalization/Standardization**:
  - **Standardization** scales the data to have a mean of 0 and a standard deviation of 1.
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])
    ```

## Model Building and Evaluation
- **Build and Evaluate Models**: Use various machine learning models and evaluate their performance using appropriate metrics.
  ```python
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.metrics import classification_report, accuracy_score

  model = DecisionTreeClassifier(random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print("Accuracy:", accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  ```

## Data Visualization
- **Count Plots**:
  ```python
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.countplot(x='education_level', hue='loan_status', data=df)
  plt.title('Loan Status by Education Level')
  plt.xticks(rotation=90)
  plt.show()
  ```

- **Other Plots**:
  - **Box Plot** for numerical features.
  - **Pair Plot** for pairwise relationships.

## Save Preprocessed Data and Model
- **Save Data**: Save the cleaned and preprocessed data for future use.
  ```python
  df.to_csv('cleaned_data.csv', index=False)
  ```
- **Save Model**: Save the trained model for deployment.
  ```python
  import joblib
  joblib.dump(model, 'model.pkl')
  ```

## Installation
To use the code in this repository, follow these steps:

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/bikuu100/Loan_elgibility_check.git
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
### EDA
1. Open the notebook `Exploratory_Data_Analysis.ipynb` to explore the dataset.
2. Run the notebook cells to perform exploratory data analysis and visualize the data.

### Model Training
1. Open the notebook `Model_Training.ipynb` to train the decision tree classifier model.
2. Follow the instructions in the notebook to load the dataset, preprocess the data, train the model, and evaluate its performance.
3. The trained model will be saved in the `models` directory.

## Contributing
Contributions to this project are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License
This project is licensed under the [MIT License](LICENSE).

---
