# Import the necessary libraries
# Import the necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the survey data into a Pandas DataFrame
df = pd.read_csv('Mall_Customers.csv')

# Use Pandas to clean and preprocess the data
df = df.dropna()  # drop rows with missing values
df = df[df['Age'] > 0]  # drop rows with invalid age values

# Use Pandas to create a bar chart showing the distribution of age groups

df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 25, 35, 45, 55, 65, 100],
                         labels=['0-18', '18-25', '25-35', '35-45', '45-55', '55-65', '65+'])
                         
age_group_counts = df.groupby('age_group')['CustomerID'].count()
age_group_counts.plot.bar()
plt.xlabel('Age Group')
plt.ylabel('Number of Responds')
plt.title('Distribution of Age Groups in the Survey')

# Use Pandas to create a bar chart showing the distribution of percentage groups
df['purchase_group'] = pd.cut(df['Purchase %'], bins=[0, 15, 30, 45, 60, 75, 85, 100],
                              labels=['0-15', '15-30', '30-45', '45-60', '60-75', '75-85', '85+'])
age_group_counts = df.groupby('purchase_group')['CustomerID'].count()
age_group_counts.plot.bar()
plt.xlabel('Purchase % group')
plt.ylabel('Number of Customers')
plt.title('Distribution of Purchase % Groups in the Survey')

# Generate summary statistics for the survey data
df.describe()

# Create a bar chart that shows the distribution of Purchasing % across different Age groups)
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', hue='purchase_group', data=df)
plt.title('Purchasing %  Across different Age groups')
plt.xlabel('Ages')
plt.ylabel('Purchaser %')
plt.show()

# Import the necessary libraries
from sklearn.linear_model import LinearRegression, LogisticRegression

# Define the features and target variable
# X = df[['age_group', 'Income', 'Gender']]
# y = df['purchase_group']

# Define the features and target variable
X = pd.get_dummies(df[['Age', 'Income', 'Gender']])
y = df['purchase_group'].apply(lambda x: 1 if x == '85+' else 0)

# Fit the classification model
model = LogisticRegression()
model.fit(X, y)

# Fit the regression model
# model = LinearRegression()
# model.fit(X, y)

# Print the coefficients
# print(model.coef_)

# Print the coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
