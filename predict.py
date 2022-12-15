# Import the necessary libraries for data analysis and visualization
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the survey data into a Pandas DataFrame
df = pd.read_csv('Mall_Customers.csv')

# Use Pandas to clean and preprocess the data
df = df.dropna()  # drop rows with missing values
df = df[df['Age'] > 0]  # drop rows with invalid age values

# Use seaborn to create a line plot
sns.lineplot(x='Purchase %', y='Age', data=df)
plt.show()

# Use Pandas to create a scatter plot showing the relationship between Purchase and Income
sns.scatterplot(x='Purchase %', y='Income', data=df)

# Show the plot
plt.show()

# Use scikit-learn to build a linear regression model to predict attitude from age, gender, and income
X = df[['Income']]
y = df['Purchase %']
plt.title('Purchasing %  Across different Age groups')
plt.xlabel('Income')
plt.ylabel('Purchase attitude')
model = LinearRegression()
model.fit(X, y)

# Evaluate the model's performance on the training data
print('R-squared:', model.score(X, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance on the test data
print('R-squared:', model.score(X_test, y_test))

# Use the model to make predictions and evaluate its performance
predictions = model.predict(X)
print('R-squared:', model.score(X, y))

# Plot the predictions as a line on top of the data points
plt.plot(X, predictions, color='red')

# Show the plot
plt.show()

