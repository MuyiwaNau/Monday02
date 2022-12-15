import unittest
import pandas as pd
import predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TestPredict(unittest.TestCase):
    def setUp(self):
        # Load the survey data into a Pandas DataFrame
        self.df = pd.read_csv('Mall_Customers.csv')

        # Use Pandas to clean and preprocess the data
        self.df = self.df.dropna()  # drop rows with missing values
        self.df = self.df[self.df['Age'] > 0]  # drop rows with invalid age values

        # Use scikit-learn to build a linear regression model
        X = self.df[['Income']]
        y = self.df['Purchase %']
        self.model = LinearRegression()

    def test_predict(self):
        # Evaluate the model's performance on the training data
        X = self.df[['Income']]
        y = self.df['Purchase %']

        # Train the model on the input data
        self.model.fit(X, y)

        # Use the model to make predictions on the input data
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        print("Mean squared error:", mse)
        print("Mean absolute error:", mae)
        self.assertTrue(mse < 1000)
        self.assertTrue(mae < 100)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fit the model on the training data and evaluate its performance on the test data
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean squared error:", mse)
        print("Mean absolute error:", mae)
        self.assertTrue(mse < 1000)
        self.assertTrue(mae < 100)


if __name__ == '__main__':
    unittest.main()
