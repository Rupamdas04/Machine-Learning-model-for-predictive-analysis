import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/train.csv', parse_dates=['date'])
data.set_index('date', inplace=True)


# Feature Engineering: Creating a supervised learning problem
data['sales_diff'] = data['sales'].diff()
data.dropna(inplace=True)

def create_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# Creating supervised data for the last 12 months
supervised_data = create_supervised(data['sales_diff'], 12)

# Splitting data into train and test sets
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]

# Scaling features
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Splitting data into input and output
x_train, y_train = train_data_scaled[:, 1:], train_data_scaled[:, 0]
x_test, y_test = test_data_scaled[:, 1:], test_data_scaled[:, 0]

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_predict_scaled = lr_model.predict(x_test)

# Inverse transform to original scale
lr_predict = scaler.inverse_transform(np.concatenate((lr_predict_scaled.reshape(-1, 1), x_test), axis=1))[:, 0]

# Calculating predicted sales
last_actual_sales = data['sales'].values[-13:-1]
predicted_sales = lr_predict + last_actual_sales

# Model Evaluation
lr_mse = mean_squared_error(last_actual_sales, predicted_sales, squared=False)
lr_mae = mean_absolute_error(last_actual_sales, predicted_sales)
lr_r2 = r2_score(last_actual_sales, predicted_sales)

print(f"Linear Regression RMSE: {lr_mse}")
print(f"Linear Regression MAE: {lr_mae}")
print(f"Linear Regression R2 Score: {lr_r2}")

# Plotting actual sales vs predicted sales
plt.figure(figsize=(15, 5))
plt.plot(data.index[-13:], data['sales'].values[-13:], label='Actual Sales')
plt.plot(data.index[-12:], predicted_sales, label='Predicted Sales', linestyle='--')
plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()