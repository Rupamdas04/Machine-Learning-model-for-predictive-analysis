Monthly Sales Prediction using Machine Learning
This project involves predicting future monthly sales using machine learning. We train a linear regression model on past sales data, perform feature engineering, and visualize the actual vs. predicted sales to evaluate the model's performance.

Table of Contents
Introduction
Installation
Usage
Results
Contributing
License
Introduction
Predicting future sales is crucial for businesses to plan their inventory, marketing strategies, and financial forecasts. This project demonstrates how to use machine learning to predict future monthly sales based on historical sales data.

Installation
Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/sales-prediction.git
cd sales-prediction
Install the required libraries:

sh
Copy code
pip install -r requirements.txt
Usage
Ensure your dataset (monthly_sales.csv) is in the same directory as the script.
Run the script:
sh
Copy code
python sales_prediction.py
The script will load the dataset, preprocess the data, train a linear regression model, and visualize the actual vs. predicted sales.

Results
The script outputs the following evaluation metrics for the linear regression model:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R2 Score
It also generates a plot comparing the actual sales with the predicted sales for the last 13 months in the dataset.

Contributing
Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.
