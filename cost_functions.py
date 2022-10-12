import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('data\salary_data.csv')

x = df['YearsExperience']
y = df['Salary']

y_predicted = 12838.39 * x + 2915.65  # The values used are the results of the gradient descent algorithm.
test = y - y_predicted
print(abs(test))


# cost functions : mae, mse, rmse

def calculate_mae(y, y_predicted):
    """ Calculates mean absolute (L1) error.

        Formula:
        MAE =  1/n *  Σ|y-y_predicted|   (n = #of observation)

        Advantages :
        * Robust to outliers.
        * Error in the same unit, and it is easy to interpret.

        Disadvantages:
        * MAE is not differentiable at zero, it is problem for the optimization algorithms.

        Python alternatives : sklearn.metrics.mean_absolute_error(y_true, y_pred)
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
    """

    errors = y - y_predicted
    absolute_errors = abs(errors)
    mean_absolute_error = np.mean(absolute_errors)
    return mean_absolute_error


def calculate_mse(y, y_predicted):
    """ Calculates mean squared (L2) error.

        Formula:
        MSE =  1/n *  Σ(y-y_predicted)^2   (n = #of observation)

        Advantages :
        * The equation is differentiable, easily converge with Gradient Descent
        * Not time-consuming

        Disadvantages:
        * Error in squared unit, not easy to interpret.
        * Not robust to outliers.
        * Penalized the error by taking square.

        Python alternatives : sklearn.metrics.mean_squared_error(y_true, y_pred)
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mse
    """

    errors = y - y_predicted
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)
    return mean_squared_error


def calculate_rmse(y, y_predicted):
    """ Calculates root mean squared error.

                Formula:
                RMSE =  1/n *  √(Σ(y-y_predicted)^2)   (n = #of observation)

                Advantages :
                * The equation is differentiable, and easily converges with optimization algorithms.
                * Error in same unit, and easy to interpret.
                * RMSE does not penalize the errors as much as MSE does due to the square root.

                Disadvantages:
                * Sensitive to outliers.

                Python alternatives : sklearn.metrics.mean_squared_error(y_true, y_pred)
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=mse

                NOT : squared : bool, default=True
                      If True returns MSE value, if False returns RMSE value.
            """

    errors = y - y_predicted
    squared_errors = errors ** 2
    mean_squared_error = np.mean(squared_errors)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error


def calculate_r2(y, y_predicted):
    y_bar = np.mean(y)

    # total sum of squares - tss
    tss = sum((y - y_bar) ** 2)

    # residual sum of square - rss
    rss = sum((y - y_predicted) ** 2)

    # r2 = 1-rss/tss
    r2 = 1 - rss / tss

    return r2


def calculate_adj_r2(y, y_predicted, num_of_predictor=1):
    y_bar = np.mean(y)
    n = len(y)
    p = num_of_predictor

    # total sum of squares - tss
    tss = sum((y - y_bar) ** 2)

    # residual sum of square - rss
    rss = sum((y - y_predicted) ** 2)

    # adj r2 = 1-(rss/tss)*(n-1/n-p-1)

    adj_r2 = 1 - (rss / tss) * ((n - 1) / (n - p - 1))

    return adj_r2


print("MAE : {} ".format(calculate_mae(y, y_predicted)))
print("MSE : {} ".format(calculate_mse(y, y_predicted)))
print("RMSE : {} ".format(calculate_rmse(y, y_predicted)))
print("R2 : {} ".format(calculate_r2(y, y_predicted)))
print("Adjusted R2 : {} ".format(calculate_adj_r2(y, y_predicted)))

# print mae, mse, rmse, r2, adj_r2 with sklearn library
print("sklearn MAE : {} ".format(mean_absolute_error(y, y_predicted)))
print("sklearn MSE : {} ".format(mean_squared_error(y, y_predicted)))
print("sklearn RMSE : {} ".format(mean_squared_error(y, y_predicted, squared=False)))
print("sklearn R2 : {} ".format(r2_score(y, y_predicted)))
# for adj_r2 ==> Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)
