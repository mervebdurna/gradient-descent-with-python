import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data\salary_data.csv')

x = df['YearsExperience']
y = df['Salary']


def gradient_descent(x, y, learning_rate=0.001, num_of_iterations=100):
    m_current = 0
    b_current = 0
    n = len(x)
    for i in range(num_of_iterations):
        # y = mx + b
        y_predicted = m_current * x + b_current

        cost_mse = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])

        # derivative of m => -2/n * sum(x*(y - y_predicted))
        # derivative of b => -2/n * sum(y - y_predicted)

        m_delta = -(2 / n) * sum(x * (y - y_predicted))
        b_delta = -(2 / n) * sum(y - y_predicted)

        # update current m and b values
        m_current = m_current - learning_rate * m_delta
        b_current = b_current - learning_rate * b_delta

        sns.scatterplot(x=x, y=y)
        sns.lineplot(x=x, y=m_current*x + b_current)

        print("m {},    b {},    cost {},   iteration {}".format(m_current, b_current, cost_mse, i))
    plt.show()


def main():
    gradient_descent(x, y)


if __name__ == '__main__':
    main()
