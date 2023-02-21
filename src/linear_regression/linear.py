import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def plot_selected_features(x: pd.DataFrame, y):
    for col in x.columns:
        plt.scatter(x[col], y)
        plt.savefig(f"linear_regression/plots/{col}_to_progress.jpeg")
        plt.close()


def plot_histograms(x: pd.DataFrame, normalized=False):
    name = "normalized_hist.jpeg" if normalized else "hist.jpeg"
    for col in x.columns:
        plt.hist(x[col], bins=50)
        plt.savefig(f"linear_regression/plots/{col}_{name}")
        plt.close()


def normalize(x_train, x_test):
    parameters = {}
    for col in x_train.columns:
        mean = x_train[col].mean()
        std = x_train[col].std()
        parameters[col] = {
            "mean": mean,
            "std": std,
        }
        x_train[col] = (x_train[col] - mean) / std
        x_test[col] = (x_test[col] - mean) / std

    return x_train, x_test, parameters


def main():
    # load and splitting the dataset
    dataset = load_diabetes()
    x, y = pd.DataFrame(dataset.data, columns=dataset.feature_names), pd.DataFrame(dataset.target, columns=["progress"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # plot relation of x to y
    plot_selected_features(x_train, y_train)
    plot_histograms(x_train)

    # print(x.describe())

    # Create linear regression object
    model = linear_model.LinearRegression()
    # Train the model using the training sets
    model.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred = model.predict(x_test)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # # Plot outputs
    # plt.scatter(x_test[["bmi"]].reset_index(drop=True), y_test.reset_index(drop=True), color="black")
    # plt.plot(x_test[["bmi"]].reset_index(drop=True), x_test[["bmi"]].reset_index(drop=True) * model.coef_[0][2], color="blue", linewidth=3)
    #
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()


if __name__ == '__main__':
    main()
