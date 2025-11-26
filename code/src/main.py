import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   

# CONSTANTS
LABEL_COLUMN = "recovery_days"
NORMALIZATION_METHODS = ["min-max", "z-score", "decimal-scaling"]

def read_raw_data(file_path: str) -> pd.DataFrame:

    raw_data = pd.read_csv(file_path)
    return raw_data


def display_data_info(data: pd.DataFrame) -> None:
    print("Data Info:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nFirst 5 Rows:")
    print(data.head())



def extract_numerical_columns(data: pd.DataFrame) -> pd.DataFrame:
    numerical_data = data.select_dtypes(include=['number'])
    return numerical_data

def correlation_analysis(data: pd.DataFrame) -> None:
    correlation_matrix = data.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
def most_correlated_pairs(data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    numberical_data = extract_numerical_columns(data)
    correlation_matrix = numberical_data.corr().abs()
    upper_mask = np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
    upper_triangle = correlation_matrix.where(upper_mask)
    correlated_pairs = (
        upper_triangle.stack()
        .reset_index()
        .rename(columns={0: "correlation"})
        .query("correlation > @threshold")
    )
    return correlated_pairs


def get_features(rawdata: pd.DataFrame) -> pd.DataFrame:
    features = rawdata.drop(columns=[LABEL_COLUMN])
    return features

def plot_feature_distributions(data: pd.DataFrame) -> None:
    numerical_data = extract_numerical_columns(data)
    numerical_data.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Feature Distributions")
    plt.show()


def split_data_train_test_validation(data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15) -> tuple:
    # using sklearn's train_test_split for simplicity
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=val_size/(1-train_size), random_state=42)
    return train_data, val_data, test_data

def normalize_column(data: pd.Series, method = "min-max") -> pd.Series:
    if method == "min-max":
        normalized = (data - data.min()) / (data.max() - data.min())
    elif method == "z-score":
        normalized = (data - data.mean()) / data.std()
    elif method == "decimal-scaling":
        j = np.ceil(np.log10(data.abs().max()))
        normalized = data / (10 ** j)
    else:
        raise ValueError(f"Normalization method '{method}' not recognized.")
    return normalized
    


def processs_data(raw_data: pd.DataFrame, method = "min-max") -> pd.DataFrame:
    processed_data = raw_data.copy()
    for column in extract_numerical_columns(raw_data).columns:
        processed_data[column] = normalize_column(raw_data[column], method)
    # make the date numerical
    if 'date' in processed_data.columns:
        processed_data['date'] = pd.to_datetime(processed_data['date'], errors='coerce')  # invalid dates → NaT
        processed_data['date'] = processed_data['date'].astype('int64') // 10**9        # safe integer timestamps
        processed_data['date'] = processed_data['date'].astype('float32')               # PyTorch compatible

    # One hot encode categorical variables
    categorical_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns, drop_first=True)
    return processed_data

def plot_most_correlated_pairs(data: pd.DataFrame, threshold: float = 0.8) -> None:
    correlated_pairs = most_correlated_pairs(data, threshold)
    for correlated_pair in correlated_pairs.itertuples():
        feature1 = correlated_pair.level_0
        feature2 = correlated_pair.level_1
        plt.figure()
        sns.scatterplot(x=data[feature1], y=data[feature2])
        plt.title(f"Scatter Plot of {feature1} vs {feature2} (Correlation: {correlated_pair.correlation:.2f})")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def print_accuracy_linear_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")


def trai_decsion_tree_model(X_train: pd.DataFrame, y_train: pd.Series):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def print_accuracy_decision_tree_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")



if __name__ == "__main__":
    file_path = "data/raw/global_disaster_response.csv"
    raw_data = read_raw_data(file_path)
    processed_data = processs_data(raw_data, method=NORMALIZATION_METHODS[0])
    train_data, val_data, test_data = split_data_train_test_validation(processed_data)
    X_train = get_features(train_data)
    y_train = train_data[LABEL_COLUMN]
    model = train_linear_model(X_train, y_train)
    print("Model trained successfully.")
    X_test = get_features(test_data)
    y_test = test_data[LABEL_COLUMN]
    print_accuracy_linear_model(model, X_test, y_test)
    dt_model = trai_decsion_tree_model(X_train, y_train)
    print("Decision Tree Model trained successfully.")
    print_accuracy_decision_tree_model(dt_model, X_test, y_test)






