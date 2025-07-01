import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_and_preprocess_data(file_path):
    """
    Load the dataset from a CSV file and preprocess it.
    
    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    
    Returns:
    X (ndarray): The feature matrix after preprocessing.
    y (ndarray): The target vector.
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing values (if any)
    data.fillna(data.mean(), inplace=True)
    
    # Separate features and target
    X = data.drop(columns=['target'])  # Replace 'target' with the actual target column name
    y = data['target'].values  # Replace 'target' with the actual target column name
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def one_hot_encode(y):
    """
    One-hot encode the target vector.
    
    Parameters:
    y (ndarray): The target vector.
    
    Returns:
    y_encoded (ndarray): The one-hot encoded target matrix.
    """
    return np.eye(np.max(y) + 1)[y]

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    X (ndarray): The feature matrix.
    y (ndarray): The target vector.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    X_train (ndarray): Training feature matrix.
    X_test (ndarray): Testing feature matrix.
    y_train (ndarray): Training target vector.
    y_test (ndarray): Testing target vector.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)