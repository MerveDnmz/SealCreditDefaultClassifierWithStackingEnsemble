import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

class FederatedLearningWithCNN:
    def __init__(self, num_clients=5):
        self.num_clients = num_clients
        self.client_data = []
        self.global_model = None
        self.scaler = MinMaxScaler()
        self.optimal_threshold = None

    def load_data(self):
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets.to_numpy().ravel()
        return X, y

    def preprocess_data(self, X, y):
        # SMOTE ile oversampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"X_resampled shape: {X_resampled.shape}")
        print(f"y_resampled shape: {y_resampled.shape}")

        # Veri setini bölme ve normalizasyon
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Veriyi federated learning için parçalara ayırma
        client_data_size = len(X_train) // self.num_clients
        self.client_data = [
            (X_train[i * client_data_size:(i + 1) * client_data_size],
             y_train[i * client_data_size:(i + 1) * client_data_size])
            for i in range(self.num_clients)
        ]
        return X_test, y_test

    def build_cnn_model(self, input_dim):
        model = Sequential()
        model.add(Input(shape=(input_dim, 1)))
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_federated_model(self):
        # Global model oluşturma
        input_dim = self.client_data[0][0].shape[1]
        self.global_model = self.build_cnn_model(input_dim)

        # Her bir client için yerel eğitim
        for client_idx, (X_client, y_client) in enumerate(self.client_data):
            print(f"Client {client_idx + 1} için eğitim başlıyor...")
            X_client = X_client.reshape((X_client.shape[0], X_client.shape[1], 1))
            local_model = self.build_cnn_model(input_dim)
            local_model.set_weights(self.global_model.get_weights())  # Global model ağırlıklarını başlat
            local_model.fit(X_client, y_client, epochs=5, batch_size=32, verbose=1)
            self.global_model.set_weights([
                np.mean([local_model.get_weights()[i] for local_model in self.client_data], axis=0)
                for i in range(len(self.global_model.get_weights()))
            ])
        print("Federated Learning tamamlandı.")

    def evaluate_model(self, X_test, y_test):
        # Test verisi üzerinde değerlendirme
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        y_pred_proba = self.global_model.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = thresholds[optimal_idx]
        print("Optimal threshold:", self.optimal_threshold)
        y_pred_opt = (y_pred_proba > self.optimal_threshold).astype(int)
        print("Federated Learning (Optimal Threshold) Classification Report:")
        print(classification_report(y_test, y_pred_opt))

if __name__ == "__main__":
    classifier = FederatedLearningWithCNN()
    X, y = classifier.load_data()
    X_test, y_test = classifier.preprocess_data(X, y)
    classifier.train_federated_model()
    classifier.evaluate_model(X_test, y_test)