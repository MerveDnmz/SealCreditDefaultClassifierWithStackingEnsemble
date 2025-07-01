import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input
from ucimlrepo import fetch_ucirepo
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, CKKSEncoder, Plaintext, Ciphertext, scheme_type, CoeffModulus
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import shap

## XGBoost/CNN Modeli
def seal_encrypt_data(data):
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(8192)
    parms.set_coeff_modulus(CoeffModulus.Create(8192, [60, 40, 40, 60]))
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    encoder = CKKSEncoder(context)
    scale = pow(2.0, 40)

    encrypted_data = []
    for row in data:
        row = np.array(row, dtype=np.float64).flatten()
        plain = encoder.encode(row, scale)
        encrypted_row = encryptor.encrypt(plain)
        encrypted_data.append(encrypted_row)

    return encrypted_data, secret_key, encoder, scale, context

def seal_decrypt_data(encrypted_data, secret_key, encoder, scale, context):
    decrypted_data = []
    decryptor = Decryptor(context, secret_key)
    for encrypted_row in encrypted_data:
        plain = Plaintext()
        decryptor.decrypt(encrypted_row, plain)
        decoded_row = np.array(encoder.decode(plain))
        decrypted_data.append(decoded_row)
    return np.array(decrypted_data)

def build_neural_network(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(512, activation='relu'))  # Daha fazla nöron
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

def build_cnn_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim, 1)))  # 1D Convolution için giriş şekli
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    return model

def focal_loss(alpha=0.25, gamma=2):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return loss

def neural_network_with_seal():
    # Load dataset
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets.to_numpy().ravel()

    # --- Özellik Analizi ve Gelişmiş Feature Engineering ---
    corr_with_target = X.corrwith(pd.Series(y, index=X.index)).abs().sort_values(ascending=False)
    top2 = corr_with_target.index[:2]
    X['feature_ratio'] = X[top2[0]] / (X[top2[1]] + 1e-5)
    X['feature_product'] = X[top2[0]] * X[top2[1]]
    X['feature_diff'] = X[top2[0]] - X[top2[1]]

    # Logaritmik özellikler (pozitif olanlar için)
    for col in X.columns:
        if (X[col] > 0).all():
            X[f'log_{col}'] = np.log1p(X[col])

    # Toplam, ortalama, std
    X['feature_sum'] = X.sum(axis=1)
    X['feature_mean'] = X.mean(axis=1)
    X['feature_std'] = X.std(axis=1)

    # --- Feature Selection: En iyi 15 özellik ---
    rf_fs = RandomForestClassifier(random_state=42)
    rf_fs.fit(X, y)
    importances = rf_fs.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15
    top_features = X.columns[indices[:top_n]]
    print("En iyi 15 özellik:", list(top_features))
    X = X[top_features].copy()

    # --- SMOTE ile Oversampling ---
    print("\n--- SMOTE ile Oversampling ---")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"X_resampled shape: {X_resampled.shape}")
    print(f"y_resampled shape: {y_resampled.shape}")

    # --- Veri Setini Bölme ---
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- XGBoost Hiperparametre Optimizasyonu ---
    print("\n--- XGBoost Hiperparametre Optimizasyonu ---")
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid_xgb, scoring='f1', cv=3, n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    print("XGBoost en iyi parametreler:", grid_xgb.best_params_)
    y_pred_xgb_grid = grid_xgb.predict(X_test)
    print("XGBoost GridSearch Classification Report:")
    print(classification_report(y_test, y_pred_xgb_grid))

    # --- RandomForest Hiperparametre Optimizasyonu ---
    print("\n--- RandomForest Hiperparametre Optimizasyonu ---")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, scoring='f1', cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    print("RandomForest en iyi parametreler:", grid_rf.best_params_)
    y_pred_rf_grid = grid_rf.predict(X_test)
    print("RandomForest GridSearch Classification Report:")
    print(classification_report(y_test, y_pred_rf_grid))

    # --- Sınıf Ağırlıkları ile Neural Network ---
    print("\n--- Neural Network ---")
    X_train_nn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_nn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = build_cnn_model(X_train_nn.shape[1])
    class_weights = {0: 1, 1: 10}  # Sınıf 1'e daha fazla ağırlık verin
    model.compile(loss=focal_loss(alpha=0.5, gamma=2), optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_nn, y_train, epochs=20, batch_size=32, verbose=1, class_weight=class_weights)

    # Model tahminleri
    y_pred = model.predict(X_test_nn)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal threshold:", optimal_threshold)
    y_pred_classes = (y_pred > optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred_classes)
    confusion = confusion_matrix(y_test, y_pred_classes)
    report = classification_report(y_test, y_pred_classes)
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(report)

    # --- Feature Selection: En iyi 10 özellik ---
    print("\n--- Feature Selection (RandomForest) ---")
    rf_fs = RandomForestClassifier(random_state=42)
    rf_fs.fit(X_train, y_train)
    importances = rf_fs.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_features = indices[:top_n]
    print("En iyi 10 özellik indexleri:", top_features)
    X_train_top = X_train[:, top_features]
    X_test_top = X_test[:, top_features]

    # --- XGBoost Hiperparametre Optimizasyonu ---
    print("\n--- XGBoost Hiperparametre Optimizasyonu ---")
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss', random_state=42), param_grid_xgb, scoring='f1', cv=3, n_jobs=-1)
    grid_xgb.fit(X_train_top, y_train)
    print("XGBoost en iyi parametreler:", grid_xgb.best_params_)
    y_pred_xgb_grid = grid_xgb.predict(X_test_top)
    print("XGBoost GridSearch Classification Report:")
    print(classification_report(y_test, y_pred_xgb_grid))

    # --- RandomForest Hiperparametre Optimizasyonu ---
    print("\n--- RandomForest Hiperparametre Optimizasyonu ---")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7]
    }
    grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, scoring='f1', cv=3, n_jobs=-1)
    grid_rf.fit(X_train_top, y_train)
    print("RandomForest en iyi parametreler:", grid_rf.best_params_)
    y_pred_rf_grid = grid_rf.predict(X_test_top)
    print("RandomForest GridSearch Classification Report:")
    print(classification_report(y_test, y_pred_rf_grid))

    # --- Stacking Ensemble (LightGBM eklendi) ---
    print("\n--- Stacking Ensemble ---")
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    estimators = [
        ('rf', grid_rf.best_estimator_),
        ('xgb', grid_xgb.best_estimator_),
        ('lgbm', LGBMClassifier(random_state=42))
    ]
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
    stacking.fit(X_train_top, y_train)
    y_pred_stack = stacking.predict(X_test_top)
    print("Stacking Ensemble Classification Report:")
    print(classification_report(y_test, y_pred_stack))

    # --- Threshold Optimizasyonu (Stacking için) ---
    y_pred_stack_proba = stacking.predict_proba(X_test_top)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_stack_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Stacking Ensemble Optimal threshold:", optimal_threshold)
    y_pred_stack_opt = (y_pred_stack_proba > optimal_threshold).astype(int)
    print("Stacking Ensemble (Optimal Threshold) Classification Report:")
    print(classification_report(y_test, y_pred_stack_opt))

    # --- SHAP ile Model Açıklanabilirliği (XGBoost) ---
    print("\n--- SHAP ile Model Açıklanabilirliği (XGBoost) ---")
    explainer = shap.TreeExplainer(grid_xgb.best_estimator_)
    shap_values = explainer.shap_values(X_test_top)
    shap.summary_plot(shap_values, X_test_top, feature_names=[f"f{i}" for i in top_features], show=False)
    plt.savefig("shap_summary.png")
    print("SHAP summary plot 'shap_summary.png' olarak kaydedildi.")

if __name__ == "__main__":
    neural_network_with_seal()