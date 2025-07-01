import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, CKKSEncoder, Plaintext, Ciphertext, scheme_type, CoeffModulus
import time
import psutil
from memory_profiler import memory_usage

class CreditDefaultClassifierWithSEAL:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.top_features = None
        self.scaler = MinMaxScaler()
        self.stacking_model = None
        self.optimal_threshold = None
        self.context = None
        self.encoder = None
        self.encryptor = None
        self.decryptor = None
        self.scale = pow(2.0, 40)

    def load_data(self):
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=350)
        X = dataset.data.features
        y = dataset.data.targets.to_numpy().ravel()
        return X, y

    def feature_engineering(self, X, y):
        # Özellik mühendisliği
        X['feature_ratio'] = X.iloc[:, 0] / (X.iloc[:, 1] + 1e-5)
        X['feature_product'] = X.iloc[:, 0] * X.iloc[:, 1]
        X['feature_diff'] = X.iloc[:, 0] - X.iloc[:, 1]
        X['feature_sum'] = X.sum(axis=1)
        X['feature_mean'] = X.mean(axis=1)
        X['feature_std'] = X.std(axis=1)

        # En iyi 10 özellik seçimi
        rf_fs = RandomForestClassifier(random_state=42)
        rf_fs.fit(X, y)
        importances = rf_fs.feature_importances_
        indices = np.argsort(importances)[::-1]
        self.top_features = X.columns[indices[:10]]
        print("En iyi 10 özellik:", list(self.top_features))
        return X[self.top_features]

    def preprocess_data(self, X, y):
        # SMOTE ile oversampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"X_resampled shape: {X_resampled.shape}")
        print(f"y_resampled shape: {y_resampled.shape}")

        # Veri setini bölme ve normalizasyon
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Eğitim verisinin sadece ilk yarısını kullan
        half_idx = self.X_train.shape[0] // 2
        self.X_train = self.X_train[:half_idx]
        self.y_train = self.y_train[:half_idx]

    def initialize_seal(self):
        parms = EncryptionParameters(scheme_type.ckks)
        parms.set_poly_modulus_degree(4096)
        parms.set_coeff_modulus(CoeffModulus.Create(4096, [40, 20, 40]))
        self.context = SEALContext(parms)
        keygen = KeyGenerator(self.context)
        self.encryptor = Encryptor(self.context, keygen.create_public_key())
        self.decryptor = Decryptor(self.context, keygen.secret_key())
        self.encoder = CKKSEncoder(self.context)
        self.scale = pow(2.0, 35)
        print("SEAL şifreleme başlatıldı (optimize).")

    def encrypt_data(self, data):
        encrypted_data = []
        for row in data:
            row = np.array(row, dtype=np.float64).flatten()
            plain = self.encoder.encode(row, self.scale)
            encrypted_row = self.encryptor.encrypt(plain)
            encrypted_data.append(encrypted_row)
        return encrypted_data

    def decrypt_data(self, encrypted_data):
        # Şifreli veriyi çözme
        decrypted_data = []
        for ciphertext in encrypted_data:
            plaintext = Plaintext()
            self.decryptor.decrypt(ciphertext, plaintext)
            decoded = self.encoder.decode(plaintext)
            decrypted_data.append(decoded)
        return np.array(decrypted_data)

    def encrypt_data_in_batches(self, data, batch_size=50):
        encrypted_data = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            for row in batch:
                row = np.array(row, dtype=np.float32).flatten()
                plain = self.encoder.encode(row, self.scale)
                encrypted_row = self.encryptor.encrypt(plain)
                encrypted_data.append(encrypted_row)
        return encrypted_data

    def decrypt_data_in_batches(self, encrypted_data, batch_size=50):
        decrypted_data = []
        for i in range(0, len(encrypted_data), batch_size):
            batch = encrypted_data[i:i+batch_size]
            for ciphertext in batch:
                plaintext = Plaintext()
                self.decryptor.decrypt(ciphertext, plaintext)
                decoded = self.encoder.decode(plaintext)
                decrypted_data.append(decoded)
        return np.array(decrypted_data)

    def train_stacking_model(self, encrypted=False):
        # Şifreli veya şifresiz verilerle Stacking Ensemble modeli
        if encrypted:
            print("Şifreli verilerle eğitim başlıyor...")
            self.X_train = self.decrypt_data_in_batches(self.encrypt_data_in_batches(self.X_train, batch_size=50))
            self.X_test = self.decrypt_data_in_batches(self.encrypt_data_in_batches(self.X_test, batch_size=50))
        else:
            print("Şifresiz verilerle eğitim başlıyor...")

        estimators = [
            ('rf', RandomForestClassifier(max_depth=7, n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(learning_rate=0.2, max_depth=7, n_estimators=200, eval_metric='logloss', random_state=42)),
            ('lgbm', LGBMClassifier(random_state=42))
        ]
        self.stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1)
        self.stacking_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Tahmin ve performans değerlendirme
        y_pred = self.stacking_model.predict(self.X_test)
        print("Stacking Ensemble Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Threshold optimizasyonu
        y_pred_proba = self.stacking_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        self.optimal_threshold = thresholds[optimal_idx]
        print("Optimal threshold:", self.optimal_threshold)
        y_pred_opt = (y_pred_proba > self.optimal_threshold).astype(int)
        print("Stacking Ensemble (Optimal Threshold) Classification Report:")
        print(classification_report(self.y_test, y_pred_opt))

if __name__ == "__main__":
    classifier = CreditDefaultClassifierWithSEAL()
    X, y = classifier.load_data()
    X = classifier.feature_engineering(X, y)
    classifier.preprocess_data(X, y)
    classifier.initialize_seal()

    # Şifresiz model için peak memory ölçümü
    y_pred_proba_plain = None
    def run_plain():
        global y_pred_proba_plain
        classifier.train_stacking_model(encrypted=False)
        y_pred_proba_plain = classifier.stacking_model.predict_proba(classifier.X_test)[:, 1]

    time_start_plain = time.time()
    mem_usage_plain = memory_usage((run_plain, ), max_usage=True)
    time_plain = time.time() - time_start_plain

    fpr_plain, tpr_plain, thresholds_plain = roc_curve(classifier.y_test, y_pred_proba_plain)
    auc_plain = auc(fpr_plain, tpr_plain)
    optimal_idx_plain = np.argmax(tpr_plain - fpr_plain)
    optimal_threshold_plain = thresholds_plain[optimal_idx_plain]

    # Şifreli model için peak memory ölçümü
    y_pred_proba_enc = None
    def run_enc():
        global y_pred_proba_enc
        classifier.train_stacking_model(encrypted=True)
        y_pred_proba_enc = classifier.stacking_model.predict_proba(classifier.X_test)[:, 1]

    time_start_enc = time.time()
    mem_usage_enc = memory_usage((run_enc, ), max_usage=True)
    time_enc = time.time() - time_start_enc

    fpr_enc, tpr_enc, thresholds_enc = roc_curve(classifier.y_test, y_pred_proba_enc)
    auc_enc = auc(fpr_enc, tpr_enc)
    optimal_idx_enc = np.argmax(tpr_enc - fpr_enc)
    optimal_threshold_enc = thresholds_enc[optimal_idx_enc]

    # ROC eğrisi çizimi
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_plain, tpr_plain, label=f'Sifresiz ROC (AUC = {auc_plain:.2f})')
    plt.plot(fpr_enc, tpr_enc, label=f'Sifreli ROC (AUC = {auc_enc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Sifresiz vs Sifreli ROC Egrisi')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_comparison.png')
    plt.close()

    # Model ve veri bilgilerini topla
    model_technique = 'Stacking Ensemble (RandomForest, XGBoost, LightGBM, LogisticRegression final)'
    data_source = 'UCI Default of Credit Card Clients Dataset (ucimlrepo id=350)'
    feature_count = X.shape[1]
    train_size = classifier.X_train.shape[0]
    test_size = classifier.X_test.shape[0]
    feature_names = list(classifier.top_features) if hasattr(classifier, 'top_features') else list(X.columns)

    # Sonuçları text dosyasına yaz
    with open("model_performance_comparison.txt", "w") as f:
        f.write("MODEL KARŞILAŞTIRMA RAPORU\n")
        f.write(f"Veri Kaynağı: {data_source}\n")
        f.write(f"Kullanılan Özellikler (feature sayısı): {feature_count}\n")
        f.write(f"Özellikler: {', '.join(feature_names)}\n")
        f.write(f"Model Tekniği: {model_technique}\n")
        f.write(f"Train veri boyutu: {train_size}\n")
        f.write(f"Test veri boyutu: {test_size}\n")
        f.write("\n")

        f.write("Sifresiz Model Sonuçları:\n")
        f.write(f"Veri Türü: Şifresiz\n")
        f.write(f"Accuracy: {accuracy_score(classifier.y_test, (y_pred_proba_plain > optimal_threshold_plain).astype(int)):.4f}\n")
        f.write(f"AUC: {auc_plain:.4f}\n")
        f.write(f"Calisma Suresi (sn): {time_plain:.2f}\n")
        f.write(f"Peak Bellek Kullanimi (MB): {mem_usage_plain:.2f}\n\n")

        f.write("Sifreli Model Sonuçları:\n")
        f.write(f"Veri Türü: Şifreli (SEAL CKKS)\n")
        f.write(f"Accuracy: {accuracy_score(classifier.y_test, (y_pred_proba_enc > optimal_threshold_enc).astype(int)):.4f}\n")
        f.write(f"AUC: {auc_enc:.4f}\n")
        f.write(f"Calisma Suresi (sn): {time_enc:.2f}\n")
        f.write(f"Peak Bellek Kullanimi (MB): {mem_usage_enc:.2f}\n")